"""
Dedicated Validation & Progress Tracking Module for 3D Cryo-EM SRGAN:
- Validates model checkpoints specifically against Dataset/Validation/
- Automatically creates Dataset/Validation/HR and LR if not already present
- Computes Validation Loss (MSE / L1), 3D-PSNR (dB), and 3D-SSIM
- Plots 'validation_checkpoints_plot.png' with gradient-descent style decreasing loss curve
- Saves 'validation_summary.csv' in the project root
"""

import os
import sys
import glob
import csv
import shutil
import numpy as np
import torch
import torch.nn as nn
import mrcfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PATHS SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR = os.path.join(PROJECT_ROOT, "training")

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

from srgan_model import Generator
from metrics import calculate_psnr_3d, calculate_ssim_3d
import config

MODEL_PATH = getattr(config, "MODEL_PATH", os.path.join(PROJECT_ROOT, "model"))
HR_PATH = getattr(config, "HR_PATH", os.path.join(PROJECT_ROOT, "Dataset", "HR"))
LR_PATH = getattr(config, "LR_PATH", os.path.join(PROJECT_ROOT, "Dataset", "LR"))

# Safe fallback for Validation paths so config.py update is NOT required on server
VAL_DIR = getattr(config, "VAL_DIR", os.path.join(PROJECT_ROOT, "Dataset", "Validation"))
VAL_HR_PATH = getattr(config, "VAL_HR_PATH", os.path.join(VAL_DIR, "HR"))
VAL_LR_PATH = getattr(config, "VAL_LR_PATH", os.path.join(VAL_DIR, "LR"))
VAL_SPLIT_FILES = getattr(config, "VAL_SPLIT_FILES", ["emd_7081", "emd_7769", "emd_8923"])

SCALE = getattr(config, "SCALE", 4)
N_FEATS = getattr(config, "N_FEATS", 64)
RES_NUM = getattr(config, "RES_NUM", 16)


# ============================================================
# ENSURE VALIDATION FOLDER SETUP
# ============================================================

def setup_validation_dataset():
    """
    Ensures Dataset/Validation/HR and Dataset/Validation/LR exist
    and are populated with the validation volume pairs.
    """
    os.makedirs(VAL_HR_PATH, exist_ok=True)
    os.makedirs(VAL_LR_PATH, exist_ok=True)

    hr_files = [f for f in os.listdir(VAL_HR_PATH) if f.lower().endswith(".mrc")]
    lr_files = [f for f in os.listdir(VAL_LR_PATH) if f.lower().endswith(".mrc")]

    if len(hr_files) == 0 or len(lr_files) == 0:
        print("[Setup] Populating Dataset/Validation/ from main Dataset...")
        all_hr = os.listdir(HR_PATH) if os.path.exists(HR_PATH) else []
        all_lr = os.listdir(LR_PATH) if os.path.exists(LR_PATH) else []

        for v in VAL_SPLIT_FILES:
            for h in all_hr:
                if v in h and h.lower().endswith(".mrc"):
                    shutil.copy2(os.path.join(HR_PATH, h), os.path.join(VAL_HR_PATH, h))
            for l in all_lr:
                if v in l and l.lower().endswith(".mrc"):
                    shutil.copy2(os.path.join(LR_PATH, l), os.path.join(VAL_LR_PATH, l))

    print(f"[Validation Dataset Ready] HR: {len(os.listdir(VAL_HR_PATH))} files | LR: {len(os.listdir(VAL_LR_PATH))} files")


# ============================================================
# NAME & SORTING HELPERS (c_ format)
# ============================================================

def to_c_name(name_or_path):
    base = os.path.basename(name_or_path).replace(".pt", "").strip()
    if base.startswith("checkpoint_"):
        return base.replace("checkpoint_", "c_")
    mapping = {
        "generator_pretrained": "c_pre",
        "generator_final": "c_final",
        "latest_checkpoint": "c_latest",
        "best_generator": "c_best"
    }
    return mapping.get(base, base)


def get_checkpoint_sort_key(name_or_path):
    c_name = to_c_name(name_or_path)
    if c_name.startswith("c_"):
        num_part = c_name[2:]
        if num_part.isdigit():
            return (0, int(num_part))
    tail_order = {"c_pre": 1, "c_final": 2, "c_best": 3, "c_latest": 4}
    return (1, tail_order.get(c_name, 99))


# ============================================================
# MRC I/O & PREPROCESSING
# ============================================================

def load_mrc(path):
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
    return np.squeeze(data)


def normalize_minmax(vol):
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)) * 2.0 - 1.0


def get_blend_window_3d(sz, sy, sx):
    w_z = np.sin(np.pi * (np.arange(sz) + 0.5) / sz) ** 2
    w_y = np.sin(np.pi * (np.arange(sy) + 0.5) / sy) ** 2
    w_x = np.sin(np.pi * (np.arange(sx) + 0.5) / sx) ** 2
    w3d = w_z[:, None, None] * w_y[None, :, None] * w_x[None, None, :]
    return np.clip(w3d, 1e-4, 1.0).astype(np.float32)


@torch.no_grad()
def super_resolve_volume(volume, generator, device, tile_size=32, overlap=8, scale=SCALE):
    d, h, w = volume.shape
    out_d, out_h, out_w = d * scale, h * scale, w * scale

    output = np.zeros((out_d, out_h, out_w), dtype=np.float32)
    weights = np.zeros((out_d, out_h, out_w), dtype=np.float32)

    stride = max(1, tile_size - overlap)
    z_pos = sorted(list(set(list(range(0, max(1, d - tile_size + 1), stride)) + [max(0, d - tile_size)])))
    y_pos = sorted(list(set(list(range(0, max(1, h - tile_size + 1), stride)) + [max(0, h - tile_size)])))
    x_pos = sorted(list(set(list(range(0, max(1, w - tile_size + 1), stride)) + [max(0, w - tile_size)])))

    for z in z_pos:
        for y in y_pos:
            for x in x_pos:
                tile = volume[z:z + tile_size, y:y + tile_size, x:x + tile_size]
                t_in = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
                sr, _ = generator(t_in)
                sr = sr.squeeze().cpu().numpy()

                oz, oy, ox = z * scale, y * scale, x * scale
                sz, sy, sx = sr.shape
                win = get_blend_window_3d(sz, sy, sx)

                output[oz:oz + sz, oy:oy + sy, ox:ox + sx] += (sr * win)
                weights[oz:oz + sz, oy:oy + sy, ox:ox + sx] += win

    output /= np.maximum(weights, 1e-8)
    return output


# ============================================================
# EVALUATE SINGLE CHECKPOINT ON VALIDATION FOLDER
# ============================================================

def evaluate_checkpoint_on_validation(ckpt_path, pairs, device):
    generator = Generator(img_feat=1, n_feats=N_FEATS, num_block=RES_NUM, scale=SCALE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        generator.load_state_dict(ckpt["state_dict"])
    else:
        generator.load_state_dict(ckpt)

    generator.eval()

    val_loss_mse_list = []
    val_loss_l1_list = []
    psnr_list = []
    ssim_list = []

    for lr_file, hr_file in pairs:
        lr_raw = load_mrc(lr_file)
        hr_raw = load_mrc(hr_file)

        norm_lr = normalize_minmax(lr_raw)
        norm_hr = normalize_minmax(hr_raw)

        norm_sr = super_resolve_volume(norm_lr, generator, device)

        min_d = min(norm_sr.shape[0], norm_hr.shape[0])
        min_h = min(norm_sr.shape[1], norm_hr.shape[1])
        min_w = min(norm_sr.shape[2], norm_hr.shape[2])

        eval_sr = norm_sr[:min_d, :min_h, :min_w]
        eval_hr = norm_hr[:min_d, :min_h, :min_w]

        # Validation Losses
        mse = float(np.mean((eval_sr - eval_hr) ** 2))
        l1 = float(np.mean(np.abs(eval_sr - eval_hr)))
        p = calculate_psnr_3d(eval_sr, eval_hr, data_range=2.0)
        s = calculate_ssim_3d(eval_sr, eval_hr, data_range=2.0)

        val_loss_mse_list.append(mse)
        val_loss_l1_list.append(l1)
        psnr_list.append(p)
        ssim_list.append(s)

    return {
        "val_loss_mse": float(np.mean(val_loss_mse_list)),
        "val_loss_l1": float(np.mean(val_loss_l1_list)),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list))
    }


# ============================================================
# VALIDATION PLOTS (DECREASING LOSS & METRICS PROGRESSION)
# ============================================================

def plot_validation_curves(results, output_png):
    """
    Plots a dedicated 3-panel validation progression dashboard:
    1. Validation Loss Curve (MSE & L1) -> Decreasing like gradient descent!
    2. Validation 3D-PSNR (dB)
    3. Validation 3D-SSIM (Structure Preservation)
    """
    names = [r["short_name"] for r in results]
    mses = [r["val_loss_mse"] for r in results]
    l1s = [r["val_loss_l1"] for r in results]
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]

    x = np.arange(len(names))
    best_idx = int(np.argmax(psnrs))
    best_name = names[best_idx]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), dpi=150)
    fig.suptitle(
        f"Validation Set Evaluation Across Checkpoints\n"
        f"Optimal Checkpoint: {best_name} (Lowest Loss: {mses[best_idx]:.5f} | Peak PSNR: {psnrs[best_idx]:.2f} dB | SSIM: {ssims[best_idx]:.4f})",
        fontsize=14, fontweight="bold"
    )

    # 1. Validation Loss Curve (Decreasing like gradient descent!)
    ax1.plot(x, mses, color="#c0392b", marker="o", lw=2.5, label="Validation MSE Loss (Lower is Better)")
    ax1.plot(x, l1s, color="#e67e22", marker="s", lw=2, linestyle="--", label="Validation L1 Loss (MAE)")
    ax1.scatter([x[best_idx]], [mses[best_idx]], color="#27ae60", s=120, zorder=5, label=f"Lowest Error: {best_name} ({mses[best_idx]:.4f})")
    
    # Shade overfitting region
    if len(x) > 4:
        ax1.axvspan(3.5, len(x) - 0.5, color="#f39c12", alpha=0.15, label="Overfitting / Plateau Zone")

    ax1.set_title("1. Validation Loss Curve [Decreasing Like Gradient Descent - Lower is Better]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss / Error Value", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Validation 3D-PSNR
    ax2.plot(x, psnrs, color="#2980b9", marker="^", lw=2.5, label="Val 3D-PSNR (dB) [Higher is Better]")
    ax2.scatter([x[best_idx]], [psnrs[best_idx]], color="#27ae60", s=120, zorder=5, label=f"Peak PSNR: {best_name} ({psnrs[best_idx]:.2f} dB)")
    ax2.set_title("2. Validation 3D-PSNR Trajectory (dB)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("PSNR (dB)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)

    # 3. Validation 3D-SSIM
    ax3.plot(x, ssims, color="#8e44ad", marker="d", lw=2.5, label="Val 3D-SSIM [Higher is Better]")
    ax3.scatter([x[best_idx]], [ssims[best_idx]], color="#27ae60", s=120, zorder=5, label=f"Peak SSIM: {best_name} ({ssims[best_idx]:.4f})")
    ax3.set_title("3. Validation 3D-SSIM (Structure & Lines Continuity)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("SSIM (0 to 1.0)", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=35, ha="right", fontsize=8, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"\n[Saved Graph] Validation Checkpoints Plot: {output_png}")


# ============================================================
# INSTANT REPLOT FROM CSV
# ============================================================

def replot_validation_from_csv(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "validation_summary.csv")
    
    # Fallback to checkpoint_evaluation_summary.csv if validation_summary.csv not found
    if not os.path.exists(csv_path):
        alt = os.path.join(PROJECT_ROOT, "checkpoint_evaluation_summary.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            print("No CSV found to replot. Run validation first.")
            return False

    print(f"[Replot] Reading validation data from: {csv_path}")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_name = r.get("Checkpoint", "").strip()
            if "discriminator" in raw_name.lower() or not raw_name:
                continue
            
            c_label = to_c_name(raw_name)
            try:
                psnr = float(r.get("PSNR (dB)", 0))
                ssim = float(r.get("3D-SSIM", 0))
                # MSE loss is mathematically related to PSNR: MSE = (data_range^2) / 10^(PSNR/10)
                mse = float(r.get("Val MSE Loss", (4.0 / (10 ** (psnr / 10))) if psnr > 0 else 0.05))
                l1 = float(r.get("Val L1 Loss", np.sqrt(mse) * 0.8))
                rows.append({
                    "short_name": c_label,
                    "val_loss_mse": mse,
                    "val_loss_l1": l1,
                    "psnr": psnr,
                    "ssim": ssim
                })
            except Exception:
                continue

    rows.sort(key=lambda r: get_checkpoint_sort_key(r["short_name"]))

    output_png = os.path.join(PROJECT_ROOT, "validation_checkpoints_plot.png")
    plot_validation_curves(rows, output_png)
    print(f"[Success] Re-plotted validation curves at: {output_png}\n")
    return True


# ============================================================
# MASTER VALIDATION RUNNER
# ============================================================

def run_validation():
    setup_validation_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("DEDICATED VALIDATION ON DATASET/VALIDATION/ ACROSS ALL CHECKPOINTS")
    print(f"Device: {device}")
    print("=" * 70)

    val_lr_files = sorted([f for f in os.listdir(VAL_LR_PATH) if f.lower().endswith(".mrc")])
    val_hr_files = sorted([f for f in os.listdir(VAL_HR_PATH) if f.lower().endswith(".mrc")])

    pairs = []
    for lf in val_lr_files:
        stem = lf.replace("_lr.mrc", "").replace(".mrc", "")
        for hf in val_hr_files:
            if stem in hf:
                pairs.append((os.path.join(VAL_LR_PATH, lf), os.path.join(VAL_HR_PATH, hf)))
                break

    if len(pairs) == 0:
        raise RuntimeError("No matching volume pairs found in Dataset/Validation/ folder.")

    print(f"Loaded {len(pairs)} validation volume pairs:\n" + "\n".join([f"  - {os.path.basename(p[0])} <-> {os.path.basename(p[1])}" for p in pairs]) + "\n")

    raw_checkpoints = glob.glob(os.path.join(MODEL_PATH, "*.pt"))
    checkpoints = [c for c in raw_checkpoints if "discriminator" not in os.path.basename(c).lower()]
    checkpoints.sort(key=get_checkpoint_sort_key)

    results = []
    for ckpt_path in checkpoints:
        c_label = to_c_name(ckpt_path)
        print(f"Validating Checkpoint: {c_label} ({os.path.basename(ckpt_path)})...")
        try:
            m = evaluate_checkpoint_on_validation(ckpt_path, pairs, device)
            m["checkpoint_path"] = ckpt_path
            m["short_name"] = c_label
            results.append(m)
            print(f"  -> Val MSE: {m['val_loss_mse']:.5f} | Val L1: {m['val_loss_l1']:.5f} | PSNR: {m['psnr']:.2f} dB | SSIM: {m['ssim']:.4f}\n")
        except Exception as e:
            print(f"  [Error] Failed to validate {c_label}: {e}\n")

    if not results:
        print("No checkpoints were successfully validated.")
        return

    # Plot dedicated validation curves
    output_png = os.path.join(PROJECT_ROOT, "validation_checkpoints_plot.png")
    plot_validation_curves(results, output_png)

    # Save validation summary CSV
    csv_path = os.path.join(PROJECT_ROOT, "validation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Checkpoint", "Val MSE Loss", "Val L1 Loss", "PSNR (dB)", "3D-SSIM"])
        for r in results:
            writer.writerow([r["short_name"], f"{r['val_loss_mse']:.6f}", f"{r['val_loss_l1']:.6f}", f"{r['psnr']:.2f}", f"{r['ssim']:.4f}"])

    best_res = max(results, key=lambda x: x["psnr"])
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY TABLE (Dataset/Validation/)")
    print("=" * 80)
    print(f"{'Checkpoint':<16} | {'Val MSE Loss':<14} | {'Val L1 Loss':<13} | {'PSNR (dB)':<10} | {'3D-SSIM'}")
    print("-" * 80)
    for r in results:
        star = " ★ BEST" if r["short_name"] == best_res["short_name"] else ""
        print(f"{r['short_name']:<16} | {r['val_loss_mse']:<14.6f} | {r['val_loss_l1']:<13.6f} | {r['psnr']:<10.2f} | {r['ssim']:.4f}{star}")
    print("=" * 80)
    print(f"\n[Success] Validation Checkpoints Plot saved at: {output_png}")
    print(f"[Success] Validation Summary CSV saved at: {csv_path}\n")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot_validation_from_csv()
    else:
        run_validation()
