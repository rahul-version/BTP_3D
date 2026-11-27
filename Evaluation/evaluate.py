import os
import sys
import argparse
import csv
import numpy as np
import mrcfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add current folder and training folder to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if TRAINING_DIR not in sys.path:
    sys.path.append(TRAINING_DIR)

DEFAULT_SR_DIR = os.path.join(PROJECT_ROOT, "result")
DEFAULT_HR_DIR = os.path.join(PROJECT_ROOT, "Dataset", "HR")

from metrics import calculate_psnr_3d, calculate_ssim_3d, calculate_fsc_3d

DEFAULT_RESULT_DIR = os.path.join(PROJECT_ROOT, "result")


# ============================================================
# MRC I/O & PREPROCESSING
# ============================================================

def load_volume_mrc(path):
    """Loads a 3D float32 MRC volume and returns (volume, voxel_size_angstrom)."""
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        if mrc.data is None:
            raise ValueError(f"Empty or corrupted MRC file: {path}")
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        data = np.squeeze(data)
        
        try:
            vx = float(mrc.voxel_size.x)
            voxel_size = vx if vx > 0.0 else 1.0
        except Exception:
            voxel_size = 1.0

    return data, voxel_size


def match_dimensions(v1, v2):
    """Aligns two 3D volumes to the common minimum dimension if boundary voxels differ."""
    min_d = min(v1.shape[0], v2.shape[0])
    min_h = min(v1.shape[1], v2.shape[1])
    min_w = min(v1.shape[2], v2.shape[2])
    return v1[:min_d, :min_h, :min_w], v2[:min_d, :min_h, :min_w]


def normalize_minmax(vol):
    """Scales volume to [-1.0, 1.0] for fair metric calculation."""
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)) * 2.0 - 1.0


# ============================================================
# FSC CURVE PLOT (SR vs HR per volume)
# ============================================================

def plot_fsc_curve(fsc_dict, volume_name, output_png):
    """
    Saves a clean FSC curve plot for a single SR/HR volume pair.

    Args:
        fsc_dict  : dict returned by calculate_fsc_3d()
        volume_name : str label for the plot title
        output_png  : absolute path to save the PNG
    """
    freqs = np.array(fsc_dict["spatial_frequencies"])
    fsc   = np.array(fsc_dict["fsc_curve"])
    res_05   = fsc_dict["resolution_05"]
    res_0143 = fsc_dict["resolution_0143"]
    nyquist  = fsc_dict["nyquist_resolution"]

    # Convert frequency axis to resolution (Å) for readability, skip zero-freq
    with np.errstate(divide="ignore"):
        res_axis = np.where(freqs > 0, 1.0 / freqs, np.inf)

    # Sort by ascending frequency (descending resolution)
    sort_idx = np.argsort(freqs)
    freqs_sorted = freqs[sort_idx]
    fsc_sorted   = fsc[sort_idx]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    ax.plot(freqs_sorted, fsc_sorted, color="#2980b9", lw=2, label="FSC (SR vs HR)")
    ax.axhline(0.5,   color="#e67e22", lw=1.4, linestyle="--", label=f"0.5 threshold  → {res_05:.2f} Å")
    ax.axhline(0.143, color="#c0392b", lw=1.4, linestyle=":",  label=f"0.143 threshold → {res_0143:.2f} Å")
    ax.axhline(0.0,   color="gray",    lw=0.8, linestyle="-",  alpha=0.4)

    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Spatial Frequency (1/Å)", fontsize=11)
    ax.set_ylabel("FSC", fontsize=11)
    ax.set_title(f"Fourier Shell Correlation — {volume_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Secondary x-axis showing resolution in Å
    ax2 = ax.twiny()
    tick_res = [100, 20, 10, 7, 5, 4, 3, 2.5, 2.0]
    tick_freq = [1.0 / r for r in tick_res if 1.0 / r <= freqs_sorted.max()]
    tick_lbl  = [f"{r}Å" for r in tick_res if 1.0 / r <= freqs_sorted.max()]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_freq)
    ax2.set_xticklabels(tick_lbl, fontsize=8)
    ax2.set_xlabel("Resolution (Å)", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    plt.close()


# ============================================================
# OVERALL EVALUATION SUMMARY PLOT (NO PER-FILE SPAM)
# ============================================================

def plot_evaluation_summary(results, output_png):
    """Generates an overall 3-panel bar dashboard saved directly in evaluate.py's parent folder."""
    try:
        names = [r["name"][:18] for r in results]
        psnrs = [r["psnr"] for r in results]
        ssims = [r["ssim"] for r in results]
        fsc0143 = [r["res_0143"] for r in results]

        x = np.arange(len(names))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
        fig.suptitle("Overall 3D Cryo-EM Super-Resolution Evaluation Summary (SR vs HR)", fontsize=14, fontweight="bold")

        # 1. PSNR
        bars1 = ax1.bar(x, psnrs, color="#2980b9", edgecolor="black", alpha=0.85)
        ax1.axhline(np.mean(psnrs), color="red", linestyle="--", label=f"Mean: {np.mean(psnrs):.2f} dB")
        ax1.set_title("3D-PSNR (dB)", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax1.set_ylabel("dB (Higher is better)")
        ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax1.legend()
        for b, v in zip(bars1, psnrs):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{v:.1f}", ha="center", fontsize=8)

        # 2. SSIM
        bars2 = ax2.bar(x, ssims, color="#8e44ad", edgecolor="black", alpha=0.85)
        ax2.axhline(np.mean(ssims), color="red", linestyle="--", label=f"Mean: {np.mean(ssims):.4f}")
        ax2.set_title("3D-SSIM (Structure Preservation)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax2.set_ylabel("SSIM (0 to 1.0)")
        ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax2.legend()
        for b, v in zip(bars2, ssims):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{v:.3f}", ha="center", fontsize=8)

        # 3. FSC 0.143 Resolution
        bars3 = ax3.bar(x, fsc0143, color="#27ae60", edgecolor="black", alpha=0.85)
        ax3.axhline(np.mean(fsc0143), color="red", linestyle="--", label=f"Mean: {np.mean(fsc0143):.2f} Å")
        ax3.set_title("FSC 0.143 Cutoff Resolution (Å)", fontsize=11, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax3.set_ylabel("Ångströms (Lower is better)")
        ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax3.legend()
        for b, v in zip(bars3, fsc0143):
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{v:.1f}Å", ha="center", fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png)
        plt.close()
        return True
    except Exception as e:
        print(f"  [Warning] Could not save evaluation summary plot: {e}")
        return False


# ============================================================
# SINGLE PAIR EVALUATION (NO INDIVIDUAL FILE PLOTS)
# ============================================================

def evaluate_pair(sr_path, hr_path):
    name = os.path.basename(sr_path)
    sr_vol, sr_vox = load_volume_mrc(sr_path)
    hr_vol, hr_vox = load_volume_mrc(hr_path)

    if sr_vol.shape != hr_vol.shape:
        sr_vol, hr_vol = match_dimensions(sr_vol, hr_vol)

    sr_norm = normalize_minmax(sr_vol)
    hr_norm = normalize_minmax(hr_vol)

    # 1. 3D-PSNR
    psnr_val = calculate_psnr_3d(sr_norm, hr_norm, data_range=2.0)

    # 2. 3D-SSIM
    ssim_val = calculate_ssim_3d(sr_norm, hr_norm, data_range=2.0)

    # 3. 3D-FSC
    fsc_dict = calculate_fsc_3d(sr_vol, hr_vol, voxel_size_angstrom=hr_vox)

    return {
        "name": name,
        "shape": sr_vol.shape,
        "voxel_size": hr_vox,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "res_05": fsc_dict["resolution_05"],
        "res_0143": fsc_dict["resolution_0143"],
        "fsc_dict": fsc_dict
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive 3D Cryo-EM Super-Resolution Evaluation (Overall Benchmark)"
    )
    parser.add_argument("--sr", type=str, default=None, help="Path to a single Super-Resolved (.mrc) file")
    parser.add_argument("--hr", type=str, default=None, help="Path to the corresponding Ground-Truth HR (.mrc) file")
    parser.add_argument("--sr_dir", type=str, default=DEFAULT_SR_DIR, help=f"Directory containing SR output files (default: {DEFAULT_SR_DIR})")
    parser.add_argument("--hr_dir", type=str, default=DEFAULT_HR_DIR, help=f"Directory containing HR reference files (default: {DEFAULT_HR_DIR})")
    parser.add_argument("--eval_checkpoints", action="store_true", help="Evaluate all model checkpoints and generate progress comparison graphs")
    args = parser.parse_args()

    if args.eval_checkpoints:
        from checkpoint_evaluator import run_checkpoint_evaluations
        run_checkpoint_evaluations(split="val")
        return

    print("=" * 70)
    print("3D CRYO-EM SUPER-RESOLUTION OVERALL EVALUATION PIPELINE")
    print("=" * 70)

    # Mode 1: Evaluate a single pair
    if args.sr and args.hr:
        if not os.path.exists(args.sr):
            raise FileNotFoundError(f"SR file not found: {args.sr}")
        if not os.path.exists(args.hr):
            raise FileNotFoundError(f"HR file not found: {args.hr}")

        res = evaluate_pair(args.sr, args.hr)
        print("\n" + "-" * 70)
        print(f"File Name                  : {res['name']}")
        print(f"3D Volume Dimensions       : {res['shape']}")
        print(f"Voxel Spacing              : {res['voxel_size']:.3f} Å")
        print("-" * 70)
        print(f"1. 3D-PSNR (Signal-to-Noise): {res['psnr']:.2f} dB   (Higher is better)")
        print(f"2. 3D-SSIM (Structure Sim) : {res['ssim']:.4f}      (Range: 0 to 1.0, Higher is better)")
        print(f"3. FSC Cutoff @ 0.500      : {res['res_05']:.2f} Å    (Conservative resolution)")
        print(f"4. FSC Cutoff @ 0.143      : {res['res_0143']:.2f} Å  (Cryo-EM Gold Standard)")
        print("=" * 70)
        return

    # Mode 2: Batch evaluation over result/ and Dataset/HR/
    if not os.path.exists(args.sr_dir) and not os.path.isabs(args.sr_dir):
        alt_sr = os.path.join(PROJECT_ROOT, args.sr_dir)
        if os.path.exists(alt_sr):
            args.sr_dir = alt_sr

    if not os.path.exists(args.hr_dir) and not os.path.isabs(args.hr_dir):
        alt_hr = os.path.join(PROJECT_ROOT, args.hr_dir)
        if os.path.exists(alt_hr):
            args.hr_dir = alt_hr

    if not os.path.exists(args.sr_dir):
        raise FileNotFoundError(f"SR directory not found: {args.sr_dir}")
    if not os.path.exists(args.hr_dir):
        raise FileNotFoundError(f"HR directory not found: {args.hr_dir}")

    sr_files = [f for f in os.listdir(args.sr_dir) if f.lower().endswith(".mrc")]
    hr_files = [f for f in os.listdir(args.hr_dir) if f.lower().endswith(".mrc")]

    if len(sr_files) == 0:
        print(f"No .mrc files found in {args.sr_dir}. Run test.py first to generate results.")
        return

    results = []
    print(f"Found {len(sr_files)} SR volume(s) in '{args.sr_dir}'. Matching with '{args.hr_dir}'...\n")

    for sf in sr_files:
        base = sf.replace("_lr_SR", "").replace("_SR", "").replace("_lr", "")
        base_clean = os.path.splitext(base)[0]
        
        match = None
        for hf in hr_files:
            if base_clean == os.path.splitext(hf)[0]:
                match = hf
                break

        if match is None:
            continue

        sr_path = os.path.join(args.sr_dir, sf)
        hr_path = os.path.join(args.hr_dir, match)

        print(f"Evaluating: {sf} <---> {match}")
        try:
            r = evaluate_pair(sr_path, hr_path)
            results.append(r)
        except Exception as e:
            print(f"  [Error] Failed to evaluate {sf}: {e}")

    if not results:
        print("\nNo matching pairs could be evaluated.")
        return

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Volume Name':<28} | {'PSNR (dB)':<10} | {'3D-SSIM':<8} | {'FSC 0.5 (Å)':<11} | {'FSC 0.143 (Å)':<13}")
    print("-" * 80)
    
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])
    avg_fsc05 = np.mean([r["res_05"] for r in results])
    avg_fsc0143 = np.mean([r["res_0143"] for r in results])

    for r in results:
        print(f"{r['name'][:28]:<28} | {r['psnr']:<10.2f} | {r['ssim']:<8.4f} | {r['res_05']:<11.2f} | {r['res_0143']:<13.2f}")

    print("-" * 80)
    print(f"{'OVERALL AVERAGE':<28} | {avg_psnr:<10.2f} | {avg_ssim:<8.4f} | {avg_fsc05:<11.2f} | {avg_fsc0143:<13.2f}")
    print("=" * 80)

    # Save summary plot and CSV in PROJECT_ROOT (parent folder of evaluate.py)
    summary_plot_path = os.path.join(PROJECT_ROOT, "evaluation_metrics_summary.png")
    plot_evaluation_summary(results, summary_plot_path)

    # Save combined FSC overlay plot (all volumes on one chart)
    combined_fsc_path = os.path.join(PROJECT_ROOT, "fsc_all_volumes.png")
    try:
        fig_c, ax_c = plt.subplots(figsize=(10, 5.5), dpi=150)
        colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#c0392b", "#16a085"]
        for i, r in enumerate(results):
            if "fsc_dict" not in r:
                continue
            fd = r["fsc_dict"]
            freqs_s = np.array(fd["spatial_frequencies"])
            fsc_s   = np.array(fd["fsc_curve"])
            sort_idx = np.argsort(freqs_s)
            lbl = os.path.splitext(r["name"])[0][:20]
            ax_c.plot(freqs_s[sort_idx], fsc_s[sort_idx],
                      color=colors[i % len(colors)], lw=1.8, label=lbl)
        ax_c.axhline(0.5,   color="#e67e22", lw=1.3, linestyle="--", label="0.5 threshold")
        ax_c.axhline(0.143, color="#c0392b", lw=1.3, linestyle=":",  label="0.143 threshold")
        ax_c.axhline(0.0,   color="gray",    lw=0.8, linestyle="-",  alpha=0.4)
        ax_c.set_xlim(left=0)
        ax_c.set_ylim(-0.05, 1.05)
        ax_c.set_xlabel("Spatial Frequency (1/Å)", fontsize=11)
        ax_c.set_ylabel("FSC", fontsize=11)
        ax_c.set_title("FSC Curves — All Evaluated Volumes (SR vs HR)", fontsize=12, fontweight="bold")
        ax_c.legend(fontsize=8, loc="upper right")
        ax_c.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(combined_fsc_path)
        plt.close()
        print(f"Combined FSC plot saved: {combined_fsc_path}")
    except Exception as e:
        print(f"  [Warning] Could not save combined FSC plot: {e}")

    csv_summary_path = os.path.join(PROJECT_ROOT, "evaluation_summary.csv")
    with open(csv_summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Volume Name", "3D-PSNR (dB)", "3D-SSIM", "FSC 0.5 (Å)", "FSC 0.143 (Å)"])
        for r in results:
            writer.writerow([r["name"], f"{r['psnr']:.2f}", f"{r['ssim']:.4f}", f"{r['res_05']:.2f}", f"{r['res_0143']:.2f}"])
        writer.writerow(["OVERALL AVERAGE", f"{avg_psnr:.2f}", f"{avg_ssim:.4f}", f"{avg_fsc05:.2f}", f"{avg_fsc0143:.2f}"])

    print(f"Overall Summary Graph saved in evaluate.py's parent folder: {summary_plot_path}")
    print(f"Overall Summary CSV saved in: {csv_summary_path}\n")


if __name__ == "__main__":
    main()
