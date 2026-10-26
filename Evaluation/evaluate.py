import os
import sys
import argparse
import numpy as np
import mrcfile

# Add current folder to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

DEFAULT_SR_DIR = os.path.join(PROJECT_ROOT, "result")
DEFAULT_HR_DIR = os.path.join(PROJECT_ROOT, "Dataset", "HR")
DEFAULT_PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

from metrics import calculate_psnr_3d, calculate_ssim_3d, calculate_fsc_3d


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
# FSC PLOTTING HELPER
# ============================================================

def plot_fsc_curve(fsc_dict, title_name, output_png):
    """Saves a publication-quality FSC curve plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        
        freqs = fsc_dict["spatial_frequencies"]
        fsc = fsc_dict["fsc_curve"]
        res_05 = fsc_dict["resolution_05"]
        res_0143 = fsc_dict["resolution_0143"]

        plt.figure(figsize=(7, 4.5), dpi=150)
        plt.plot(freqs, fsc, color="#0066cc", lw=2, label="FSC (SR vs HR)")
        plt.axhline(0.5, color="red", linestyle="--", alpha=0.7, label=f"FSC=0.5 ({res_05:.2f} Å)")
        plt.axhline(0.143, color="green", linestyle="--", alpha=0.7, label=f"FSC=0.143 ({res_0143:.2f} Å)")

        plt.title(f"Fourier Shell Correlation (FSC) - {title_name}", fontsize=12, fontweight="bold")
        plt.xlabel("Spatial Frequency (1/Å)", fontsize=10)
        plt.ylabel("Correlation Coefficient", fontsize=10)
        plt.ylim(-0.1, 1.05)
        plt.xlim(0, max(freqs) if len(freqs) > 0 else 1)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
        return True
    except Exception as e:
        return False


# ============================================================
# SINGLE PAIR EVALUATION
# ============================================================

def evaluate_pair(sr_path, hr_path, plot_dir=None):
    """
    Evaluates one (SR, HR) pair and prints a clean report.
    Returns dict with calculated metrics.
    """
    name = os.path.basename(sr_path)
    sr_vol, sr_vox = load_volume_mrc(sr_path)
    hr_vol, hr_vox = load_volume_mrc(hr_path)

    # Align dimensions if downsampling caused integer boundary crop
    if sr_vol.shape != hr_vol.shape:
        print(f"  [Info] Aligning shapes: SR {sr_vol.shape} -> HR {hr_vol.shape}")
        sr_vol, hr_vol = match_dimensions(sr_vol, hr_vol)

    # Normalized copies for signal metrics
    sr_norm = normalize_minmax(sr_vol)
    hr_norm = normalize_minmax(hr_vol)

    # 1. 3D-PSNR
    psnr_val = calculate_psnr_3d(sr_norm, hr_norm, data_range=2.0)

    # 2. 3D-SSIM
    ssim_val = calculate_ssim_3d(sr_norm, hr_norm, data_range=2.0)

    # 3. 3D-FSC
    fsc_dict = calculate_fsc_3d(sr_vol, hr_vol, voxel_size_angstrom=hr_vox)

    # Plot FSC
    plot_saved = None
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        base_clean = os.path.splitext(name)[0]
        plot_path = os.path.join(plot_dir, f"fsc_curve_{base_clean}.png")
        if plot_fsc_curve(fsc_dict, base_clean, plot_path):
            plot_saved = plot_path

    return {
        "name": name,
        "shape": sr_vol.shape,
        "voxel_size": hr_vox,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "res_05": fsc_dict["resolution_05"],
        "res_0143": fsc_dict["resolution_0143"],
        "plot_path": plot_saved
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive 3D Cryo-EM Super-Resolution Evaluation (3D-PSNR, 3D-SSIM, 3D-FSC)"
    )
    parser.add_argument("--sr", type=str, default=None, help="Path to a single Super-Resolved (.mrc) file")
    parser.add_argument("--hr", type=str, default=None, help="Path to the corresponding Ground-Truth HR (.mrc) file")
    parser.add_argument("--sr_dir", type=str, default=DEFAULT_SR_DIR, help=f"Directory containing SR output files (default: {DEFAULT_SR_DIR})")
    parser.add_argument("--hr_dir", type=str, default=DEFAULT_HR_DIR, help=f"Directory containing HR reference files (default: {DEFAULT_HR_DIR})")
    parser.add_argument("--plot_dir", type=str, default=DEFAULT_PLOT_DIR, help=f"Directory to save FSC curve plots (default: {DEFAULT_PLOT_DIR})")
    args = parser.parse_args()

    print("=" * 70)
    print("3D CRYO-EM SUPER-RESOLUTION EVALUATION PIPELINE")
    print("=" * 70)

    # Mode 1: Evaluate a single pair
    if args.sr and args.hr:
        if not os.path.exists(args.sr):
            raise FileNotFoundError(f"SR file not found: {args.sr}")
        if not os.path.exists(args.hr):
            raise FileNotFoundError(f"HR file not found: {args.hr}")

        res = evaluate_pair(args.sr, args.hr, plot_dir=args.plot_dir)
        print("\n" + "-" * 70)
        print(f"File Name                  : {res['name']}")
        print(f"3D Volume Dimensions       : {res['shape']}")
        print(f"Voxel Spacing              : {res['voxel_size']:.3f} Å")
        print("-" * 70)
        print(f"1. 3D-PSNR (Signal-to-Noise): {res['psnr']:.2f} dB   (Higher is better)")
        print(f"2. 3D-SSIM (Structure Sim) : {res['ssim']:.4f}      (Range: 0 to 1.0, Higher is better)")
        print(f"3. FSC Cutoff @ 0.500      : {res['res_05']:.2f} Å    (Conservative resolution)")
        print(f"4. FSC Cutoff @ 0.143      : {res['res_0143']:.2f} Å  (Cryo-EM Gold Standard)")
        if res['plot_path']:
            print(f"   FSC Plot Saved at       : {res['plot_path']}")
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
        # Match name: e.g. 'emd_3228_lr_SR.mrc' -> 'emd_3228.mrc'
        base = sf.replace("_lr_SR", "").replace("_SR", "").replace("_lr", "")
        base_clean = os.path.splitext(base)[0]
        
        # Look for matching HR file
        match = None
        for hf in hr_files:
            if base_clean == os.path.splitext(hf)[0]:
                match = hf
                break

        if match is None:
            print(f"  [Skipping] No matching HR file found for: {sf}")
            continue

        sr_path = os.path.join(args.sr_dir, sf)
        hr_path = os.path.join(args.hr_dir, match)

        print(f"Evaluating: {sf} <---> {match}")
        try:
            r = evaluate_pair(sr_path, hr_path, plot_dir=args.plot_dir)
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
    print(f"All FSC curve graphs saved in: {args.plot_dir}\n")


if __name__ == "__main__":
    main()
