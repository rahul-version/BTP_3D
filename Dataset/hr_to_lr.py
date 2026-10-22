import os
import sys
import time
import gc
import argparse
import numpy as np
import mrcfile
import torch
import torch.nn.functional as F


# ============================================================
# DIRECTORY PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HR_DIR = os.path.join(SCRIPT_DIR, "HR")
DEFAULT_LR_DIR = os.path.join(SCRIPT_DIR, "LR2")


# ============================================================
# MRC I/O HELPERS
# ============================================================

def load_mrc(path):
    """
    Safely loads 3D volume data and voxel dimensions from an MRC file.
    """
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        voxel_size = (
            float(mrc.voxel_size.x),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.z)
        )
    return data, voxel_size


def save_mrc(output_path, volume, voxel_size):
    """
    Saves a 3D float32 volume to an MRC file and writes updated voxel size & header stats.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(volume, dtype=np.float32))
        mrc.voxel_size = voxel_size
        mrc.update_header_stats()
        mrc.update_header_from_data()


# ============================================================
# NOISE AND DEGRADATION FUNCTIONS
# ============================================================

def add_gaussian_noise(volume, noise_level=0.05, relative=True):
    """
    Adds Additive White Gaussian Noise (AWGN).
    If relative is True, noise std is (noise_level * signal_std).
    Example: noise_level=0.05 adds 5% noise.
    """
    if noise_level <= 0.0:
        return volume

    if relative:
        v_std = float(np.std(volume))
        sigma = noise_level * (v_std if v_std > 1e-8 else 1.0)
    else:
        sigma = float(noise_level)

    noise = np.random.normal(0.0, sigma, size=volume.shape).astype(np.float32)
    return volume + noise


def add_poisson_noise(volume, peak_factor=100.0):
    """
    Simulates Poisson (shot) noise prevalent in Cryo-EM electron counting.
    Normalizes to positive range, applies Poisson sampling, then shifts back.
    """
    v_min = float(np.min(volume))
    v_max = float(np.max(volume))
    spread = v_max - v_min

    if spread < 1e-8:
        return volume

    # Shift to [0, 1] range
    normalized = (volume - v_min) / spread

    # Scale by peak count, sample Poisson, scale back
    lam = np.clip(normalized * peak_factor, 0, None)
    noisy_norm = np.random.poisson(lam).astype(np.float32) / peak_factor

    # Restore original scale and shift
    return (noisy_norm * spread + v_min).astype(np.float32)


def apply_gaussian_blur_3d(volume, kernel_size=5, sigma=1.0):
    """
    Applies separable 3D Gaussian blur filter using PyTorch conv3d.
    Simulates optical Point Spread Function (PSF) / CTF envelope attenuation.
    """
    if sigma <= 0.0 or kernel_size < 3:
        return volume

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    k_d = kernel_1d.view(1, 1, -1, 1, 1)
    k_h = kernel_1d.view(1, 1, 1, -1, 1)
    k_w = kernel_1d.view(1, 1, 1, 1, -1)

    t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, D, H, W)

    with torch.no_grad():
        out = F.conv3d(t, k_d, padding=(radius, 0, 0))
        out = F.conv3d(out, k_h, padding=(0, radius, 0))
        out = F.conv3d(out, k_w, padding=(0, 0, radius))

    return out.squeeze().numpy().astype(np.float32)


# ============================================================
# DOWNSAMPLING FUNCTION
# ============================================================

def downsample_volume(volume, scale=4, method="area"):
    """
    Downsamples a 3D volume by integer scale factor (e.g. 4x).
    
    Methods:
    - 'area': Average pooling / box binning (ideal anti-aliasing for cryo-EM voxel binning).
    - 'trilinear': 3D trilinear interpolation.
    """
    d, h, w = volume.shape

    # Crop boundary voxels if dimension is not divisible by scale
    d_crop = (d // scale) * scale
    h_crop = (h // scale) * scale
    w_crop = (w // scale) * scale

    if (d_crop, h_crop, w_crop) != (d, h, w):
        print(f"    [Info] Cropping ({d}, {h}, {w}) -> ({d_crop}, {h_crop}, {w_crop}) for integer {scale}x downsampling")
        volume = volume[:d_crop, :h_crop, :w_crop]

    out_d = d_crop // scale
    out_h = h_crop // scale
    out_w = w_crop // scale

    if method == "area":
        # Fast, low-memory block averaging using reshape & mean
        downsampled = volume.reshape(
            out_d, scale,
            out_h, scale,
            out_w, scale
        ).mean(axis=(1, 3, 5))
        return downsampled.astype(np.float32)

    elif method == "trilinear":
        t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        with torch.no_grad():
            downsampled = F.interpolate(
                t,
                size=(out_d, out_h, out_w),
                mode="trilinear",
                align_corners=False
            ).squeeze().numpy()
        return downsampled.astype(np.float32)

    else:
        raise ValueError(f"Unknown downsampling method: {method}. Choose 'area' or 'trilinear'.")


# ============================================================
# SINGLE VOLUME PROCESSING PIPELINE
# ============================================================

def process_file(
    filename,
    hr_dir,
    output_dir,
    scale=4,
    method="area",
    noise_type="gaussian",
    noise_level=0.05,
    noise_order="after_downsample",
    apply_blur=False,
    blur_sigma=1.0
):
    hr_path = os.path.join(hr_dir, filename)

    # Naming convention: ensure output file has '_lr.mrc' suffix
    base_name, ext = os.path.splitext(filename)
    if not base_name.endswith("_lr"):
        out_filename = f"{base_name}_lr{ext}"
    else:
        out_filename = filename

    out_path = os.path.join(output_dir, out_filename)

    print(f"\n--> Processing: {filename}", flush=True)
    t0 = time.time()

    # 1. Load HR MRC
    hr_data, hr_voxel_size = load_mrc(hr_path)
    print(f"    HR Shape: {hr_data.shape} | Voxel size: {hr_voxel_size}", flush=True)
    print(f"    HR Stats: min={hr_data.min():.4f}, max={hr_data.max():.4f}, std={hr_data.std():.4f}", flush=True)

    # 2. Optional pre-blur (simulate microscope optical PSF / low-pass filter)
    if apply_blur:
        print(f"    Applying Gaussian Blur (sigma={blur_sigma})...", flush=True)
        hr_data = apply_gaussian_blur_3d(hr_data, sigma=blur_sigma)

    # 3. Add noise BEFORE downsampling (if requested)
    if noise_order == "before_downsample":
        if noise_type == "gaussian" and noise_level > 0:
            print(f"    Adding Gaussian Noise before downsample (level={noise_level})...", flush=True)
            hr_data = add_gaussian_noise(hr_data, noise_level=noise_level)
        elif noise_type == "poisson":
            print("    Adding Poisson Noise before downsample...", flush=True)
            hr_data = add_poisson_noise(hr_data)

    # 4. Downsample by scale (4x)
    print(f"    Downsampling {scale}x using '{method}' method...", flush=True)
    lr_data = downsample_volume(hr_data, scale=scale, method=method)

    # Free up HR memory immediately
    del hr_data
    gc.collect()

    # 5. Add noise AFTER downsampling (Default / Physically Realistic Sensor Noise)
    if noise_order == "after_downsample":
        if noise_type == "gaussian" and noise_level > 0:
            print(f"    Adding Gaussian Noise to LR (level={noise_level})...", flush=True)
            lr_data = add_gaussian_noise(lr_data, noise_level=noise_level)
        elif noise_type == "poisson":
            print("    Adding Poisson Noise to LR...", flush=True)
            lr_data = add_poisson_noise(lr_data)

    # 6. Compute new voxel size: physical voxel length increases by scale
    lr_voxel_size = (
        float(hr_voxel_size[0] * scale),
        float(hr_voxel_size[1] * scale),
        float(hr_voxel_size[2] * scale)
    )

    # 7. Save output LR MRC
    save_mrc(out_path, lr_data, lr_voxel_size)

    elapsed = time.time() - t0
    print(f"    LR Shape: {lr_data.shape} | Voxel size: {lr_voxel_size}", flush=True)
    print(f"    LR Stats: min={lr_data.min():.4f}, max={lr_data.max():.4f}, std={lr_data.std():.4f}", flush=True)
    print(f"    Saved: {out_path} (Done in {elapsed:.2f}s)", flush=True)

    del lr_data
    gc.collect()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 4x Low-Resolution (LR) Cryo-EM volumes from HR volumes with noise injection."
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
        default=DEFAULT_HR_DIR,
        help=f"Path to input HR MRC directory (default: {DEFAULT_HR_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_LR_DIR,
        help=f"Path to output LR MRC directory (default: {DEFAULT_LR_DIR})"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Downsampling scale factor (default: 4)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="area",
        choices=["area", "trilinear"],
        help="Downsampling method: 'area' (box avg binning) or 'trilinear' (default: area)"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "poisson", "none"],
        help="Type of noise to inject: 'gaussian', 'poisson', or 'none' (default: gaussian)"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.05,
        help="Relative standard deviation of Gaussian noise (default: 0.05, i.e., 5%% of volume std)"
    )
    parser.add_argument(
        "--noise_order",
        type=str,
        default="after_downsample",
        choices=["after_downsample", "before_downsample"],
        help="Whether to inject noise after or before downsampling (default: after_downsample)"
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Apply mild 3D Gaussian blur before downsampling to simulate optical PSF/CTF"
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=1.0,
        help="Gaussian blur sigma (default: 1.0)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a single specific .mrc file (optional)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit the number of files to process (optional)"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("HR TO LR VOLUME GENERATOR")
    print("=" * 65)
    print(f"Input HR Directory : {args.hr_dir}")
    print(f"Output LR Directory: {args.output_dir}")
    print(f"Scale Factor       : {args.scale}x")
    print(f"Downsample Method  : {args.method}")
    print(f"Noise Type         : {args.noise_type}")
    if args.noise_type == "gaussian":
        print(f"Noise Level (std)  : {args.noise_level} ({args.noise_level * 100:.1f}% relative)")
    print(f"Noise Order        : {args.noise_order}")
    print(f"Pre-Blur           : {args.blur} (sigma={args.blur_sigma if args.blur else 'N/A'})")
    print("=" * 65)

    if not os.path.exists(args.hr_dir):
        raise FileNotFoundError(f"HR directory not found: {args.hr_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect .mrc files
    if args.file:
        files = [args.file]
    else:
        files = sorted([
            f for f in os.listdir(args.hr_dir)
            if f.lower().endswith(".mrc")
        ])

    if len(files) == 0:
        raise RuntimeError(f"No .mrc files found in HR directory: {args.hr_dir}")

    if args.max_files is not None:
        files = files[:args.max_files]

    print(f"Found {len(files)} MRC file(s) to process.")

    total_start = time.time()
    for idx, fname in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Processing {fname}...")
        process_file(
            filename=fname,
            hr_dir=args.hr_dir,
            output_dir=args.output_dir,
            scale=args.scale,
            method=args.method,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
            noise_order=args.noise_order,
            apply_blur=args.blur,
            blur_sigma=args.blur_sigma
        )

    total_time = time.time() - total_start
    print("\n" + "=" * 65)
    print(f"ALL FILES PROCESSED SUCCESSFULLY in {total_time:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
