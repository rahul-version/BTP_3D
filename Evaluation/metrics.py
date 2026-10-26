import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. 3D PEAK SIGNAL-TO-NOISE RATIO (3D-PSNR)
# ============================================================

def calculate_psnr_3d(sr_volume, hr_volume, data_range=None):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in dB between two 3D volumes.
    Higher value indicates closer reconstruction fidelity.
    
    Args:
        sr_volume (np.ndarray): Super-resolved 3D volume.
        hr_volume (np.ndarray): Ground-truth High-Resolution 3D volume.
        data_range (float, optional): Dynamic range (max - min). If None, calculated from HR.
        
    Returns:
        float: PSNR value in dB.
    """
    sr = np.asarray(sr_volume, dtype=np.float64)
    hr = np.asarray(hr_volume, dtype=np.float64)

    if sr.shape != hr.shape:
        raise ValueError(f"Shape mismatch for PSNR: SR {sr.shape} vs HR {hr.shape}")

    mse = np.mean((sr - hr) ** 2)
    if mse == 0.0 or mse < 1e-12:
        return 100.0  # Identical volumes

    if data_range is None:
        data_range = float(np.max(hr) - np.min(hr))
        if data_range < 1e-8:
            data_range = 1.0

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


# ============================================================
# 2. 3D STRUCTURAL SIMILARITY INDEX MEASURE (3D-SSIM)
# ============================================================

def _gaussian_kernel_3d(kernel_size=7, sigma=1.5):
    """Generates a normalized 3D Gaussian kernel for volumetric SSIM."""
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    k1d = np.exp(-0.5 * (coords / sigma) ** 2)
    k1d /= np.sum(k1d)
    
    # 3D outer product
    k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
    k3d /= np.sum(k3d)
    return torch.from_numpy(k3d).unsqueeze(0).unsqueeze(0)  # (1, 1, K, K, K)


def calculate_ssim_3d(sr_volume, hr_volume, data_range=None, kernel_size=7, sigma=1.5):
    """
    Calculates Structural Similarity Index Measure (SSIM) in 3D.
    Measures luminance, contrast, and structural preservation.
    
    Args:
        sr_volume (np.ndarray): Super-resolved 3D volume.
        hr_volume (np.ndarray): Ground-truth High-Resolution 3D volume.
        data_range (float, optional): Dynamic range. If None, calculated from HR.
        kernel_size (int): Gaussian filter window size (odd integer).
        sigma (float): Gaussian filter standard deviation.
        
    Returns:
        float: 3D-SSIM value between -1.0 and 1.0 (higher is better, 1.0 = identical).
    """
    sr = np.asarray(sr_volume, dtype=np.float32)
    hr = np.asarray(hr_volume, dtype=np.float32)

    if sr.shape != hr.shape:
        raise ValueError(f"Shape mismatch for SSIM: SR {sr.shape} vs HR {hr.shape}")

    if data_range is None:
        data_range = float(np.max(hr) - np.min(hr))
        if data_range < 1e-8:
            data_range = 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Convert to PyTorch tensors for fast separable 3D convolution
    t_sr = torch.from_numpy(sr).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    t_hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)

    kernel = _gaussian_kernel_3d(kernel_size, sigma)
    pad = kernel_size // 2

    with torch.no_grad():
        mu_sr = F.conv3d(t_sr, kernel, padding=pad)
        mu_hr = F.conv3d(t_hr, kernel, padding=pad)

        mu_sr_sq = mu_sr ** 2
        mu_hr_sq = mu_hr ** 2
        mu_sr_hr = mu_sr * mu_hr

        sigma_sr_sq = F.conv3d(t_sr ** 2, kernel, padding=pad) - mu_sr_sq
        sigma_hr_sq = F.conv3d(t_hr ** 2, kernel, padding=pad) - mu_hr_sq
        sigma_sr_hr = F.conv3d(t_sr * t_hr, kernel, padding=pad) - mu_sr_hr

        ssim_map = ((2.0 * mu_sr_hr + c1) * (2.0 * sigma_sr_hr + c2)) / \
                   ((mu_sr_sq + mu_hr_sq + c1) * (sigma_sr_sq + sigma_hr_sq + c2))

        return float(ssim_map.mean().item())


# ============================================================
# 3. 3D FOURIER SHELL CORRELATION (FSC) & CRYO-EM RESOLUTION
# ============================================================

def calculate_fsc_3d(sr_volume, hr_volume, voxel_size_angstrom=1.0):
    """
    Calculates Fourier Shell Correlation (FSC) across 3D frequency shells.
    Standard gold-standard biophysical metric in Cryo-EM structural determination.
    
    Args:
        sr_volume (np.ndarray): Super-resolved 3D volume.
        hr_volume (np.ndarray): Ground-truth High-Resolution 3D volume.
        voxel_size_angstrom (float): Physical voxel spacing in Ångströms (Å).
        
    Returns:
        dict:
            - 'spatial_frequencies': Frequency array in 1/Å.
            - 'fsc_curve': Correlation coefficient per shell (values in [-1, 1]).
            - 'resolution_05': Estimated resolution in Å at FSC = 0.5 threshold.
            - 'resolution_0143': Estimated resolution in Å at FSC = 0.143 threshold (Gold standard).
            - 'nyquist_resolution': Theoretical limit in Å (2 * voxel_size).
    """
    sr = np.asarray(sr_volume, dtype=np.float32)
    hr = np.asarray(hr_volume, dtype=np.float32)

    if sr.shape != hr.shape:
        raise ValueError(f"Shape mismatch for FSC: SR {sr.shape} vs HR {hr.shape}")

    d, h, w = sr.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    max_radius = min(cz, cy, cx)

    # 1. Compute 3D Fast Fourier Transforms
    f_sr = np.fft.fftshift(np.fft.fftn(sr))
    f_hr = np.fft.fftshift(np.fft.fftn(hr))

    # 2. Build radial 3D frequency distance grid
    z, y, x = np.ogrid[:d, :h, :w]
    r_grid = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    r_int = np.round(r_grid).astype(np.int32)

    # 3. Compute correlation per spherical shell
    radii = np.arange(1, max_radius)
    fsc_values = []
    frequencies = []

    for r in radii:
        mask = (r_int == r)
        if not np.any(mask):
            continue

        f1 = f_sr[mask]
        f2 = f_hr[mask]

        numerator = np.real(np.sum(f1 * np.conj(f2)))
        denominator = np.sqrt(np.sum(np.abs(f1) ** 2) * np.sum(np.abs(f2) ** 2))

        fsc = numerator / (denominator + 1e-12)
        fsc_values.append(float(np.clip(fsc, -1.0, 1.0)))

        # Frequency in 1/Å: spatial_freq = r / (box_size * voxel_size)
        box_size = max_radius * 2
        freq = r / (box_size * voxel_size_angstrom)
        frequencies.append(float(freq))

    frequencies = np.array(frequencies)
    fsc_values = np.array(fsc_values)

    # 4. Find resolution cutoffs at FSC = 0.5 and FSC = 0.143
    def _find_cutoff(thresh):
        below = np.where(fsc_values < thresh)[0]
        if len(below) > 0:
            idx = below[0]
            if idx == 0:
                return float(1.0 / (frequencies[0] + 1e-8))
            # Linear interpolation for sub-voxel accuracy
            f1, f2 = frequencies[idx - 1], frequencies[idx]
            v1, v2 = fsc_values[idx - 1], fsc_values[idx]
            if v1 != v2:
                interp_freq = f1 + (thresh - v1) * (f2 - f1) / (v2 - v1)
            else:
                interp_freq = f2
            return float(1.0 / (interp_freq + 1e-8))
        else:
            # Reached Nyquist
            return float(2.0 * voxel_size_angstrom)

    res_05 = _find_cutoff(0.5)
    res_0143 = _find_cutoff(0.143)
    nyquist = float(2.0 * voxel_size_angstrom)

    return {
        "spatial_frequencies": frequencies,
        "fsc_curve": fsc_values,
        "resolution_05": res_05,
        "resolution_0143": res_0143,
        "nyquist_resolution": nyquist
    }
