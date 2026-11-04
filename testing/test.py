import os
import gc
import sys
import argparse
import numpy as np
import torch
import mrcfile

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "training"
    )
)

from srgan_model import Generator


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "input"
)

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "result"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "checkpoint_1200.pt"
)

# Alternative fallback checkpoints if pre_trained_model_final.pt isn't present
FALLBACK_CHECKPOINTS = [
    os.path.join(PROJECT_ROOT, "model", "latest_checkpoint.pt"),
    os.path.join(PROJECT_ROOT, "model", "generator_final.pt"),
    os.path.join(PROJECT_ROOT, "model", "generator_pretrained.pt"),
    os.path.join(PROJECT_ROOT, "model", "best_generator.pt")
]


# ============================================================
# SETTINGS
# ============================================================

SCALE = 4
N_FEATS = 64
RES_NUM = 16
TILE_SIZE = 32
OVERLAP = 8


# ============================================================
# MRC I/O & NORMALIZATION
# ============================================================

def load_mrc(path):
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        try:
            vx = float(mrc.voxel_size.x) if float(mrc.voxel_size.x) > 0 else 1.0
            vy = float(mrc.voxel_size.y) if float(mrc.voxel_size.y) > 0 else 1.0
            vz = float(mrc.voxel_size.z) if float(mrc.voxel_size.z) > 0 else 1.0
            voxel_size = (vx, vy, vz)
        except Exception:
            voxel_size = (1.0, 1.0, 1.0)
    return data, voxel_size


def normalize(volume):
    """Normalizes volume to [-1.0, 1.0] and returns (normalized_vol, vmin, vmax)."""
    vmin = float(volume.min())
    vmax = float(volume.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax

    volume = (volume - vmin) / (vmax - vmin)
    volume = volume * 2.0 - 1.0
    return volume.astype(np.float32), vmin, vmax


def denormalize(volume, vmin, vmax):
    """Denormalizes volume from [-1.0, 1.0] back to original Cryo-EM density [vmin, vmax]."""
    if vmax - vmin < 1e-8:
        return volume + vmin
    volume = (volume + 1.0) / 2.0
    volume = volume * (vmax - vmin) + vmin
    return volume.astype(np.float32)


# ============================================================
# SMOOTH 3D WINDOW FOR SEAMLESS TILE BLENDING (Prevents Gaps)
# ============================================================

def get_3d_blend_window(sz, sy, sx):
    """
    Constructs a smooth 3D Hann / Bartlett taper window.
    Eliminates boundary discontinuities and 3D grid line gaps.
    """
    w_z = np.sin(np.pi * (np.arange(sz) + 0.5) / sz) ** 2
    w_y = np.sin(np.pi * (np.arange(sy) + 0.5) / sy) ** 2
    w_x = np.sin(np.pi * (np.arange(sx) + 0.5) / sx) ** 2
    w3d = w_z[:, None, None] * w_y[None, :, None] * w_x[None, None, :]
    return np.clip(w3d, 1e-4, 1.0).astype(np.float32)


# ============================================================
# TILED INFERENCE WITH SMOOTH BLENDING
# ============================================================

@torch.no_grad()
def tiled_inference(volume, generator, device, tile_size=TILE_SIZE, overlap=OVERLAP, scale=SCALE):
    depth, height, width = volume.shape

    output_shape = (
        depth * scale,
        height * scale,
        width * scale
    )

    output_path = os.path.join(RESULT_PATH, "_temporary_sr.dat")
    os.makedirs(RESULT_PATH, exist_ok=True)

    output = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=output_shape
    )

    weights = np.memmap(
        output_path + "_weights",
        dtype=np.float32,
        mode="w+",
        shape=output_shape
    )

    output[:] = 0
    weights[:] = 0

    stride = max(1, tile_size - overlap)

    z_positions = list(range(0, max(1, depth - tile_size + 1), stride))
    y_positions = list(range(0, max(1, height - tile_size + 1), stride))
    x_positions = list(range(0, max(1, width - tile_size + 1), stride))

    if not z_positions or z_positions[-1] != depth - tile_size:
        z_positions.append(max(0, depth - tile_size))
    if not y_positions or y_positions[-1] != height - tile_size:
        y_positions.append(max(0, height - tile_size))
    if not x_positions or x_positions[-1] != width - tile_size:
        x_positions.append(max(0, width - tile_size))

    # Remove duplicates while preserving order
    z_positions = sorted(list(set(z_positions)))
    y_positions = sorted(list(set(y_positions)))
    x_positions = sorted(list(set(x_positions)))

    total_tiles = len(z_positions) * len(y_positions) * len(x_positions)
    tile_number = 0

    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                tile_number += 1
                if tile_number % 5 == 0 or tile_number == total_tiles:
                    print(f"\r  Processing tile {tile_number}/{total_tiles} ({tile_number/total_tiles*100:.1f}%)", end="", flush=True)

                tile = volume[
                    z:z + tile_size,
                    y:y + tile_size,
                    x:x + tile_size
                ]

                tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)

                sr, _ = generator(tile_tensor)
                sr = sr.squeeze().cpu().numpy()

                oz = z * scale
                oy = y * scale
                ox = x * scale

                sz, sy, sx = sr.shape

                # Smooth 3D blending window to avoid seams and boundary line gaps
                window = get_3d_blend_window(sz, sy, sx)

                output[oz:oz + sz, oy:oy + sy, ox:ox + sx] += (sr * window)
                weights[oz:oz + sz, oy:oy + sy, ox:ox + sx] += window

                del tile_tensor
                del sr
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print()

    output /= np.maximum(weights, 1e-8)
    output.flush()
    weights.flush()
    del weights
    gc.collect()

    return output, output_path


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_file, generator, device, input_dir=INPUT_PATH, result_dir=RESULT_PATH, tile_size=TILE_SIZE, overlap=OVERLAP):
    input_path = os.path.join(input_dir, input_file)
    print(f"\nProcessing: {input_file}")

    volume, voxel_size = load_mrc(input_path)
    print(f"  Input shape: {volume.shape} | Voxel size: {voxel_size}")

    # 1. Normalize and preserve density limits
    norm_vol, vmin, vmax = normalize(volume)

    # 2. Tiled 3D inference with smooth 3D Hann blending
    sr_norm, temporary_path = tiled_inference(
        norm_vol,
        generator=generator,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        scale=SCALE
    )

    # 3. Denormalize back to authentic physical Cryo-EM density range
    sr_volume = denormalize(sr_norm, vmin, vmax)

    output_name = os.path.splitext(input_file)[0] + "_SR.mrc"
    output_path = os.path.join(result_dir, output_name)
    os.makedirs(result_dir, exist_ok=True)

    new_voxel_size = (
        voxel_size[0] / SCALE,
        voxel_size[1] / SCALE,
        voxel_size[2] / SCALE
    )

    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(sr_volume, dtype=np.float32))
        mrc.voxel_size = new_voxel_size
        mrc.update_header_stats()
        mrc.update_header_from_data()

    del sr_norm
    del sr_volume
    gc.collect()

    # Clean up temporary memmap files
    for p in [temporary_path, temporary_path + "_weights"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    print(f"  Saved Super-Resolved MRC: {output_path}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="3D Cryo-EM SRGAN Seamless Inference")
    parser.add_argument("--model", type=str, default=None, help="Path to generator model checkpoint")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Input directory or single .mrc file")
    parser.add_argument("--output", type=str, default=RESULT_PATH, help="Directory to save SR outputs")
    parser.add_argument("--tile_size", type=int, default=TILE_SIZE, help="Inference tile size in LR voxels")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help="Overlap between tiles in LR voxels")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("3D CRYO-EM SUPER-RESOLUTION INFERENCE (SEAMLESS 3D BLENDING)")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Locate model checkpoint
    model_to_load = args.model
    if not model_to_load or not os.path.exists(model_to_load):
        candidates = [MODEL_PATH] + FALLBACK_CHECKPOINTS
        for c in candidates:
            if os.path.exists(c):
                model_to_load = c
                break

    if not model_to_load or not os.path.exists(model_to_load):
        raise FileNotFoundError(f"No checkpoint found at '{args.model or MODEL_PATH}' or fallbacks.")

    print(f"Loading checkpoint: {model_to_load}")
    generator = Generator(
        img_feat=1,
        n_feats=N_FEATS,
        num_block=RES_NUM,
        scale=SCALE
    ).to(device)

    checkpoint = torch.load(model_to_load, map_location=device)
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['state_dict'])
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()
    print("Generator loaded successfully.")

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        process_file(
            os.path.basename(args.input),
            generator=generator,
            device=device,
            input_dir=os.path.dirname(args.input),
            result_dir=args.output,
            tile_size=args.tile_size,
            overlap=args.overlap
        )
    else:
        files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(".mrc")])
        if len(files) == 0:
            raise RuntimeError(f"No .mrc files found in {args.input}")
        print(f"Total MRC volumes to process: {len(files)}")
        for f in files:
            process_file(
                f,
                generator=generator,
                device=device,
                input_dir=args.input,
                result_dir=args.output,
                tile_size=args.tile_size,
                overlap=args.overlap
            )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE - All SR volumes saved.")
    print(f"Output directory: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
