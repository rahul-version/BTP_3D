import os
import gc

import numpy as np
import torch
import mrcfile


import sys

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
    "checkpoint_600.pt"
)
# SETTINGS
SCALE = 4

N_FEATS = 64

RES_NUM = 16

TILE_SIZE = 32

OVERLAP = 8

# DEVICE
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("3D SRGAN TESTING")
print("=" * 60)

print(
    "Device:",
    device
)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()



# LOAD GENERATOR
generator = Generator(
    img_feat=1,
    n_feats=N_FEATS,
    num_block=RES_NUM,
    scale=SCALE
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and 'generator' in checkpoint:
    generator.load_state_dict(checkpoint['generator'])
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    generator.load_state_dict(checkpoint['state_dict'])
else:
    generator.load_state_dict(checkpoint)

generator.eval()

print(
    "Generator loaded successfully."
)

print()

# LOAD MRC
def load_mrc(path):

    with mrcfile.open(
        path,
        mode="r",
        permissive=True
    ) as mrc:

        data = np.asarray(
            mrc.data,
            dtype=np.float32
        ).copy()

        voxel_size = (
            float(mrc.voxel_size.x),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.z)
        )

    return data, voxel_size

# NORMALIZE
def normalize(volume):

    vmin = volume.min()

    vmax = volume.max()

    if vmax - vmin < 1e-8:

        return np.zeros_like(
            volume,
            dtype=np.float32
        )

    volume = (
        volume - vmin
    ) / (
        vmax - vmin
    )

    volume = volume * 2.0 - 1.0

    return volume.astype(
        np.float32
    )

# TILED INFERENCE
@torch.no_grad()
def tiled_inference(volume):

    depth, height, width = volume.shape

    output_shape = (
        depth * SCALE,
        height * SCALE,
        width * SCALE
    )

    output_path = os.path.join(
        RESULT_PATH,
        "_temporary_sr.dat"
    )

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

    stride = TILE_SIZE - OVERLAP

    z_positions = list(
        range(
            0,
            max(1, depth - TILE_SIZE + 1),
            stride
        )
    )

    y_positions = list(
        range(
            0,
            max(1, height - TILE_SIZE + 1),
            stride
        )
    )

    x_positions = list(
        range(
            0,
            max(1, width - TILE_SIZE + 1),
            stride
        )
    )

    if not z_positions or z_positions[-1] != depth - TILE_SIZE:
        z_positions.append(
            max(0, depth - TILE_SIZE)
        )

    if not y_positions or y_positions[-1] != height - TILE_SIZE:
        y_positions.append(
            max(0, height - TILE_SIZE)
        )

    if not x_positions or x_positions[-1] != width - TILE_SIZE:
        x_positions.append(
            max(0, width - TILE_SIZE)
        )

    total_tiles = (
        len(z_positions)
        * len(y_positions)
        * len(x_positions)
    )

    tile_number = 0

    for z in z_positions:

        for y in y_positions:

            for x in x_positions:

                tile_number += 1

                print(
                    f"\rProcessing tile "
                    f"{tile_number}/{total_tiles}",
                    end=""
                )

                tile = volume[
                    z:z + TILE_SIZE,
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ]

                tile_tensor = torch.from_numpy(
                    tile
                ).float()

                tile_tensor = tile_tensor.unsqueeze(
                    0
                ).unsqueeze(
                    0
                )

                tile_tensor = tile_tensor.to(
                    device
                )

                sr, _ = generator(
                    tile_tensor
                )

                sr = sr.squeeze().cpu().numpy()

                oz = z * SCALE
                oy = y * SCALE
                ox = x * SCALE

                sz = sr.shape[0]
                sy = sr.shape[1]
                sx = sr.shape[2]

                output[
                    oz:oz + sz,
                    oy:oy + sy,
                    ox:ox + sx
                ] += sr

                weights[
                    oz:oz + sz,
                    oy:oy + sy,
                    ox:ox + sx
                ] += 1.0

                del tile_tensor
                del sr

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print()

    output /= np.maximum(
        weights,
        1e-8
    )

    output.flush()
    weights.flush()

    del weights

    gc.collect()

    return output, output_path

# PROCESS ONE FIL

def process_file(input_file):

    input_path = os.path.join(
        INPUT_PATH,
        input_file
    )

    print()
    print(
        "Processing:",
        input_file
    )

    volume, voxel_size = load_mrc(
        input_path
    )

    print(
        "Input shape:",
        volume.shape
    )

    print(
        "Input voxel size:",
        voxel_size
    )

    volume = normalize(
        volume
    )

    sr_volume, temporary_path = tiled_inference(
        volume
    )

    output_name = (
        os.path.splitext(input_file)[0]
        + "_SR.mrc"
    )

    output_path = os.path.join(
        RESULT_PATH,
        output_name
    )

    print(
        "Saving:",
        output_path
    )

    new_voxel_size = (
        voxel_size[0] / SCALE,
        voxel_size[1] / SCALE,
        voxel_size[2] / SCALE
    )

    with mrcfile.new(
        output_path,
        overwrite=True
    ) as mrc:

        mrc.set_data(
            np.asarray(
                sr_volume,
                dtype=np.float32
            )
        )

        mrc.voxel_size = new_voxel_size

        mrc.update_header_stats()

        mrc.update_header_from_data()

    del sr_volume

    gc.collect()

    # Remove temporary files
    try:
        os.remove(
            temporary_path
        )
    except:
        pass

    try:
        os.remove(
            temporary_path + "_weights"
        )
    except:
        pass

    print(
        "Saved successfully."
    )

# MAIN

def main():

    os.makedirs(
        RESULT_PATH,
        exist_ok=True
    )

    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            "pre_trained_model_final.pt not found in model folder. "
            "Train the model first."
        )

    files = sorted([
        f
        for f in os.listdir(INPUT_PATH)
        if f.lower().endswith(".mrc")
    ])

    if len(files) == 0:

        raise RuntimeError(
            "No .mrc files found in input folder."
        )

    print(
        "Input MRC files:",
        len(files)
    )

    for file in files:

        process_file(file)

    print()
    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

    print(
        "SR files saved in:",
        RESULT_PATH
    )


if __name__ == "__main__":

    main()