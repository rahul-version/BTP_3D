import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import mrcfile

# LOAD MRC

def load_volume(path):

    with mrcfile.open(
        path,
        mode="r",
        permissive=True
    ) as mrc:

        data = np.asarray(
            mrc.data,
            dtype=np.float32
        ).copy()

    return data



# NORMALIZATION

def normalize(volume):

    vmin = volume.min()
    vmax = volume.max()

    if vmax - vmin < 1e-8:

        return np.zeros_like(
            volume,
            dtype=np.float32
        )

    volume = (
        (volume - vmin)
        / (vmax - vmin)
    )

    volume = volume * 2.0 - 1.0

    return volume.astype(
        np.float32
    )



# AUGMENTATION

def augment(lr, hr):

    # Flip depth
    if random.random() < 0.5:

        lr = np.flip(
            lr,
            axis=0
        ).copy()

        hr = np.flip(
            hr,
            axis=0
        ).copy()

    # Flip height
    if random.random() < 0.5:

        lr = np.flip(
            lr,
            axis=1
        ).copy()

        hr = np.flip(
            hr,
            axis=1
        ).copy()

    # Flip width
    if random.random() < 0.5:

        lr = np.flip(
            lr,
            axis=2
        ).copy()

        hr = np.flip(
            hr,
            axis=2
        ).copy()

    # Rotation
    k = random.randint(0, 3)

    if k > 0:

        lr = np.rot90(
            lr,
            k=k,
            axes=(1, 2)
        ).copy()

        hr = np.rot90(
            hr,
            k=k,
            axes=(1, 2)
        ).copy()

    return lr, hr


# TRAINING DATASET
class PairedMRCDataset(Dataset):

    def __init__(
        self,
        lr_path,
        hr_path,
        scale=4,
        patch_size=16
    ):
        self.lr_path = lr_path
        self.hr_path = hr_path

        self.scale = scale
        self.patch_size = patch_size

        self.lr_files = sorted([
            f
            for f in os.listdir(lr_path)
            if f.lower().endswith(".mrc")
        ])

        self.hr_files = sorted([
            f
            for f in os.listdir(hr_path)
            if f.lower().endswith(".mrc")
        ])

        if len(self.lr_files) == 0:
            raise RuntimeError(
                "No LR MRC files found in Dataset/LR"
            )

        if len(self.hr_files) == 0:
            raise RuntimeError(
                "No HR MRC files found in Dataset/HR"
            )

        if len(self.lr_files) != len(self.hr_files):
            raise RuntimeError(
                f"LR files ({len(self.lr_files)}) "
                f"and HR files ({len(self.hr_files)}) "
                "count do not match."
            )

    def __len__(self):

        return len(self.lr_files)

    def __getitem__(self, index):

        lr_file = self.lr_files[index]
        hr_file = self.hr_files[index]

        lr = load_volume(
            os.path.join(
                self.lr_path,
                lr_file
            )
        )

        hr = load_volume(
            os.path.join(
                self.hr_path,
                hr_file
            )
        )

        lr = normalize(lr)
        hr = normalize(hr)

        lr_size = self.patch_size
        hr_size = self.patch_size * self.scale

        if (
            lr.shape[0] < lr_size
            or lr.shape[1] < lr_size
            or lr.shape[2] < lr_size
        ):
            raise RuntimeError(
                f"LR volume {lr_file} is smaller "
                f"than patch size {lr_size}"
            )

        if (
            hr.shape[0] < hr_size
            or hr.shape[1] < hr_size
            or hr.shape[2] < hr_size
        ):
            raise RuntimeError(
                f"HR volume {hr_file} is smaller "
                f"than patch size {hr_size}"
            )

        # Random LR crop
        z = random.randint(
            0,
            lr.shape[0] - lr_size
        )

        y = random.randint(
            0,
            lr.shape[1] - lr_size
        )

        x = random.randint(
            0,
            lr.shape[2] - lr_size
        )

        lr_patch = lr[
            z:z + lr_size,
            y:y + lr_size,
            x:x + lr_size
        ]

        # Corresponding HR crop
        hz = z * self.scale
        hy = y * self.scale
        hx = x * self.scale

        hr_patch = hr[
            hz:hz + hr_size,
            hy:hy + hr_size,
            hx:hx + hr_size
        ]

        lr_patch, hr_patch = augment(
            lr_patch,
            hr_patch
        )

        lr_patch = torch.from_numpy(
            lr_patch
        ).float().unsqueeze(0)

        hr_patch = torch.from_numpy(
            hr_patch
        ).float().unsqueeze(0)

        return lr_patch, hr_patch
