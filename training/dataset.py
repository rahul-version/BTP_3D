import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import mrcfile


# ============================================================
# LOAD MRC
# ============================================================

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


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# AUGMENTATION
# ============================================================

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


# ============================================================
# PAIRED MRC DATASET (Train, Validation, Test Support)
# ============================================================

class PairedMRCDataset(Dataset):

    def __init__(
        self,
        lr_path,
        hr_path,
        scale=4,
        patch_size=16,
        split="all",
        val_files=None,
        test_files=None,
        patches_per_volume=4
    ):
        self.lr_path = lr_path
        self.hr_path = hr_path

        self.scale = scale
        self.patch_size = patch_size
        self.split = split
        self.patches_per_volume = patches_per_volume if split == "train" else 1

        all_lr = sorted([
            f
            for f in os.listdir(lr_path)
            if f.lower().endswith(".mrc")
        ])

        all_hr = sorted([
            f
            for f in os.listdir(hr_path)
            if f.lower().endswith(".mrc")
        ])

        if len(all_lr) == 0:
            raise RuntimeError(
                f"No LR MRC files found in {lr_path}"
            )

        if len(all_hr) == 0:
            raise RuntimeError(
                f"No HR MRC files found in {hr_path}"
            )

        val_set = set(val_files) if val_files else set()
        test_set = set(test_files) if test_files else set()

        def get_stem(filename):
            return filename.replace("_lr.mrc", "").replace(".mrc", "")

        matched_pairs = []
        for lf in all_lr:
            l_stem = get_stem(lf)
            for hf in all_hr:
                if get_stem(hf) == l_stem:
                    matched_pairs.append((lf, hf, l_stem))
                    break

        if split == "val":
            self.pairs = [p for p in matched_pairs if any(v in p[2] for v in val_set)]
            if len(self.pairs) == 0:
                # Fallback: take last 3 if specific names not matched
                self.pairs = matched_pairs[-3:] if len(matched_pairs) >= 3 else matched_pairs
        elif split == "test":
            self.pairs = [p for p in matched_pairs if any(t in p[2] for t in test_set)]
            if len(self.pairs) == 0:
                # Fallback: take previous 3
                self.pairs = matched_pairs[-6:-3] if len(matched_pairs) >= 6 else matched_pairs
        elif split == "train":
            exclude = val_set | test_set
            self.pairs = [p for p in matched_pairs if not any(e in p[2] for e in exclude)]
            if len(self.pairs) == 0:
                self.pairs = matched_pairs[:-6] if len(matched_pairs) >= 6 else matched_pairs
        else:  # "all"
            self.pairs = matched_pairs

        self.lr_files = [p[0] for p in self.pairs]
        self.hr_files = [p[1] for p in self.pairs]

        print(f"[{split.upper()} Dataset] Loaded {len(self.pairs)} volume pairs.")

    def __len__(self):
        return len(self.pairs) * self.patches_per_volume

    def __getitem__(self, index):
        vol_idx = index // self.patches_per_volume
        lr_file = self.lr_files[vol_idx]
        hr_file = self.hr_files[vol_idx]

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
                f"LR volume {lr_file} is smaller than patch size {lr_size}"
            )

        # For validation / test: deterministic center crops so metrics are stable & comparable
        if self.split in ["val", "test"]:
            z = max(0, (lr.shape[0] - lr_size) // 2)
            y = max(0, (lr.shape[1] - lr_size) // 2)
            x = max(0, (lr.shape[2] - lr_size) // 2)
            lr_patch = lr[z:z + lr_size, y:y + lr_size, x:x + lr_size]
            hz, hy, hx = z * self.scale, y * self.scale, x * self.scale
            hr_patch = hr[hz:hz + hr_size, hy:hy + hr_size, hx:hx + hr_size]
        else:
            # Training: Foreground-aware sampling (avoid sampling pure empty solvent)
            best_lr, best_hr = None, None
            max_std = -1.0
            for _ in range(5):
                z = random.randint(0, lr.shape[0] - lr_size)
                y = random.randint(0, lr.shape[1] - lr_size)
                x = random.randint(0, lr.shape[2] - lr_size)

                cand_lr = lr[z:z + lr_size, y:y + lr_size, x:x + lr_size]
                hz, hy, hx = z * self.scale, y * self.scale, x * self.scale
                cand_hr = hr[hz:hz + hr_size, hy:hy + hr_size, hx:hx + hr_size]

                current_std = float(cand_hr.std())
                if current_std > max_std:
                    max_std = current_std
                    best_lr, best_hr = cand_lr, cand_hr
                # If standard deviation indicates dense macromolecular structure, accept immediately
                if current_std > 0.10:
                    break

            lr_patch, hr_patch = best_lr, best_hr
            lr_patch, hr_patch = augment(lr_patch, hr_patch)

        lr_patch = torch.from_numpy(lr_patch).float().unsqueeze(0)
        hr_patch = torch.from_numpy(hr_patch).float().unsqueeze(0)

        return lr_patch, hr_patch
