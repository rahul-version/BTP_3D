import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from config import (
    HR_PATH,
    LR_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    PLOTS_DIR,
    SCALE,
    PATCH_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_FEAT,
    N_FEATS,
    RES_NUM,
    NUM_D_BLOCK,
    PRE_TRAIN_EPOCH,
    FINE_TRAIN_EPOCH,
    LEARNING_RATE,
    L2_COEFF,
    ADV_COEFF,
    GRAD_COEFF,
    SPECTRAL_COEFF,
    TV_LOSS_COEFF,
    VAL_SPLIT_FILES,
    TEST_SPLIT_FILES,
    GENERATOR_PRETRAINED,
    GENERATOR_FINAL,
    DISCRIMINATOR_FINAL,
    BEST_GENERATOR
)

from dataset import PairedMRCDataset
from srgan_model import Generator, Discriminator
from losses import GradientLoss3D, SpectralLoss3D, TVLoss3D
from utils import set_seed, get_device, ensure_dirs, count_parameters, save_checkpoint


# ============================================================
# METRICS HELPER (PSNR & SSIM)
# ============================================================

def calc_psnr_batch(pred, target, data_range=2.0):
    """Calculates average PSNR in dB over batch of normalized volumes [-1, 1]."""
    mse = torch.mean((pred - target) ** 2, dim=(-3, -2, -1)).view(-1)
    psnrs = []
    for m in mse:
        val = m.item()
        if val <= 1e-10:
            psnrs.append(100.0)
        else:
            psnrs.append(10.0 * np.log10((data_range ** 2) / val))
    return float(np.mean(psnrs))


def calc_ssim_batch(pred, target, data_range=2.0):
    """Calculates mean 3D structural similarity index over batch."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = pred.mean(dim=(-3, -2, -1), keepdim=True)
    mu_y = target.mean(dim=(-3, -2, -1), keepdim=True)
    sigma_x_sq = ((pred - mu_x) ** 2).mean(dim=(-3, -2, -1), keepdim=True)
    sigma_y_sq = ((target - mu_y) ** 2).mean(dim=(-3, -2, -1), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-3, -2, -1), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return float(ssim_map.mean().item())


# ============================================================
# VALIDATION EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_validation(generator, val_loader, device, mse_loss_fn):
    generator.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    n = len(val_loader)

    if n == 0:
        return 0.0, 0.0, 0.0

    for lr, hr in val_loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr, _ = generator(lr)

        loss = mse_loss_fn(sr, hr)
        val_loss += loss.item()
        val_psnr += calc_psnr_batch(sr, hr, data_range=2.0)
        val_ssim += calc_ssim_batch(sr, hr, data_range=2.0)

    return val_loss / n, val_psnr / n, val_ssim / n


# ============================================================
# PLOTTING TRAINING & VALIDATION DASHBOARD (IN PARENT FOLDER)
# ============================================================

def plot_training_curves(history, output_png):
    try:
        import matplotlib.pyplot as plt
        if not history or "epoch" not in history or len(history["epoch"]) == 0:
            return

        epochs = history["epoch"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        fig.suptitle("3D Cryo-EM SRGAN Training & Validation Progress", fontsize=15, fontweight="bold")

        # 1. Generator & Discriminator Losses
        axes[0, 0].plot(epochs, history.get("train_g_loss", []), label="Generator Loss", color="#0066cc", lw=2)
        if any(d > 0 for d in history.get("train_d_loss", [])):
            axes[0, 0].plot(epochs, history.get("train_d_loss", []), label="Discriminator Loss", color="#e67e22", lw=2)
        axes[0, 0].set_title("Training Loss Trajectory", fontsize=11, fontweight="bold")
        axes[0, 0].set_xlabel("Overall Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, linestyle=":", alpha=0.6)
        axes[0, 0].legend()

        # 2. Train MSE vs Validation MSE
        axes[0, 1].plot(epochs, history.get("train_mse", []), label="Train MSE", color="#2980b9", lw=2)
        axes[0, 1].plot(epochs, history.get("val_loss", []), label="Val MSE", color="#c0392b", lw=2, linestyle="--")
        axes[0, 1].set_title("Reconstruction Error (Train vs Validation)", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Overall Epoch")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)
        axes[0, 1].legend()

        # 3. Validation PSNR
        val_psnr = history.get("val_psnr", [])
        axes[1, 0].plot(epochs, val_psnr, label="Val 3D-PSNR (dB)", color="#27ae60", lw=2)
        if len(val_psnr) > 0:
            best_idx = int(np.argmax(val_psnr))
            axes[1, 0].scatter([epochs[best_idx]], [val_psnr[best_idx]], color="red", s=60, zorder=5,
                               label=f"Best: {val_psnr[best_idx]:.2f} dB (Ep {epochs[best_idx]})")
        axes[1, 0].set_title("Validation 3D-PSNR", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Overall Epoch")
        axes[1, 0].set_ylabel("PSNR (dB)")
        axes[1, 0].grid(True, linestyle=":", alpha=0.6)
        axes[1, 0].legend()

        # 4. Validation 3D-SSIM
        val_ssim = history.get("val_ssim", [])
        axes[1, 1].plot(epochs, val_ssim, label="Val 3D-SSIM", color="#8e44ad", lw=2)
        if len(val_ssim) > 0:
            best_ssim_idx = int(np.argmax(val_ssim))
            axes[1, 1].scatter([epochs[best_ssim_idx]], [val_ssim[best_ssim_idx]], color="red", s=60, zorder=5,
                               label=f"Best: {val_ssim[best_ssim_idx]:.4f} (Ep {epochs[best_ssim_idx]})")
        axes[1, 1].set_title("Validation 3D-SSIM (Structure Preservation)", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Overall Epoch")
        axes[1, 1].set_ylabel("SSIM (0 - 1.0)")
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)
        axes[1, 1].legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png)
        plt.close()
    except Exception as e:
        print(f"  [Warning] Could not save training curves plot: {e}")


def save_history_csv(history, csv_path):
    if not history or "epoch" not in history:
        return
    keys = list(history.keys())
    n = len(history["epoch"])
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([history[k][i] if i < len(history[k]) else "" for k in keys])


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    set_seed(42)
    device = get_device()
    ensure_dirs(MODEL_PATH)
    ensure_dirs(PLOTS_DIR)

    print("=" * 60)
    print("3D CRYO-EM SRGAN TRAINING WITH FULL VALIDATION & TRACKING")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 1. Datasets & Loaders: Train, Validation, Testing sets
    train_dataset = PairedMRCDataset(
        lr_path=LR_PATH,
        hr_path=HR_PATH,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        split="train",
        val_files=VAL_SPLIT_FILES,
        test_files=TEST_SPLIT_FILES,
        patches_per_volume=4
    )

    val_dataset = PairedMRCDataset(
        lr_path=LR_PATH,
        hr_path=HR_PATH,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        split="val",
        val_files=VAL_SPLIT_FILES,
        test_files=TEST_SPLIT_FILES,
        patches_per_volume=1
    )

    test_dataset = PairedMRCDataset(
        lr_path=LR_PATH,
        hr_path=HR_PATH,
        scale=SCALE,
        patch_size=PATCH_SIZE,
        split="test",
        val_files=VAL_SPLIT_FILES,
        test_files=TEST_SPLIT_FILES,
        patches_per_volume=1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    print(f"Dataset Partitioning: Train={len(train_dataset.pairs)} vols | Val={len(val_dataset.pairs)} vols | Test={len(test_dataset.pairs)} vols\n")

    # 2. Models
    generator = Generator(
        img_feat=IMG_FEAT,
        n_feats=N_FEATS,
        num_block=RES_NUM,
        scale=SCALE
    ).to(device)

    discriminator = Discriminator(
        img_feat=IMG_FEAT,
        n_feats=N_FEATS,
        num_block=NUM_D_BLOCK
    ).to(device)

    print("Generator parameters:", count_parameters(generator))
    print("Discriminator parameters:", count_parameters(discriminator))
    print()

    # 3. Losses
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    gradient_loss = GradientLoss3D()
    spectral_loss = SpectralLoss3D()
    tv_loss = TVLoss3D()

    # 4. Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=2000, gamma=0.1)

    # 5. Checkpoint & History Loading
    LATEST_CHECKPOINT = os.path.join(MODEL_PATH, "latest_checkpoint.pt")
    HISTORY_JSON = os.path.join(PROJECT_ROOT, "training_history.json")
    HISTORY_CSV = os.path.join(PROJECT_ROOT, "training_history.csv")
    PARENT_PLOT_PNG = os.path.join(PROJECT_ROOT, "training_validation_curves.png")

    history = {
        "epoch": [],
        "cycle": [],
        "phase": [],
        "train_g_loss": [],
        "train_d_loss": [],
        "train_mse": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "improved": []
    }

    if os.path.exists(HISTORY_JSON):
        try:
            with open(HISTORY_JSON, "r") as f:
                history = json.load(f)
        except Exception:
            pass

    start_cycle = 1
    start_phase = 1
    start_epoch = 0
    best_val_psnr = max(history["val_psnr"]) if len(history.get("val_psnr", [])) > 0 else -1e9

    if os.path.exists(LATEST_CHECKPOINT):
        print("=" * 60)
        print("LOADING CHECKPOINT")
        print("=" * 60)
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device)
        start_cycle = checkpoint.get('cycle', 1)
        start_phase = checkpoint.get('phase', 1)
        start_epoch = checkpoint.get('epoch', -1) + 1

        if 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        if 'g_optimizer' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        if 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if 'd_optimizer' in checkpoint:
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        print(f"Resuming from Cycle {start_cycle}, Phase {start_phase}, Epoch {start_epoch}")
        print()
    else:
        print("No checkpoint found. Starting training from scratch.\n")

    # 6. Training Loop with Integrated Validation & Graphing
    cycle = start_cycle
    target_cycle = start_cycle + 5

    while cycle < target_cycle:
        print(f"\n{'*' * 60}")
        print(f"STARTING CYCLE {cycle}")
        print(f"{'*' * 60}\n")

        # ----------------------------------------------------
        # PHASE 1: GENERATOR MSE PRE-TRAINING
        # ----------------------------------------------------
        if start_phase == 1:
            print("=" * 60)
            print("PHASE 1: GENERATOR PRE-TRAINING")
            print("=" * 60)

            target_epoch = PRE_TRAIN_EPOCH
            for epoch in range(start_epoch, target_epoch):
                generator.train()
                abs_epoch = (cycle - 1) * (PRE_TRAIN_EPOCH + FINE_TRAIN_EPOCH) + epoch + 1
                epoch_loss = 0.0

                n_batches = len(train_loader)
                for i, (lr, hr) in enumerate(train_loader):
                    lr = lr.to(device, non_blocking=True)
                    hr = hr.to(device, non_blocking=True)

                    g_optimizer.zero_grad()
                    sr, _ = generator(lr)
                    loss = mse_loss(sr, hr)
                    loss.backward()
                    g_optimizer.step()

                    epoch_loss += loss.item()
                    percent = (i + 1) / n_batches * 100
                    print(f"\r[Cycle {cycle}] Epoch {abs_epoch} (Phase 1: {epoch + 1}/{target_epoch}) [{i + 1}/{n_batches}] - {percent:.1f}% | Loss: {loss.item():.6f} ", end="", flush=True)

                epoch_loss /= n_batches

                # Validation step after each pre-train epoch
                val_l, val_p, val_s = evaluate_validation(generator, val_loader, device, mse_loss)
                improved = val_p > best_val_psnr
                if improved:
                    best_val_psnr = val_p
                    save_checkpoint(generator, BEST_GENERATOR)

                print(f"\n[Cycle {cycle}] Epoch {abs_epoch} Completed | Train MSE: {epoch_loss:.6f} | Val MSE: {val_l:.6f} | Val PSNR: {val_p:.2f} dB | Val SSIM: {val_s:.4f}" + (" [★ IMPROVED]" if improved else ""))

                # Record history
                history["epoch"].append(abs_epoch)
                history["cycle"].append(cycle)
                history["phase"].append(1)
                history["train_g_loss"].append(epoch_loss)
                history["train_d_loss"].append(0.0)
                history["train_mse"].append(epoch_loss)
                history["val_loss"].append(val_l)
                history["val_psnr"].append(val_p)
                history["val_ssim"].append(val_s)
                history["improved"].append(1 if improved else 0)

                # Save history, plots, and checkpoint
                with open(HISTORY_JSON, "w") as f:
                    json.dump(history, f, indent=2)
                save_history_csv(history, HISTORY_CSV)
                plot_training_curves(history, PARENT_PLOT_PNG)

                torch.save({
                    'cycle': cycle,
                    'phase': 1,
                    'epoch': epoch,
                    'total_epochs': abs_epoch,
                    'generator': generator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'val_psnr': val_p,
                    'val_ssim': val_s
                }, LATEST_CHECKPOINT)

            save_checkpoint(generator, GENERATOR_PRETRAINED)
            print("\nPretrained generator saved:", GENERATOR_PRETRAINED, "\n")
            start_phase = 2
            start_epoch = 0

        # ----------------------------------------------------
        # PHASE 2: ADVERSARIAL FINE-TUNING
        # ----------------------------------------------------
        if start_phase == 2:
            print("=" * 60)
            print("PHASE 2: ADVERSARIAL FINE-TUNING")
            print("=" * 60)

            target_epoch = FINE_TRAIN_EPOCH
            for epoch in range(start_epoch, target_epoch):
                abs_epoch = (cycle - 1) * (PRE_TRAIN_EPOCH + FINE_TRAIN_EPOCH) + PRE_TRAIN_EPOCH + epoch + 1

                generator.train()
                discriminator.train()

                total_g_loss = 0.0
                total_d_loss = 0.0
                total_mse = 0.0

                n_batches = len(train_loader)
                for i, (lr, hr) in enumerate(train_loader):
                    lr = lr.to(device, non_blocking=True)
                    hr = hr.to(device, non_blocking=True)

                    # 1. Forward Generator
                    sr, _ = generator(lr)

                    # 2. Discriminator Step
                    d_optimizer.zero_grad()
                    real_output = discriminator(hr)
                    fake_output = discriminator(sr.detach())

                    real_labels = torch.ones_like(real_output)
                    fake_labels = torch.zeros_like(fake_output)

                    real_loss = bce_loss(real_output, real_labels)
                    fake_loss = bce_loss(fake_output, fake_labels)
                    d_loss = (real_loss + fake_loss) * 0.5
                    d_loss.backward()
                    d_optimizer.step()

                    # 3. Generator Step
                    g_optimizer.zero_grad()
                    fake_output = discriminator(sr)

                    mse = mse_loss(sr, hr)
                    adv = bce_loss(fake_output, real_labels)
                    grad = gradient_loss(sr, hr)
                    spec = spectral_loss(sr, hr)
                    tv = tv_loss(sr)

                    g_loss = (
                        L2_COEFF * mse
                        + ADV_COEFF * adv
                        + GRAD_COEFF * grad
                        + SPECTRAL_COEFF * spec
                        + TV_LOSS_COEFF * tv
                    )
                    g_loss.backward()
                    g_optimizer.step()
                    scheduler.step()

                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()
                    total_mse += mse.item()

                    percent = (i + 1) / n_batches * 100
                    print(f"\r[Cycle {cycle}] Epoch {abs_epoch} (Phase 2: {epoch + 1}/{target_epoch}) [{i + 1}/{n_batches}] - {percent:.1f}% ", end="", flush=True)

                total_g_loss /= n_batches
                total_d_loss /= n_batches
                total_mse /= n_batches

                # Validation step after each fine-tune epoch
                val_l, val_p, val_s = evaluate_validation(generator, val_loader, device, mse_loss)
                improved = val_p > best_val_psnr
                if improved:
                    best_val_psnr = val_p
                    save_checkpoint(generator, BEST_GENERATOR)

                print(
                    f"\n[Cycle {cycle}] Epoch {abs_epoch} Completed | "
                    f"G: {total_g_loss:.4f} | D: {total_d_loss:.4f} | MSE: {total_mse:.6f} | "
                    f"Val PSNR: {val_p:.2f} dB | Val SSIM: {val_s:.4f}" + (" [★ IMPROVED]" if improved else "")
                )

                # Record history
                history["epoch"].append(abs_epoch)
                history["cycle"].append(cycle)
                history["phase"].append(2)
                history["train_g_loss"].append(total_g_loss)
                history["train_d_loss"].append(total_d_loss)
                history["train_mse"].append(total_mse)
                history["val_loss"].append(val_l)
                history["val_psnr"].append(val_p)
                history["val_ssim"].append(val_s)
                history["improved"].append(1 if improved else 0)

                with open(HISTORY_JSON, "w") as f:
                    json.dump(history, f, indent=2)
                save_history_csv(history, HISTORY_CSV)
                plot_training_curves(history, PARENT_PLOT_PNG)

                torch.save({
                    'cycle': cycle,
                    'phase': 2,
                    'epoch': epoch,
                    'total_epochs': abs_epoch,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_psnr': val_p,
                    'val_ssim': val_s
                }, LATEST_CHECKPOINT)

            start_phase = 1
            start_epoch = 0

            save_checkpoint(generator, GENERATOR_FINAL)
            save_checkpoint(discriminator, DISCRIMINATOR_FINAL)

            total_epochs = cycle * (PRE_TRAIN_EPOCH + FINE_TRAIN_EPOCH)
            checkpoint_filename = f"checkpoint_{total_epochs}.pt"
            checkpoint_path = os.path.join(MODEL_PATH, checkpoint_filename)

            checkpoint_data = {
                'cycle': cycle + 1,
                'phase': 1,
                'epoch': -1,
                'total_epochs': total_epochs,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_psnr': val_p,
                'val_ssim': val_s
            }
            torch.save(checkpoint_data, checkpoint_path)
            torch.save(checkpoint_data, LATEST_CHECKPOINT)

            print()
            print("=" * 60)
            print(f"CYCLE {cycle} COMPLETE - MILESTONE CHECKPOINT SAVED")
            print(f"Checkpoint Name: {checkpoint_filename}")
            print(f"Validation PSNR: {val_p:.2f} dB | SSIM: {val_s:.4f}")
            print(f"Plots updated in: {PARENT_PLOT_PNG}")
            print("=" * 60)
            print()

            cycle += 1


if __name__ == "__main__":
    main()
