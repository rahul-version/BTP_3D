import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    HR_PATH,
    LR_PATH,
    MODEL_PATH,
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
    GENERATOR_PRETRAINED,
    GENERATOR_FINAL,
    DISCRIMINATOR_FINAL
)

from dataset import PairedMRCDataset
from srgan_model import Generator, Discriminator
from losses import GradientLoss3D, SpectralLoss3D, TVLoss3D
from utils import set_seed, get_device, ensure_dirs, count_parameters, save_checkpoint

def main():
    # ============================================================
    # SETUP
    # ============================================================

    set_seed(42)
    device = get_device()
    ensure_dirs(MODEL_PATH)

    print("=" * 60)
    print("3D SRGAN TRAINING WITH CHECKPOINTS")
    print("=" * 60)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print()

    # ============================================================
    # DATASET
    # ============================================================

    dataset = PairedMRCDataset(
        lr_path=LR_PATH,
        hr_path=HR_PATH,
        scale=SCALE,
        patch_size=PATCH_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    print("Training volumes:", len(dataset))
    print()

    # ============================================================
    # MODELS
    # ============================================================

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

    # ============================================================
    # LOSSES
    # ============================================================

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    gradient_loss = GradientLoss3D()
    spectral_loss = SpectralLoss3D()
    tv_loss = TVLoss3D()

    # ============================================================
    # OPTIMIZERS
    # ============================================================

    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE
    )

    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.StepLR(
        g_optimizer,
        step_size=2000,
        gamma=0.1
    )

    # ============================================================
    # CHECKPOINT LOADING
    # ============================================================
    
    LATEST_CHECKPOINT = os.path.join(MODEL_PATH, "latest_checkpoint.pt")
    
    start_phase = 1
    start_epoch = 0

    if os.path.exists(LATEST_CHECKPOINT):
        print("=" * 60)
        print("LOADING CHECKPOINT")
        print("=" * 60)
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device)
        start_phase = checkpoint.get('phase', 1)
        start_epoch = checkpoint.get('epoch', -1) + 1
        
        if 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        if 'g_optimizer' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        
        if start_phase == 2:
            if 'discriminator' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator'])
            if 'd_optimizer' in checkpoint:
                d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                
        print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
        print()
    else:
        print("No checkpoint found. Starting training from scratch.")
        print()

    # ============================================================
    # PHASE 1: MSE PRE-TRAINING
    # ============================================================

    if start_phase == 1:
        print("=" * 60)
        print("PHASE 1: GENERATOR PRE-TRAINING")
        print("=" * 60)

        generator.train()

        for epoch in range(start_epoch, PRE_TRAIN_EPOCH):
            epoch_loss = 0.0

            for lr, hr in loader:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)

                g_optimizer.zero_grad()
                sr, _ = generator(lr)
                loss = mse_loss(sr, hr)
                loss.backward()
                g_optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)

            print(f"Pretrain Epoch {epoch + 1}/{PRE_TRAIN_EPOCH} - MSE: {epoch_loss:.6f}")
            
            # Save checkpoint
            torch.save({
                'phase': 1,
                'epoch': epoch,
                'generator': generator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
            }, LATEST_CHECKPOINT)

        # Save pretrained generator
        save_checkpoint(generator, GENERATOR_PRETRAINED)
        print("\nPretrained generator saved:", GENERATOR_PRETRAINED, "\n")
        
        # Setup for Phase 2
        start_phase = 2
        start_epoch = 0

    # ============================================================
    # PHASE 2: ADVERSARIAL FINE-TUNING
    # ============================================================

    if start_phase == 2:
        print("=" * 60)
        print("PHASE 2: ADVERSARIAL FINE-TUNING")
        print("=" * 60)

        for epoch in range(start_epoch, FINE_TRAIN_EPOCH):
            generator.train()
            discriminator.train()

            total_g_loss = 0.0
            total_d_loss = 0.0
            total_mse = 0.0
            total_adv = 0.0
            total_grad = 0.0
            total_spec = 0.0
            total_tv = 0.0

            for lr, hr in loader:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)

                # GENERATE SR
                sr, _ = generator(lr)

                # DISCRIMINATOR
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

                # GENERATOR
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

                # RECORD LOSSES
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                total_mse += mse.item()
                total_adv += adv.item()
                total_grad += grad.item()
                total_spec += spec.item()
                total_tv += tv.item()

            # EPOCH AVERAGE
            n = len(loader)
            total_g_loss /= n
            total_d_loss /= n
            total_mse /= n
            total_adv /= n
            total_grad /= n
            total_spec /= n
            total_tv /= n

            print(
                f"Fine Epoch {epoch + 1}/{FINE_TRAIN_EPOCH} | "
                f"G: {total_g_loss:.6f} | D: {total_d_loss:.6f} | "
                f"MSE: {total_mse:.6f} | ADV: {total_adv:.6f} | "
                f"GRAD: {total_grad:.6f} | SPEC: {total_spec:.6f} | "
                f"TV: {total_tv:.6f}"
            )

            # Save checkpoint
            torch.save({
                'phase': 2,
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, LATEST_CHECKPOINT)

        # ============================================================
        # SAVE FINAL MODELS
        # ============================================================

        save_checkpoint(generator, GENERATOR_FINAL)
        save_checkpoint(discriminator, DISCRIMINATOR_FINAL)

        # Save checkpoint 300 (100 training + 200 fine tuning)
        total_epochs = PRE_TRAIN_EPOCH + FINE_TRAIN_EPOCH
        checkpoint_filename = f"checkpoint_{total_epochs}.pt"
        checkpoint_path = os.path.join(MODEL_PATH, checkpoint_filename)
        torch.save({
            'phase': 2,
            'epoch': FINE_TRAIN_EPOCH - 1,
            'total_epochs': total_epochs,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, checkpoint_path)

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print("Generator:", GENERATOR_FINAL)
        print("Discriminator:", DISCRIMINATOR_FINAL)
        print("=" * 60)


if __name__ == "__main__":
    main()
