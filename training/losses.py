import torch
import torch.nn as nn


# 3D GRADIENT LOSS
class GradientLoss3D(nn.Module):

    def __init__(self):
        super(GradientLoss3D, self).__init__()

    def forward(self, pred, target):

        pred_d = torch.abs(
            pred[:, :, 1:, :, :]
            - pred[:, :, :-1, :, :]
        )

        target_d = torch.abs(
            target[:, :, 1:, :, :]
            - target[:, :, :-1, :, :]
        )

        pred_h = torch.abs(
            pred[:, :, :, 1:, :]
            - pred[:, :, :, :-1, :]
        )

        target_h = torch.abs(
            target[:, :, :, 1:, :]
            - target[:, :, :, :-1, :]
        )

        pred_w = torch.abs(
            pred[:, :, :, :, 1:]
            - pred[:, :, :, :, :-1]
        )

        target_w = torch.abs(
            target[:, :, :, :, 1:]
            - target[:, :, :, :, :-1]
        )

        loss = (
            torch.mean(
                torch.abs(
                    pred_d - target_d
                )
            )
            +
            torch.mean(
                torch.abs(
                    pred_h - target_h
                )
            )
            +
            torch.mean(
                torch.abs(
                    pred_w - target_w
                )
            )
        )

        return loss

# SPECTRAL LOSS

class SpectralLoss3D(nn.Module):

    def __init__(self):
        super(SpectralLoss3D, self).__init__()

    def forward(self, pred, target):

        pred_fft = torch.fft.fftn(
            pred,
            dim=(-3, -2, -1)
        )

        target_fft = torch.fft.fftn(
            target,
            dim=(-3, -2, -1)
        )

        pred_power = torch.abs(
            pred_fft
        ) ** 2

        target_power = torch.abs(
            target_fft
        ) ** 2

        pred_power = torch.log1p(
            pred_power
        )

        target_power = torch.log1p(
            target_power
        )

        return torch.mean(
            (pred_power - target_power) ** 2
        )


# TOTAL VARIATION LOSS
class TVLoss3D(nn.Module):

    def __init__(self):
        super(TVLoss3D, self).__init__()

    def forward(self, x):

        loss_d = torch.mean(
            torch.abs(
                x[:, :, 1:, :, :]
                - x[:, :, :-1, :, :]
            )
        )

        loss_h = torch.mean(
            torch.abs(
                x[:, :, :, 1:, :]
                - x[:, :, :, :-1, :]
            )
        )

        loss_w = torch.mean(
            torch.abs(
                x[:, :, :, :, 1:]
                - x[:, :, :, :, :-1]
            )
        )

        return (
            loss_d
            + loss_h
            + loss_w
        )