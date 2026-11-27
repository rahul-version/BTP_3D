import torch
import torch.nn as nn
import torch.nn.functional as F


class _conv3d(nn.Conv3d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True
    ):
        super(_conv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.weight.data = torch.normal(
            torch.zeros(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                kernel_size
            ),
            0.02
        )

        if self.bias is not None:
            self.bias.data = torch.zeros(out_channels)

        for p in self.parameters():
            p.requires_grad = True


class conv(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        BN=False,
        act=None,
        stride=1,
        bias=True
    ):
        super(conv, self).__init__()

        padding = kernel_size // 2

        layers = []

        layers.append(
            _conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        )

        if BN:
            layers.append(
                nn.BatchNorm3d(
                    num_features=out_channel
                )
            )

        if act is not None:
            layers.append(act)

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size,
        act=nn.ReLU(inplace=True),
        bias=True
    ):
        super(ResBlock, self).__init__()

        layers = [
            conv(
                channels,
                channels,
                kernel_size,
                BN=True,
                act=act,
                bias=bias
            ),

            conv(
                channels,
                channels,
                kernel_size,
                BN=True,
                act=None,
                bias=bias
            )
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x) + x


# ============================================================
# RESNET-50 3D BOTTLENECK BLOCK (He et al., CVPR 2016 / Med3D)
# ============================================================

class Bottleneck3D(nn.Module):
    """
    3D Bottleneck Residual Block for ResNet-50.
    Based on:
      - He et al. (CVPR 2016): 'Deep Residual Learning for Image Recognition'
      - Chen et al. (2019): 'Med3D: Transfer Learning for 3D Medical Image Analysis' (MedicalNet)
      
    Structure:
      1. 1x1x1 conv (channel compression: in_channels -> planes) + BN + Act
      2. 3x3x3 conv (spatial 3D feature extraction: planes -> planes) + BN + Act
      3. 1x1x1 conv (channel expansion: planes -> planes * 4) + BN
      + Identity or 1x1x1 projection shortcut connection
    """
    expansion = 4

    def __init__(
        self,
        in_channels,
        planes,
        stride=1,
        act=nn.PReLU(),
        bias=False
    ):
        super(Bottleneck3D, self).__init__()
        out_channels = planes * self.expansion

        # 1x1x1 conv: channel compression
        self.conv1 = conv(
            in_channel=in_channels,
            out_channel=planes,
            kernel_size=1,
            BN=True,
            act=act,
            stride=1,
            bias=bias
        )

        # 3x3x3 conv: 3D spatial feature extraction
        self.conv2 = conv(
            in_channel=planes,
            out_channel=planes,
            kernel_size=3,
            BN=True,
            act=act,
            stride=stride,
            bias=bias
        )

        # 1x1x1 conv: channel expansion (x4 expansion factor)
        self.conv3 = conv(
            in_channel=planes,
            out_channel=out_channels,
            kernel_size=1,
            BN=True,
            act=None,
            stride=1,
            bias=bias
        )

        # Shortcut connection (1x1x1 projection if in_channels != out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = conv(
                in_channel=in_channels,
                out_channel=out_channels,
                kernel_size=1,
                BN=True,
                act=None,
                stride=stride,
                bias=bias
            )
        else:
            self.shortcut = nn.Identity()

        self.act = act

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + residual
        return self.act(out) if self.act is not None else out


class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_res_block,
        act=nn.ReLU(inplace=True)
    ):
        super(BasicBlock, self).__init__()

        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size,
            BN=False,
            act=act
        )

        layers = [
            ResBlock(
                out_channels,
                kernel_size,
                act
            )
            for _ in range(num_res_block)
        ]

        layers.append(
            conv(
                out_channels,
                out_channels,
                kernel_size,
                BN=True,
                act=None
            )
        )

        self.body = nn.Sequential(*layers)

    def forward(self, x):

        res = self.conv(x)

        return self.body(res) + res


class Upsampler(nn.Module):

    def __init__(
        self,
        channel,
        kernel_size,
        scale,
        act=nn.ReLU(inplace=True)
    ):
        super(Upsampler, self).__init__()

        self.scale = scale

        self.refine = conv(
            channel,
            channel,
            kernel_size,
            BN=False,
            act=act
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="trilinear",
            align_corners=False
        )

        return self.refine(x)


class discrim_block(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        kernel_size,
        act=nn.LeakyReLU(
            0.2,
            inplace=True
        )
    ):
        super(discrim_block, self).__init__()

        layers = [
            conv(
                in_feats,
                out_feats,
                kernel_size,
                BN=True,
                act=act
            ),

            conv(
                out_feats,
                out_feats,
                kernel_size,
                BN=True,
                act=act,
                stride=2
            )
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)