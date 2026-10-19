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