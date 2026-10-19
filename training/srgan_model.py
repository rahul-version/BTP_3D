import torch
import torch.nn as nn

from ops import conv, ResBlock, Upsampler, discrim_block



# GENERATOR
class Generator(nn.Module):

    def __init__(
        self,
        img_feat=1,
        n_feats=64,
        kernel_size=3,
        num_block=16,
        act=nn.PReLU(),
        scale=4
    ):
        super(Generator, self).__init__()

        self.conv01 = conv(
            in_channel=img_feat,
            out_channel=n_feats,
            kernel_size=9,
            BN=False,
            act=act
        )

        resblocks = [
            ResBlock(
                channels=n_feats,
                kernel_size=3,
                act=act
            )
            for _ in range(num_block)
        ]

        self.body = nn.Sequential(*resblocks)

        self.conv02 = conv(
            in_channel=n_feats,
            out_channel=n_feats,
            kernel_size=3,
            BN=True,
            act=None
        )

        if scale == 4:

            upsample_blocks = [
                Upsampler(
                    channel=n_feats,
                    kernel_size=3,
                    scale=2,
                    act=act
                ),
                Upsampler(
                    channel=n_feats,
                    kernel_size=3,
                    scale=2,
                    act=act
                )
            ]

        else:

            upsample_blocks = [
                Upsampler(
                    channel=n_feats,
                    kernel_size=3,
                    scale=scale,
                    act=act
                )
            ]

        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = conv(
            in_channel=n_feats,
            out_channel=img_feat,
            kernel_size=3,
            BN=False,
            act=nn.Tanh()
        )

    def forward(self, x):

        x = self.conv01(x)

        skip = x

        x = self.body(x)

        x = self.conv02(x)

        feat = x + skip

        x = self.tail(feat)

        out = self.last_conv(x)

        return out, feat



# DISCRIMINATOR
class Discriminator(nn.Module):

    def __init__(
        self,
        img_feat=1,
        n_feats=64,
        num_block=3
    ):
        super(Discriminator, self).__init__()

        layers = []

        in_channels = img_feat
        out_channels = n_feats

        for i in range(num_block):

            layers.append(
                discrim_block(
                    in_feats=in_channels,
                    out_feats=out_channels,
                    kernel_size=3
                )
            )

            in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*layers)

        final_channels = n_feats * (2 ** (num_block - 1))

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                final_channels,
                1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x