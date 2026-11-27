import torch
import torch.nn as nn

from ops import conv, ResBlock, Bottleneck3D, Upsampler, discrim_block


# ============================================================
# 3D RESNET-50 GENERATOR (He et al., CVPR 2016 / Med3D / SRGAN)
# ============================================================

class ResNet50_Generator(nn.Module):
    """
    3D ResNet-50 Super-Resolution Generator.
    Replaces 16-block SRResNet with an exact 50-layer deep 3D Residual Network.

    Academic References & Valid Parameters:
      1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
         "Deep Residual Learning for Image Recognition." CVPR 2016.
         - Exact Layer Distribution: [3, 4, 6, 3] = 16 Bottleneck blocks (48 conv layers)
         - Bottleneck Expansion Factor: 4
         - Feature Stages: planes = [64, 128, 256, 512] -> [256, 512, 1024, 2048]
      2. Sihang Chen et al.
         "Med3D: Transfer Learning for 3D Medical Image Analysis." arXiv:1904.00625 (Tencent MedicalNet).
         - Volumetric 3D Convolutions (Conv3d + BatchNorm3d) for 3D volumetric tensors
      3. Christian Ledig et al.
         "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." CVPR 2017.
         - Stride = 1 across all stages (Dense 3D ResNet without spatial downsampling to preserve voxel-level detail)
         - Global Residual Connection (feat = body(x) + skip)
         - 4x 3D Trilinear Refinement Upsampler Head
    """

    def __init__(
        self,
        img_feat=1,
        n_feats=64,
        kernel_size=3,
        num_block=16,
        act=nn.PReLU(),
        scale=4,
        layers=[3, 4, 6, 3],
        base_planes=64
    ):
        super(ResNet50_Generator, self).__init__()

        # Layer 1: Initial 3D Conv (maps input Cryo-EM volume to n_feats)
        self.conv01 = conv(
            in_channel=img_feat,
            out_channel=n_feats,
            kernel_size=kernel_size,
            BN=False,
            act=act
        )

        # ResNet-50 4-Stage Bottleneck Backbone (layers = [3, 4, 6, 3])
        # Total Bottleneck blocks = 3 + 4 + 6 + 3 = 16 blocks (16 * 3 = 48 conv layers)
        self.in_channels = n_feats

        # Stage 1: 3 Bottleneck blocks (planes = 64, out = 256)
        self.layer1 = self._make_layer(Bottleneck3D, planes=base_planes, blocks=layers[0], act=act)
        # Stage 2: 4 Bottleneck blocks (planes = 128, out = 512)
        self.layer2 = self._make_layer(Bottleneck3D, planes=base_planes * 2, blocks=layers[1], act=act)
        # Stage 3: 6 Bottleneck blocks (planes = 256, out = 1024)
        self.layer3 = self._make_layer(Bottleneck3D, planes=base_planes * 4, blocks=layers[2], act=act)
        # Stage 4: 3 Bottleneck blocks (planes = 512, out = 2048)
        self.layer4 = self._make_layer(Bottleneck3D, planes=base_planes * 8, blocks=layers[3], act=act)

        final_resnet_channels = (base_planes * 8) * Bottleneck3D.expansion  # 512 * 4 = 2048

        # Layer 50: Post-ResNet-50 Projection (maps 2048 channels back to n_feats)
        self.conv02 = conv(
            in_channel=final_resnet_channels,
            out_channel=n_feats,
            kernel_size=1,
            BN=True,
            act=None
        )

        # 4x Upsampling Head: Two 2x Trilinear Upsamplers
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

        # Final Output Reconstruction Layer
        self.last_conv = conv(
            in_channel=n_feats,
            out_channel=img_feat,
            kernel_size=3,
            BN=False,
            act=nn.Tanh()
        )

    def _make_layer(self, block, planes, blocks, act):
        layers = []
        layers.append(block(in_channels=self.in_channels, planes=planes, stride=1, act=act))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels=self.in_channels, planes=planes, stride=1, act=act))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv01(x)
        skip = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv02(x)
        feat = x + skip  # Global Residual Skip Connection

        x = self.tail(feat)
        out = self.last_conv(x)
        return out, feat


# ============================================================
# LEGACY 16-BLOCK SRRESNET GENERATOR (For Comparison)
# ============================================================

class SRResNet_Generator(nn.Module):
    def __init__(
        self,
        img_feat=1,
        n_feats=64,
        kernel_size=3,
        num_block=16,
        act=nn.PReLU(),
        scale=4
    ):
        super(SRResNet_Generator, self).__init__()

        self.conv01 = conv(
            in_channel=img_feat,
            out_channel=n_feats,
            kernel_size=kernel_size,
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


# Active Generator is now the 3D ResNet-50 Generator
Generator = ResNet50_Generator


# ============================================================
# DISCRIMINATOR
# ============================================================

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