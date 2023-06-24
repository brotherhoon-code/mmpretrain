import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmpretrain.models.modules.SelfConv2d_6 import SelfConv2d

from ..builder import BACKBONES


class PatchEmbedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super(PatchEmbedBlock, self).__init__()
        self.patch_size = patch_size
        self.embed_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.embed_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x


class SpatialMixBlock(nn.Module):
    def __init__(self, kernel_size: int, dim: int, is_self: bool, **kwargs):
        super(SpatialMixBlock, self).__init__()
        if is_self == True:
            self.spatial_mix_layer = SelfConv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                temp=kwargs["temp"],
                pooling_resolution=7,
                bottle_ratio=4,
            )
        else:
            self.spatial_mix_layer = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                groups=dim,
                padding="same",
            )

        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)
        self.dim = dim

    def forward(self, x):
        x = self.spatial_mix_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x


class ChannelMixBlock(nn.Module):
    def __init__(self, dim: int):
        super(ChannelMixBlock, self).__init__()
        self.dim = dim
        self.channel_mix_layer = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)

    def forward(self, x):
        x = self.channel_mix_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, kernel_size: int, dim: int, is_self: bool = False, **kwargs):
        super(MixerBlock, self).__init__()
        # print(kwargs)
        self.spatial_mix_block = SpatialMixBlock(kernel_size, dim, is_self, **kwargs)
        self.channel_mix_block = ChannelMixBlock(dim)

    def forward(self, x):
        x = self.spatial_mix_block(x) + x
        x = self.channel_mix_block(x)
        return x


@BACKBONES.register_module()
class B4(nn.Module):
    def __init__(
        self,
        stage_channels: list = [128, 256, 512, 1024],
        stage_blocks: list = [3, 3, 9, 3],
        patch_size: list = [4, 2, 2, 2],
        kernel_size: int = 9,
        last_self_block: bool = False,
        self_conv_stages: list = [True, True, True, True],
        temp=20.0,
    ):
        super().__init__()

        if (last_self_block == True) and (self_conv_stages != None):
            raise ValueError(
                f"if last_self_block is True, self_conv_stages must be None"
            )
        if self_conv_stages is None:
            self_conv_stages = [False, False, False, False]

        ## s1
        self.s1_patch_embed = PatchEmbedBlock(
            in_channels=3, out_channels=stage_channels[0], patch_size=patch_size[0]
        )

        if last_self_block == False:
            self.stage1 = nn.Sequential(
                *[
                    MixerBlock(
                        kernel_size,
                        stage_channels[0],
                        is_self=self_conv_stages[0],
                        temp=temp,  # kwargs
                    )
                    for _ in range(stage_blocks[0])
                ]
            )
        else:
            blocks = [
                MixerBlock(
                    kernel_size,
                    stage_channels[0],
                    is_self=False,
                    temp=temp,  # kwargs
                )
                for _ in range(stage_blocks[0] - 1)
            ]
            blocks.extend(
                [
                    MixerBlock(
                        kernel_size,
                        stage_channels[0],
                        is_self=True,
                        temp=temp,  # kwargs
                    )
                ]
            )
            self.stage1 = nn.Sequential(*blocks)

        ## s2
        self.s2_patch_embed = PatchEmbedBlock(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            patch_size=patch_size[1],
        )

        if last_self_block == False:
            self.stage2 = nn.Sequential(
                *[
                    MixerBlock(
                        kernel_size,
                        stage_channels[1],
                        is_self=self_conv_stages[1],
                        temp=temp,  # kwargs
                    )
                    for _ in range(stage_blocks[1])
                ]
            )
        else:
            blocks = [
                MixerBlock(
                    kernel_size,
                    stage_channels[1],
                    is_self=False,
                    temp=temp,  # kwargs
                )
                for _ in range(stage_blocks[1] - 1)
            ]
            blocks.extend(
                [
                    MixerBlock(
                        kernel_size,
                        stage_channels[1],
                        is_self=True,
                        temp=temp,  # kwargs
                    )
                ]
            )
            self.stage2 = nn.Sequential(*blocks)

        ## s3
        self.s3_patch_embed = PatchEmbedBlock(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            patch_size=patch_size[2],
        )

        if last_self_block == False:
            self.stage3 = nn.Sequential(
                *[
                    MixerBlock(
                        kernel_size,
                        stage_channels[2],
                        is_self=self_conv_stages[2],
                        temp=temp,  # kwargs
                    )
                    for _ in range(stage_blocks[2])
                ]
            )
        else:
            blocks = [
                MixerBlock(
                    kernel_size,
                    stage_channels[2],
                    is_self=False,
                    temp=temp,  # kwargs
                )
                for _ in range(stage_blocks[2] - 1)
            ]
            blocks.extend(
                [
                    MixerBlock(
                        kernel_size,
                        stage_channels[2],
                        is_self=True,
                        temp=temp,  # kwargs
                    )
                ]
            )
            self.stage3 = nn.Sequential(*blocks)

        ## s4
        self.s4_patch_embed = PatchEmbedBlock(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            patch_size=patch_size[3],
        )

        if last_self_block == False:
            self.stage4 = nn.Sequential(
                *[
                    MixerBlock(
                        kernel_size,
                        stage_channels[3],
                        is_self=self_conv_stages[3],
                        temp=temp,  # kwargs
                    )
                    for _ in range(stage_blocks[3])
                ]
            )
        else:
            blocks = [
                MixerBlock(
                    kernel_size,
                    stage_channels[3],
                    is_self=False,
                    temp=temp,  # kwargs
                )
                for _ in range(stage_blocks[3] - 1)
            ]
            blocks.extend(
                [
                    MixerBlock(
                        kernel_size,
                        stage_channels[3],
                        is_self=True,
                        temp=temp,  # kwargs
                    )
                ]
            )
            self.stage4 = nn.Sequential(*blocks)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        x = self.s1_patch_embed(x)
        x = self.stage1(x)
        outs.append(x)

        x = self.s2_patch_embed(x)
        x = self.stage2(x)
        outs.append(x)

        x = self.s3_patch_embed(x)
        x = self.stage3(x)
        outs.append(x)

        x = self.s4_patch_embed(x)
        x = self.stage4(x)
        outs.append(x)

        return tuple(outs)


def count_model_parameters(model):
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)
    return num_params


if __name__ == "__main__":
    m = B4(
        stage_channels=[128, 256, 512, 1024],
        stage_blocks=[3, 3, 9, 3],
        patch_size=[4, 2, 2, 2],
        kernel_size=9,
        last_self_block=False,
        self_conv_stages=[True, True, True, True],
        temp=40.0,
    )

    if True:
        summary(m, (3, 224, 224), batch_size=1, device="cpu")
    # if True:
    #     input_img = torch.Tensor(64, 3, 224, 224)
    #     flops = FlopCountAnalysis(m, input_img)
    #     print(flop_count_table(flops))
    #     formatted_number = "{:.2f}G".format(flops.total() / 1e9)
    #     print(f"total FLOPs: {formatted_number}")
