import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmpretrain.models.backbones.custom_modules.GCconv3_Sigmoid import GCconv3

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
    def __init__(
        self,
        kernel_size: int,
        dim: int,
    ):
        super(SpatialMixBlock, self).__init__()
        self.spatial_mix_layer = GCconv3(in_channels=dim, out_channels=dim, kernel_size=kernel_size)
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)

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
    def __init__(
        self,
        kernel_size: int,
        dim: int,
    ):
        super(MixerBlock, self).__init__()
        self.spatial_mix_block = SpatialMixBlock(
            kernel_size, 
            dim, 
        )
        self.channel_mix_block = ChannelMixBlock(dim)

    def forward(self, x):
        x = self.spatial_mix_block(x) + x
        x = self.channel_mix_block(x)
        return x


@BACKBONES.register_module()
class A33(nn.Module):
    def __init__(
        self,
        stage_channels: int = [96, 192, 384, 768],
        stage_blocks: int = [2, 2, 2, 2],
        patch_size: int = [4, 2, 2, 2],
        kernel_size: int = 7,
    ):
        super().__init__()
        self.s1_patch_embed = PatchEmbedBlock(
            in_channels=3, out_channels=stage_channels[0], patch_size=patch_size[0]
        )
        self.stage1 = nn.Sequential(
            *[
                MixerBlock(
                    kernel_size,
                    stage_channels[0],
                )
                for _ in range(stage_blocks[0])
            ]
        )

        self.s2_patch_embed = PatchEmbedBlock(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            patch_size=patch_size[1],
        )
        self.stage2 = nn.Sequential(
            *[
                MixerBlock(
                    kernel_size,
                    stage_channels[1],
                )
                for _ in range(stage_blocks[1])
            ]
        )

        self.s3_patch_embed = PatchEmbedBlock(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            patch_size=patch_size[2],
        )

        self.stage3 = nn.Sequential(
            *[
                MixerBlock(
                    kernel_size,
                    stage_channels[2],
                )
                for _ in range(stage_blocks[2])
            ]
        )

        self.s4_patch_embed = PatchEmbedBlock(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            patch_size=patch_size[3],
        )
        self.stage4 = nn.Sequential(
            *[
                MixerBlock(
                    kernel_size,
                    stage_channels[3],
                )
                for _ in range(stage_blocks[3])
            ]
        )
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
    m = A33(
        stage_channels=[96*2, 192*2, 384*2, 768*2],
        stage_blocks=[2, 2, 6, 2],
        patch_size=[4, 2, 2, 2],
        kernel_size=7,
    )
    if True:
        summary(m, (3, 224, 224), batch_size=64, device="cpu")
    if False:
        input_img = torch.Tensor(64, 3, 224, 224)
        flops = FlopCountAnalysis(m, input_img)
        print(flop_count_table(flops))
        formatted_number = "{:.2f}G".format(flops.total() / 1e9)
        print(f"total FLOPs: {formatted_number}")
