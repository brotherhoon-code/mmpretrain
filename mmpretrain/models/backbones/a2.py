import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmpretrain.models.backbones.custom_modules.attention import SEModule
from ..builder import BACKBONES

## 임배딩 블록 리스케일링
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
        self.rescale = SEModule(in_channels=out_channels, r=16, activ_func="ReLU", importance="mean")

    def forward(self, x):
        x = self.embed_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        x = self.rescale(x)
        return x


class SpatialMixBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, groups=None):
        super(SpatialMixBlock, self).__init__()
        self.dim = dim
        if groups == None:
            groups = dim
        self.kernel_size = kernel_size
        self.spatial_mix_layer = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=groups,
            padding="same",
        )
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
    def __init__(self, dim: int, kernel_size: int):
        super(MixerBlock, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.spatial_mix_block = SpatialMixBlock(dim, kernel_size)
        self.channel_mix_block = ChannelMixBlock(dim)

    def forward(self, x):
        x = self.spatial_mix_block(x) + x
        x = self.channel_mix_block(x)
        return x


@BACKBONES.register_module()
class A2(nn.Module):
    def __init__(self,
                 stage_channels: int=[96, 192, 384, 768],
                 stage_blocks: int=[3, 3, 9, 3],
                 patch_size: int=[4, 2, 2, 2],
                 kernel_size: int=7):
        super(A2, self).__init__()
        self.s1_patch_embed = PatchEmbedBlock(in_channels=3, out_channels=stage_channels[0], patch_size=patch_size[0])
        self.stage1 = nn.Sequential(*[MixerBlock(stage_channels[0], kernel_size) for _ in range(stage_blocks[0])])
        
        self.s2_patch_embed = PatchEmbedBlock(in_channels=stage_channels[0], out_channels=stage_channels[1], patch_size=patch_size[1])
        self.stage2 = nn.Sequential(*[MixerBlock(stage_channels[1], kernel_size) for _ in range(stage_blocks[1])])
        
        self.s3_patch_embed = PatchEmbedBlock(in_channels=stage_channels[1], out_channels=stage_channels[2], patch_size=patch_size[2])
        self.stage3 = nn.Sequential(*[MixerBlock(stage_channels[2], kernel_size) for _ in range(stage_blocks[2])])
        
        self.s4_patch_embed = PatchEmbedBlock(in_channels=stage_channels[2], out_channels=stage_channels[3], patch_size=patch_size[3])
        self.stage4 = nn.Sequential(*[MixerBlock(stage_channels[3], kernel_size) for _ in range(stage_blocks[3])])

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


if __name__ == "__main__":
    m = A2()
    input_img = torch.Tensor(1, 3, 224, 224)
    flops = FlopCountAnalysis(m, input_img)

    summary(m, (3, 224, 224), batch_size=256, device="cpu") # torchsummary
    print(flop_count_table(flops)) # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌
    formatted_number = "{:.2f}G".format(flops.total() / 1e9)
    print(f"total FLOPs: {formatted_number}") # kb단위로 모델전체 FLOPs 출력해줌
