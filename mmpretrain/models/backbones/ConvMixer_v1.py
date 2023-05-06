import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from ..builder import BACKBONES


class PatchEmbedBlock(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 out_channels:int, 
                 patch_size:int):
        super(PatchEmbedBlock, self).__init__()
        self.patch_size=patch_size
        self.embed_layer = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=out_channels, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        x = self.embed_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x


class SpatialMixBlock(nn.Module):
    def __init__(self, 
                 dim:int, 
                 kernel_size:int, 
                 groups=None):
        super(SpatialMixBlock, self).__init__()
        self.dim = dim
        if groups == None:
            groups=dim
        self.kernel_size = kernel_size
        self.spatial_mix_layer = nn.Conv2d(in_channels=dim,
                                           out_channels=dim,
                                           kernel_size=kernel_size,
                                           groups=groups,
                                           padding="same")
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)
        
    def forward(self, x):
        x = self.spatial_mix_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x


class ChannelMixBlock(nn.Module):
    def __init__(self, 
                 dim:int):
        super(ChannelMixBlock, self).__init__()
        self.dim = dim
        self.channel_mix_layer = nn.Conv2d(dim, dim ,kernel_size=(1,1))
        self.active_func = nn.GELU()
        self.bn_layer = nn.BatchNorm2d(num_features=dim)
    
    def forward(self, x):
        x = self.channel_mix_layer(x)
        x = self.active_func(x)
        x = self.bn_layer(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, 
                 dim:int, 
                 kernel_size:int):
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
class ConvMixer_v1(nn.Module):
    def __init__(self, 
                 dim:int, 
                 depth: int, 
                 patch_size:int, 
                 kernel_size:int):
        super(ConvMixer_v1, self).__init__()
        self.patch_embed_layer = PatchEmbedBlock(3, dim, patch_size)
        self.mixer_layers = nn.Sequential(*[MixerBlock(dim, kernel_size)]*depth)

    def forward(self, x):
        x = self.patch_embed_layer(x)
        outs = []
        for stage in self.mixer_layers:
            x = stage(x)
            outs.append(x)
        return tuple(outs)


if __name__ == "__main__":
    convmixer = ConvMixer_v1(dim=768,
                             depth=32,
                             patch_size=7,
                             kernel_size=7)
    summary(convmixer, (3, 224, 224), batch_size=256, device="cpu")
    
    




