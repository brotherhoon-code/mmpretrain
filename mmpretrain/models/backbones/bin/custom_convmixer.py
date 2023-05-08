import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from ...builder import BACKBONES


class ResidualBlock(nn.Module):
    def __init__(self,
                 block:nn.Module,
                 skip_conn=True):
        super(ResidualBlock, self).__init__()
        self.block = block
        self.skip_conn = skip_conn
        
    def forward(self, x):
        return self.block(x)+x


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
        return x # b dim h/patch_size w/patch_size


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
                 kernel_size:int,
                 block_type:str="dw-p"):
        super(MixerBlock, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.block_type = block_type

        if block_type == "dw-p":
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, dim) # dw
            self.channel_mix_block = ChannelMixBlock(dim) # p

        elif block_type == "p-dw":
            # self.channel_mix_block = ChannelMixBlock(dim) # p
            self.channel_mix_block = nn.Conv2d(dim, dim, 1, bias=False) # BSCond
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, dim) # dw

        elif block_type == "dw-p-p":
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, dim) # dw
            self.channel_mix_block = nn.Sequential(ChannelMixBlock(dim), ChannelMixBlock(dim)) # p, p

        elif block_type == "p-p-dw":
            # self.channel_mix_block = nn.Sequential(ChannelMixBlock(dim), ChannelMixBlock(dim)) # p, p
            self.channel_mix_block = nn.Sequential(nn.Conv2d(dim, dim//4, 1, bias=False), 
                                                   nn.Conv2d(dim//4, dim, 1, bias=False)) # Subspace BSConv
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, dim) # dw

        elif block_type == "p-dw-p":
            self.channel_mix_block1 = ChannelMixBlock(dim) # p
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, dim) # dw
            self.channel_mix_block2 = nn.Conv2d(dim,dim,1) # p

        elif block_type == "r-p":
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, 1) # r
            self.channel_mix_block = ChannelMixBlock(dim) # p

        elif block_type == "r-p-p":
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, 1) # r
            self.channel_mix_block = nn.Sequential(ChannelMixBlock(dim), ChannelMixBlock(dim)) # p, p

        elif block_type == "p-p-r":
            self.channel_mix_block = nn.Sequential(ChannelMixBlock(dim), ChannelMixBlock(dim)) # p, p
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, 1) # r

        elif block_type == "p-r-p":
            self.channel_mix_block1 = ChannelMixBlock(dim) # p
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, 1) # r
            self.channel_mix_block2 = nn.Conv2d(dim,dim,1) # p

        elif block_type == "r":
            self.spatial_mix_block = SpatialMixBlock(dim, kernel_size, 1) # r
            self.channel_mix_block = nn.Sequential()

    
    def forward(self, x):
        if self.block_type.split("-")[0] != 'p': # dw-p, r, r-p-p, dw-p-p
            # identity = x
            x = self.spatial_mix_block(x) + x
            x = self.channel_mix_block(x)
            if len(self.block_type) == 1: # r case
                return x
            # x += identity
            
        elif self.block_type.split("-")[0] == 'p' and self.block_type.split("-")[-1] != 'p': # p-p-dw, p-p-r, p-dw
            # identity = x
            x = self.channel_mix_block(x)
            x = self.spatial_mix_block(x) + x
            # x += identity
            
        elif self.block_type.split("-")[0] == 'p' and self.block_type.split("-")[-1] == 'p': # p-dw-p, p-r-p
            # identity = x
            x = self.channel_mix_block1(x)
            x = self.spatial_mix_block(x) + x
            x = self.channel_mix_block2(x)
            # x += identity
            
        else:
            raise ValueError(f"block_type error: {self.block_type} not in case")
        return x


@BACKBONES.register_module()
class CustomConvMixer(nn.Module):
    def __init__(self, 
                 block_type=["dw-p", "dw-p", "dw-p", "dw-p"], # homo case
                 block_repeat="homo",
                 in_stage_block_type = ["dw-p", "dw-p", "dw-p"], # inhomo case
                 stage_in_channels:list=[96, 192, 384, 768],
                 stage_blocks:list=[3,3,3,3], 
                 patch_size:int=4,
                 kernel_size:int=3,
                 **kwargs):
        super(CustomConvMixer, self).__init__()
        self.patch_size = patch_size
        self.stage_blocks = stage_blocks
        self.kernel_Size = kernel_size
        
        if block_repeat == "inhomo":
            stage_blocks = [i//3 for i in stage_blocks]
        
        self.stem_block1 = PatchEmbedBlock(3, stage_in_channels[0], patch_size)
        if block_repeat == "homo":
            self.stage1_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[0], kernel_size, block_type[0])]*stage_blocks[0]
                )
        elif block_repeat == "inhomo":
            self.stage1_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[0], kernel_size, in_stage_block_type[0]),
                  MixerBlock(stage_in_channels[0], kernel_size, in_stage_block_type[1]),
                  MixerBlock(stage_in_channels[0], kernel_size, in_stage_block_type[2]),]*stage_blocks[0]
                )
        
        
        self.stem_block2 = PatchEmbedBlock(stage_in_channels[0], stage_in_channels[1], 2)
        if block_repeat == "homo":
            self.stage2_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[1], kernel_size, block_type[1])]*stage_blocks[1]
                )
        elif block_repeat == "inhomo":
            self.stage2_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[1], kernel_size, in_stage_block_type[0]),
                  MixerBlock(stage_in_channels[1], kernel_size, in_stage_block_type[1]),
                  MixerBlock(stage_in_channels[1], kernel_size, in_stage_block_type[2])]*stage_blocks[1]
                )
        
        
        self.stem_block3 = PatchEmbedBlock(stage_in_channels[1], stage_in_channels[2], 2)
        if block_repeat == "homo":
            self.stage3_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[2], kernel_size, block_type[2])]*stage_blocks[2]
                )
        elif block_repeat == "inhomo":
            self.stage3_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[2], kernel_size, in_stage_block_type[0]),
                  MixerBlock(stage_in_channels[2], kernel_size, in_stage_block_type[1]),
                  MixerBlock(stage_in_channels[2], kernel_size, in_stage_block_type[2]),]*stage_blocks[2]
                )
        
        
        self.stem_block4 = PatchEmbedBlock(stage_in_channels[2], stage_in_channels[3], 2)
        if block_repeat == "homo":
            self.stage4_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[3], kernel_size, block_type[3])]*stage_blocks[3]
                )
        elif block_repeat == "inhomo":
            self.stage4_block = nn.Sequential(
                *[MixerBlock(stage_in_channels[3], kernel_size, in_stage_block_type[0]),
                  MixerBlock(stage_in_channels[3], kernel_size, in_stage_block_type[1]),
                  MixerBlock(stage_in_channels[3], kernel_size, in_stage_block_type[2]),]*stage_blocks[3]
                )
        print(self.stem_block1)
        print(self.stage1_block)
        print(self.stem_block2)
        print(self.stage2_block)
        print(self.stem_block3)
        print(self.stage3_block)
        print(self.stem_block4)
        print(self.stage4_block)
        
    def forward(self, x):
        outs = []
        
        x = self.stem_block1(x)
        x = self.stage1_block(x)
        outs.append(x)
        
        x = self.stem_block2(x)
        x = self.stage2_block(x)
        outs.append(x)
        
        x = self.stem_block3(x)
        x = self.stage3_block(x)
        outs.append(x)
        
        x = self.stem_block4(x)
        x = self.stage4_block(x)
        outs.append(x)
        
        return tuple(outs)


if __name__ == "__main__":
    convmixer = CustomConvMixer(block_type= ["r-p-p","r-p-p","r-p-p","r-p-p"], 
                                block_repeat="homo",
                                in_stage_block_type=["dw-p","dw-p",'r-p'], # not using
                                stage_in_channels=[96, 192, 384, 768],
                                stage_blocks = [3,3,3,3], 
                                patch_size = 4, 
                                kernel_size = 3)
    
    summary(convmixer, (3,224,224), batch_size=256, device="cpu")
    
    




