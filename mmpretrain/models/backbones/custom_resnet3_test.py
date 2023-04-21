import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from typing import Any, Callable, List, Optional, Type, Union
# from ..builder import BACKBONES


'''
CustomResNet3
    def make_stage(block_type, num_blocks, downsample, in_channels, out_channels)
    
BottleneckResBlock

StemBlock

'''
def getActFunc(type:str="ReLU"):
    function = None
    if type == "ReLU":
        function = nn.ReLU(inplace=True)
    elif type == "GELU":
        function = nn.GELU()

    if function == None:
        raise ValueError(f"{type}is not implemented")
    return function


class ResnetStemBlock(nn.Module):
    def __init__(self,
                 out_channels,
                 deep_stem=True,
                 act_func = "ReLU",
                 **kwrags):
        super().__init__()
        stem = []
        if deep_stem:
            stem.append(nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=(7,7), stride=2, padding=3))
            stem.append(nn.BatchNorm2d(out_channels))
            stem.append(getActFunc(act_func))
            stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            stem.append(nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1))
            stem.append(nn.BatchNorm2d(out_channels))
            stem.append(getActFunc(act_func))
        self.stem = nn.Sequential(*stem)
    
    def forward(self, x):
       return self.stem(x)
   

class IBttleneckResBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 inter_channels,
                 out_channels,
                 stride,
                 act_func = "ReLU",
                 **kwargs):
        super().__init__()
        
        isDepthwise = kwargs.get('isDepthwise', False)
        if isDepthwise == False:
            raise ValueError("IBttleneckResBlock require Depthwise Conv")
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, stride),
            nn.BatchNorm2d(inter_channels),
            getActFunc(act_func),
            
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, groups=inter_channels, padding=1),
            nn.BatchNorm2d(inter_channels),
            getActFunc(act_func),
            
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels))
        
        skip_conn = []
        if in_channels != out_channels and stride==1: # stage1_1
            skip_conn.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
            skip_conn.append(nn.BatchNorm2d(out_channels))
        
        elif in_channels != out_channels and stride!=1:
            skip_conn.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            skip_conn.append(nn.BatchNorm2d(out_channels))
        self.skip_conn = nn.Sequential(*skip_conn)
        
        self.active_func = getActFunc(act_func)
        
        
        
    def forward(self, x):
        return x


class BottleneckResBlock(nn.Module):
    block_out_channels = None
    def __init__(self, 
                 in_channels,
                 inter_channels,
                 out_channels,
                 stride,
                 act_func = "ReLU",
                 **kwargs):
        super().__init__()
        
        dw = kwargs.get('dw', False)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, stride),
            nn.BatchNorm2d(inter_channels),
            getActFunc(act_func),
            
            nn.Conv2d(inter_channels, 
                      inter_channels, 
                      kernel_size=3, 
                      padding=1) if dw == False else nn.Conv2d(inter_channels,
                                                               inter_channels,
                                                               kernel_size=3,
                                                               groups=inter_channels,
                                                               padding=1),
            nn.BatchNorm2d(inter_channels),
            getActFunc(act_func),
            
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels))
        
        skip_conn = []
        if in_channels != out_channels and stride==1: # stage1_1
            skip_conn.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
            skip_conn.append(nn.BatchNorm2d(out_channels))
        
        elif in_channels != out_channels and stride!=1:
            skip_conn.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            skip_conn.append(nn.BatchNorm2d(out_channels))
        self.skip_conn = nn.Sequential(*skip_conn)
        
        self.active_func = getActFunc(act_func)
        
        
    def forward(self, x):
        identity = self.skip_conn(x)
        outs = self.block(x)
        outs = outs + identity
        outs = self.active_func(outs)
        return outs


# @BACKBONES.register_module()
class CustomResNet3(nn.Module):
    def __init__(self, 
                 block_type:str = "BottleneckResBlock", 
                 stem_type:str = "Resnet", 
                 stem_channels:int = 64,
                 stage_blocks:list = [3, 4, 6, 3], 
                 feature_channels:list = [64, 128, 256, 512], # 3x3 stage1, stage2, stage3, stage4
                 stage_out_channels:list = [256, 512, 1024, 2048], # last_channels stage1, stage2, stage3, stage4
                 strides = [1,2,2,2],
                 act_func = "ReLU",
                 **kwargs):
        super().__init__()
        
        if stem_type == "Resnet":
            self.stem = ResnetStemBlock(stem_channels, True, act_func) # out_channel, deep_stem, act_func
        
        if block_type == "BottleneckResBlock":
            block = BottleneckResBlock
        
        # block_type, n_stage_blocks, stride, in_channels, feature_channels, out_channels
        self.stage1 = self._make_stage(block, stage_blocks[0], strides[0], stem_channels, feature_channels[0], stage_out_channels[0], act_func, 0, **kwargs)
        self.stage2 = self._make_stage(block, stage_blocks[1], strides[1], stage_out_channels[0], feature_channels[1], stage_out_channels[1], act_func, 1, **kwargs)
        self.stage3 = self._make_stage(block, stage_blocks[2], strides[2], stage_out_channels[1], feature_channels[2], stage_out_channels[2], act_func, 2, **kwargs)
        self.stage4 = self._make_stage(block, stage_blocks[3], strides[3], stage_out_channels[2], feature_channels[3], stage_out_channels[3], act_func, 3, **kwargs)

        print(self.stage1)
        print(self.stage2)
        print(self.stage3)
        print(self.stage4)
    
    def forward(self, x):
        outs = []
        out = self.stem(x)
        out = self.stage1(out)
        outs.append(out)
        out = self.stage2(out)
        outs.append(out)
        out = self.stage3(out)
        outs.append(out)
        out = self.stage4(out)
        outs.append(out)
        return tuple(outs)


    # 이 함수에서 모든 keyword 처리
    @staticmethod
    def _make_stage(block:Union[BottleneckResBlock, IBttleneckResBlock],
                    n_blocks:int,
                    stride:int,
                    in_channels:int,
                    feature_channels:int,
                    out_channels:int, 
                    act_func:str,
                    stage_order:int,
                    **kwargs):
        blocks = []
        current_channels = None
        ''' 
        in_channels,
        inter_channels,
        out_channels,
        stride,
        '''
        
        isDepthwise = kwargs.get('isDepthwise', False)[stage_order] # 생성하는 스테이지의 DW여부를 확인한다.
        
        n_repeat = n_blocks-2 # total blocks에서 sub-block_last, sub-block_last을 제외한 sub블록의 반복 횟수
        
        blocks.append(block(in_channels, feature_channels, out_channels, stride, act_func, dw=isDepthwise)) # first sub-block
        current_channels = out_channels
        
        for stage_order in range(n_repeat):
            blocks.append(block(current_channels, feature_channels, out_channels, 1, act_func, dw=isDepthwise))
            current_channels = out_channels
        
        blocks.append(block(current_channels, feature_channels, out_channels, 1, act_func, dw=isDepthwise)) # last sub-block
        
        return nn.Sequential(*blocks)

# R R R R
# DW DW R R
# R R DW DW


if __name__ == "__main__":
    m = CustomResNet3(block_type = "BottleneckResBlock",
                      stem_type = "Resnet",
                      stem_channels = 64,
                      stage_blocks = [3, 3, 3, 3],
                      feature_channels = [64, 128, 256, 512],
                      stage_out_channels = [256, 512, 1024, 2048],
                      strides = [1, 2, 2, 2],
                      act_func="ReLU",
                      isDepthwise=[True, False, True, False])
    
    # summary(m, (3,224,224), device="cpu", batch_size=1)