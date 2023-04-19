import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from ..builder import BACKBONES


'''
ref: https://github.com/xxxnell/spatial-smoothing/blob/master/models/resnet.py
'''


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, 
                 in_channels, 
                 channels, 
                 stride=1, 
                 groups=1, 
                 width_per_group=64,
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_groups = width_per_group
        
        if groups != 1 or width_per_group != 64:
            raise ValueError("Basic only supperts groups=1 and base_width=64")
        self.width = int(channels*(width_per_group/64.))*groups
        
        self.shortcut=[]
        
        # 차원을 맞추기 위해서 1번째 블록인 경우 체널을 확장시킵니다
        if self.stride!=1 or self.in_channels != self.channels*self.expansion:
            self.shortcut.append(nn.Conv2d(self.in_channels, 
                                           self.channels*self.expansion, 
                                           kernel_size=(1,1),
                                           stride=self.stride))
            self.shortcut.append(nn.BatchNorm2d(self.channels*self.expansion))
            
        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(3,3), stride=self.stride, padding=1), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.width, out_channels=self.channels*self.expansion, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(self.channels*self.expansion), 
        )
        
    def forward(self, x):
        skip = self.shortcut(x)
        x = self.block(x) + skip
        x = F.relu(x)
        return x
    

class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, 
                 in_channels:int, 
                 channels:int, 
                 stride=1, 
                 groups=1, 
                 width_per_group=64, 
                 **kwagrs):
        super(BottleNeckBlock, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        
        self.width = int(channels * (width_per_group/64.)) * groups
        self.shortcut = []
        
        if stride!=1 or in_channels!=channels*self.expansion:
            self.shortcut.append(nn.Conv2d(self.in_channels,
                                           self.channels*self.expansion, 
                                           kernel_size=(1,1),
                                           stride=self.stride))
            self.shortcut.append(nn.BatchNorm2d(self.channels*self.expansion))
        
        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1,1)), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(), 
            # 3x3 padding=1 conv
            nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(3,3), stride=self.stride, groups=self.groups, padding=1), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(), 
            # expasion 1x1 conv
            nn.Conv2d(in_channels=self.width, out_channels=self.channels*self.expansion, kernel_size=(1,1)), 
            nn.BatchNorm2d(self.channels*self.expansion), 
            # output channel dim is self.channels*expansion
        )
        
        
    def forward(self, x):
        skip = self.shortcut(x)
        x = self.block(x) + skip
        x = F.relu(x)
        return x
    
@BACKBONES.register_module()
class CustomResNet(nn.Module):
    def __init__(self, 
                 block_cls=BottleNeckBlock, 
                 deep_stem = True, # for cifar
                 n_blocks=[3,4,6,3],
                 n_classes=100, 
                 **kwargs):
        super(CustomResNet, self).__init__()
        self.block_cls = block_cls
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.expansion = block_cls.expansion
        
        self.stem = []
        if deep_stem:
            self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3))
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU())
            self.stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1))
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU())
        
        self.stem_block = nn.Sequential(*self.stem)
        self.stage1_block = self._make_stage(self.block_cls, 64, 64, n_blocks[0], 1)
        self.stage2_block = self._make_stage(self.block_cls, 64*self.expansion, 128, n_blocks[1], 2)
        self.stage3_block = self._make_stage(self.block_cls, 128*self.expansion, 256, n_blocks[2], 2)
        self.stage4_block = self._make_stage(self.block_cls, 256*self.expansion, 512, n_blocks[3], 2)
        # self.neck = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier = nn.Linear(512*self.expansion, self.n_classes)
        
        
    def forward(self, x):
        outs=[]
        x = self.stem_block(x)
        x = self.stage1_block(x)
        outs.append(x)
        x = self.stage2_block(x)
        outs.append(x)
        x = self.stage3_block(x)
        outs.append(x)
        x = self.stage4_block(x)
        outs.append(x)
        return tuple(outs)
    
    @staticmethod
    def _make_stage(block, in_channels, out_channels, num_blocks, stride):
        stride_arr = [stride] + [1] * (num_blocks-1) # 각 스테이지의 첫번째 블록만 stride 반영
        layers, channels = [], in_channels
        
        layers.append(block(channels, out_channels, stride_arr[0]))
        channels = out_channels*block.expansion
        
        for stride in stride_arr[1:]:
            layers.append(block(channels, out_channels, stride))
            channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    

if __name__ == "__main__":
    m = CustomResNet(BottleNeckBlock, True, [3,4,6,3], 10)
    summary(m, (3,224,224), device='cpu', batch_size=1)