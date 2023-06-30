import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat
from torchsummary import summary
from typing import Literal


class SelfDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.norm0 = nn.BatchNorm2d(num_features=in_channels)
        self.reshape0 = Rearrange('b c kh kw -> b c (kh kw)')
        self.linear0 = nn.Linear(in_features=kernel_size**2, out_features=kernel_size, bias=False)
        self.norm1 = nn.LayerNorm([in_channels, kernel_size])
        self.activ_func = nn.Sigmoid()
        self.linear1 = nn.Linear(in_features=kernel_size, out_features=kernel_size**2, bias=False)
        self.reshape1 = Rearrange('b c (kh kw) -> b c kh kw', kh=kernel_size, kw=kernel_size)
        
    def forward(self, x):
        input = x
        B, C, _, _ = input.size()
        x = self.pooling(x)
        x = self.norm0(x)
        x = self.reshape0(x)
        x = self.linear0(x)
        x = self.norm1(x)
        x = self.activ_func(x)
        x = self.linear1(x)
        x = self.reshape1(x)
        
        out = F.conv2d(
            Rearrange('b c h w -> 1 (b c) h w')(input),
            weight = Rearrange('b c kh kw -> (b c) 1 kh kw')(x),
            bias=None,
            stride=1,
            padding="same",
            groups=B*C,
        )
        
        out = Rearrange('1 (b c) h w -> b c h w',b=B, c=C)(out)
        return out
    
if __name__ == "__main__":
    fm = torch.randn(64,96,58,58)
    self_dwconv = SelfDWConv2d(96,96,7)
    out = self_dwconv(fm)
    print(out.shape)
