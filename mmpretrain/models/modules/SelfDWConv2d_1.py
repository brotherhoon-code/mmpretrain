import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal


class SelfDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.layer_norm0 = nn.LayerNorm([in_channels, kernel_size, kernel_size])
        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
            padding="same",
        )
        self.layer_norm1 = nn.LayerNorm([in_channels, kernel_size, kernel_size])
        self.activ_func = nn.ReLU(inplace=True)
        # self.bias = nn.Parameter(torch.zeros(in_channels), requires_grad=True)
    
    def forward(self, x):
        input = x
        B, C, _, _ = input.size()
        x = self.pooling(x)
        x = self.layer_norm0(x)
        x = self.conv0(x)
        x = self.layer_norm1(x)
        x = self.activ_func(x)
        # bias = self.bias.unsqueeze(0).expand(B, -1)
        
        out = F.conv2d(
            Rearrange('b c h w -> 1 (b c) h w')(input),
            weight = Rearrange('b c kh kw -> (b c) 1 kh kw')(x),
            bias=None, #Rearrange("b c -> (b c)")(bias),
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
