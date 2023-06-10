import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal

class GCconv(nn.Module):
    def __init__(self, batch, in_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.reshape0 = Rearrange('b c h w -> b c (h w)')
        self.K = nn.Conv2d(in_channels=in_channels,
                           out_channels=kernel_size*kernel_size,
                           kernel_size=1)
        self.reshape1 = Rearrange('b kk h w -> b (h w) kk')
        self.reshape2 = Rearrange('b c (k_h k_w) -> b c k_h k_w',
                                  k_h=kernel_size,
                                  k_w=kernel_size)
        
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.LN = nn.LayerNorm([in_channels, kernel_size, kernel_size])
        self.activ_func = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        self.reshape3 = Rearrange('b c k_h k_w -> (b c) 1 k_h k_w',
                                  k_h=kernel_size,
                                  k_w=kernel_size)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # --- make kernel --- #
        query = self.reshape0(x) # b c N
        key = self.reshape1(self.K(x)) # b N kk
        key = F.softmax(key, dim=1) # b N kk
        filters = torch.matmul(query, key) # b c kk
        filters = self.reshape2(filters) # b c k k
        filters = self.conv0(filters)
        filters = self.LN(filters)
        filters = self.activ_func(filters)
        filters = self.conv1(filters)
        filters = self.reshape3(filters)
        
        # --- convolution --- #
        x = x.reshape(1, -1, H, W)
        out = F.conv2d(
            x,
            weight=filters,
            bias=None,
            stride=1,
            padding=int(self.kernel_size//2),
            groups=B*C
        )
        
        out = out.view(B, C, H, W)
        
        return out

if __name__ == "__main__":
    output = GCconv(batch=64, in_channels=32, kernel_size=7)(torch.Tensor(64,32,128,128))
    print(output.shape)
