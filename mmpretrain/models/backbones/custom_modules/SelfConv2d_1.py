import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary

class SelfConv2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 resolution:int,
                 temperature:float=2.,
                 reduction:int=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size        
        self.resolution = resolution
        self.temperature = temperature
        if in_channels != out_channels:
            raise ValueError(f"in_channels must be same out_channels")
        
        self.q_layer = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 bias=False)
        self.q_reshaper = Rearrange('b c h w -> b c (h w)')
        
        self.k_layer = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 bias=False)
        self.k_reshaper = Rearrange('b c h w -> b (h w) c')
        
        self.v_layer0 = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 bias=False)
        self.v_reshaper = Rearrange('b c h w -> b c (h w)')
        self.v_layer1 = nn.Linear(in_features=resolution**2,
                                  out_features=kernel_size**2,
                                  bias=False)
        
        self.filter_shaper = nn.Sequential(
            Reduce('b c kk -> c kk', reduction="mean"),
            Rearrange('c (k_h k_w) -> c k_h k_w', k_h=kernel_size, k_w=kernel_size))
        
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                               out_channels=int(out_channels//reduction),
                               kernel_size=1),
            nn.LayerNorm([int(out_channels//reduction), kernel_size, kernel_size]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(out_channels//reduction),
                               out_channels=out_channels,
                               kernel_size=1),
            Rearrange('b c k_h k_w -> c b k_h k_w')
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x):
        # make Q, K, V
        Q = self.q_reshaper(self.q_layer(x))
        K = self.k_reshaper(self.k_layer(x))
        V = self.v_layer1(self.v_reshaper(self.v_layer0(x)))

        # get attention_score # b, c, c
        attn_score = torch.matmul(Q, K)
        attn_prob = torch.softmax(attn_score/self.temperature, dim=1)
        
        # make filters
        filters = torch.matmul(attn_prob, V)
        
        # fusion
        filters = self.filter_shaper(filters)
        filters = filters.unsqueeze(dim=0)
        filters = self.transform(filters)
        
        # convolution
        out = F.conv2d(
            x,
            weight=filters,
            bias=self.bias,
            stride=1,
            padding=int(self.kernel_size // 2),
            groups=self.in_channels,
        )
        
        return out
    

if __name__ == "__main__":
    output = SelfConv2d(96, 96, 7, 52)(torch.Tensor(64,96,52,52))
    print(f'output.shape={output.shape}')