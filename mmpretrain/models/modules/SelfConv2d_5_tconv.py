import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal
"""
pooling -> point conv
kernel_component = "Tconv"
"""

class SelfConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, temp=7.0, pooling_resolution=7, bottle_ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temp = temp
        self.bottle_ratio=bottle_ratio

        if in_channels != out_channels:
            raise ValueError(
                f"in_channels({in_channels}) must be same out_channels{out_channels}"
            )
            
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(pooling_resolution, pooling_resolution))
        self.layer_norm0 = nn.LayerNorm([in_channels, pooling_resolution, pooling_resolution])
        
        self.q_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.q_reshaper = Rearrange("b c p_h p_w -> b c (p_h p_w)")  # b c 49

        self.k_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.k_reshaper = Rearrange("b c p_h p_w -> b (p_h p_w) c")  # b n c
        
        self.kernel_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size-pooling_resolution+1, groups=in_channels, bias=False)
        self.kernel_reshaper = Rearrange('b c p_h p_w -> b c (p_h p_w)') # b c 49
        
        self.filter_reshaper = Rearrange('b c (k_h k_w) -> b c k_h k_w', k_h=kernel_size, k_w=kernel_size)
        
        self.fusion_conv0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//bottle_ratio, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm([in_channels//bottle_ratio, kernel_size, kernel_size])
        self.activ_func = nn.ReLU(inplace=True)
        self.fusion_conv1 = nn.Conv2d(in_channels=in_channels//bottle_ratio, out_channels=out_channels, kernel_size=1)

    def _get_query(self, pooled_feature_map: torch.Tensor):
        Q = self.q_layer(pooled_feature_map)
        Q = self.q_reshaper(Q) # b c n
        return Q

    def _get_key(self, pooled_feature_map: torch.Tensor):
        K = self.k_layer(pooled_feature_map)
        K = self.k_reshaper(K) # b n c
        return K
    
    def _get_kernel(self, pooled_feature_map: torch.Tensor):
        kernel_weights = self.kernel_layer(pooled_feature_map)
        # print(kernel_weights.shape) # 64, 96, 9, 9
        kernel_weights = self.kernel_reshaper(kernel_weights)
        return kernel_weights # b c k**2

    def forward(self, x: torch.Tensor):
        # get_query
        input = x
        B, C, _, _ = input.size()
        
        x = self.avgpool(x) # b c 7 7
        x = self.layer_norm0(x)
        
        Q = self._get_query(x) # b c 49
        K = self._get_key(x) # b 49 c
        
        channel_attn = torch.matmul(Q,K) # b c c
        channel_attn = F.softmax(channel_attn/self.temp, dim=1) # b c c
        
        kernel_weights = self._get_kernel(x) # b c k**2
        kernel_weights = torch.matmul(channel_attn, kernel_weights) # b c k**2
        kernel_weights = self.filter_reshaper(kernel_weights)
        
        ## fusion
        kernel_weights = self.fusion_conv0(kernel_weights)
        kernel_weights = self.layer_norm1(kernel_weights)
        kernel_weights = self.activ_func(kernel_weights)
        kernel_weights = self.fusion_conv1(kernel_weights)
        kernel_weights = Rearrange('b c k_h k_w -> (b c) 1 k_h k_w')(kernel_weights)
        
        out = F.conv2d(
            Rearrange('b c h w -> 1 (b c) h w')(input),
            weight=kernel_weights,
            bias=None,
            stride=1,
            padding=int(self.kernel_size // 2),
            groups=x.size(0)*x.size(1),
        )
        
        out = Rearrange('1 (b c) h w -> b c h w', b=B, c=C)(out)
        
        return out

def count_model_parameters(model):
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)
    return num_params


def compare_modules(input, m1, m2, m3):
    print('-'*50)
    print('<input shape>')
    print(f'{input.shape}')
    print('<m1 module>')
    print(f'model params: {count_model_parameters(m1)}')
    print('-'*50)
    print('<m2 module>')
    print(f'model params: {count_model_parameters(m2)}')
    print('-'*50)
    print('<m3 module>')
    print(f'model params: {count_model_parameters(m3)}')
    print('-'*50)


if __name__ == "__main__":
    input = torch.Tensor(64, 96, 32, 32)
    B, C, H, W = input.shape
    self_conv = SelfConv2d(in_channels=C, out_channels=C, kernel_size=9)
    dw_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=7, 
                              padding=7//2, groups=C)
    r_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=7, 
                              padding=7//2, groups=1)
    
    # self_conv(input)
    compare_modules(input, self_conv, dw_conv, r_conv)


