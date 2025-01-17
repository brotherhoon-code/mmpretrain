import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal
"""
SelfConv2d_5 대비 다른점
* Layernorm을 each channel별로 수행(왜냐하면, 체널까지 LN에 넣을경우 파라미터의 증가가 과함)
* projection -> 풀링
"""

class SelfConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, temp=20.0, pooling_resolution=7, bottle_ratio=4):
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
        
        if in_channels == 128:
            norm_resolution = 56
            pass
        elif in_channels == 256:
            norm_resolution = 28
            pass
        elif in_channels == 512:
            norm_resolution = 14
            pass
        elif in_channels == 1024:
            norm_resolution = 7
            pass
        
        self.layer_norm0 = nn.LayerNorm([norm_resolution, norm_resolution]) # feature map norm
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(pooling_resolution, pooling_resolution))
        
        self.q_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.q_reshaper = Rearrange("b c p_h p_w -> b c (p_h p_w)")  # b c 49

        self.k_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.k_reshaper = Rearrange("b c p_h p_w -> b (p_h p_w) c")  # b n c
        
        self.kernel_reshaper = Rearrange('b c p_h p_w -> b c (p_h p_w)') # b c 49
        self.kernel_layer = nn.Linear(in_features=pooling_resolution**2, out_features=kernel_size**2, bias=False)
        
        self.filter_reshaper = Rearrange('b c (k_h k_w) -> b c k_h k_w', k_h=kernel_size, k_w=kernel_size)
        
        self.fusion_conv0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//bottle_ratio, kernel_size=1)
        self.layer_norm1 = nn.LayerNorm([in_channels//bottle_ratio, kernel_size, kernel_size])
        
        self.activ_func = nn.ReLU(inplace=True)
        self.fusion_conv1 = nn.Conv2d(in_channels=in_channels//bottle_ratio, out_channels=out_channels, kernel_size=1)

    def _get_query(self, x: torch.Tensor):
        Q = self.q_layer(x)
        Q = self.avgpool(Q)
        Q = self.q_reshaper(Q) # b c n
        return Q

    def _get_key(self, x: torch.Tensor):
        K = self.k_layer(x)
        K = self.avgpool(K)
        K = self.k_reshaper(K) # b n c
        return K
    
    def _get_kernel(self, pooled_feature_map: torch.Tensor):
        kernel_weights = self.kernel_reshaper(pooled_feature_map)
        kernel_weights = self.kernel_layer(kernel_weights)
        return kernel_weights # b c k**2

    def forward(self, x: torch.Tensor):
        # get_query
        input = x
        B, C, _, _ = input.size()
        
        x = self.layer_norm0(x)
        
        Q = self._get_query(x) # b c 49
        K = self._get_key(x) # b 49 c
        
        channel_attn = torch.matmul(Q,K) # b c c
        channel_attn = F.softmax(channel_attn/self.temp, dim=2) # b c c
        
        x = self.avgpool(x) # b c 7 7
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
    input = torch.Tensor(64, 128, 32, 32)
    B, C, H, W = input.shape
    conv = SelfConv2d(in_channels=C, out_channels=C, kernel_size=7)
    dw_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=7, 
                              padding=7//2, groups=C)
    r_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=7, 
                              padding=7//2, groups=1)
    
    # compare_modules(input, conv, dw_conv, r_conv)
    summary(conv, (128, 56, 56), batch_size=64, device='cpu')


