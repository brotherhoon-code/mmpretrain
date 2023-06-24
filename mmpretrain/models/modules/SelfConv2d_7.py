import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal
"""
1. LN
- channel이 아니라 height, width에서만 실시

2. 유사도
- attention과 동일한 dot similarity

3. 풀링
- 7,7 로 통일

4. 퓨전
- 퓨전 미실시

※. 파라미터
- 35,267
"""

class SelfConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, temp=20.0, pooling_resolution=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temp = temp

        if in_channels != out_channels:
            raise ValueError(
                f"in_channels({in_channels}) must be same out_channels{out_channels}"
            )
            
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(pooling_resolution, pooling_resolution))
        self.layer_norm0 = nn.LayerNorm([pooling_resolution, pooling_resolution]) # LayerNorm을 Channel별로 독립으로 실시
        
        self.q_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.q_reshaper = Rearrange("b c p_h p_w -> b c (p_h p_w)")  # b c 49

        self.k_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.k_reshaper = Rearrange("b c p_h p_w -> b (p_h p_w) c")  # b n c
        
        self.kernel_reshaper = Rearrange('b c p_h p_w -> b c (p_h p_w)') # b c 49
        self.kernel_layer = nn.Linear(in_features=pooling_resolution**2, out_features=kernel_size**2, bias=False)
        
        self.filter_reshaper = Rearrange('b c (k_h k_w) -> b c k_h k_w', k_h=kernel_size, k_w=kernel_size)
        

    def _get_query(self, pooled_feature_map: torch.Tensor):
        Q = self.q_layer(pooled_feature_map)
        Q = self.q_reshaper(Q) # b c n
        return Q

    def _get_key(self, pooled_feature_map: torch.Tensor):
        K = self.k_layer(pooled_feature_map)
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
        
        x = self.avgpool(x) # b c 7 7
        x = self.layer_norm0(x)
        
        Q = self._get_query(x) # b c 49
        K = self._get_key(x) # b 49 c
        
        channel_attn = torch.matmul(Q,K) # b c c
        channel_attn = F.softmax(channel_attn/self.temp, dim=2) # b c c
        
        kernel_weights = self._get_kernel(x) # b c k**2
        kernel_weights = torch.matmul(channel_attn, kernel_weights) # b c k**2
        kernel_weights = self.filter_reshaper(kernel_weights)
        
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
    input = torch.Tensor(64, 128, 56, 56)
    B, C, H, W = input.shape
    conv = SelfConv2d(in_channels=C, out_channels=C, kernel_size=7)
    dw_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=7, padding=7//2, groups=C)
    r_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=7, padding=7//2, groups=1)
    compare_modules(input, conv, dw_conv, r_conv)
    summary(conv, (128, 56, 56), batch_size=64, device='cpu')


