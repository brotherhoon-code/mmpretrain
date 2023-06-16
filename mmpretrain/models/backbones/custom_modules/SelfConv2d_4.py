import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal


class SelfConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        reduction=16,
        temperature=2.0,
        pooling_resolution=7, # last stage의 feature_map resolution
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = temperature

        if in_channels != out_channels:
            raise ValueError(
                f"in_channels({in_channels}) must be same out_channels{out_channels}"
            )

        self.q_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.q_reshaper = Rearrange("b c h w -> b c (h w)")  # b c n

        self.k_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.k_reshaper = Rearrange("b c h w -> b (h w) c")  # b n c
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(pooling_resolution, pooling_resolution))

        self.fusion = nn.Sequential(
            nn.Linear(
                in_features=in_channels, out_features=int(in_channels // reduction)
            ),
            nn.LayerNorm([in_channels, int(in_channels // reduction)]),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=int(in_channels // reduction),
                out_features=kernel_size * kernel_size,
            ),
            Reduce("b c kk -> c kk", reduction="mean"),
            Rearrange("c (kh kw) -> c 1 kh kw", kh=kernel_size, kw=kernel_size),
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _get_query(self, x: torch.Tensor):
        x = self.q_layer(x)
        x = self.avgpool(x)
        K = self.q_reshaper(x)  # b n c_p
        return K

    def _get_key(self, x: torch.Tensor):
        x = self.k_layer(x)
        x = self.avgpool(x)
        V = self.k_reshaper(x)  # b c n_p
        return V

    def forward(self, x: torch.Tensor):
        # --- make self-convolution weights --- #
        Q = self._get_query(x)  # b c r**2
        K = self._get_key(x)  # b r**2 c
        channel_score = torch.matmul(Q, K)  # b c c
        channel_score = F.softmax(channel_score / self.temperature, dim=1)  # space 제한
        weights = self.fusion(channel_score)

        out = F.conv2d(
            x,weight=weights,
            bias=self.bias,
            stride=1,
            padding=int(self.kernel_size // 2),
            groups=self.in_channels,
        )
        
        return out


if __name__ == "__main__":
    output:torch.Tensor = SelfConv2d(in_channels=64, out_channels=64, kernel_size=7)(
        torch.Tensor(64, 64, 32, 32)
        )
    print(f"output.shape: {output.shape}")