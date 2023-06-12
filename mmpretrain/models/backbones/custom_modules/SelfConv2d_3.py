import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal


class SelfConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, reduction=4, temperature=2.0
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
        
        self.point_conv0 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels//4), kernel_size=1, bias=False)
        self.regular_conv = nn.Conv2d(in_channels=int(in_channels//4), out_channels=int(in_channels//4), kernel_size=kernel_size, padding="same", groups=1, bias=True)
        self.point_conv1 = nn.Conv2d(in_channels=int(in_channels//4), out_channels=out_channels, kernel_size=1, bias=False)

    def _get_query(self, x: torch.Tensor):
        K = self.q_reshaper(self.q_layer(x))  # b n c
        return K

    def _get_key(self, x: torch.Tensor):
        V = self.k_reshaper(self.k_layer(x))  # b c n
        return V

    def forward(self, x: torch.Tensor):
        # --- make self-convolution weights --- #
        Q = self._get_query(x)  # b c n
        K = self._get_key(x)  # b n c
        channel_score = torch.matmul(Q, K)  # b c c
        channel_score = F.softmax(channel_score / self.temperature, dim=1)  # space 제한
        weights = self.fusion(channel_score)

        out1 = F.conv2d(
            x,weight=weights,
            bias=None,
            stride=1,
            padding=int(self.kernel_size // 2),
            groups=self.in_channels,
        )
        
        out2 = self.point_conv1(self.regular_conv(self.point_conv0(x)))
        return (out1+out2)/2


if __name__ == "__main__":
    output = SelfConv2d(in_channels=64, out_channels=64, kernel_size=7)(
        torch.Tensor(64, 64, 32, 32)
    )
    print(f"output.shape: {output.shape}")
