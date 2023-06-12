import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal


class SelfConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, reduction=4, temperature=2.):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = temperature
        if in_channels != out_channels:
            raise ValueError(
                f"in_channels({in_channels}) must be same out_channels{out_channels}"
            )

        self.key_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.key_reshaper = Rearrange("b c h w -> b (h w) c")  # b n c

        self.value_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.value_reshaper = Rearrange("b c h w -> b c (h w)")  # b c n

        self.linear0 = nn.Linear(
            in_features=in_channels, out_features=int(in_channels // reduction)
        )  # b c c//r
        self.layer_norm = nn.LayerNorm([in_channels, int(in_channels // reduction)])
        self.activ_func = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(
            in_features=int(in_channels // reduction),
            out_features=kernel_size * kernel_size,
        )  # b c kk
        self.reducer = Reduce("b c kk -> c kk", reduction="mean")
        self.kernel_shaper = Rearrange(
            "c (kh kw) -> c 1 kh kw", kh=kernel_size, kw=kernel_size
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _get_key(self, x: torch.Tensor):
        K = self.key_reshaper(self.key_layer(x))  # b n c
        return K

    def _get_value(self, x: torch.Tensor):
        V = self.value_reshaper(self.value_layer(x))  # b c n
        return V

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        # --- make kernel --- #
        V = self._get_value(x)  # b c n
        K = self._get_key(x)  # b n c
        channel_score = torch.matmul(V, K)  # b c c
        channel_score = F.softmax(channel_score/self.temperature, dim=1) # space 제한
        filters = self.linear0(channel_score) # b c c//r
        filters = self.layer_norm(filters) # b c c//r
        filters = self.activ_func(filters) # b c c//r
        filters = self.linear1(filters) # b c kk
        filters = self.reducer(filters) # c kk
        filters = self.kernel_shaper(filters) # c 1 k k
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
    output = SelfConv2d(in_channels=64, out_channels=64, kernel_size=7)(
        torch.Tensor(64, 64, 32, 32)
    )
    print(f"output.shape: {output.shape}")
