import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal


class GCconv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            raise ValueError(
                f"in_channels({in_channels}) must be same out_channels{out_channels}"
            )

        self.key_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=kernel_size * kernel_size,
            kernel_size=1,
            bias=False,
        )
        self.key_reshaper = Rearrange("b kk h w -> b (h w) kk")  # b n kk

        self.value_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.value_reshaper = Rearrange("b c h w -> b c (h w)")  # b c n

        self.weight_reshaper = Rearrange(
            "b c (k_h k_w) -> (b c) 1 k_h k_w", k_h=kernel_size, k_w=kernel_size
        )

    def _get_key(self, x: torch.Tensor):
        K = self.key_reshaper(self.key_layer(x))  # b n kk
        K = F.softmax(K, dim=1)
        return K

    def _get_value(self, x: torch.Tensor):
        V = self.value_reshaper(self.value_layer(x))  # b c n
        return V

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        # --- make kernel --- #
        K = self._get_key(x) # b n kk
        V = self._get_value(x) # b c n
        weights = torch.matmul(V, K)  # b c kk
        weights = self.weight_reshaper(weights)  # b c k k

        # --- convolution --- #
        x = x.reshape(1, -1, H, W)
        out = F.conv2d(
            input=x,
            weight=weights,
            bias=None,
            stride=1,
            padding=int(self.kernel_size // 2),
            groups=B * C,
        )
        out = out.view(B, C, H, W)
        return out


if __name__ == "__main__":
    output = GCconv2(in_channels=64, out_channels=64, kernel_size=7)(
        torch.Tensor(64, 64, 32, 32)
    )
    print(output.shape)
