import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal


def get_activ_func(function_name:Literal["ReLU", "GELU", "Sigmoid", "None"]="GELU"):
    if function_name == "ReLU":
        activ_func = nn.ReLU(inplace=True)
    elif function_name == "GELU":
        activ_func = nn.GELU()
    elif function_name == "None":
        activ_func = nn.Sequential()
    elif function_name == "Sigmoid":
        activ_func = nn.Sigmoid()
    else:
        raise ValueError(f"function_name = {function_name} is not registered")
    return activ_func

class SpatialSelfConv(nn.Module):
    def __init__(
        self,
        batch: int,
        channel: int,
        height: int,
        width: int,
        kernel_size: int,
        hidden_dim: int,
        bias = True,
        dropout_ratio: float = 0.2,
        activ_func:Literal["ReLU", "GELU", "Sigmoid", "None"] = "GELU"
    ):
        super().__init__()
        self.batch = batch
        self.channel = channel
        self.kernel_size = kernel_size

        self.reshape0 = Rearrange("B C H W -> B C (H W)")
        self.layernorm = nn.LayerNorm(height * width)
        self.mlp0 = nn.Linear(height * width, hidden_dim)
        self.activ_func = get_activ_func(activ_func)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.mlp1 = nn.Linear(hidden_dim, kernel_size * kernel_size)
        self.reshape1 = Rearrange(
            "B C (K_H K_W) -> (B C) 1 K_H K_W", K_H=kernel_size, K_W=kernel_size
        )
        self.attn_bias = nn.Parameter(torch.zeros(batch*channel), requires_grad=True) if bias == True else None
        # print(self.attn_bias.shape if self.attn_bias != None else "None")

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        
        ## (start) make kernel
        attn = self.reshape0(x)  # B C H*W
        attn = self.layernorm(attn)  # B C H*W
        attn = self.mlp0(attn)  # B C H
        attn = self.activ_func(attn)  # B C H
        attn = self.dropout(attn)  # B C H
        attn = self.mlp1(attn)  # B C K*K
        attn_weight = self.reshape1(attn)  # (B C) 1 K K
        ## (end) make kernel
        
        x = x.reshape(1, -1, H, W)  # 1 (B C) H W
        out = F.conv2d(
            x,
            weight=attn_weight,
            bias=self.attn_bias,
            stride=1,
            padding=int(self.kernel_size // 2),
            groups=C * B,
        )
        out = out.view(B, C, H, W)
        return out


if __name__ == "__main__":
    input = torch.randn(32, 96, 52, 52)
    m = SpatialSelfConv(
        batch=32,
        channel=96,
        height=52,
        width=52,
        kernel_size=7,
        hidden_dim=96,
        bias=True,
        activ_func="ReLU"
    )
    output = m(input)
    print(output.shape)
    # summary(m, input_size=(96, 52, 52), batch_size=128, device="cpu")
