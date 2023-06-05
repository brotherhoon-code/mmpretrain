import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal

class MultipleTanh(nn.Module):
    def __init__(self, scalar:float=1.0, temperature:float=1.0):
        super().__init__()
        self.activ_func = nn.Tanh()
        self.scalar = scalar
        self.temp = temperature
        
    def forward(self, x):
        x = self.scalar * self.activ_func(x/self.temp)
        return x

class MultipleSigmoid(nn.Module):
    def __init__(self, scalar:float):
        super().__init__()
        self.activ_func = nn.Sigmoid()
        self.scalar = scalar
        
    def forward(self, x):
        x = self.scalar*self.activ_func(x)
        return x


def get_activ_func(function_name:str="GELU"):
    if function_name == "ReLU":
        activ_func = nn.ReLU(inplace=True)
    elif function_name == "GELU":
        activ_func = nn.GELU()
    elif function_name == "None":
        activ_func = nn.Sequential()
    elif function_name == "Sigmoid":
        activ_func = nn.Sigmoid()
    elif function_name == "Sigmoid10":
        activ_func = MultipleSigmoid(scalar=10.)
    elif function_name == "Sigmoid20":
        activ_func = MultipleSigmoid(scalar=20.)
    elif function_name == "Sigmoid30":
        activ_func = MultipleSigmoid(scalar=30.)
    elif function_name == "Tanh":
        activ_func = MultipleTanh(scalar=1.0, temperature=1.0)
    elif function_name == "Tanh10":
        activ_func = MultipleTanh(scalar=10., temperature=10.)
    elif function_name == "Tanh100":
        activ_func = MultipleTanh(scalar=100., temperature=10.0)
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
        dropout_ratio: float = 0.2,
        activ_func:str = "GELU",
        bias:bool = False
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
        self.attn_bias = (
            nn.Parameter(torch.zeros(channel*batch), requires_grad=True)
            if bias == True
            else None
        )

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
    
def count_model_parameters(model):
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)
    return num_params

if __name__ == "__main__":
    input = torch.randn(32, 96, 52, 52)
    m = SpatialSelfConv(
        batch=32,
        channel=96,
        height=52,
        width=52,
        kernel_size=7,
        hidden_dim=96,
        activ_func="ReLU",
        bias=False,
    )
    output:torch.Tensor = m(input)
    print(output.shape)
    print(count_model_parameters(m))