import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import repeat


def getActFunc(type: str = "ReLU"):
    function = None
    if type == "ReLU":
        function = nn.ReLU(inplace=True)
    elif type == "GELU":
        function = nn.GELU()
    if function == None:
        raise ValueError(f"{type}is not implemented")
    return function


class SEModule(nn.Module):
    def __init__(self, in_channels, r=16, activ_func="ReLU", importance="mean"):
        super().__init__()
        hidden_features = int(in_channels // r)
        self.squeeze = Reduce("B C H W -> B C 1 1", reduction="mean")
        self.flatten = nn.Flatten()
        self.mlp1 = nn.Linear(in_features=in_channels, out_features=hidden_features)
        self.activ_func = getActFunc(activ_func)
        self.mlp2 = nn.Linear(in_features=hidden_features, out_features=in_channels)
        self.prob_func = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape
        attn = self.squeeze(x)  # B C 1 1
        attn = self.flatten(attn)  # B C
        attn = self.mlp1(attn)  # B C//r
        attn = self.activ_func(attn)  # B C//r
        attn = self.mlp2(attn)  # B C
        attn = self.prob_func(attn)  # B C
        attn = attn.unsqueeze(dim=-1).unsqueeze(dim=-1)  # B C 1 1
        attn = repeat(attn, pattern="B C 1 1 -> B C H W", H=H, W=W)  # B C H W
        x = x * attn  # B C H W
        return x


class CBAM_C(nn.Module):
    def __init__(self, in_channels, r=16, activ_func="ReLU"):
        super().__init__()
        hidden_features = in_channels // r
        self.avgpool = Reduce(pattern="B C H W -> B C 1 1", reduction="mean")
        self.maxpool = Reduce(pattern="B C H W -> B C 1 1", reduction="max")
        self.flatten = nn.Flatten()
        self.mlp1 = nn.Linear(in_features=in_channels, out_features=hidden_features)
        self.activ_func = getActFunc(activ_func)
        self.mlp2 = nn.Linear(in_features=hidden_features, out_features=in_channels)
        self.prob_func = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # avg attn
        attn1 = self.avgpool(x)  # B C 1 1
        attn1 = self.flatten(attn1)  # B C
        attn1 = self.mlp1(attn1)  # B C//r
        attn1 = self.activ_func(attn1)  # B C//r
        attn1 = self.mlp2(attn1)  # B C
        # max attn
        attn2 = self.maxpool(x)  # B C 1 1
        attn2 = self.flatten(attn2)  # B C
        attn2 = self.mlp1(attn2)  # B C//r
        attn2 = self.activ_func(attn2)  # B C//r
        attn2 = self.mlp2(attn2)  # B C
        
        attn = attn1 + attn2  # B C
        attn = self.prob_func(attn)  # B C
        attn = attn.unsqueeze(dim=-1).unsqueeze(dim=-1)  # B C 1 1
        attn = repeat(attn, pattern="B C 1 1 -> B C H W", H=H, W=W)  # B C H W

        x = x * attn
        return x


class CBAM_S(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.avgpool = Reduce(pattern="B C H W -> B 1 H W", reduction="mean")
        self.maxpool = Reduce(pattern="B C H W -> B 1 H W", reduction="max")
        self.conv0 = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=kernel_size, padding="same"
        )
        self.prob_func = nn.Sigmoid()

    def forward(self, x):
        _, C, _, _ = x.shape
        attn = torch.concat([self.avgpool(x), self.maxpool(x)], dim=1)  # B 2 H W
        attn = self.conv0(attn)  # B 1 H W
        attn = self.prob_func(attn)  # B 1 H W
        attn = repeat(attn, pattern="B 1 H W -> B C H W", C=C)  # B C H W
        x = x * attn  # B C H W
        return x



if __name__ == "__main__":
    input = torch.arange(32 * 5 * 5).reshape(1, 32, 5, 5).float()
    output = CBAM_S(kernel_size=7)(input)
    print(input.shape)
    print(output.shape)
