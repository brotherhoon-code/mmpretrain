import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

class channel_pooling_module(nn.Module):
    def __init__(self, type="avg"):
        super().__init__()
        self.pooling_list = nn.ModuleList()
        # avg, max, mix
        if type == "avg":
            layer = nn.AdaptiveAvgPool2d((1,1))
            self.pooling_list.append(layer)
        elif type == "max":
            layer = nn.AdaptiveMaxPool2d((1,1))
            self.pooling_list.append(layer)
        else:
            layer = [nn.AdaptiveAvgPool2d((1,1)),
                     nn.AdaptiveMaxPool2d((1,1))]
            self.pooling_list.extend(layer)
    
    def forward(self, x):
        outs = []
        for layer in self.pooling_list:
            out = layer(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        return x
    
class spatial_attention(nn.Module):
    def __init__(self, hidden_channels, out_channels, type="avg"):
        super().__init__()
        self.pooling_list = nn.ModuleList()
        # avg, max, mix
        if type == "avg":
            layer = Reduce('B C H W -> B 1 H W', reduction="mean")
            self.pooling_list.append(layer)
        elif type == "max":
            layer = Reduce('B C H W -> B 1 H W', reduction="max")
            self.pooling_list.append(layer)
        else:
            layer = [Reduce('B C H W -> B 1 H W', reduction="mean"),
                     Reduce('B C H W -> B 1 H W', reduction="max")]
            self.pooling_list.extend(layer)
        
        self.conv1 = nn.Conv2d(in_channels=2 if type=="mix" else 1,
                               out_channels=hidden_channels,
                               kernel_size=7,
                               stride=1,
                               padding="same")
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_channels=hidden_channels,
                              out_channels=out_channels,
                              kernel_size=1)
        self.reshape_layer = Rearrange("B C H W -> B C (H W)")
    
    def forward(self, x):
        outs = []
        for layer in self.pooling_list:
            out = layer(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.reshape_layer(x)
        x = F.softmax(x, dim=2)
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
        return x
    

if __name__ == "__main__":
    print("hello world!")
    m = spatial_attention(hidden_channels=12, out_channels=4, type="avg")
    print(m(torch.Tensor(64,3,224,224)).shape)
        