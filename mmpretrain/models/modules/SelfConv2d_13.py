import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torchsummary import summary
from typing import Literal

# Output = (Input - Filter + 2*padding)/stride + 1
class SelfConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        
    def forward(self, x:torch.Tensor):
        return x
    
