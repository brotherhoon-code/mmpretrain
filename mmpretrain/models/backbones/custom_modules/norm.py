import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape:int,
                 eps=1e-6,
                 data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = normalized_shape
        
    def forward(self, x:torch.Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(input=x, 
                                normalized_shape=self.normalized_shape,
                                weight=self.weight,
                                bias=self.bias,
                                eps=self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(dim=1, keepdim=True)
            s = (x-u).pow(2).mean(1, keepdim=True)
            x = (x-u)/torch.sqrt(s + self.eps)
            x = self.weight[:, None, None]*x + self.bias[:, None, None]
            return x
        
if __name__ == "__main__":
    layer_norm = LayerNorm(normalized_shape=96, data_format="channels_first")
    output:torch.Tensor = layer_norm(torch.Tensor(64,96,56,56))
    print(output.shape)