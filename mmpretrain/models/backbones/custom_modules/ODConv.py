import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import repeat
import torch.autograd
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import Literal


def fusion(input, attn, fusion: str):
    output = None
    if fusion == "scale":
        output = input * attn
    elif fusion == "add":
        output = input + attn
    else:
        raise ValueError(f"fusion={fusion} is not registered")
    return output


class Attention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups=1,
        reduction=16,
        kernel_num=1,
        min_channels=16,
        temperature=1.0,
        attn_type: Literal["spatial", "channel"] = "spatial",
    ):
        super().__init__()
        attention_channels = max(int(in_channels // reduction), min_channels)

        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temperature

        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=attention_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(num_features=attention_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attn_type = attn_type

        if attn_type == "channel":
            self.channel_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=True,
            )
            self.filter_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            )
            self.func_channel = self.get_channel_attention
            self.func_filter = self.get_filter_attention

        if attn_type == "spatial":
            self.spatial_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=kernel_size * kernel_size,
                kernel_size=1,
                bias=True,
            )
            self.func_spatial = self.get_spatial_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_channel_attention(self, x: torch.Tensor):
        channel_attention = torch.sigmoid(
            self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature
        )
        return channel_attention

    def get_filter_attention(self, x: torch.Tensor):
        filter_attention = torch.sigmoid(
            self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature
        )
        return filter_attention

    def get_spatial_attention(self, x: torch.Tensor):
        spatial_attention = self.spatial_fc(x).view(
            x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size
        )
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.attn_type == "spatial":
            return self.func_spatial(x)  # spatial attn
        elif self.attn_type == "channel":
            return self.func_filter(x), self.func_channel(x)  # out_c, in_c attention


class Spatial_ODconv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        reduction=16,
        kernel_num=1,
        temperature=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num

        self.attention = Attention(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            reduction=reduction,
            kernel_num=kernel_num,
            temperature=temperature,
            attn_type="spatial"
        )
        
        self.weight = nn.Parameter(
            torch.randn(kernel_num, 
                        out_channels, 
                        in_channels//groups, 
                        kernel_size, 
                        kernel_size), # K n_filters, filter_c, k, k
            requires_grad=True
        )
        
        # weight initialize
        self._initialize_weights()
        
        # depthwise    
        self._forward_impl = self._forward_impl_common
    
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
    
    def _forward_impl_common(self, x:torch.Tensor):
        spatial_attn = self.attention(x)
        B, C, H, W = x.size()
        x = x.reshape(1, -1, H, W)
        aggregate_weight = spatial_attn * self.weight.unsqueeze(dim=0)
        # aggregate_weight = [128, 1, 64, 1, 3, 3]
        # spatial_attn = [128, 1, 1, 1, 3, 3]
        # weight.unsqueeze = [1, 1, 64, 1, 3, 3]
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_channels//self.groups, self.kernel_size, self.kernel_size]
        )
        # aggregate_weight = [B*64, 1, 3, 3]
        output = F.conv2d(x, # [B, 64, 32, 32]
                          weight=aggregate_weight, # [B*64, 1, 3, 3]
                          bias=None,
                          stride=self.stride,
                          padding=self.padding,
                          dilation=self.dilation,
                          groups=self.groups*B # 64*B
                          )
        output = output.view(B, self.out_channels, output.size(-2), output.size(-1))
        return output
        
        
        
    def forward(self, x):
        return self._forward_impl(x)
        
        
class Channel_ODconv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 reduction=16,
                 kernel_num=1,
                 temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        
        self.attention = Attention(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            reduction=reduction,
            kernel_num=kernel_num,
            temperature=temperature,
            attn_type = "channel"
        )
        
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels//groups, kernel_num, kernel_num),
            requires_grad=True
        )
        self._initialize_weights()
        
        if self.kernel_size==1 and self.kernel_num==1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            raise ValueError(f"kernel_size={kernel_size}, kernel_num={kernel_num}")
    
    
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode="fan_out", nonlinearity="relu")
    
    def _forward_impl_pw1x(self, x):
        #  self.func_filter(x), self.func_channel(x)  # out_c, in_c attention
        filter_attn, channel_attn = self.attention(x)
        x = x*channel_attn
        output = F.conv2d(x,
                          weight=self.weight.squeeze(dim=0),
                          bias=None,
                          stride=self.stride,
                          padding=self.padding,
                          dilation=self.dilation,
                          groups=self.groups)
        output = output*filter_attn
        return output
    
    def forward(self, x):
        return self._forward_impl(x)
    
    
        
        


if __name__ == "__main__":
    input = torch.randn(128, 64, 32, 32)
    
    depthwise_conv = Spatial_ODconv(64, 64, 3, 1, 1, 1, 64, 16, 1, 1)
    output = depthwise_conv(input)
    print(output.shape)
    
    pointwise_conv = Channel_ODconv(64, 128, 1, 1, 0, 1, 1, 16, 1, 1.)
    output = pointwise_conv(input)
    print(output.shape)
