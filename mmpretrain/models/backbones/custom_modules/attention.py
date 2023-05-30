import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import repeat
import torch.autograd


def getActFunc(type: str = "ReLU"):
    function = None
    if type == "ReLU":
        function = nn.ReLU(inplace=True)
    elif type == "GELU":
        function = nn.GELU()
    if function == None:
        raise ValueError(f"{type}is not implemented")
    return function

def fusion(input, attn, fusion:str):
    output = None
    if fusion == "scale":
        output = input * attn
    elif fusion == "add":
        output = input + attn
    else:
        raise ValueError(f"fusion={fusion} is not registered")
    return output
    

class SEModule(nn.Module):
    def __init__(self, in_channels, r=16, activ_func="ReLU", importance="mean", fusion="scale"):
        super().__init__()
        hidden_features = int(in_channels // r)
        self.fusion = fusion
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
        x = fusion(x, attn, self.fusion)
        return x


class CBAM_C(nn.Module):
    def __init__(self, in_channels, r=16, activ_func="ReLU", fusion="scale"):
        super().__init__()
        hidden_features = in_channels // r
        self.fusion = fusion
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
        x = fusion(x, attn, self.fusion)
        return x


class CBAM_S(nn.Module):
    def __init__(self, kernel_size=7, fusion="scale"):
        super().__init__()
        self.fusion = fusion
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
        x = fusion(x, attn, self.fusion)
        return x

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channels=16, temperature=1.0):
        super(Attention, self).__init__()
        attention_channels = max(
            int(in_channels * reduction), min_channels
        )  # reduction과 min중에서 큰값을 attention 체널로 정의
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temperature

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=attention_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(attention_channels)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(
            in_channels=attention_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True,
        )
        self.func_channel = self.get_channel_attention  # pass

        if (
            in_channels == groups and in_channels == out_channels
        ):  # if depthwise not using func filter
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            )
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # if point-wise not using func_spatial
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=kernel_size * kernel_size,
                kernel_size=1,
                bias=True,
            )
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(
                in_channels=attention_channels,
                out_channels=kernel_num,
                kernel_size=1,
                bias=True,
            )
            self.func_kernel = self.get_kernel_attention

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

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

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

    def get_kernel_attention(self, x: torch.Tensor):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return (
            self.func_channel(x),
            self.func_filter(x),
            self.func_spatial(x),
            self.func_kernel(x),
        )


class ODconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, reduction=0.0625, kernel_num=4, temperature=1.0):
        super(ODconv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        
        self.attention = Attention(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   groups=groups, reduction=reduction, kernel_num=kernel_num, temperature=temperature)
        
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True,
        )
        
        self._initialize_weights()
        
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common
        
    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature=temperature)
        
    def _forward_impl_common(self, x:torch.Tensor):
        channel_attn, filter_attn, spatial_attn, kernel_attn = self.attention(x)
        B, C, H, W = x.size()
        x = x*channel_attn
        x = x.reshape(1, -1, H, W)
        aggregate_weight = spatial_attn * kernel_attn * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_channels//self.groups, self.kernel_size, self.kernel_size]
        )
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups*B)
        output = output.view(B, self.out_channels, output.size(-2), output.size(-1))
        output = output*filter_attn
        return output
    
    def _forward_impl_pw1x(self, x):
        channel_attn, filter_attn, spatial_attn, kernel_attn = self.attention(x)
        x = x*channel_attn
        output =F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride,
                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = output*filter_attn
        return output
    
    def forward(self, x):
        return self._forward_impl(x)