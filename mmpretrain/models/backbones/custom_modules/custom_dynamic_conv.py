import torch
import torch.nn as nn
import torch.nn.functional as F

class pooling_module(nn.Module):
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
        else: # type "mix"
            layer1 = nn.AdaptiveAvgPool2d((1,1))
            layer2 = nn.AdaptiveMaxPool2d((1,1))
            self.pooling_list.append(layer1)
            self.pooling_list.append(layer2)
    
    def forward(self, x):
        outs = []
        for layer in self.pooling_list:
            out = layer(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        return x
        
        

class channel_attention_module(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio,
                 K,
                 temperature,
                 pool_mode="avg",
                 init_weight=True):
        super().__init__()
        assert temperature%3 == 1
        self.pool = pooling_module(pool_mode)
        if pool_mode == "mix":
            in_channels = in_channels*2
        
        if in_channels != 3:
            hidden_channels = int(in_channels*ratio) + 1
        else:
            hidden_channels = K
        # projection
        self.fc1 = nn.Conv2d(in_channels=in_channels,
                             out_channels=hidden_channels,
                             kernel_size=(1,1),
                             bias=False) 
        # importance
        self.fc2 = nn.Conv2d(in_channels=hidden_channels,
                             out_channels=K,
                             kernel_size=(1,1))
        self.temperature = temperature
        
        if init_weight:
            self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=34,
        pool_mode="mix",
        init_weight=True,
    ):
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = channel_attention_module(
            in_channels=in_channels,
            ratio=ratio,
            K=K,
            temperature=temperature,
            pool_mode=pool_mode,
            init_weight=True
        )

        # [K, out_c, in_c, k, k]
        self.weight = nn.Parameter(
            torch.randn(
                K, out_channels, in_channels // groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels))
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_normal_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        attention_score = self.attention(x)
        B, C, H, W = x.size()
        x = x.view(1, -1, H, W)  # [1, B*C, H, W]
        weight = self.weight.view(self.K, -1)  # [K, out_c*in_c*k*k]

        aggregate_weight = torch.mm(attention_score, weight).view(
            B * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(attention_score, self.bias).view(-1)
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * B,
            )
        output = output.view(B, self.out_channels, output.size(-2), output.size(-1))
        return output

def count_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    input = torch.randn((128, 96, 52, 52))
    # m = attention2d(in_channels=96, ratio=0.25, K=4, temperature=34, init_weight=True)
    # print(f"{m(input).shape}") # 128, 4
    
    m = Dynamic_conv2d(in_channels=96,
                       out_channels=96,
                       kernel_size=3,
                       ratio=0.25,
                       K=4,
                       temperature=34,
                       pool_mode="avg")
    print(count_parameters(m), "params")
    print(m(input).shape)
    
    
    
# torch.Size([128, 96, 1, 1])
# torch.Size([128, 25, 1, 1])
# torch.Size([128, 4, 1, 1])
# torch.Size([128, 4])
# torch.Size([128, 4]


# torch.Size([128, 192, 1, 1])
# torch.Size([128, 49, 1, 1])
# torch.Size([128, 4, 1, 1])
# torch.Size([128, 4])
# torch.Size([128, 4])