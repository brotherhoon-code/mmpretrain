import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, stride=1, padding=0):
        super(DepthwiseConv, self).__init__()
        
        self.depth_multiplier = depth_multiplier
        self.weight = nn.Parameter(torch.randn(in_channels * depth_multiplier, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(in_channels * depth_multiplier))
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        x = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=x.size(1))
        return x

def count_params(model:nn.Module):
    return sum(p.numel() for p in model.parameters())
    

if __name__ == "__main__":
    B, C, H, W = 64, 192, 56, 56
    
    # 입력 이미지의 Shape: (B, C, H, W)
    input_image = torch.randn(B, C, H, W)

    # Depthwise Convolution을 위한 설정
    depth_multiplier = 1  # 입력 채널 수에 대한 배수 설정
    kernel_size = 3


    # Depthwise Convolution 수행
    depthwise_conv = DepthwiseConv(C, depth_multiplier, kernel_size)
    output = depthwise_conv(input_image)
    print(count_params(depthwise_conv))
