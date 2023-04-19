# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from torchsummary import summary

@BACKBONES.register_module()
class Carrot(nn.Module):
    def __init__(self):
        super(Carrot, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=1024, kernel_size=(3,3), padding=1)

    # backbone에서 return 타입에 따라 다르게 해야합니다.
    # 1. tuple 타입의 경우 각 레이어에서 나오는 텐서들을 리스트로 묶고 튜플로 변환시켜 리턴해야 합니다.
    # 2. tensor 타입일 경우 그냥 리턴하면 됩니다.
    def forward(self, x):
        outs = []
        x = self.conv1(x)
        outs.append(x)
        x = self.conv2(x)
        outs.append(x)
        return tuple(outs)


''' for model test
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    m = Carrot().to(device)
    summary(m, (3, 32, 32), device="cuda")
'''