import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from typing import Any, Callable, List, Optional, Type, Union
from ..builder import BACKBONES
from mmpretrain.models.backbones.custom_modules.dynamic_conv_decomp import conv_dy


def getActFunc(type: str = "ReLU"):
    function = None
    if type == "ReLU":
        function = nn.ReLU(inplace=True)
    elif type == "GELU":
        function = nn.GELU()
    if function == None:
        raise ValueError(f"{type}is not implemented")
    return function


class ResnetStemBlock(nn.Module):
    def __init__(self, out_channels, deep_stem=True, act_func="ReLU", **kwrags):
        super().__init__()
        stem = []
        if deep_stem:
            stem.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=(7, 7),
                    stride=2,
                    padding=3,
                )
            )
            stem.append(nn.BatchNorm2d(out_channels))
            stem.append(getActFunc(act_func))
            stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            stem.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                )
            )
            stem.append(nn.BatchNorm2d(out_channels))
            stem.append(getActFunc(act_func))
        self.stem = nn.Sequential(*stem)

    def forward(self, x):
        return self.stem(x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        act_func: str = "ReLU",
    ):
        super().__init__()
        self.embed_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ_func = getActFunc(act_func)

    def forward(self, x):
        x = self.embed_layer(x)
        x = self.activ_func(x)
        x = self.bn(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels,
        out_channels,
        stride,
        act_func="ReLU",
        **kwargs,
    ):
        super().__init__()

        dynamic = kwargs.get("dynamic", False)
        dw = kwargs.get("dw", False)

        if dynamic == True:
            self.block = nn.Sequential(
                conv_dy(
                    inplanes=inter_channels,
                    planes=inter_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(inter_channels),
                getActFunc(act_func),
                
                
                conv_dy(
                    inplanes=in_channels,
                    planes=inter_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2d(inter_channels),
                getActFunc(act_func),
                
                
                conv_dy(
                    inplanes=inter_channels,
                    planes=inter_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                )
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1, stride),
                nn.BatchNorm2d(inter_channels),
                getActFunc(act_func),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
                if dw == False
                else nn.Conv2d(
                    inter_channels,
                    inter_channels,
                    kernel_size=3,
                    groups=inter_channels,
                    padding=1,
                ),
                nn.BatchNorm2d(inter_channels),
                getActFunc(act_func),
                nn.Conv2d(inter_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

        skip_conn = []
        if in_channels != out_channels and stride == 1:  # stage1_1
            skip_conn.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
            skip_conn.append(nn.BatchNorm2d(out_channels))

        elif in_channels != out_channels and stride != 1:
            skip_conn.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )
            skip_conn.append(nn.BatchNorm2d(out_channels))
        self.skip_conn = nn.Sequential(*skip_conn)

        self.active_func = getActFunc(act_func)

    def forward(self, x):
        identity = self.skip_conn(x)
        outs = self.block(x)
        outs = outs + identity
        outs = self.active_func(outs)
        return outs


@BACKBONES.register_module()
class DynamicDecompResNet(nn.Module):
    def __init__(
        self,
        stem_channels: int = 96,
        stage_blocks: list = [2, 2, 6, 2],
        feature_channels: list = [
            96,
            192,
            384,
            768,
        ],  # 3x3 stage1, stage2, stage3, stage4
        stage_out_channels: list = [
            96,
            192,
            384,
            768,
        ],  # last_channels stage1, stage2, stage3, stage4
        strides=[1, 1, 1, 1],
        act_func="ReLU",
        dw=[False, False, False, False],
        dynamic=[False, False, False, False],
        **kwargs,
    ):
        super().__init__()

        # 3 -> 96
        self.patch_embed1 = PatchEmbed(3, stem_channels, 4, act_func)
        # 96 -> 192
        self.patch_embed2 = PatchEmbed(
            stage_out_channels[0], stage_out_channels[0] * 2, 2, act_func
        )
        # 192 -> 384
        self.patch_embed3 = PatchEmbed(
            stage_out_channels[1], stage_out_channels[1] * 2, 2, act_func
        )
        # 384 -> 768
        self.patch_embed4 = PatchEmbed(
            stage_out_channels[2], stage_out_channels[2] * 2, 2, act_func
        )

        self.stage1 = nn.Sequential(
            *[
                ResBlock(
                    stem_channels,
                    feature_channels[0],
                    stage_out_channels[0],
                    1,
                    act_func,
                    dw=dw[0],
                    dynamic=dynamic[0],
                ),
                ResBlock(
                    stem_channels,
                    feature_channels[0],
                    stage_out_channels[0],
                    1,
                    act_func,
                    dw=dw[0],
                    dynamic=dynamic[0],
                ),
            ]
        )
        self.stage2 = nn.Sequential(
            *[
                ResBlock(
                    stage_out_channels[0] * 2,
                    feature_channels[1],
                    stage_out_channels[1],
                    1,
                    act_func,
                    dw=dw[1],
                    dynamic=dynamic[1],
                ),
                ResBlock(
                    stage_out_channels[0] * 2,
                    feature_channels[1],
                    stage_out_channels[1],
                    1,
                    act_func,
                    dw=dw[1],
                    dynamic=dynamic[1],
                ),
            ]
        )
        self.stage3 = nn.Sequential(
            *[
                ResBlock(
                    stage_out_channels[1] * 2,
                    feature_channels[2],
                    stage_out_channels[2],
                    1,
                    act_func,
                    dw=dw[2],
                    dynamic=dynamic[2],
                ),
                ResBlock(
                    stage_out_channels[1] * 2,
                    feature_channels[2],
                    stage_out_channels[2],
                    1,
                    act_func,
                    dw=dw[2],
                    dynamic=dynamic[2],
                ),
                ResBlock(
                    stage_out_channels[1] * 2,
                    feature_channels[2],
                    stage_out_channels[2],
                    1,
                    act_func,
                    dw=dw[2],
                    dynamic=dynamic[2],
                ),
                ResBlock(
                    stage_out_channels[1] * 2,
                    feature_channels[2],
                    stage_out_channels[2],
                    1,
                    act_func,
                    dw=dw[2],
                    dynamic=dynamic[2],
                ),
                ResBlock(
                    stage_out_channels[1] * 2,
                    feature_channels[2],
                    stage_out_channels[2],
                    1,
                    act_func,
                    dw=dw[2],
                    dynamic=dynamic[2],
                ),
                ResBlock(
                    stage_out_channels[1] * 2,
                    feature_channels[2],
                    stage_out_channels[2],
                    1,
                    act_func,
                    dw=dw[2],
                    dynamic=dynamic[2],
                ),
            ]
        )
        self.stage4 = nn.Sequential(
            *[
                ResBlock(
                    stage_out_channels[2] * 2,
                    feature_channels[3],
                    stage_out_channels[3],
                    1,
                    act_func,
                    dw=dw[3],
                    dynamic=dynamic[3],
                ),
                ResBlock(
                    stage_out_channels[2] * 2,
                    feature_channels[3],
                    stage_out_channels[3],
                    1,
                    act_func,
                    dw=dw[3],
                    dynamic=dynamic[3],
                ),
            ]
        )

    def forward(self, x):
        outs = []

        out = self.patch_embed1(x)
        out = self.stage1(out)
        outs.append(out)

        out = self.patch_embed2(out)
        out = self.stage2(out)
        outs.append(out)

        out = self.patch_embed3(out)
        out = self.stage3(out)
        outs.append(out)

        out = self.patch_embed4(out)
        out = self.stage4(out)
        outs.append(out)

        return tuple(outs)


if __name__ == "__main__":
    m = DynamicDecompResNet(dynamic=[False, False, False, False])
    summary(m, (3, 224, 224), device="cpu", batch_size=1)
