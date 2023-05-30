# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .beit import BEiTViT
from .conformer import Conformer
from .convmixer import ConvMixer
from .convnext import ConvNeXt
from .cspnet import CSPDarkNet, CSPNet, CSPResNet, CSPResNeXt
from .davit import DaViT
from .deit import DistilledVisionTransformer
from .deit3 import DeiT3
from .densenet import DenseNet
from .edgenext import EdgeNeXt
from .efficientformer import EfficientFormer
from .efficientnet import EfficientNet
from .efficientnet_v2 import EfficientNetV2
from .hornet import HorNet
from .hrnet import HRNet
from .inception_v3 import InceptionV3
from .lenet import LeNet5
from .levit import LeViT
from .mixmim import MixMIMTransformer
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mobileone import MobileOne
from .mobilevit import MobileViT
from .mvit import MViT
from .poolformer import PoolFormer
from .regnet import RegNet
from .replknet import RepLKNet
from .repmlp import RepMLPNet
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .revvit import RevVisionTransformer
from .riformer import RIFormer
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tinyvit import TinyViT
from .tnt import TNT
from .twins import PCPVT, SVT
from .van import VAN
from .vgg import VGG
from .vig import PyramidVig, Vig
from .vision_transformer import VisionTransformer
from .xcit import XCiT

from .carrot import Carrot
from .custom_resnet import CustomResNet
from .custom_resnet_swinlike import SwinLikeResNet
from .custom_resnet_swinlike_dw import SwinLikeDWResNet
from .custom_resnet_swinlike_dw_rescale import SwinLikeRescaleDWResNet
from .custom_resnet_swinlike_dw_odconv import ODSwinLikeDWResNet
from .custom_convmixer import CustomConvMixer
from .a1 import A1
from .a2 import A2
from .a3 import A3
from .a4 import A4
from .a5 import A5
from .a6 import A6
from .a7 import A7
from .a8 import A8
from .a9 import A9
from .a10 import A10
from .a11 import A11
from .a12 import A12
from .a13 import A13
from .a14 import A14
from .a15 import A15


__all__ = [
    'LeNet5',
    'AlexNet',
    'VGG',
    'RegNet',
    'ResNet',
    'ResNeXt',
    'ResNetV1d',
    'ResNeSt',
    'ResNet_CIFAR',
    'SEResNet',
    'SEResNeXt',
    'ShuffleNetV1',
    'ShuffleNetV2',
    'MobileNetV2',
    'MobileNetV3',
    'VisionTransformer',
    'SwinTransformer',
    'TNT',
    'TIMMBackbone',
    'T2T_ViT',
    'Res2Net',
    'RepVGG',
    'Conformer',
    'MlpMixer',
    'DistilledVisionTransformer',
    'PCPVT',
    'SVT',
    'EfficientNet',
    'EfficientNetV2',
    'ConvNeXt',
    'HRNet',
    'ResNetV1c',
    'ConvMixer',
    'EdgeNeXt',
    'CSPDarkNet',
    'CSPResNet',
    'CSPResNeXt',
    'CSPNet',
    'RepLKNet',
    'RepMLPNet',
    'PoolFormer',
    'RIFormer',
    'DenseNet',
    'VAN',
    'InceptionV3',
    'MobileOne',
    'EfficientFormer',
    'SwinTransformerV2',
    'MViT',
    'DeiT3',
    'HorNet',
    'MobileViT',
    'DaViT',
    'BEiTViT',
    'RevVisionTransformer',
    'MixMIMTransformer',
    'TinyViT',
    'LeViT',
    'Vig',
    'PyramidVig',
    'XCiT',
    
    'Carrot',
    
    'CustomResNet',
    'SwinLikeResNet',
    'SwinLikeDWResNet',
    'SwinLikeRescaleDWResNet',
    'ODSwinLikeDWResNet',
    'CustomConvMixer',
    'A1',
    'A2',
    'A3',
    'A4',
    'A5',
    'A6',
    'A7',
    'A8',
    'A9',
    'A10',
    'A11',
    'A12',
    'A13',
    'A14',
    'A15'
    ]