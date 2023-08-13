# # Copyright (c) OpenMMLab. All rights reserved.
# from functools import partial
# from itertools import chain
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as cp
# from mmcv.cnn.bricks import DropPath
# from mmengine.model import BaseModule, ModuleList, Sequential

# from mmpretrain.registry import MODELS
# from ..utils import GRN, build_norm_layer
# from .base_backbone import BaseBackbone

# class ConvNeXtBlock(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  dw_conv_cfg=dict(kernel_size=7, padding=3),
#                  norm_cfg=dict(type='LN2d', eps=1e-6),
#                  act_cfg=dict(type='GELU'),
#                  mlp_ratio=4.,
#                  linear_pw_conv=True,
#                  drop_path_rate=0.,
#                  layer_scale_init_value=1e-6,
#                  use_grn=False,
#                  with_cp=False):
#         super().__init__()
#         self.with_cp = with_cp

#         self.depthwise_conv = nn.Conv2d(
#             in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

#         self.linear_pw_conv = linear_pw_conv
#         self.norm = build_norm_layer(norm_cfg, in_channels)

#         mid_channels = int(mlp_ratio * in_channels)
#         if self.linear_pw_conv:
#             # Use linear layer to do pointwise conv.
#             pw_conv = nn.Linear
#         else:
#             pw_conv = partial(nn.Conv2d, kernel_size=1)

#         self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
#         self.act = MODELS.build(act_cfg)
#         self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

#         if use_grn:
#             self.grn = GRN(mid_channels)
#         else:
#             self.grn = None

#         self.gamma = nn.Parameter(
#             layer_scale_init_value * torch.ones((in_channels)),
#             requires_grad=True) if layer_scale_init_value > 0 else None

#         self.drop_path = DropPath(
#             drop_path_rate) if drop_path_rate > 0. else nn.Identity()

#     def forward(self, x):

#         def _inner_forward(x):
#             shortcut = x
#             x = self.depthwise_conv(x)

#             if self.linear_pw_conv:
#                 x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#                 x = self.norm(x, data_format='channel_last')
#                 x = self.pointwise_conv1(x)
#                 x = self.act(x)
#                 if self.grn is not None:
#                     x = self.grn(x, data_format='channel_last')
#                 x = self.pointwise_conv2(x)
#                 x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#             else:
#                 x = self.norm(x, data_format='channel_first')
#                 x = self.pointwise_conv1(x)
#                 x = self.act(x)

#                 if self.grn is not None:
#                     x = self.grn(x, data_format='channel_first')
#                 x = self.pointwise_conv2(x)

#             if self.gamma is not None:
#                 x = x.mul(self.gamma.view(1, -1, 1, 1))

#             x = shortcut + self.drop_path(x)
#             return x

#         if self.with_cp and x.requires_grad:
#             x = cp.checkpoint(_inner_forward, x)
#         else:
#             x = _inner_forward(x)
#         return x


# @MODELS.register_module()
# class Custom_ConvNeXt(BaseBackbone):
#     arch_settings = {
#         'atto': {
#             'depths': [2, 2, 6, 2],
#             'channels': [40, 80, 160, 320]
#         },
#         'femto': {
#             'depths': [2, 2, 6, 2],
#             'channels': [48, 96, 192, 384]
#         },
#         'pico': {
#             'depths': [2, 2, 6, 2],
#             'channels': [64, 128, 256, 512]
#         },
#         'nano': {
#             'depths': [2, 2, 8, 2],
#             'channels': [80, 160, 320, 640]
#         },
#         'tiny': {
#             'depths': [3, 3, 9, 3],
#             'channels': [96, 192, 384, 768]
#         },
#         'small': {
#             'depths': [3, 3, 27, 3],
#             'channels': [96, 192, 384, 768]
#         },
#         'base': {
#             'depths': [3, 3, 27, 3],
#             'channels': [128, 256, 512, 1024]
#         },
#         'large': {
#             'depths': [3, 3, 27, 3],
#             'channels': [192, 384, 768, 1536]
#         },
#         'xlarge': {
#             'depths': [3, 3, 27, 3],
#             'channels': [256, 512, 1024, 2048]
#         },
#         'huge': {
#             'depths': [3, 3, 27, 3],
#             'channels': [352, 704, 1408, 2816]
#         }
#     }

#     def __init__(self,
#                  arch='tiny',
#                  in_channels=3,
#                  stem_patch_size=4,
#                  norm_cfg=dict(type='LN2d', eps=1e-6),
#                  act_cfg=dict(type='GELU'),
#                  linear_pw_conv=True,
#                  use_grn=False, # v2, not use in v1
#                  drop_path_rate=0., # v1, v2
#                  layer_scale_init_value=1e-6, # v2 not, use in v1
#                  out_indices=-1,
#                  frozen_stages=0,
#                  gap_before_final_norm=True,
#                  with_cp=False,
#                  init_cfg=[
#                      dict(
#                          type='TruncNormal',
#                          layer=['Conv2d', 'Linear'],
#                          std=.02,
#                          bias=0.),
#                      dict(
#                          type='Constant', layer=['LayerNorm'], val=1.,
#                          bias=0.),
#                  ]):
#         super().__init__(init_cfg=init_cfg)

#         if isinstance(arch, str):
#             assert arch in self.arch_settings, \
#                 f'Unavailable arch, please choose from ' \
#                 f'({set(self.arch_settings)}) or pass a dict.'
#             arch = self.arch_settings[arch]
#         elif isinstance(arch, dict):
#             assert 'depths' in arch and 'channels' in arch, \
#                 f'The arch dict must have "depths" and "channels", ' \
#                 f'but got {list(arch.keys())}.'

#         self.depths = arch['depths']
#         self.channels = arch['channels']
#         assert (isinstance(self.depths, Sequence)
#                 and isinstance(self.channels, Sequence)
#                 and len(self.depths) == len(self.channels)), \
#             f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
#             'should be both sequence with the same length.'

#         self.num_stages = len(self.depths)

#         if isinstance(out_indices, int):
#             out_indices = [out_indices]
#         assert isinstance(out_indices, Sequence), \
#             f'"out_indices" must by a sequence or int, ' \
#             f'get {type(out_indices)} instead.'
#         for i, index in enumerate(out_indices):
#             if index < 0:
#                 out_indices[i] = 4 + index
#                 assert out_indices[i] >= 0, f'Invalid out_indices {index}'
#         self.out_indices = out_indices

#         self.frozen_stages = frozen_stages
#         self.gap_before_final_norm = gap_before_final_norm

#         # stochastic depth decay rule
#         dpr = [
#             x.item()
#             for x in torch.linspace(0, drop_path_rate, sum(self.depths))
#         ]
#         block_idx = 0

#         # 4 downsample layers between stages, including the stem layer.
#         self.downsample_layers = ModuleList()
#         stem = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 self.channels[0],
#                 kernel_size=stem_patch_size,
#                 stride=stem_patch_size),
#             build_norm_layer(norm_cfg, self.channels[0]),
#         )
#         self.downsample_layers.append(stem)

#         # 4 feature resolution stages, each consisting of multiple residual
#         # blocks
#         self.stages = nn.ModuleList()

#         for i in range(self.num_stages):
#             depth = self.depths[i]
#             channels = self.channels[i]

#             if i >= 1:
#                 downsample_layer = nn.Sequential(
#                     build_norm_layer(norm_cfg, self.channels[i - 1]),
#                     nn.Conv2d(
#                         self.channels[i - 1],
#                         channels,
#                         kernel_size=2,
#                         stride=2),
#                 )
#                 self.downsample_layers.append(downsample_layer)

#             stage = Sequential(*[
#                 ConvNeXtBlock(
#                     in_channels=channels,
#                     drop_path_rate=dpr[block_idx + j],
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     linear_pw_conv=linear_pw_conv,
#                     layer_scale_init_value=layer_scale_init_value,
#                     use_grn=use_grn,
#                     with_cp=with_cp) for j in range(depth)
#             ])
#             block_idx += depth

#             self.stages.append(stage)

#             if i in self.out_indices:
#                 norm_layer = build_norm_layer(norm_cfg, channels)
#                 self.add_module(f'norm{i}', norm_layer)

#         self._freeze_stages()

#     def forward(self, x):
#         outs = []
#         for i, stage in enumerate(self.stages):
#             x = self.downsample_layers[i](x)
#             x = stage(x)
#             if i in self.out_indices:
#                 norm_layer = getattr(self, f'norm{i}')
#                 if self.gap_before_final_norm:
#                     gap = x.mean([-2, -1], keepdim=True)
#                     outs.append(norm_layer(gap).flatten(1))
#                 else:
#                     outs.append(norm_layer(x))

#         return tuple(outs)

#     def _freeze_stages(self):
#         for i in range(self.frozen_stages):
#             downsample_layer = self.downsample_layers[i]
#             stage = self.stages[i]
#             downsample_layer.eval()
#             stage.eval()
#             for param in chain(downsample_layer.parameters(),
#                                stage.parameters()):
#                 param.requires_grad = False

#     def train(self, mode=True):
#         super(Custom_ConvNeXt, self).train(mode)
#         self._freeze_stages()
