import torch
from torch import nn
from torch.nn import functional as F
from ..builder import BACKBONES

'''
ref: https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
'''
class NLBlockND(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels=None, 
                 mode='embedded', 
                 dimension=2, 
                 bn_layer=False):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, 
                 in_channels:int, 
                 channels:int, 
                 stride=1, 
                 groups=1, 
                 width_per_group=64, 
                 **kwagrs):
        super(BottleNeckBlock, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        
        
        # width: 1x1에 의해서 축소될 체널입니다.
        # 최종적으로는 self.channels*self.expansion dim으로 out됩니다.
        self.width = int(channels * (width_per_group/64.)) * groups
        self.shortcut = []
        
        if stride!=1 or in_channels!=channels*self.expansion:
            self.shortcut.append(nn.Conv2d(self.in_channels,
                                           self.channels*self.expansion, 
                                           kernel_size=(1,1),
                                           stride=self.stride))
            self.shortcut.append(nn.BatchNorm2d(self.channels*self.expansion))
        
        self.shortcut = nn.Sequential(*self.shortcut)
        self.block = nn.Sequential(
            # contraction 1x1 conv
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1,1)), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(inplace=True), 
            # 3x3 padding=1 conv
            nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(3,3), stride=self.stride, groups=self.groups, padding=1), 
            nn.BatchNorm2d(self.width), 
            nn.ReLU(inplace=True), 
            # expasion 1x1 conv
            nn.Conv2d(in_channels=self.width, out_channels=self.channels*self.expansion, kernel_size=(1,1)), 
            nn.BatchNorm2d(self.channels*self.expansion), 
            # output channel dim is self.channels*expansion
        )
        
        
    def forward(self, x):
        skip = self.shortcut(x)
        x = self.block(x) + skip
        x = F.relu(x)
        return x


@BACKBONES.register_module()
class CustomNonLocalResNet2(nn.Module):
    def __init__(self, 
                 block_cls=BottleNeckBlock, 
                 deep_stem = True, # for cifar
                 n_blocks=[3,4,6,3],
                 n_classes=100, 
                 mode="embedded", 
                 n_local_stage_idx=[1,2,3,4],
                 **kwargs):
        super(CustomNonLocalResNet2, self).__init__()
        self.block_cls = block_cls
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.expansion = block_cls.expansion
        self.non_local_blocks = nn.ModuleList()
        n_local_cls = NLBlockND
            
        self.stem = []
        if deep_stem:
            self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3))
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU(inplace=True))
            self.stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.stem.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1))
            self.stem.append(nn.BatchNorm2d(64))
            self.stem.append(nn.ReLU(inplace=True))
        
        self.stem_block = nn.Sequential(*self.stem)
        self.stage1_block = self._make_stage(self.block_cls, 64, 64, n_blocks[0], 1)
        self.non_local_blocks.append(n_local_cls(in_channels=64*self.expansion, mode=mode)) if 1 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())
        
        self.stage2_block = self._make_stage(self.block_cls, 64*self.expansion, 128, n_blocks[1], 2)
        self.non_local_blocks.append(n_local_cls(in_channels=128*self.expansion, mode=mode)) if 2 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())
        
        self.stage3_block = self._make_stage(self.block_cls, 128*self.expansion, 256, n_blocks[2], 2)
        self.non_local_blocks.append(n_local_cls(in_channels=256*self.expansion, mode=mode)) if 3 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())

        self.stage4_block = self._make_stage(self.block_cls, 256*self.expansion, 512, n_blocks[3], 2)
        self.non_local_blocks.append(n_local_cls(in_channels=512*self.expansion, mode=mode)) if 4 in n_local_stage_idx else self.non_local_blocks.append(nn.Sequential())
        
        # self.neck = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier = nn.Linear(512*self.expansion, self.n_classes)
        
    def forward(self, x):
        outs=[]
        x = self.stem_block(x)

        x = self.stage1_block(x)
        x = self.non_local_blocks[0](x)
        outs.append(x)   
        
        x = self.stage2_block(x)
        x = self.non_local_blocks[1](x)
        outs.append(x)   
        
        x = self.stage3_block(x)
        x = self.non_local_blocks[2](x)
        outs.append(x)   
        
        x = self.stage4_block(x)
        x = self.non_local_blocks[3](x)
        outs.append(x)   
        
        # x = self.neck(x)
        # x = x.view(x.size()[0], -1)
        # x = self.classifier(x)
        return tuple(outs)
    
    @staticmethod
    def _make_stage(block, in_channels, out_channels, num_blocks, stride):
        stride_arr = [stride] + [1] * (num_blocks-1) # 각 스테이지의 첫번째 블록만 stride 반영
        layers, channels = [], in_channels
        for stride in stride_arr:
            layers.append(block(channels, out_channels, stride))
            channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    
'''
class CustomNonLocalResNet2(nn.Module):
    def __init__(self, 
                 block_cls=BottleNeckBlock, 
                 deep_stem = True, # for cifar
                 n_blocks=[3,4,6,3],
                 n_classes=100, 
                 n_local_cls="embedded", 
                 n_local_stage_idx=[1,2,3,4],
                 **kwargs):

'''
# gaussian, embedded, dot, concatenate
# n_cla
if __name__ == '__main__':
    import torch
    img = torch.zeros(1, 3, 224, 224)
    net = CustomNonLocalResNet2(n_local_cls='dot', n_local_stage_idx=[2,3])
    out = net(img)
    print(out[-1].size())