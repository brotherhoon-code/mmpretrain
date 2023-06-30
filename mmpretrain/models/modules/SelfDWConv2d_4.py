import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat
from torchsummary import summary
from typing import Literal
import math


class SelfDWConv2d(nn.Module):
    def __init__(self, channels, kernel_size, n_kernels):
        super().__init__()
        
        self.temp = channels*2
        self.bn = nn.BatchNorm2d(num_features=channels)
        self.q_layer = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False) # b c h w
        self.q_reshape = Rearrange('b c h w -> b c (h w)')
        self.k_layer = nn.Conv2d(in_channels=channels, out_channels=n_kernels, kernel_size=1, bias=False) # b n_k h w
        self.k_reshape = Rearrange('b nk h w -> b (h w) nk')
        
        self.weight = nn.Parameter(torch.randn(n_kernels, kernel_size**2), requires_grad=True) # n_k k**2
        self.W_reshape = Rearrange('b c (k_h k_w) -> b c k_h k_w', k_h=kernel_size, k_w=kernel_size)
        
        self.activ_func = nn.Sigmoid()
        self.Bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
        
    
    def compute_similarity_matrix(self, A, B):
        similarity_matrix = torch.zeros(A.size(0), A.size(1), B.size(1))
        batch, n_tokens, _ = A.size()
        for b in range(batch):
            for q in range(n_tokens):
                for k in range(n_tokens):
                    query = A[b,q,:]
                    key = B[b,k,:]
                    similarity = cosine_similarity(query.unsqueeze(0), key.unsqueeze(0))
                    similarity_matrix[b,q,k] = similarity

        return similarity_matrix

    def forward(self, x:torch.Tensor):
        input = x
        B, C, _, _ = x.size()
        x = self.bn(x)
        Q = self.q_reshape(self.q_layer(x))
        K = self.k_reshape(self.k_layer(x))
        attn_map = torch.matmul(Q,K) 
        attn_score = F.softmax(attn_map/self.temp, dim=2) # b c nk
        
        '''
        print(attn_score.shape)
        print(attn_score[0][0])
        for score in sorted(attn_score[0][0],reverse=True)[:6]:
            print(score)
        print('MAX', max(attn_score[0][0]))
        print('MIN', min(attn_score[0][0]))
        '''
        
        weight = torch.matmul(attn_score, self.weight) # (b c n_k) x (n_k k**2) -> (b c k**2)
        weight = self.W_reshape(weight) # b c k k
        bias = self.Bias.unsqueeze(0).expand(B, -1)
        
        out = F.conv2d(
            Rearrange("b c h w -> 1 (b c) h w")(input),
            weight=Rearrange("b c kh kw -> (b c) 1 kh kw")(weight),
            bias= Rearrange("b c -> (b c)")(bias),
            stride=1,
            padding="same",
            groups=B * C,
        )
        
        out = Rearrange("1 (b c) h w -> b c h w", b=B, c=C)(out)
        return out


if __name__ == "__main__":
    CHANNELS = 96*4
    fm = torch.randn(64, CHANNELS, 58, 58)
    self_dwconv = SelfDWConv2d(CHANNELS, 7, CHANNELS*4)
    out = self_dwconv(fm)
    # print(out.shape)
