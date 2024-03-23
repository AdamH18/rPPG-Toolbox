"""This file is a combination of Physformer.py and transformer_layer.py
   in the official PhysFormer implementation here:
   https://github.com/ZitongYu/PhysFormer

   model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

import numpy as np
from typing import Optional
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import math

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


class MultiHeadedCrossAndSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_kf = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_vf = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        self.proj_ks = nn.Sequential(
            CDC_T(dim*2, dim*2, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim*2),
        )
        self.proj_vs = nn.Sequential(
            nn.Conv3d(dim*2, dim*2, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, xf, xs, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=xf.shape
        [Bs, Ps, Cs]=xs.shape
        xf = xf.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        xs = xs.transpose(1, 2).view(Bs, Cs, Ps//16, 4, 4)
        q, kf, vf, ks, vs = self.proj_q(xf), self.proj_kf(xf), self.proj_vf(xf), self.proj_ks(xs), self.proj_vs(xs)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        kf = kf.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        vf = vf.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        ks = ks.flatten(1).view(B, C, P).transpose(1, 2)  # [B, 4*4*40, dim]
        vs = vs.flatten(1).view(B, C, P).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, kf, vf, ks, vs = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, kf, vf, ks, vs])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scoresf = q @ kf.transpose(-2, -1) / gra_sharp
        scoress = q @ ks.transpose(-2, -1) / gra_sharp

        scoresf = self.drop(F.softmax(scoresf, dim=-1))
        scoress = self.drop(F.softmax(scoress, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        hf = (scoresf @ vf).transpose(1, 2).contiguous()
        hs = (scoress @ vs).transpose(1, 2).contiguous()
        h = hs + hf
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scoresf
        return h, scoresf


class MultiHeadedPeriodicAndSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta, t):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        self.R = nn.Embedding(1, num_heads*dim*t)
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.lmbda = 0.5
        self.s = None
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        d = q @ k.transpose(-2, -1)
        r = self.R(torch.LongTensor([0]).to("cuda:0")).view(1, q.shape[1], q.shape[2], q.shape[3]).expand(q.shape[0], -1, -1, -1) # Get periodicity encoding
        s = q @ r.transpose(-2, -1)
        self.s = s
        scores = (d + self.lmbda*s) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        return x

class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score

class Block_ST_TDC_CA_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedCrossAndSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim*2, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, xf, xs, gra_sharp):
        Atten, Score = self.attn(self.norm1(xf), self.norm2(xs), gra_sharp)
        h = self.drop(self.proj(Atten))
        xf = xf + h
        h = self.drop(self.pwff(self.norm3(xf)))
        xf = xf + h
        return xf, Score
    
class Block_ST_TDC_Per_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta, t):
        super().__init__()
        self.attn = MultiHeadedPeriodicAndSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta, t)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, self.attn.s, Score

class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score
    
class Transformer_ST_TDC_CA_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_CA_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, xf, xss, gra_sharp):
        for i, block in enumerate(self.blocks):
            xf, Score = block(xf, xss[i], gra_sharp)
        return xf, Score

class Transformer_ST_TDC_Per_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta, t):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_Per_gra_sharp(dim, num_heads, ff_dim, dropout, theta, t) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        xss = []
        ss = []
        for block in self.blocks:
            x, s, Score = block(x, gra_sharp)
            xss.append(x)
            ss.append(s)
        return x, xss, ss, Score

# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_PP_gra_sharp(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None
    ):
        super().__init__()

        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding_slow = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        self.patch_embedding_fast = nn.Conv3d(dim, dim//2, kernel_size=(ft//2, fh, fw), stride=(ft//2, fh, fw))
        
        # Transformer
        self.fast_transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim//2, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.fast_transformer2 = Transformer_ST_TDC_CA_gra_sharp(num_layers=num_layers//3, dim=dim//2, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.fast_transformer3 = Transformer_ST_TDC_CA_gra_sharp(num_layers=num_layers//3, dim=dim//2, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        # Transformer
        self.slow_transformer1 = Transformer_ST_TDC_Per_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta, t=frame)
        # Transformer
        self.slow_transformer2 = Transformer_ST_TDC_Per_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta, t=frame)
        # Transformer
        self.slow_transformer3 = Transformer_ST_TDC_Per_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta, t=frame)
        
        
        
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Conv1 = nn.Conv3d(dim//2, dim//2, [3, 1, 1], stride=(2, 1, 1), padding=(1, 0, 0))
        self.Conv2 = nn.Conv3d(dim*3//2, dim, 1, stride=1, padding=0, groups=1, bias=False)
           
        self.pre_f_s = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ReLU(),
        )
        self.pre_f_f = nn.Sequential(
            nn.Conv3d(dim//2, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ReLU(),
        )
 
        self.ConvBlockLast = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim*3//2, dim*3//4, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim*3//4),
            nn.ReLU(),
            nn.AvgPool3d((1, 4, 4)),
            nn.Flatten(2),
            nn.Conv1d(dim*3//4, 1, 1, stride=1, padding=0)
        )

        self.SAverage = nn.AdaptiveAvgPool2d((frame//4, frame//4))
        
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp):

        # b is batch number, c channels, t frame, fh frame height, and fw frame width
        #x = x[:, :, :160, :, :]
        b, c, t, fh, fw = x.shape
        
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]
        
        xs = self.patch_embedding_slow(x)  # [B, 64, 40, 4, 4]
        xf = self.patch_embedding_fast(x)
        xs = xs.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]
        xf = xf.flatten(2).transpose(1, 2)
        

        xs, _, ss1, _ = self.slow_transformer1(xs, gra_sharp)
        xf, _ =  self.fast_transformer1(xf, gra_sharp)  # [B, 4*4*40, 64]
        xs, xss, ss2, _ = self.slow_transformer2(xs, gra_sharp)
        xf, _ =  self.fast_transformer2(xf, xss, gra_sharp)  # [B, 4*4*40, 64]
        xs = xs.transpose(1, 2).view(b, self.dim, t//4, 4, 4)
        xf = xf.transpose(1, 2).view(b, self.dim//2, t//2, 4, 4) # [B, 64, 40, 4, 4]
        xs = self.Conv2(torch.concat((xs, self.Conv1(xf)), dim=1)) # Lateral connection
        xs = xs.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]
        xf = xf.flatten(2).transpose(1, 2)
        xs, xss, ss3, _ = self.slow_transformer3(xs, gra_sharp)
        xf, _ =  self.fast_transformer3(xf, xss, gra_sharp)  # [B, 4*4*40, 64]
        
        # upsampling heads
        #features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 64, 40, 4, 4]
        xs = xs.transpose(1, 2).view(b, self.dim, t//4, 4, 4)
        xf = xf.transpose(1, 2).view(b, self.dim//2, t//2, 4, 4) # [B, 64, 40, 4, 4]
        
        xs = self.pre_f_s(xs)
        xf = self.pre_f_f(xf)
        x = torch.concat((xs, xf), dim=1)
        rPPG = self.ConvBlockLast(x)
        
        rPPG = rPPG.squeeze(1)

        ss1 = torch.cat(ss1, dim=1)
        ss2 = torch.cat(ss2, dim=1)
        ss3 = torch.cat(ss3, dim=1)
        ss = self.SAverage(torch.cat((ss1, ss2, ss3), dim=1))
        
        return rPPG, ss
