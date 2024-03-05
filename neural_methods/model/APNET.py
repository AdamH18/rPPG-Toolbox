import torch
from torch import nn
import torch.nn.functional as F
from timm.models.maxxvit import MaxxVitBlock, MaxxVitTransformerCfg

# https://paperswithcode.com/method/spatial-attention-module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class APNET(nn.Module):
    def __init__(self,
                 hwDim: int = 128,
                 tDim: int = 128,
                 dropout: float = 0.0):
        super(APNET, self).__init__()
        self.hwDim = hwDim
        self.tDim = tDim

        self.t_conv = nn.Sequential(
            nn.Conv2d(3, 3, [3, 3], 2, 1),
            nn.Conv2d(3, 3, [3, 3], 2, 1)
        )
        self.h_conv = nn.Sequential(
            nn.Conv2d(3, 3, [3, 3], 2, 1),
            nn.Conv2d(3, 3, [3, 3], 2, 1)
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(3, 3, [3, 3], 2, 1),
            nn.Conv2d(3, 3, [3, 3], 2, 1)
        )

        cfg = MaxxVitTransformerCfg()
        cfg.dim_head = 3
        cfg.window_size = (4, 4)
        cfg.grid_size = (4, 4)
        cfg.attn_drop = dropout
        cfg.proj_drop = dropout

        self.t_spatial = nn.Sequential(
            MaxxVitBlock(3, 3, transformer_cfg=cfg),
            SpatialGate(),
            nn.AdaptiveAvgPool2d((self.tDim, 16))
        )
        self.h_spatial = nn.Sequential(
            MaxxVitBlock(3, 3, transformer_cfg=cfg),
            SpatialGate(),
            nn.AdaptiveAvgPool2d((self.hwDim, 16))
        )
        self.w_spatial = nn.Sequential(
            MaxxVitBlock(3, 3, transformer_cfg=cfg),
            SpatialGate(),
            nn.AdaptiveAvgPool2d((self.hwDim, 16))
        )

        self.end_2dconv = nn.Conv2d(3, tDim, kernel_size=3, padding='same')
        self.end_1dconv = nn.Conv1d(tDim, 1, kernel_size=3, padding='same')

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv1d(tDim, tDim, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(tDim),
            torch.nn.Conv1d(tDim, tDim, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(tDim),
        )

    def forward(self, x):
        b, c, t, h, w = x.shape

        xt = x.transpose(1, 2).reshape((b*t, c, h, w))
        xh = x.transpose(1, 3).transpose(2, 3).reshape((b*h, c, t, w))
        xw = x.transpose(1, 4).transpose(2, 4).transpose(3, 4).reshape((b*w, c, t, h))

        xt = self.t_conv(xt)
        xh = self.h_conv(xh)
        xw = self.w_conv(xw)

        xt = xt.reshape((b, t, c, h//4, w//4)).transpose(1, 2).reshape(b, c, t, (h//4)*(w//4))
        xh = xh.reshape((b, h, c, t//4, w//4)).transpose(1, 2).reshape(b, c, h, (t//4)*(w//4))
        xw = xw.reshape((b, w, c, t//4, h//4)).transpose(1, 2).reshape(b, c, w, (t//4)*(h//4))

        xt = self.t_spatial(xt).reshape(b, c, t, 4, 4)
        xh = self.h_spatial(xh).reshape(b, c, h, 4, 4)
        xw = self.w_spatial(xw).reshape(b, c, w, 4, 4)

        #Feature mix with case 1. Skip interpolation as all dims should be 128
        mix = (((xw*xh) @ xt) + xt).reshape(b, c, t, 16)

        mix = self.end_2dconv(mix)
        mix = torch.mean(mix, dim=3)
        mix = self.residual_block(mix)
        mix = self.end_1dconv(mix).squeeze(1)

        return mix
