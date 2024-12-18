import torch
import torch.nn as nn
from .layers import conv, self_attn
import torch.nn.functional as F

class PAA_d(nn.Module):
    # dense decoder, it can be replaced by other decoder previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(PAA_d, self).__init__()
        self.conv1 = conv(channel * 3, channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

    def forward(self, f1, f2, f3):
        f1 = self.upsample(f1, f3.shape[-2:])
        f2 = self.upsample(f2, f3.shape[-2:])
        f3 = torch.cat([f1, f2, f3], dim=1)
        f3 = self.conv1(f3)

        Hf3 = self.Hattn(f3)
        Wf3 = self.Wattn(f3)

        f3 = self.conv2(Hf3 + Wf3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        out = self.conv5(f3)

        return f3, out


class PAA_d_v2(nn.Module):
    # dense decoder, it can be replaced by other decoder previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(PAA_d_v2, self).__init__()
        self.conv1 = conv(channel * 3, channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

    def forward(self, f1, f2, f3):
        f1_ = self.upsample(f1, f3.shape[-2:])
        f2_ = self.upsample(f2, f3.shape[-2:])
        f3_ = f3
        f3 = torch.cat([f1_, f2_, f3_], dim=1)
        f3 = self.conv1(f3)

        Hf3 = self.Hattn(f3)
        Wf3 = self.Wattn(f3)

        f3 = self.conv2(Hf3 + Wf3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        out = self.conv5(f3)

        return f3, out, f1_, f2_, f3_


class PAA_d_v3(nn.Module):
    # dense decoder, it can be replaced by other decoder previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(PAA_d_v3, self).__init__()
        self.conv1 = conv(channel * 3, channel, 3)
        self.conv2 = conv(channel, channel, 3)
        self.conv3 = conv(channel, channel, 3)
        self.conv4 = conv(channel, channel, 3)
        self.conv5 = conv(channel, 1, 3, bn=False)

        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

    def forward(self, f1, f2, f3):
        f1_ = self.upsample(f1, f3.shape[-2:])
        f2_ = self.upsample(f2, f3.shape[-2:])
        f3_ = f3
        f3 = torch.cat([f1_, f2_, f3_], dim=1)
        f3 = self.conv1(f3)
        f3 = self.conv2(f3)
        f3 = self.conv3(f3)
        out = self.conv5(f3)

        return f3, out