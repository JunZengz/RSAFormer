from lib.RSAFormer.pvtv2 import pvt_v2_b4
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder_module import PAA_d
from .uper_head import UPerHead_v2
from .layers import conv, Conv, BNPReLU, self_attn
from .losses import bce_iou_loss
from .data_process import get_data_augmentation, get_transforms

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class PAA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(PAA_kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x


class PAA_e(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PAA_e, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x


class PDD(nn.Module):
    def __init__(self, channels=32):
        super(PDD, self).__init__()
        self.context2 = PAA_e(128, channels)
        self.context3 = PAA_e(320, channels)
        self.context4 = PAA_e(512, channels)
        self.decoder = PAA_d(channels)

    def forward(self, x2, x3, x4):
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.decoder(x4, x3, x2)

        return f5


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class RSA(nn.Module):
    def __init__(self, in_channel, channel):
        super(RSA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.ca1 = ChannelAttention(channel)
        self.sa1 = SpatialAttention()

        self.ca2 = ChannelAttention(channel)
        self.sa2 = SpatialAttention()

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)
        self.conv_out5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        fg = torch.sigmoid(map)
        p = fg - .5
        fg = torch.clip(p, 0, 1)  # foreground
        bg = torch.clip(-p, 0, 1)  # background
        cg = .5 - torch.abs(p)  # confusion area

        prob = torch.cat([fg, bg, cg], dim=1)
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c

        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        a = self.conv_out4(x)

        x = self.ca1(x) * x  # channel attention
        x = self.sa1(x) * x  # spatial attention
        a2 = self.conv_out5(x)
        a = a + a2 + map
        return a


class RSAFormer(nn.Module):
    def __init__(self, channel=32):
        super(RSAFormer, self).__init__()

        self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.CFP_2 = CFPModule(128, d=8)
        self.CFP_3 = CFPModule(320, d=8)
        self.CFP_4 = CFPModule(512, d=8)

        self.Translayer2_0 = BasicConv2d(128, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.Translayer_low = BasicConv2d(64, channel, 2, 2)

        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.decode_head = UPerHead_v2(num_classes=1, in_channels=[64, 128, 320, 512],
                                       in_index=[0, 1, 2, 3], channels=128,
                                       norm_cfg=dict(type='BN', requires_grad=True), dropout_ratio=0.1,
                                       align_corners=False, decoder_params=dict(embed_dim=768))

        self.pdd_decoder = PAA_d(channel)

        self.rsa = RSA(3 * channel, channel)

        self.out_simple = nn.Conv2d(channel, 1, 1)
        self.out_prediction = nn.Conv2d(4, 1, 1)

        self.loss_fn = bce_iou_loss
        self.transforms = get_transforms()
        self.data_augmentation = get_data_augmentation()

    def forward(self, sample):
        x = sample['images']
        y = sample['masks']

        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1 = self.ca(x1) * x1  # channel attention
        edge_feature = self.sa(x1) * x1  # spatial attention
        edge_feature = self.Translayer_low(edge_feature)

        uper_feature, prediction1 = self.decode_head([x1, x2, x3, x4])
        uper_feature = self.Translayer2_0(uper_feature)

        cfp_out_2 = self.CFP_2(x2)
        cfp_out_3 = self.CFP_3(x3)
        cfp_out_4 = self.CFP_4(x4)
        x2_t = self.Translayer2_1(cfp_out_2)
        x3_t = self.Translayer3_1(cfp_out_3)
        x4_t = self.Translayer4_1(cfp_out_4)
        pdd_feature, prediction2 = self.pdd_decoder(x4_t, x3_t, x2_t)

        fused_feature = torch.cat([uper_feature, pdd_feature, edge_feature], dim=1)
        prediction3 = self.rsa(fused_feature, prediction1)
        prediction4 = self.rsa(fused_feature, prediction2)

        prediction1 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
        prediction2 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')
        prediction4 = F.interpolate(prediction4, scale_factor=8, mode='bilinear')
        prediction = prediction1 + prediction2 + prediction3 + prediction4

        loss1 = self.loss_fn(prediction1, y)
        loss2 = self.loss_fn(prediction2, y)
        loss3 = self.loss_fn(prediction3, y)
        loss4 = self.loss_fn(prediction4, y)

        loss = loss1 + loss2 + loss3 + loss4

        return {'prediction': prediction, 'loss': loss}


if __name__ == '__main__':
    model = RSAFormer().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    out = model(input_tensor)
    print(out['prediction'].size())
