from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ltr.models.segmentation.utils import adaptive_cat, interpolate


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p  # , p2, p3, p4


class DecoderResnet18(nn.Module):
    def __init__(self, filter_out_dim, mdim):
        super(DecoderResnet18, self).__init__()
        self.convFM = nn.Conv2d(256 + filter_out_dim, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(64, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, filter_output, feat_list, out_shape=None, output_layers=()):
        assert filter_output.dim() == 5
        r4 = feat_list['layer3']
        r3 = feat_list['layer2']
        r2 = feat_list['layer1']

        outputs = OrderedDict()

        num_sequences = filter_output.shape[1]
        num_frames = filter_output.shape[0]

        filter_output = filter_output.view(-1, *filter_output.shape[-3:])
        r4_f = torch.cat((filter_output, r4), dim=1)
        if 'r4_f' in output_layers:
            outputs['r4_f'] = r4_f

        m4 = self.ResMM(self.convFM(r4_f))
        if 'm4' in output_layers:
            outputs['m4'] = m4

        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        if 'm3' in output_layers:
            outputs['m3'] = m3

        m2 = self.RF2(r2, m3)  # out: 1/4, 256
        if 'm2' in output_layers:
            outputs['m2'] = m2

        p2 = self.pred2(F.relu(m2))
        if 'p2' in output_layers:
            outputs['p2'] = p2

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)

        p = p.view(num_frames, num_sequences, *p.shape[-2:])
        return p, outputs  # , p2, p3, p4


class DecoderResnet18Mask(nn.Module):
    def __init__(self, filter_out_dim, mdim):
        super(DecoderResnet18Mask, self).__init__()
        self.convFM = nn.Conv2d(256 + filter_out_dim, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128 + filter_out_dim, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(64 + filter_out_dim, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, filter_output, feat_list, out_shape=None, output_layers=()):
        assert filter_output.dim() == 5
        r4 = feat_list['layer3']
        r3 = feat_list['layer2']
        r2 = feat_list['layer1']

        outputs = OrderedDict()

        num_sequences = filter_output.shape[1]
        num_frames = filter_output.shape[0]

        filter_output = filter_output.view(-1, *filter_output.shape[-3:])
        r4_f = torch.cat((filter_output, r4), dim=1)
        if 'r4_f' in output_layers:
            outputs['r4_f'] = r4_f

        m4 = self.ResMM(self.convFM(r4_f))
        if 'm4' in output_layers:
            outputs['m4'] = m4

        filter_output_3 = interpolate(filter_output, r3.shape[-2:])
        r3_f = torch.cat((filter_output_3, r3), dim=1)
        m3 = self.RF3(r3_f, m4)  # out: 1/8, 256
        if 'm3' in output_layers:
            outputs['m3'] = m3

        filter_output_2 = interpolate(filter_output, r2.shape[-2:])
        r2_f = torch.cat((filter_output_2, r2), dim=1)
        m2 = self.RF2(r2_f, m3)  # out: 1/4, 256
        if 'm2' in output_layers:
            outputs['m2'] = m2

        p2 = self.pred2(F.relu(m2))
        if 'p2' in output_layers:
            outputs['p2'] = p2

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)

        p = p.view(num_frames, num_sequences, *p.shape[-2:])
        return p, outputs  # , p2, p3, p4


class DecoderResnet50(nn.Module):
    def __init__(self, filter_out_dim, mdim):
        super(DecoderResnet50, self).__init__()
        self.convFM = nn.Conv2d(1024 + filter_out_dim, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, filter_output, feat_list, out_shape=None, output_layers=()):
        assert filter_output.dim() == 5
        r4 = feat_list['layer3']
        r3 = feat_list['layer2']
        r2 = feat_list['layer1']

        outputs = OrderedDict()

        num_sequences = filter_output.shape[1]
        num_frames = filter_output.shape[0]

        filter_output = filter_output.view(-1, *filter_output.shape[-3:])
        r4_f = torch.cat((filter_output, r4), dim=1)
        if 'r4_f' in output_layers:
            outputs['r4_f'] = r4_f

        m4 = self.ResMM(self.convFM(r4_f))
        if 'm4' in output_layers:
            outputs['m4'] = m4

        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        if 'm3' in output_layers:
            outputs['m3'] = m3

        m2 = self.RF2(r2, m3)  # out: 1/4, 256
        if 'm2' in output_layers:
            outputs['m2'] = m2

        p2 = self.pred2(F.relu(m2))
        if 'p2' in output_layers:
            outputs['p2'] = p2

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)

        p = p.view(num_frames, num_sequences, *p.shape[-2:])
        return p, outputs  # , p2, p3, p4


class DecoderResnet50Mask(nn.Module):
    def __init__(self, filter_out_dim, mdim):
        super(DecoderResnet50Mask, self).__init__()
        self.convFM = nn.Conv2d(1024 + filter_out_dim, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512 + filter_out_dim, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256 + filter_out_dim, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, filter_output, feat_list, out_shape=None, output_layers=()):
        assert filter_output.dim() == 5
        r4 = feat_list['layer3']
        r3 = feat_list['layer2']
        r2 = feat_list['layer1']

        outputs = OrderedDict()

        num_sequences = filter_output.shape[1]
        num_frames = filter_output.shape[0]

        filter_output = filter_output.view(-1, *filter_output.shape[-3:])
        r4_f = torch.cat((filter_output, r4), dim=1)
        if 'r4_f' in output_layers:
            outputs['r4_f'] = r4_f

        m4 = self.ResMM(self.convFM(r4_f))
        if 'm4' in output_layers:
            outputs['m4'] = m4

        filter_output_3 = interpolate(filter_output, r3.shape[-2:])
        r3_f = torch.cat((filter_output_3, r3), dim=1)
        m3 = self.RF3(r3_f, m4)  # out: 1/8, 256
        if 'm3' in output_layers:
            outputs['m3'] = m3

        filter_output_2 = interpolate(filter_output, r2.shape[-2:])
        r2_f = torch.cat((filter_output_2, r2), dim=1)
        m2 = self.RF2(r2_f, m3)  # out: 1/4, 256
        if 'm2' in output_layers:
            outputs['m2'] = m2

        p2 = self.pred2(F.relu(m2))
        if 'p2' in output_layers:
            outputs['p2'] = p2

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)

        p = p.view(num_frames, num_sequences, *p.shape[-2:])
        return p, outputs  # , p2, p3, p4
