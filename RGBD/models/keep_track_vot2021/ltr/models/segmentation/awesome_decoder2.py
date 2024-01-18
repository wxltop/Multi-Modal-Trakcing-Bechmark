from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.models.segmentation.utils import adaptive_cat, interpolate
from collections import OrderedDict


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    return nn.Conv2d(ic, oc, ksize, padding=ksize // 2, bias=bias, dilation=dilation, stride=stride)


def relu(negative_slope=0.0, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace=inplace)


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()

        self.reduce = nn.Sequential(conv(fc+ic, oc, 3), relu(), conv(oc, oc, 3), relu())  # Reduce number of feature dimensions
        #nc = ic + oc
        #self.transform = nn.Sequential(conv(nc, nc, 3), relu(),
        #                               conv(nc, nc, 3), relu(),
        #                               conv(nc, oc, 3), relu())

    def forward(self, ft, score):
        h = adaptive_cat((ft, score), dim=1, ref_tensor=0)
        h = self.reduce(h)
        #hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x

        #h = self.transform(h)
        return h


class CAB(nn.Module):

    def __init__(self, oc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2*oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower, att_vec=None):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        #deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        deeper_pool = F.adaptive_avg_pool2d(deeper, (1, 1))
        if att_vec is not None:
            global_pool = torch.cat([shallow_pool, deeper_pool, att_vec], dim=1)
        else:
            global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])

        return out


class RRB(nn.Module):

    def __init__(self, oc, us_bn=False):
        super().__init__()

        self.conv1x1 = conv(oc, oc, 1)
        if us_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class FelixUpsampler(nn.Module):
    def __init__(self, in_channels=64, reduce_factor=2):
        super().__init__()
        mid_channels = in_channels//reduce_factor
        self.conv1 = conv(in_channels, mid_channels, 3)
        self.conv2 = conv(mid_channels, 1, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2*x.shape[-2], 2*x.shape[-1]), mode='bilinear', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x


class SegNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=64, ft_channels=None, use_bn=False, skip_rrb=None, final_reduce_factor=2):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()

        if skip_rrb is None:
            self.skip_rrb = dict()
        else:
            self.skip_rrb = skip_rrb
        ic = in_channels
        #oc = out_channels

        oc = {'layer1': 1, 'layer2': 2, 'layer3': 2, 'layer4': 4}
        out_feature_channels = {}

        if 'layer4' in ft_channels.keys():
            last_layer = 'layer4'
        else:
            last_layer = 'layer3'

        prev_layer = None
        for L, fc in self.ft_channels.items():
            if not L==last_layer:
                self.proj[L] = nn.Sequential(conv(oc[prev_layer]*out_channels, oc[L]*out_channels, 1), relu())
            else:
                self.proj[L] = nn.Sequential(conv(self.ft_channels[L], oc[L]*out_channels, 1), relu())

            self.TSE[L] = TSE(fc, ic, oc[L]*out_channels)
            self.RRB1[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            self.CAB[L] = CAB(oc[L]*out_channels, L == last_layer)
            if L not in self.skip_rrb:
                self.RRB2[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L]*out_channels
            prev_layer = L

        self.project = FelixUpsampler(out_channels, reduce_factor=final_reduce_factor)
        self._out_feature_channels = out_feature_channels

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=()):
        assert scores.dim() == 5  # frames, seq, ch, h, w
        outputs = OrderedDict()

        scores = scores.view(-1, *scores.shape[-3:])
        num_targets = 1
        # num_targets = scores[0].shape[0]
        num_fmaps = features[next(iter(self.ft_channels))].shape[0]
        # if num_targets > num_fmaps:
        #     multi_targets = True
        # else:
        #     multi_targets = False
        multi_targets = False

        x = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]

            if x is None:
                x = ft

            x = self.proj[L](x)

            if multi_targets:
                h = self.TSE[L](ft.repeat(num_targets,1,1,1), scores)
            else:
                h = self.TSE[L](ft, scores)

            h = self.RRB1[L](h)
            h = self.CAB[L](x, h)
            if L not in self.skip_rrb:
                x = self.RRB2[L](h)
            else:
                x = h

            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x

        x = self.project(x, image_size)
        return x, outputs
