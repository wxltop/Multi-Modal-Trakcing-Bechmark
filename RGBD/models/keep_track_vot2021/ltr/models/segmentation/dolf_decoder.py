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

        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))  # Reduce number of feature dimensions
        nc = ic + oc
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(),
                                       conv(nc, nc, 3), relu(),
                                       conv(nc, oc, 3), relu())

    def forward(self, ft, score, x=None):

        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):

    def __init__(self, oc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2*oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower, att_vec=None):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
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
    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels//2, 3)
        self.conv2 = conv(in_channels//2, 1, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2*x.shape[-2], 2*x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class RefelixNetwork2(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, new_upsampler=False, upsampler=None,
                 use_bn=False, use_backbone_feat=True):

        super().__init__()

        #self.use_backbone_feat = use_backbone_feat
        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()

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

            self.TSE[L] = TSE(fc, ic, oc[L]*out_channels)
            self.RRB1[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            self.CAB[L] = CAB(oc[L]*out_channels, L == last_layer)
            self.RRB2[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L]*out_channels
            prev_layer = L

        self.project = FelixUpsampler(out_channels)
        self._out_feature_channels = out_feature_channels

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5  # frames, seq, ch, h, w
        else:
            assert scores.dim() == 6  # frames, seq, obj, ch, h, w
        outputs = OrderedDict()

        scores = scores.view(-1, *scores.shape[-3:])

        x = None
        g = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            if not getattr(self, 'use_backbone_feat', True):
                ft = ft * 0.0


            s = interpolate(scores, ft.shape[-2:])
            if not x is None:
                x = self.proj[L](x)

            if num_objects is not None:
                h, hpool = self.TSE[L](ft.view(ft.shape[0], 1, *ft.shape[-3:]).repeat(1, num_objects, 1, 1, 1).view(-1, *ft.shape[-3:]), s, x)
            else:
                h, hpool = self.TSE[L](ft, s, x)

            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)
            if L == 'layer3':
                g = x

            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x

        x = self.project(x, image_size)
        return x, outputs


class RefelixNetwork2Enc(nn.Module):

    def __init__(self, in_channels=1, num_filters=32, out_channels=32, ft_channels=None, new_upsampler=False, upsampler=None,
                 use_bn=False, use_backbone_feat=True):

        super().__init__()

        self.use_backbone_feat = use_backbone_feat
        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()

        ic = in_channels
        #oc = out_channels

        oc = {'layer1': 1, 'layer2': 2, 'layer3': 2, 'layer4': 4}
        out_feature_channels = {}

        if 'layer4' in ft_channels.keys():
            last_layer = 'layer4'
        else:
            last_layer = 'layer3'

        prev_layer = None

        self.mask_enc_predictor = nn.Sequential(conv(ic, num_filters, 3), relu())

        ic = num_filters

        for L, fc in self.ft_channels.items():
            if not L==last_layer:
                self.proj[L] = nn.Sequential(conv(oc[prev_layer]*out_channels, oc[L]*out_channels, 1), relu())

            self.TSE[L] = TSE(fc, ic, oc[L]*out_channels)
            self.RRB1[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            self.CAB[L] = CAB(oc[L]*out_channels, L == last_layer)
            self.RRB2[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L]*out_channels
            prev_layer = L

        self.project = FelixUpsampler(out_channels)
        self._out_feature_channels = out_feature_channels

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5  # frames, seq, ch, h, w
        else:
            assert scores.dim() == 6  # frames, seq, obj, ch, h, w
        outputs = OrderedDict()

        scores = scores.view(-1, *scores.shape[-3:])
        scores_enc = self.mask_enc_predictor(scores)
        x = None
        g = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            if not self.use_backbone_feat:
                ft = ft * 0.0

            s = interpolate(scores_enc, ft.shape[-2:])
            if not x is None:
                x = self.proj[L](x)

            if num_objects is not None:
                h, hpool = self.TSE[L](ft.view(ft.shape[0], 1, *ft.shape[-3:]).repeat(1, num_objects, 1, 1, 1).view(-1, *ft.shape[-3:]), s, x)
            else:
                h, hpool = self.TSE[L](ft, s, x)

            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)
            if L == 'layer3':
                g = x

            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x

        x = self.project(x, image_size)
        return x, outputs



class RefelixNetwork2Att(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, new_upsampler=False, upsampler=None,
                 use_bn=False, att_dim=256):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.filter_att = nn.ModuleDict()
        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()

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

            if att_dim > 0:
                self.filter_att[L] = nn.Sequential(conv(in_channels, att_dim, 3), relu(),
                                                   conv(att_dim, fc, 3), nn.Sigmoid())
            else:
                self.filter_att[L] = nn.Sequential(conv(in_channels, fc, 3), nn.Sigmoid())

            self.TSE[L] = TSE(fc, ic, oc[L]*out_channels)
            self.RRB1[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            self.CAB[L] = CAB(oc[L]*out_channels, L == last_layer)
            self.RRB2[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L]*out_channels
            prev_layer = L

        self.project = FelixUpsampler(out_channels)
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
        g = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            # s = [interpolate(ss, ft.shape[-2:]) for ss in scores]  # Resample scores to match features size
            s = interpolate(scores, ft.shape[-2:])

            attention_weights = self.filter_att[L](s)
            ft_att = ft * attention_weights
            # s = torch.cat(s, dim=1)
            if not x is None:
                x = self.proj[L](x)
            if multi_targets:
                h, hpool = self.TSE[L](ft_att.repeat(num_targets,1,1,1), s, x)
            else:
                h, hpool = self.TSE[L](ft_att, s, x)

            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)
            if L == 'layer3':
                g = x

            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x

        x = self.project(x, image_size)
        return x, outputs


class NaiveUpsample(nn.Module):

    def __init__(self):

        super().__init__()

        self._out_feature_channels = {}

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5  # frames, seq, ch, h, w
        else:
            assert scores.dim() == 6  # frames, seq, obj, ch, h, w
        outputs = OrderedDict()

        scores = scores.view(-1, *scores.shape[-3:])

        x = F.interpolate(scores, image_size[-2:], mode='bicubic', align_corners=False)
        return x, outputs


class RefelixDecoder(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None,
                 use_bn=False, use_backbone_feat=True):

        super().__init__()

        self.use_backbone_feat = use_backbone_feat
        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()

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

            self.TSE[L] = TSE(fc, ic, oc[L]*out_channels)
            self.RRB1[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            self.CAB[L] = CAB(oc[L]*out_channels, L == last_layer)
            self.RRB2[L] = RRB(oc[L]*out_channels,us_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L]*out_channels
            prev_layer = L

        self._out_feature_channels = out_feature_channels

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5  # frames, seq, ch, h, w
        else:
            assert scores.dim() == 6  # frames, seq, obj, ch, h, w
        outputs = OrderedDict()

        scores = scores.view(-1, *scores.shape[-3:])

        x = None
        g = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            if not getattr(self, 'use_backbone_feat', True):
                ft = ft * 0.0

            s = interpolate(scores, ft.shape[-2:])
            if not x is None:
                x = self.proj[L](x)

            if num_objects is not None:
                h, hpool = self.TSE[L](ft.view(ft.shape[0], 1, *ft.shape[-3:]).repeat(1, num_objects, 1, 1, 1).view(-1, *ft.shape[-3:]), s, x)
            else:
                h, hpool = self.TSE[L](ft, s, x)

            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)
            if L == 'layer3':
                g = x

            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x

        return x, outputs
