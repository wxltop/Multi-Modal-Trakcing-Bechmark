import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    return nn.Conv2d(ic, oc, ksize, padding=ksize // 2, bias=bias, dilation=dilation, stride=stride)

def interpolate(t, sz):
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    return F.interpolate(t, sz, mode='bilinear', align_corners=False) if t.shape[-2:] != sz else t

def relu(negative_slope=0.0, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace=inplace)

def adaptive_cat(seq, dim=0, ref_tensor=0):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz) for t in seq], dim=dim)
    return t


class Upsample2x(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, (2*x.shape[-2], 2*x.shape[-1]), mode='bilinear', align_corners=False)



class TSE_DOLF(nn.Module):

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


class TSE(nn.Module):
    def __init__(self, fc, oc, num_transform_layers=3):
        super().__init__()

        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))  # Reduce number of feature dimensions

        layers = []
        for i in range(num_transform_layers):
            layers.extend([conv(oc, oc, 3), relu()])
        self.transform = nn.Sequential(*layers)

    def forward(self, ft, x=None):
        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):

    def __init__(self, oc, deepest=None):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        # self.deepest = deepest

    def forward(self, deeper, shallower):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        # deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        deeper_pool = F.adaptive_avg_pool2d(deeper, (1, 1))
        global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])

        return out


class RRB(nn.Module):

    def __init__(self, oc):
        super().__init__()
        self.conv1x1 = conv(oc, oc, 1)
        self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class DOLFRefinementNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()

        ic = in_channels
        oc = out_channels

        for L, fc in self.ft_channels.items():
            self.TSE[L] = TSE_DOLF(fc, ic, oc)
            self.RRB1[L] = RRB(oc)
            self.CAB[L] = CAB(oc, L == 'layer5')
            self.RRB2[L] = RRB(oc)

        self.project = nn.Sequential(relu(), conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False), conv(oc, 1, 1))

    def forward(self, score, features):
        x = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            h, hpool = self.TSE[L](ft, score, x)
            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)

        x = self.project(x)
        return x



class SegRefinementNetwork(nn.Module):

    def __init__(self, out_channels=64, ft_channels=None, num_TSE_tf_layers=3, final_upsample=1):
        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()

        oc = out_channels

        for L, fc in self.ft_channels.items():
            self.TSE[L] = TSE(fc, oc, num_TSE_tf_layers)
            self.RRB1[L] = RRB(oc)
            self.CAB[L] = CAB(oc)
            self.RRB2[L] = RRB(oc)

        # self.project = nn.Sequential(relu(), conv(oc, oc, 3), relu(), conv(oc, 1, 3))

        layers = []
        if final_upsample == 4:
            layers.append(Upsample2x())
        layers.append(conv(oc, oc, 3))
        if final_upsample >= 2:
            layers.append(Upsample2x())
        layers.extend([relu(), conv(oc, 1, 3)])
        self.project = nn.Sequential(*layers)

    def forward(self, features):
        x = None
        for L in list(self.ft_channels.keys())[::-1]:
            ft = features[L]
            h, hpool = self.TSE[L](ft, x)
            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)

        x = self.project(x)
        return x