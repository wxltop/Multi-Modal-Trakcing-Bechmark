import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.layers.blocks import conv_block
from ltr.models.segmentation.utils import adaptive_cat
import ltr.models.target_classifier.features as target_cls_features
import math


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class BackboneWithHead(nn.Module):
    def __init__(self, backbone, head):
        super(BackboneWithHead, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        b_feat = self.backbone(x)
        out = self.head(b_feat)

        return out


class SegBlockPPM_keepsize(nn.Module):
    def __init__(self, feature_dim=256, num_blocks=1, l2norm=True, final_conv=True, norm_scale=1.0, out_dim=None,
                 interp_cat=False, use_res_block=False):
        super().__init__()
        if out_dim is None:
            out_dim = feature_dim

        if use_res_block:
            self.res_block = target_cls_features.residual_basic_block(num_blocks=1, l2norm=False)
        else:
            self.res_block = None
        self.conv_init = nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)

        self.final_conv = nn.Conv2d(4 * out_dim, out_dim, kernel_size=3, padding=1, bias=False)

        self.l2norm = l2norm
        if self.l2norm:
            self.norm = InstanceL2Norm(scale=norm_scale)

    def forward(self, x):

        x = F.relu(self.conv_init(x))

        xpool1 = self.conv1(F.avg_pool2d(x, kernel_size=2, stride=1))
        xpool2 = self.conv2(F.avg_pool2d(x, kernel_size=3, stride=1))
        xpool3 = self.conv3(F.avg_pool2d(x, kernel_size=6, stride=1))
        x = adaptive_cat([x, xpool1, xpool2, xpool3], dim=1, ref_tensor=0)

        x = self.final_conv(F.relu(x))
        if self.l2norm:
            x = self.norm(x)

        return x

class SegBlockSimple(nn.Module):
    def __init__(self, feature_dim=256, l2norm=True, norm_scale=1.0, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = feature_dim
        self.conv_init = nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.final_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)

        self.l2norm = l2norm
        if self.l2norm:
            self.norm = InstanceL2Norm(scale=norm_scale)

    def forward(self, x):
        x = F.relu(self.conv_init(x))
        x = self.final_conv(x)
        if self.l2norm:
            x = self.norm(x)

        return x

class SegBlockBoring(nn.Module):
    def __init__(self, feature_dim=256, l2norm=True, norm_scale=1.0, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = feature_dim

        self.final_conv = nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False)

        self.l2norm = l2norm
        if self.l2norm:
            self.norm = InstanceL2Norm(scale=norm_scale)

    def forward(self, x):
        x = self.final_conv(x)
        if self.l2norm:
            x = self.norm(x)

        return x


class SPP(nn.Module):
    def __init__(self, feature_dim=256, inter_dim=128, out_dim=None, use_bn=False):
        super().__init__()
        if out_dim is None:
            out_dim = feature_dim

        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(8, 8))

        self.conv1 = conv_block(feature_dim, inter_dim, kernel_size=1, padding=0, bias=False, batch_norm=use_bn, relu=True)
        self.conv2 = conv_block(feature_dim, inter_dim, kernel_size=1, padding=0, bias=False, batch_norm=use_bn, relu=True)
        self.conv3 = conv_block(feature_dim, inter_dim, kernel_size=1, padding=0, bias=False, batch_norm=use_bn, relu=True)
        self.conv4 = conv_block(feature_dim, inter_dim, kernel_size=1, padding=0, bias=False, batch_norm=use_bn, relu=True)

        self.final_conv = conv_block(4 * inter_dim + feature_dim, out_dim, kernel_size=3, padding=1, bias=False,
                                     batch_norm=use_bn, relu=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xpool1 = self.conv1(self.pool1(x))
        xpool2 = self.conv2(self.pool2(x))
        xpool3 = self.conv3(self.pool3(x))
        xpool4 = self.conv4(self.pool4(x))

        x = adaptive_cat([x, xpool1, xpool2, xpool3, xpool4], dim=1, ref_tensor=0)
        x = self.final_conv(x)
        return x
