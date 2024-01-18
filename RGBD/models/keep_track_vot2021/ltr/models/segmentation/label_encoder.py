import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from ltr.models.backbone.resnet import BasicBlock
from ltr.models.layers.blocks import conv_block
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.segmentation.features import SPP
from ltr.models.segmentation.utils import interpolate


class ResidualDS16(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return out.view(label_shape[0], label_shape[1], *out.shape[-3:])


class PoolDS16(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.pool(self.pool(self.pool(label_mask))))

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return out.view(label_shape[0], label_shape[1], *out.shape[-3:])


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label_mask, feature=None):
        assert label_mask.dim() == 4
        return label_mask.unsqueeze(2)


class ResidualDS16Feat(nn.Module):
    def __init__(self, layer_dims, feat_dim, instance_norm=False, norm_scale=1.0):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2)

        ds3 = nn.Conv2d(layer_dims[2] + feat_dim, layer_dims[3], kernel_size=3, padding=1, stride=1)
        self.res3 = BasicBlock(layer_dims[2] + feat_dim, layer_dims[3], stride=1, downsample=ds3)

        self.norm = None
        if instance_norm:
            self.norm = InstanceL2Norm(scale=norm_scale)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feat):
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))

        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat((mask_enc, feat), dim=1)
        out = self.res3(feat_mask_enc)

        if self.norm is not None:
            out = self.norm(out)

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return out.view(label_shape[0], label_shape[1], *out.shape[-3:])


class ResidualDS16SW(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w

class ResidualDS16SWFGBG(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Sequential(conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=True, batch_norm=use_bn), conv_block(layer_dims[3], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=False, batch_norm=use_bn), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        bg_prob = 1.0 - F.adaptive_max_pool2d(label_mask, (1,1))

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)

        spatial_weight = bg_prob * interpolate(label_mask, out.shape[-2:])
        sample_w = self.samp_w_pred(spatial_weight * out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w

class ResidualDS16SWv2(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block1 = conv_block(1, layer_dims[0], kernel_size=7, stride=2, padding=3, batch_norm=use_bn)
        self.conv_block2 = conv_block(layer_dims[0], layer_dims[1], kernel_size=5, stride=2, padding=2, batch_norm=use_bn)

        ds1 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[2], layer_dims[3], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[3], layer_dims[4], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.conv_block2(self.conv_block1(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w


class ResidualDS16NoSW(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc


class Conv1DS16(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=17, stride=16, padding=8, batch_norm=use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.conv_block(label_mask)
        label_enc = out.view(label_shape[0], label_shape[1], *out.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc


class Conv1DS16SWConv(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=17, stride=16, padding=8, batch_norm=use_bn)
        self.samp_w_pred = nn.Conv2d(1, layer_dims[0], kernel_size=17, padding=8, stride=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.conv_block(label_mask)
        label_enc = out.view(label_shape[0], label_shape[1], *out.shape[-3:])
        sample_w = self.samp_w_pred(label_mask)

        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w


class Conv1DS16SWRes(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.label_pred = conv_block(1, layer_dims[3], kernel_size=17, stride=16, padding=8, batch_norm=use_bn)

        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        label_enc = self.label_pred(label_mask)

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w


class IdentityDS16SW(nn.Module):
    def __init__(self, layer_dims, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        sample_w = self.samp_w_pred(out)

        label_mask = label_mask.view(label_shape[0], label_shape[1], 1, *label_mask.shape[-2:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_mask, sample_w


class ResidualDS16FeatDeep1(nn.Module):
    def __init__(self, layer_dims, feat_dim, instance_norm=False, norm_scale=1.0):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2)

        self.feat_conv = conv_block(feat_dim, 256, kernel_size=3, stride=1, padding=1)
        ds3 = nn.Conv2d(layer_dims[2] + 256, layer_dims[3], kernel_size=3, padding=1, stride=1)
        self.res3 = BasicBlock(layer_dims[2] + 256, layer_dims[3], stride=1, downsample=ds3)

        self.norm = None
        if instance_norm:
            self.norm = InstanceL2Norm(scale=norm_scale)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, label_mask, feat):
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))

        feat = feat.view(-1, *feat.shape[-3:])
        feat_c = self.feat_conv(feat)
        feat_mask_enc = torch.cat((mask_enc, feat_c), dim=1)
        out = self.res3(feat_mask_enc)

        if self.norm is not None:
            out = self.norm(out)

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return out.view(label_shape[0], label_shape[1], *out.shape[-3:])


class ResidualDS16FeatSW(nn.Module):
    def __init__(self, layer_dims, feat_dim,  use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        ds3 = nn.Conv2d(layer_dims[2] + feat_dim, layer_dims[3], kernel_size=3, padding=1, stride=1)
        self.res3 = BasicBlock(layer_dims[2] + feat_dim, layer_dims[3], stride=1, downsample=ds3, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[3], layer_dims[4], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feat):
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))

        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat((mask_enc, feat), dim=1)
        out = self.res3(feat_mask_enc)

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w

class ResidualDS16FeatSWAtt(nn.Module):
    def __init__(self, layer_dims, feat_dim,  use_final_relu=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2)

        ds3 = nn.Conv2d(layer_dims[2] + feat_dim, layer_dims[3], kernel_size=3, padding=1, stride=1)
        self.res3 = BasicBlock(layer_dims[2] + feat_dim, layer_dims[3], stride=1, downsample=ds3)

        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu)

        self.samp_w_pred = nn.Conv2d(layer_dims[3], layer_dims[4], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feat):
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))

        feat = feat.view(-1, *feat.shape[-3:])
        feat = feat * interpolate(label_mask, feat.shape[-2:])
        feat_mask_enc = torch.cat((mask_enc, feat), dim=1)
        out = self.res3(feat_mask_enc)

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w

def interpolate(t, sz, mode='bilinear'):
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    align = {} if mode == 'nearest' else dict(align_corners=False)
    return F.interpolate(t, sz, mode=mode, **align) if t.shape[-2:] != sz else t

def adaptive_cat(seq, dim=0, ref_tensor=0, mode='bilinear'):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz, mode=mode) for t in seq], dim=dim)
    return t

class ResLabelGeneratorLabelConv(nn.Module):
    def __init__(self, layer_dims, feature_dim=256, use_bn=False):
        super().__init__()
        self.label_conv = nn.Sequential(*[conv_block(in_planes=1, out_planes=16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, relu=True), conv_block(in_planes=16, out_planes=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, relu=True)])
        self.conv1 = conv_block(in_planes=feature_dim+32, out_planes=layer_dims[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=use_bn, relu=True)

        self.res_conv = BasicBlock(layer_dims[0], layer_dims[1], stride=1, use_bn=use_bn)
        self.label_pred = conv_block(layer_dims[1], layer_dims[2], kernel_size=3, stride=1, padding=1,
                                     relu=False, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=1)

    def forward(self, label_mask, feat):
        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        l = self.label_conv(label_mask)
        feat = feat.view(-1, *feat.shape[-3:])
        h = adaptive_cat((feat, l), dim=1, ref_tensor=0)
        h = self.conv1(h)
        h = self.res_conv(h)
        label_enc = self.label_pred(h)
        sample_w = self.samp_w_pred(h)
        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])
        return label_enc, sample_w

class ResidualDS16FeatSWv2(nn.Module):
    def __init__(self, layer_dims, feat_dim,  use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[-1], kernel_size=3, padding=1, stride=1)

        self.conv3 = conv_block(layer_dims[2] + feat_dim, layer_dims[3], kernel_size=3, stride=1, padding=1, batch_norm=use_bn)
        self.conv4 = conv_block(layer_dims[3], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                batch_norm=use_bn)

        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feat):
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))

        sample_w = self.samp_w_pred(mask_enc)

        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat((mask_enc, feat), dim=1)
        out = self.conv4(self.conv3(feat_mask_enc))

        label_enc = self.label_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w



class ResidualDS16SWMulti(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(3, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None, object_ids=None):
        # label_mask: frames, seq, num_obj, h, w
        # returns as [frames, seq, obj, dim, h, w]
        assert label_mask.dim() == 5

        num_frames = label_mask.shape[0]
        num_sequences = label_mask.shape[1]
        num_objects = label_mask.shape[2]

        target_mask = label_mask.unsqueeze(3)                                       # frames, seq, num_obj, 1, h, w

        bg_mask = 1.0 - label_mask.sum(dim=2, keepdim=True)                         # frames, seq, 1, h, w
        bg_mask = bg_mask.unsqueeze(3).repeat(1, 1, target_mask.shape[2], 1, 1, 1)  # frames, seq, num_obj, 1, h, w

        distractor_mask = label_mask.sum(dim=2, keepdim=True) - label_mask          # frames, seq, num_obj, h, w
        distractor_mask = distractor_mask.unsqueeze(3)                              # frames, seq, num_obj, 1, h, w

        label_enc_in = torch.cat((target_mask, bg_mask, distractor_mask), dim=3)    # frames, seq, num_obj, 3, h, w

        if object_ids is not None:
            label_enc_in = label_enc_in[:, :, object_ids, ...]
            num_objects = len(object_ids)

        label_enc_in = label_enc_in.view(num_frames*num_sequences*num_objects, 3, *label_mask.shape[-2:]).float()

        out = self.pool(self.conv_block(label_enc_in))
        out = self.res2(self.res1(out))


        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)


        label_enc = label_enc.view(num_frames, num_sequences, num_objects, -1, *label_enc.shape[-2:])
        sample_w = sample_w.view(num_frames, num_sequences, num_objects, -1, *sample_w.shape[-2:])

        # Out dim is (num_seq, num_frames, num_obj, layer_dims[-1], h, w)
        return label_enc, sample_w


class ResidualDS16SWSPP(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.spp = SPP(feature_dim=layer_dims[2], inter_dim=(layer_dims[2] // 4), out_dim=layer_dims[3], use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[3], layer_dims[4], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))
        out = self.spp(out)
        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w


class ResidualDS16FeatSWBox(nn.Module):
    def __init__(self, layer_dims, feat_dim,  use_final_relu=True, use_gauss=True, use_bn=False):
        super().__init__()

        self.use_gauss = use_gauss
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        ds3 = nn.Conv2d(layer_dims[2] + feat_dim, layer_dims[3], kernel_size=3, padding=1, stride=1)
        self.res3 = BasicBlock(layer_dims[2] + feat_dim, layer_dims[3], stride=1, downsample=ds3, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu)

        self.samp_w_pred = nn.Conv2d(layer_dims[3], layer_dims[4], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def bbox_to_mask(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0],1,*sz), dtype=torch.float32, device=bbox.device)
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            x1 = int(x1+0.5)
            y1 = int(y1+0.5)
            h = int(h+0.5)
            w = int(w+0.5)
            mask[i, :, y1:(y1+h), x1:(x1+w)] = 1.0
        return mask

    def bbox_to_gauss(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0],1,*sz), dtype=torch.float32, device=bbox.device)
        x_max, y_max = sz[-1], sz[-2]
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            cx, cy = x1+w/2, y1+h/2
            xcoords = torch.arange(0, x_max).unsqueeze(dim=0).to(bbox.device).float()
            ycoords = torch.arange(0, y_max).unsqueeze(dim=0).T.to(bbox.device).float()
            d_xcoords = xcoords - cx
            d_ycoords = ycoords - cy
            dtotsqr = d_xcoords**2/(0.25*w)**2 + d_ycoords**2/(0.25*h)**2
            mask[i,0] = torch.exp(-0.5*dtotsqr)
        return mask

    def forward(self, bb, feat, sz):
        assert bb.dim() == 3
        num_frames = bb.shape[0]
        batch_sz = bb.shape[1]
        bb = bb.view(-1, 4)
        if self.use_gauss:
            label_mask = self.bbox_to_gauss(bb, sz[-2:])
        else:
            label_mask = self.bbox_to_mask(bb, sz[-2:])

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))

        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat((mask_enc, feat), dim=1)
        out = self.res3(feat_mask_enc)

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(num_frames, batch_sz, *label_enc.shape[-3:])
        sample_w = sample_w.view(num_frames, batch_sz, *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w


class ResidualDS16FeatSWBoxCatMultiBlock(nn.Module):
    def __init__(self, layer_dims, feat_dim,  use_final_relu=True, use_gauss=True, use_bn=True,
                 non_default_init=True, init_bn=1, gauss_scale=0.25):
        super().__init__()
        in_layer_dim = (feat_dim+1,) + tuple(list(layer_dims)[:-2])
        out_layer_dim = tuple(list(layer_dims)[:-1])
        self.use_gauss = use_gauss
        res = []
        for in_d, out_d in zip(in_layer_dim, out_layer_dim):
            ds = nn.Conv2d(in_d, out_d, kernel_size=3, padding=1, stride=1)
            res.append(BasicBlock(in_d, out_d, stride=1, downsample=ds, use_bn=use_bn))

        self.res = nn.Sequential(*res)
        self.label_pred = conv_block(layer_dims[-2], layer_dims[-1], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu)
        self.gauss_scale = gauss_scale
        if non_default_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(init_bn)
                    m.bias.data.zero_()

    def bbox_to_mask(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0],1,*sz), dtype=torch.float32, device=bbox.device)
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            x1 = int(x1+0.5)
            y1 = int(y1+0.5)
            h = int(h+0.5)
            w = int(w+0.5)
            mask[i, :, y1:(y1+h), x1:(x1+w)] = 1.0
        return mask

    def bbox_to_gauss(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0],1,*sz), dtype=torch.float32, device=bbox.device)
        x_max, y_max = sz[-1], sz[-2]
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            cx, cy = x1+w/2, y1+h/2
            xcoords = torch.arange(0, x_max).unsqueeze(dim=0).to(bbox.device).float()
            ycoords = torch.arange(0, y_max).unsqueeze(dim=0).T.to(bbox.device).float()
            d_xcoords = xcoords - cx
            d_ycoords = ycoords - cy
            dtotsqr = d_xcoords**2/(self.gauss_scale*w)**2 + d_ycoords**2/(self.gauss_scale*h)**2
            mask[i,0] = torch.exp(-0.5*dtotsqr)
        return mask

    def forward(self, bb, feat, sz):
        #assert bb.dim() == 3

        if self.use_gauss:
            label_mask = self.bbox_to_gauss(bb, sz[-2:])
        else:
            label_mask = self.bbox_to_mask(bb, sz[-2:])

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat([feat, interpolate(label_mask, feat.shape[-2:])], dim=1)
        out = self.res(feat_mask_enc)

        label_enc = self.label_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])

        return label_enc
