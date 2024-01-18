import math
import torch.nn as nn
import os
import torch
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
import ltr.admin.settings as ws_settings
from .base import Backbone
from ltr.external.dcn.deform_conv import DeformConv, ModulatedDeformConv


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_norm(norm_type, num_channels):
    if norm_type is None:
        bn = Identity()
    elif norm_type == 'bn':
        bn = nn.BatchNorm2d(num_channels)
    elif norm_type == 'gn':
        bn = nn.GroupNorm(32, num_channels)
    else:
        raise Exception

    return bn


class Bottleneck(nn.Module):
    def __init__(self, inplanes, bottleneck_planes, out_planes, stride=1, downsample=None, dilation=1,
                 num_groups=1, stride_in_1x1=False, norm='bn'):
        super(Bottleneck, self).__init__()

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride_1x1, bias=False)
        self.bn1 = get_norm(norm, bottleneck_planes)
        self.conv2 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=stride_3x3,
                               padding=dilation, bias=False, dilation=dilation,
                               groups=num_groups)
        self.bn2 = get_norm(norm, bottleneck_planes)
        self.conv3 = nn.Conv2d(bottleneck_planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeformableBottleneck(nn.Module):
    def __init__(self, inplanes, bottleneck_planes, out_planes, stride=1, downsample=None, dilation=1, num_groups=1,
                 deform_modulated=False, deform_num_groups=1, stride_in_1x1=False, norm='bn'):
        super(DeformableBottleneck, self).__init__()
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride_1x1, bias=False)
        self.bn1 = get_norm(norm, bottleneck_planes)

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = nn.Conv2d(bottleneck_planes, offset_channels * deform_num_groups, kernel_size=3, stride=stride_3x3,
                                      padding=1 * dilation, dilation=dilation)

        self.conv2 = deform_conv_op(bottleneck_planes, bottleneck_planes, kernel_size=3, stride=stride_3x3, padding=1 * dilation,
                                    bias=False, groups=num_groups, dilation=dilation,
                                    deformable_groups=deform_num_groups,
                                    )

        self.bn2 = get_norm(norm, bottleneck_planes)
        self.conv3 = nn.Conv2d(bottleneck_planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.deform_modulated = deform_modulated
        # self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNext(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, stage_params, output_layers, stem_inplanes=64, dilation_factor=1, frozen_layers=(),
                 norm='bn'):
        super(ResNext, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, stem_inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(norm, stem_inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Change! stride condition
        stride = [1, 2, 2]
        # stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(stage_params[0]['block'], stage_params[0]['in_planes'],
                                       stage_params[0]['bottleneck_planes'], stage_params[0]['out_planes'],
                                       stage_params[0]['num_layers'],
                                       groups=stage_params[0]['num_groups'], stride=stride[0],
                                       dilation=max(dilation_factor//8, 1),
                                       stride_in_1x1=stage_params[0].get('stride_in_1x1', False),
                                       norm=norm)
        self.layer2 = self._make_layer(stage_params[1]['block'], stage_params[1]['in_planes'],
                                       stage_params[1]['bottleneck_planes'], stage_params[1]['out_planes'],
                                       stage_params[1]['num_layers'],
                                       groups=stage_params[1]['num_groups'],
                                       stride=stride[1], dilation=max(dilation_factor//4, 1),
                                       stride_in_1x1=stage_params[1].get('stride_in_1x1', False),
                                       norm=norm)
        self.layer3 = self._make_layer(stage_params[2]['block'], stage_params[2]['in_planes'],
                                       stage_params[2]['bottleneck_planes'], stage_params[2]['out_planes'],
                                       stage_params[2]['num_layers'],
                                       groups=stage_params[2]['num_groups'], stride=stride[2],
                                       dilation=max(dilation_factor//2, 1),
                                       stride_in_1x1=stage_params[2].get('stride_in_1x1', False),
                                       norm=norm)
        self.layer4 = self._make_layer(stage_params[3]['block'], stage_params[3]['in_planes'],
                                       stage_params[3]['bottleneck_planes'], stage_params[3]['out_planes'],
                                       stage_params[3]['num_layers'],
                                       groups=stage_params[3]['num_groups'], stride=stride[2],
                                       dilation=dilation_factor,
                                       stride_in_1x1=stage_params[3].get('stride_in_1x1', False),
                                       norm=norm)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 8, 'layer3': 16,
                               'layer4': 32}

        if isinstance(self.layer1[0], Bottleneck):
            out_feature_channels = {'conv1': stem_inplanes, 'layer1': stage_params[0]['out_planes'],
                                    'layer2': stage_params[1]['out_planes'],
                                    'layer3': stage_params[2]['out_planes'],
                                    'layer4': stage_params[3]['out_planes']}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, in_planes, bottleneck_planes, out_planes, blocks, stride=1, groups=1, dilation=1,
                    stride_in_1x1=False, norm='bn'):
        downsample = None
        # Change! No condition for stride 1
        if in_planes != out_planes:
            down_stride = stride if dilation == 1 else 1
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=down_stride, bias=False),
                get_norm(norm, out_planes),
            )

        # Change! reset stride
        if dilation > 1:
            stride = 1

        # Change! Stride
        layers = []
        layers.append(block(in_planes, bottleneck_planes, out_planes, stride, downsample, dilation=dilation,
                            num_groups=groups, stride_in_1x1=stride_in_1x1, norm=norm))

        for i in range(1, blocks):
            layers.append(block(out_planes, bottleneck_planes, out_planes, num_groups=groups, norm=norm))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # if self._add_output_and_check('fc', x, outputs, output_layers):
        #     return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnext_152_32x8d_dconv(output_layers=None, pretrained=False, frozen_layers=(), **kwargs):
    # Backbone for cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                raise ValueError('Unknown layer: {}'.format(l))

    stage_params = [{'block': Bottleneck, 'num_layers': 3, 'num_groups': 32, 'in_planes': 64,
                     'bottleneck_planes': 256, 'out_planes': 256},
                    {'block': DeformableBottleneck, 'num_layers': 8, 'num_groups': 32, 'in_planes': 256,
                     'bottleneck_planes': 512, 'out_planes': 512},
                    {'block': DeformableBottleneck, 'num_layers': 36, 'num_groups': 32, 'in_planes': 512,
                     'bottleneck_planes': 1024, 'out_planes': 1024},
                    {'block': DeformableBottleneck, 'num_layers': 3, 'num_groups': 32, 'in_planes': 1024,
                     'bottleneck_planes': 2048, 'out_planes': 2048}]
    model = ResNext(stage_params, output_layers, frozen_layers=frozen_layers, **kwargs)

    if pretrained:
        print('Pre-trained weights not available. Load it manually')

    return model


def resnet50(output_layers=None, pretrained=False, frozen_layers=(), **kwargs):
    # Backbone for cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                raise ValueError('Unknown layer: {}'.format(l))

    stage_params = [{'block': Bottleneck, 'num_layers': 3, 'num_groups': 1, 'in_planes': 64,
                     'bottleneck_planes': 64, 'out_planes': 256, 'stride_in_1x1': True},
                    {'block': Bottleneck, 'num_layers': 4, 'num_groups': 1, 'in_planes': 256,
                     'bottleneck_planes': 128, 'out_planes': 512, 'stride_in_1x1': True},
                    {'block': Bottleneck, 'num_layers': 6, 'num_groups': 1, 'in_planes': 512,
                     'bottleneck_planes': 256, 'out_planes': 1024, 'stride_in_1x1': True},
                    {'block': Bottleneck, 'num_layers': 3, 'num_groups': 1, 'in_planes': 1024,
                     'bottleneck_planes': 512, 'out_planes': 2048, 'stride_in_1x1': True}]
    model = ResNext(stage_params, output_layers, frozen_layers=frozen_layers, **kwargs)

    if pretrained:
        print('Pre-trained weights not available. Load it manually')

    return model


def resnet50_gn(output_layers=None, pretrained=False, frozen_layers=(), **kwargs):
    # Backbone for cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                raise ValueError('Unknown layer: {}'.format(l))

    stage_params = [{'block': Bottleneck, 'num_layers': 3, 'num_groups': 1, 'in_planes': 64,
                     'bottleneck_planes': 64, 'out_planes': 256, 'stride_in_1x1': False},
                    {'block': Bottleneck, 'num_layers': 4, 'num_groups': 1, 'in_planes': 256,
                     'bottleneck_planes': 128, 'out_planes': 512, 'stride_in_1x1': False},
                    {'block': Bottleneck, 'num_layers': 6, 'num_groups': 1, 'in_planes': 512,
                     'bottleneck_planes': 256, 'out_planes': 1024, 'stride_in_1x1': False},
                    {'block': Bottleneck, 'num_layers': 3, 'num_groups': 1, 'in_planes': 1024,
                     'bottleneck_planes': 512, 'out_planes': 2048, 'stride_in_1x1': False}]
    model = ResNext(stage_params, output_layers, frozen_layers=frozen_layers, norm='gn', **kwargs)

    if pretrained:
        print('Pre-trained weights not available. Load it manually')

    return model
