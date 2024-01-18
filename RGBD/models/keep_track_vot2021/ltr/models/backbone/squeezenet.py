import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from torchvision.models.squeezenet import Fire, model_urls

def remap_weight_keys(model_dict, key_map):
    new_model_dict = {}
    for key, value in model_dict.items():
        new_key = None
        for old_prefix, new_prefix in key_map.items():
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix)
                break

        if new_key is not None:
            new_model_dict[new_key] = value
        else:
            raise ValueError

    return new_model_dict

class SqueezeNet(nn.Module):

    def __init__(self, output_layers, version=1.1, num_classes=1000):
        self.output_layers = output_layers
        super(SqueezeNet, self).__init__()
        if version not in [1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, padding=1)

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.layer1 = nn.Sequential(
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
            )

            self.layer2 = nn.Sequential(
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
            )

            self.layer3 = nn.Sequential(
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)
        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)
        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.classifier(x)

        return x.view(x.size(0), self.num_classes)

def squeezenet1_1(output_layers=None, pretrained=False):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    key_map = {'features.0.': 'conv1.', 'features.3.': 'layer1.0.', 'features.4.': 'layer1.1.',
               'features.6.': 'layer2.0.', 'features.7.': 'layer2.1.',
               'features.9.': 'layer3.0.', 'features.10.': 'layer3.1.',
               'features.11.': 'layer3.2.', 'features.12.': 'layer3.3.',
               'classifier.1.': 'classifier.1.'}

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = SqueezeNet(output_layers=output_layers, version=1.1)

    if pretrained:
        model.load_state_dict(remap_weight_keys(model_zoo.load_url(model_urls['squeezenet1_1']), key_map))


    return model

def resnet18(output_layers=None, pretrained=False):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_d16(output_layers=None, pretrained=False):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, inplanes=16)

    if pretrained:
        raise NotImplementedError
    return model

def resnet50(output_layers=None, pretrained=False):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model