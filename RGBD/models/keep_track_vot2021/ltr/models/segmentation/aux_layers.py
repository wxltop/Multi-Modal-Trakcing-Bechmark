import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ltr.models.layers.blocks import conv_block


class ConvPredictor(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.predictor = nn.Conv2d(num_channels, 1, 3, padding=1, bias=True, stride=1)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ft, out_size):
        assert ft.dim() == 4

        ft = ft.view(-1, *ft.shape[-3:])
        x = self.predictor(ft)
        x = F.interpolate(x, out_size[-2:], mode='bicubic', align_corners=False)
        return x.view(-1, 1, *x.shape[-2:])


class DeConvPredictor4x(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.predictor = nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels // 2, 3,
                                                          padding=1, bias=True, stride=1),
                                       nn.ReLU(),
                                       nn.ConvTranspose2d(num_channels // 2, 1, 3,
                                                          padding=1, bias=True, stride=1)
                                       )

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ft, out_size):
        assert ft.dim() == 4

        ft = ft.view(-1, *ft.shape[-3:])
        x = self.predictor(ft)
        x = F.interpolate(x, out_size[-2:], mode='bicubic', align_corners=False)
        return x.view(-1, 1, *x.shape[-2:])


class BBRegressor(nn.Module):
    def __init__(self, num_channels, pool_sz, conv_channels, fc_channels=512, scale_factor=100.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.pool_sz = pool_sz
        self.conv_block = conv_block(num_channels, conv_channels, kernel_size=3, padding=1)

        self.pool_rows = nn.AdaptiveMaxPool2d((pool_sz[0], 1))
        self.pool_cols = nn.AdaptiveMaxPool2d((1, pool_sz[1]))

        self.predictor_r = nn.Sequential(conv_block(conv_channels, fc_channels, kernel_size=(pool_sz[0], 1), padding=0),
                                         conv_block(fc_channels, 2, kernel_size=1, padding=0, batch_norm=False, relu=False))
        self.predictor_c = nn.Sequential(conv_block(conv_channels, fc_channels, kernel_size=(1, pool_sz[1]), padding=0),
                                         conv_block(fc_channels, 2, kernel_size=1, padding=0, batch_norm=False, relu=False))

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        assert x.dim() == 4
        x = self.conv_block(x)

        pred_r1r2 = self.predictor_r(self.pool_rows(x)).view(-1, 2) / self.scale_factor
        pred_c1c2 = self.predictor_c(self.pool_cols(x)).view(-1, 2) / self.scale_factor

        pred_box = torch.cat((pred_r1r2, pred_c1c2), dim=1)
        return pred_box


class BBRegressorFC(nn.Module):
    def __init__(self, num_channels, scale_factor=100.0):
        super().__init__()
        pool_sz = (32, 52)

        self.scale_factor = scale_factor
        self.pool_sz = pool_sz
        self.conv_block1 = conv_block(num_channels, 32, stride=2, kernel_size=3, padding=1, batch_norm=False)
        self.conv_block2 = conv_block(32, 32, stride=2, kernel_size=3, padding=1, batch_norm=False)

        self.pool_layer = nn.AdaptiveMaxPool2d((pool_sz[0], pool_sz[1]))

        input_sz = (pool_sz[0] // 4, pool_sz[1] // 4)
        self.fc1 = conv_block(32, 512, kernel_size=input_sz, padding=0, batch_norm=False)
        self.predictor = conv_block(512, 4, kernel_size=1, padding=0, batch_norm=False, relu=False)

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        assert x.dim() == 4
        pooled_feat = self.pool_layer(x)

        feat = self.conv_block2(self.conv_block1(pooled_feat))
        pred_box = self.predictor(self.fc1(feat))

        return pred_box