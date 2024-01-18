import torch.nn as nn
from ltr.models.layers.blocks import conv_block
import torch.nn.functional as F


class Predictorv1(nn.Module):
    def __init__(self, input_dim, inter_dim, window_sz, use_bn=False):
        super().__init__()
        self.conv1 = conv_block(input_dim, inter_dim, kernel_size=3, stride=1, padding=1, batch_norm=use_bn)
        self.mask_predictor = conv_block(inter_dim, 1, kernel_size=3, stride=1,
                                         padding=1, batch_norm=False, relu=False)

        self.conv2 = conv_block(inter_dim, 2*inter_dim, kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.conv3 = conv_block(2 * inter_dim, 2 * inter_dim, kernel_size=3, stride=1, padding=1, batch_norm=use_bn)

        self.score_predictor = conv_block(2 * inter_dim, 1, kernel_size=3, stride=1, padding=1, batch_norm=False,
                                          relu=False)
        self.conv4 = conv_block(2 * inter_dim, 4 * inter_dim, kernel_size=3, stride=2, padding=1, batch_norm=use_bn)

        self.box_predictor = conv_block(4 * inter_dim * window_sz**2, 4, kernel_size=1, stride=1, padding=0, batch_norm=False,
                                        relu=False)

        self.window_sz = window_sz

    def forward(self, mask_enc):
        shape = mask_enc.shape
        mask_enc = mask_enc.view(-1, *mask_enc.shape[-3:])
        feat = self.conv1(mask_enc)
        mask_pred = self.mask_predictor(feat)
        feat = self.conv3(self.conv2(feat))

        center_scores = self.score_predictor(feat)

        feat = self.conv4(feat)
        feat_unfold = F.unfold(feat, kernel_size=self.window_sz, padding=self.window_sz//2)

        box_pred = self.box_predictor(feat_unfold.view(feat.shape[0], -1, *feat.shape[-2:]))

        output = {'bbox': box_pred, 'center_score': center_scores,
                  'mask': mask_pred}
        return output
