import torch
from torch import nn
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.bbreg.bbox_to_roi import bbox_to_roi


class SemanticClassifier(nn.Module):
    def __init__(self, feat_dim, feat_stride, pool_sz, class_dict):
        super().__init__()
        self.pooler = PrRoIPool2D(pool_sz, pool_sz, 1.0 / feat_stride)

        self.classifiers = nn.ModuleDict({k: nn.Conv2d(feat_dim, len(v), kernel_size=pool_sz)
                                          for k, v in class_dict.items()})

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat, bb):
        assert bb.dim() == 3

        num_sequences = bb.shape[1]
        num_frames = bb.shape[0]
        feat = feat.view(-1, *feat.shape[-3:])
        bb = bb.view(-1, 4)

        roi = bbox_to_roi(bb)

        feat_pooled = self.pooler(feat, roi)

        class_pred = {}

        for k, classifier in self.classifiers.items():
            class_pred[k] = classifier(feat_pooled).view(num_frames, num_sequences, -1)

        return class_pred
