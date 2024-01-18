import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d


def prpool_features(feat, box, pool_size, stride):
    box = box.view(1, 4)

    batch_index = torch.tensor([0.0]).view(1, 1).to(box.device)
    box_xyxy = torch.cat((box[:, 0:2], box[:, 0:2] + box[:, 2:4]), dim=1)

    # Add batch index
    roi = torch.cat((batch_index, box_xyxy), dim=1)
    roi = roi.view(-1, 5).to(feat.device)

    roi_features = prroi_pool2d(feat, roi, pool_size[0], pool_size[1], 1.0 / stride)

    return roi_features


def shift_features(feat, relative_translation_vector):
    T_mat = torch.eye(2).repeat(feat.shape[0], 1, 1).to(feat.device)
    T_mat = torch.cat((T_mat, relative_translation_vector.view(-1, 2, 1)), dim=2)

    grid = F.affine_grid(T_mat, feat.shape)

    feat_out = F.grid_sample(feat, grid)
    return feat_out


class CenterShiftFeatures(nn.Module):
    def __init__(self, feature_stride):
        super().__init__()
        self.feature_stride = feature_stride

    def forward(self, feat, anno):
        anno = anno.view(-1, 4)
        c_x = (anno[:, 0] + anno[:, 2] * 0.5) / self.feature_stride
        c_y = (anno[:, 1] + anno[:, 3] * 0.5) / self.feature_stride

        t_x = 2 * (c_x - feat.shape[-1] * 0.5) / feat.shape[-1]
        t_y = 2 * (c_y - feat.shape[-2] * 0.5) / feat.shape[-2]

        t = torch.cat((t_x.view(-1, 1), t_y.view(-1, 1)), dim=1)

        feat_out = shift_features(feat, t)
        return feat_out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat):
        return feat


class Nothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat):
        return torch.zeros(feat.shape[0], 0, feat.shape[2], feat.shape[3]).to(feat.device)
