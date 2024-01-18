import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BBRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, predictions, init_box, gt_box):
        assert predictions.dim() == init_box.dim() == gt_box.dim()

        if predictions.dim() == 2:
            predictions = predictions.view(-1, 1, 4)
            init_box = init_box.view(-1, 1, 4)
            gt_box = gt_box.view(-1, 1, 4)

        init_box_center = init_box[:, :, :2] + 0.5*init_box[:, :, 2:]
        gt_box_center = gt_box[:, :, :2] + 0.5 * gt_box[:, :, 2:]

        # Targets
        t_xy = (gt_box_center - init_box_center) / init_box[:, :, 2:]
        t_wh = (gt_box[:, :, 2:] / init_box[:, :, 2:]).log()

        target = torch.cat((t_xy, t_wh), dim=2)

        L = self.loss_fn(predictions, target)

        return L
