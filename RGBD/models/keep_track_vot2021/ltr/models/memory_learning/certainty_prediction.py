import torch
import torch.nn as nn
import torch.nn.functional as F

from ltr.models.layers.distance import DistanceValueEncoder
from ltr import model_constructor

@model_constructor
def learned_binary_overlap_predictor():
    net = BinOverlapPredictionFromProjModule()
    return net



class BinOverlapPredictionFromProjModule(nn.Module):
    def __init__(self, num_gth_mem=15):
        super().__init__()
        self.num_gth_mem = num_gth_mem

        self.kernel_size = 3
        # self.mean_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)
        # self.std_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)
        #
        self.layers = [
            nn.Conv2d(in_channels=14, out_channels=64, kernel_size=self.kernel_size), #(n,c,21,21)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (n,c,10,10)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size), #(n,c,8,8)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size), #(n,c,6,6)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=self.kernel_size), #(n,c,4,4)
            nn.AdaptiveMaxPool2d((1,1))
        ]
        self.model = nn.Sequential(*self.layers)

    def predict_certainties(self, target_scores, ptm, pmt, predicted_iou, ngthmem=15, **kwargs):
        # scores_raw torch.Size([1, 1, 23, 23])
        # pmt torch.Size([15, 1, 23, 23])
        # ptm torch.Size([15, 1, 23, 23])
        # predicted_iou tensor([3.6767])
        N = 1
        nmem, c, h, w = ptm.shape

        target_scores = target_scores.reshape(N, 1, h, w)
        ptm = ptm.reshape(N, nmem, 1, h, w)
        pmt = pmt.reshape(N, nmem, 1, h, w)
        predicted_iou = predicted_iou.reshape(N, 1, 1, 1)

        max_target_scores = torch.max(target_scores.reshape(N, h * w), dim=1)[0]

        pmt_mean = torch.mean(pmt, dim=1).reshape(N, 1, h, w)
        pmt_std = torch.std(pmt, dim=1).reshape(N, 1, h, w)

        pmt_gth_mean = torch.mean(pmt[:, :ngthmem], dim=1).reshape(N, 1, h, w)
        pmt_gth_std = torch.std(pmt[:, :ngthmem], dim=1, unbiased=False).reshape(N, 1, h, w)

        ptm_max = torch.max(ptm.reshape(N, nmem, w*h), dim=2)[0]

        ptm_mean_max = torch.mean(ptm_max, dim=1)
        ptm_std_max = torch.std(ptm_max, dim=1, unbiased=False)

        ptm_gth_mean_max = torch.mean(ptm_max[:, :ngthmem], dim=1)
        ptm_gth_std_max = torch.std(ptm_max[:, :ngthmem], dim=1, unbiased=False)


        if ptm_max.shape[1] > ngthmem:
            ptm_other_mean_max = torch.mean(ptm_max[:, ngthmem:], dim=1)
            ptm_other_std_max = torch.std(ptm_max[:, ngthmem:], dim=1, unbiased=False)

            pmt_other_mean = torch.mean(pmt[:, ngthmem:], dim=1).reshape(N, 1, h, w)
            pmt_other_std = torch.std(pmt[:, ngthmem:], dim=1, unbiased=False).reshape(N, 1, h, w)
        else:
            ptm_other_mean_max = torch.ones_like(ptm_gth_mean_max)
            ptm_other_std_max = torch.zeros_like(ptm_gth_std_max)

            pmt_other_mean = torch.ones_like(pmt_gth_mean)
            pmt_other_std = torch.zeros_like(pmt_gth_std)

        max_target_scores = max_target_scores.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        predicted_iou = predicted_iou.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        ptm_mean_max = ptm_mean_max.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        ptm_std_max = ptm_std_max.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        ptm_gth_mean_max = ptm_gth_mean_max.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        ptm_gth_std_max = ptm_gth_std_max.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        ptm_other_mean_max = ptm_other_mean_max.reshape(N, 1, 1, 1).repeat(1, 1, h, w)
        ptm_other_std_max = ptm_other_std_max.reshape(N, 1, 1, 1).repeat(1, 1, h, w)

        input = torch.cat([
            max_target_scores, predicted_iou,
            pmt_mean, pmt_std, pmt_gth_mean, pmt_gth_std, pmt_other_mean, pmt_other_std,
            ptm_mean_max, ptm_std_max, ptm_gth_mean_max, ptm_gth_std_max, ptm_other_mean_max, ptm_other_std_max
        ], dim=1)

        out = self.model(input)

        return torch.sigmoid(out)



    def forward(self, target_scores, ptm, pmt, predicted_iou, mem_mask, **kwargs):
        ngthmem = self.num_gth_mem
        nframes, nseq, nmem, c, h, w = ptm.shape

        target_scores = target_scores.reshape(nframes*nseq, 1, h, w)
        pmt = pmt.reshape(nframes*nseq, nmem, 1, h, w)
        ptm = ptm.reshape(nframes*nseq, nmem, c, h, w) # nmem in [15, 50]
        mem_mask = mem_mask.reshape(nframes*nseq, nmem)
        predicted_iou = predicted_iou.reshape(nframes*nseq, 1, 1, 1)

        # max target score feature
        max_target_scores = torch.max(target_scores.reshape(nframes*nseq, h*w), dim=1)[0]

        _iter = range(nframes * nseq)
        # PMT, feature: on all, gth, and dynamic memory cells
        pmt_mean = torch.stack([torch.mean(pmt[i, mem_mask[i]], dim=0) for i in _iter], dim=0)
        pmt_std = torch.stack([torch.std(pmt[i, mem_mask[i]], dim=0, unbiased=False) for i in _iter], dim=0)

        pmt_gth_mean = torch.mean(pmt[:, :ngthmem], dim=1).reshape(nframes*nseq, 1, h, w)
        pmt_gth_std = torch.std(pmt[:, :ngthmem], dim=1, unbiased=False).reshape(nframes*nseq, 1, h, w)

        pmt_other_mean = torch.stack([torch.mean(pmt[i, mem_mask[i]][ngthmem:], dim=0) for i in _iter], dim=0)
        pmt_other_std = torch.stack([torch.std(pmt[i, mem_mask[i]][ngthmem:], dim=0, unbiased=False) for i in _iter], dim=0)

        # PTM: features on all, gth, and dynamic memory cells
        ptm_max = torch.max(ptm.reshape(nframes * nseq, nmem, c * w * h), dim=2)[0]

        ptm_mean_max = torch.stack([torch.mean(ptm_max[i, mem_mask[i]], dim=0) for i in _iter], dim=0)
        ptm_std_max = torch.stack([torch.std(ptm_max[i, mem_mask[i]], dim=0, unbiased=False) for i in _iter], dim=0)

        ptm_gth_mean_max = torch.stack([torch.mean(ptm_max[i, mem_mask[i]][:ngthmem], dim=0) for i in _iter], dim=0)
        ptm_gth_std_max = torch.stack([torch.std(ptm_max[i, mem_mask[i]][:ngthmem], dim=0, unbiased=False) for i in _iter], dim=0)

        ptm_other_mean_max = torch.stack([torch.mean(ptm_max[i, mem_mask[i]][ngthmem:], dim=0) for i in _iter], dim=0)
        ptm_other_std_max = torch.stack([torch.std(ptm_max[i, mem_mask[i]][ngthmem:], dim=0, unbiased=False) for i in _iter], dim=0)

        # repeat values for tensor
        max_target_scores = max_target_scores.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        predicted_iou = predicted_iou.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        ptm_mean_max = ptm_mean_max.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        ptm_std_max = ptm_std_max.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        ptm_gth_mean_max = ptm_gth_mean_max.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        ptm_gth_std_max = ptm_gth_std_max.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        ptm_other_mean_max = ptm_other_mean_max.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)
        ptm_other_std_max = ptm_other_std_max.reshape(nframes*nseq, 1, 1, 1).repeat(1, 1, h, w)


        input = torch.cat([
            max_target_scores, predicted_iou,
            pmt_mean, pmt_std, pmt_gth_mean, pmt_gth_std, pmt_other_mean, pmt_other_std,
            ptm_mean_max, ptm_std_max, ptm_gth_mean_max, ptm_gth_std_max, ptm_other_mean_max, ptm_other_std_max
        ], dim=1)

        out = self.model(input)
        return out


class BinOverlapPredictionFromMaxProjManualModule(nn.Module):
    def __init__(self, gth_only=False, num_gth_mem=15):
        super().__init__()
        self.gth_only = gth_only
        self.num_gth_mem = num_gth_mem

    def forward(self, ptm, mem_mask, **kwargs):
        nframes, nseq, nmem, c, w, h = ptm.shape
        ptm = ptm.reshape(nframes * nseq, nmem, c, w, h)
        mem_mask = mem_mask.reshape(nframes * nseq, nmem)

        ptm_max = torch.max(ptm.reshape(nframes*nseq, nmem, c*w*h), dim=2)[0]
        if self.gth_only:
            ptm_max_avg = torch.stack([torch.mean(ptm_max[i, mem_mask[i]][:self.num_gth_mem], dim=0)
                                       for i in range(0, nframes*nseq)], dim=0)
        else:
            ptm_max_avg = torch.stack([torch.mean(ptm_max[i, mem_mask[i]], dim=0)
                                       for i in range(0, nframes*nseq)], dim=0)

        return ptm_max_avg


class BinOverlapPredictionFromMaxTargetScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_scores, **kwargs):
        nframes, nseq, nmem, c, w, h = target_scores.shape
        target_scores = target_scores.reshape(nframes*nseq, w*h)

        max_target_scores = torch.max(target_scores, dim=1)[0]

        return max_target_scores