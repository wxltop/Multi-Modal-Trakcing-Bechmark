import math

import torch
from torch import nn
import torch.nn.functional as F

from ltr.models.layers.distance import DistanceValueEncoder, DistanceMap


def compute_cosine_similarity(train_feat, test_feat):
    tr_nframes, nseq, c, w, h = train_feat.size()
    te_nframes, _, _, _, _ = test_feat.size()

    train_reshuffle = train_feat.permute(1, 2, 0, 3, 4)  # (nframes, nseq, C, W, H) -> (nseq, C, nframes, W, H)
    test_reshuffle = test_feat.permute(1, 2, 0, 3, 4)  # (nframes, nseq, C, W, H) -> (nseq, C, nframes, W, H)
    train_reshuffle = train_reshuffle.reshape(nseq, c, tr_nframes * w * h)  # merge dimensions into one patch dimension
    test_reshuffle = test_reshuffle.reshape(nseq, c, te_nframes * w * h)  # merge dimensions into one patch dimension

    train_norm = torch.sqrt(
        torch.einsum('bij,bij->bj', train_reshuffle, train_reshuffle)
    ).view(nseq, 1, tr_nframes * w * h)

    test_norm = torch.sqrt(
        torch.einsum('bij,bij->bj', test_reshuffle, test_reshuffle)
    ).view(nseq, 1, te_nframes * w * h)

    train_normalized = train_reshuffle / train_norm  # (nseq, C, tr_nframes*W*H)
    test_normalized = test_reshuffle / test_norm  # (nseq, C, te_nframes*W*H)

    return torch.einsum('bij,bik->bjk', test_normalized, train_normalized)  # (nseq, te_nframes*w*h, tr_nframes*w*h)


def compute_proj_test_score_on_memory_frames_feature(sim, test_label, train_label, feat_size=None, out_size=None):
    # sim  (nseq, te_nframes*W*H, tr_nframes*W*H)
    # test_label_reshuffle (nseq, te_nframes*W*H, 1)
    if feat_size is None:
        feat_size = (22, 22)

    if out_size is None:
        out_size = (23, 23)

    p = torch.softmax(50. * sim, dim=1)  # (nseq, te_nframes*w*h, tr_nframes*w*h)

    test_label = F.interpolate(test_label, size=feat_size, mode='bilinear') # (22,22)

    tr_nframes, _, _, _ = train_label.size()
    te_nframes, nseq, w, h = test_label.size()
    test_label_reshuffle = test_label.permute(1, 0, 2, 3) # (nframes, nseq, W, H) -> (nseq, nframes, W, H)
    # (nseq, nframes*W*H, 1) merge dimensions into one patch dimension
    test_label_reshuffle = test_label_reshuffle.reshape(nseq, te_nframes * w * h, 1)

    proj_test_label_on_mem = torch.einsum('bij,bik->bkj', test_label_reshuffle, p)  # (nseq, nframes*w*h, 1)
    proj_test_label_on_mem_reshuffle = proj_test_label_on_mem.reshape(nseq, tr_nframes, w, h, 1)
    proj_test_label_on_mem_reshuffle = proj_test_label_on_mem_reshuffle.permute(1, 0, 4, 2, 3)
    proj_test_label_on_mem_reshuffle = proj_test_label_on_mem_reshuffle.reshape(tr_nframes * nseq, 1, w, h)

    proj_test_label_on_mem_reshuffle = F.interpolate(proj_test_label_on_mem_reshuffle,
                                                     size=out_size, mode='bilinear') # (23,23)

    return proj_test_label_on_mem_reshuffle


class MemoryLabelCertaintyFeature(nn.Module):
    def __init__(self, num_bins=16, out_size=None, **kwargs):
        super().__init__()
        self.num_bins = num_bins
        self.out_size = out_size if out_size is not None else (23, 23)
        self.d_value_encoder = DistanceValueEncoder(self.num_bins, min_val=0., max_val=1.)
        self.tanh = nn.Tanh()

    @property
    def feat_dim(self):
        return self.num_bins
    def forward(self, train_certainty, **kwargs):
        # train_certainty (nseq, nframes, 1)
        feats = self.compute_feature(train_certainty)
        return self.encode(feats)

    def encode(self, inp):
        return self.d_value_encoder(0.5*(self.tanh(inp) + 1))

    def compute_feature(self, train_certainty, **kwargs):
        train_certainty = train_certainty.repeat(1, 1, self.out_size[0], self.out_size[1])
        return train_certainty.reshape(-1, 1, self.out_size[0], self.out_size[1])


class RawTrainFeaturesFeature(nn.Module):
    def __init__(self, out_dim=16, input_feat_dim=512, kernel_size=1, use_batch_norm=True, out_size=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.input_feat_dim = input_feat_dim
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.out_size = out_size if out_size is not None else (23, 23)
        self.conv_layer = nn.Conv2d(self.input_feat_dim, self.out_dim,kernel_size=self.kernel_size,
                                    padding=self.kernel_size // 2)
        if self.use_batch_norm:
            self.bn_layer = nn.BatchNorm2d(self.out_dim)

    @property
    def feat_dim(self):
        return self.out_dim

    def forward(self, train_feat, **kwargs):
        feats = self.compute_feature(train_feat)
        return self.encode(feats)

    def encode(self, inp):
        h = self.conv_layer(inp)
        bn = self.bn_layer(h) if self.use_batch_norm else h
        return F.relu(bn)

    def compute_feature(self, train_feat, **kwargs):
        nframes, nseq, c, w, h = train_feat.size()
        train_feat_reshuffle = train_feat.reshape(nframes * nseq, c, w, h)
        train_feat_reshuffle = F.interpolate(train_feat_reshuffle, size=self.out_size, mode='bilinear')

        return train_feat_reshuffle


class BBoxDistanceMapFeature(nn.Module):
    def __init__(self, feat_stride, out_size=None, num_bins=16, **kwargs):
        super().__init__()
        self.out_size = out_size if out_size is not None else (23, 23)
        self.feat_stride = feat_stride
        self.num_bins = num_bins
        self.dist_map = DistanceMap(self.num_bins, math.sqrt(2) * self.out_size[0] / (self.num_bins - 1))

    @property
    def feat_dim(self):
        return self.num_bins

    def forward(self, train_bb, **kwargs):
        return self.compute_feature(train_bb)

    def encode(self, inp):
        return inp

    def compute_feature(self, train_bb, **kwargs):
        center = ((train_bb[..., :2] + train_bb[..., 2:] / 2) / self.feat_stride).flip((-1,))
        return self.dist_map(center, self.out_size)


class ProjTestScoreOnMemoryFramesFeature(nn.Module):
    def __init__(self, feat_size=None, out_size=None, softmax_temp_init=50, num_bins=16, requires_grad=True, **kwargs):
        super().__init__()
        self.out_size = out_size if out_size is not None else (23, 23)
        self.feat_size = feat_size if feat_size is not None else (22, 22)
        self.num_bins = num_bins
        self.softmax_temp_init = softmax_temp_init
        self.a = nn.Parameter(softmax_temp_init * torch.ones(1), requires_grad=requires_grad)
        self.d_value_encoder = DistanceValueEncoder(self.num_bins)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    @property
    def feat_dim(self):
        return self.num_bins

    def forward(self, sim, test_label, train_label, **kwargs):
        feats = self.compute_feature(sim, test_label, train_label)
        return self.encode(feats)

    def encode(self, inp):
        return self.d_value_encoder(self.tanh(inp))

    def compute_feature(self, sim, test_label, train_label, **kwargs):
        # sim  (nseq, te_nframes*W*H, tr_nframes*W*H)
        # test_label_reshuffle (nseq, te_nframes*W*H, 1)
        p = self.softmax(self.a * sim)  # (nseq, te_nframes*w*h, tr_nframes*w*h)

        test_label = F.interpolate(test_label, size=self.feat_size, mode='bilinear') # (22,22)

        tr_nframes, _, _, _ = train_label.size()
        te_nframes, nseq, w, h = test_label.size()
        test_label_reshuffle = test_label.permute(1, 0, 2, 3) # (nframes, nseq, W, H) -> (nseq, nframes, W, H)
        # (nseq, nframes*W*H, 1) merge dimensions into one patch dimension
        test_label_reshuffle = test_label_reshuffle.reshape(nseq, te_nframes * w * h, 1)

        proj_test_label_on_mem = torch.einsum('bij,bik->bkj', test_label_reshuffle, p)  # (nseq, nframes*w*h, 1)
        proj_test_label_on_mem_reshuffle = proj_test_label_on_mem.reshape(nseq, tr_nframes, w, h, 1)
        proj_test_label_on_mem_reshuffle = proj_test_label_on_mem_reshuffle.permute(1, 0, 4, 2, 3)
        proj_test_label_on_mem_reshuffle = proj_test_label_on_mem_reshuffle.reshape(tr_nframes * nseq, 1, w, h)

        proj_test_label_on_mem_reshuffle = F.interpolate(proj_test_label_on_mem_reshuffle,
                                                         size=self.out_size, mode='bilinear') # (23,23)

        return proj_test_label_on_mem_reshuffle

# ======================================================================================================================
#
# DEBUGGING AND VISUALIZATION CLASSES
#
# ======================================================================================================================

class ProjMemLabelsOnTestFrameFeature(nn.Module):
    def __init__(self, num_mem_frames=50, feat_size=None, out_size=None, **kwargs):
        super().__init__()
        self.num_mem_frames = num_mem_frames
        self.feat_size = feat_size if feat_size is not None else (22, 22)
        self.out_size = out_size if out_size is not None else (23, 23)

    @property
    def feat_dim(self):
        return 1

    def forward(self):
        raise NotImplementedError("Not designed for training")

    def compute_feature(self, sim, test_label, train_label, **kwargs):
        # sim  (nseq, te_nframes*W*H, tr_nframes*W*H)
        # test_label_reshuffle (nseq, te_nframes*W*H, 1)
        train_label = F.interpolate(train_label, size=self.feat_size, mode='bilinear') # (22,22)

        tr_nframes, nseq, w, h = train_label.size()
        te_nframes, _, _, _ = test_label.size()

        p = F.softmax(1.*sim[:, :, :self.num_mem_frames*w*h], dim=2)  # (nseq, te_nframes*w*h, tr_nframes*w*h)

        train_label_reshuffle = train_label.permute(1, 0, 2, 3)  # (nframes, nseq, W, H) -> (nseq, nframes, W, H)

        train_label_reshuffle = train_label_reshuffle[:, :self.num_mem_frames, :, :]

        train_label_reshuffle = train_label_reshuffle.reshape(nseq, -1, 1)  # (nseq, nframes*W*H, 1) merge dimensions into one patch dimension

        proj_mem_on_test_label = torch.einsum('bij,bki->bkj', train_label_reshuffle, p)  # (nseq, nframes*w*h, 1)
        proj_mem_on_test_label_reshuffle = proj_mem_on_test_label.reshape(nseq, te_nframes, w, h, 1)
        proj_mem_on_test_label_reshuffle = proj_mem_on_test_label_reshuffle.permute(1, 0, 4, 2, 3)
        proj_mem_on_test_label_reshuffle = proj_mem_on_test_label_reshuffle.reshape(te_nframes*nseq, 1, w, h)

        proj_mem_on_test_label_reshuffle = F.interpolate(proj_mem_on_test_label_reshuffle,
                                                         size=self.out_size, mode='bilinear')  # (23,23)

        return proj_mem_on_test_label_reshuffle


class ProjWholeMemLabelsOnTestFrameFeature(ProjMemLabelsOnTestFrameFeature):
    def __init__(self, **kwargs):
        super().__init__(num_mem_frames=50, **kwargs)

class ProjGthMemLabelsOnTestFrameFeature(ProjMemLabelsOnTestFrameFeature):
    def __init__(self, **kwargs):
        super().__init__(num_mem_frames=15, **kwargs)

