import torch
from torch import nn
import torch.nn.functional as F

from ltr.models.layers.distance import DistanceValueEncoder


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


def proj_test_to_mem(sim, test, softmax_temp=50.):
    '''
    Params:
    - sim: (M, Nseq, W*H, W*H)
    - test: (1, Nseq, C, W, H) or (1, Nseq, W, H)

    Returns:
    - proj: (M, Nseq, C, W, H)

    '''
    m = sim.shape[0]

    ndim = len(test.shape)
    if ndim == 5:
        _, nseq, c, w, h = test.shape
    elif ndim == 4:
        _, nseq, w, h = test.shape
        c = 1
    else:
        raise ValueError()

    test = test.view(nseq, c, w * h)  # (Nseq, C, H*W)
    p = torch.softmax(softmax_temp * sim, dim=2)

    z = torch.einsum('nci,mnij->mncj', test, p)
    return z.reshape(m, nseq, c, w, h)


def proj_mem_to_test(sim, mem, softmax_temp=50.):
    '''
    Params:
    - sim: (M, Nseq, W*H, W*H)
    - mem: (M, Nseq, C, W, H) or (M, Nseq, W, H)

    Returns:
    - proj: (M, Nseq, C, W, H)

    '''

    ndim = len(mem.shape)
    if ndim == 5:
        m, nseq, c, w, h = mem.shape
    elif ndim == 4:
        m, nseq, w, h = mem.shape
        c = 1
    else:
        raise ValueError()

    mem = mem.view(m, nseq, c, w * h)  # (Nseq, M, C, H*W)
    p = torch.softmax(softmax_temp * sim, dim=3)

    z = torch.einsum('mnij,mncj->mnci', p, mem)
    return z.reshape(m, nseq, c, w, h)

# ---------------------------------------------------------------------------------------------------
#
# Manually Tuned Modules
#
# ---------------------------------------------------------------------------------------------------

class AttentionAverageStdScalingModule(nn.Module):
    def __init__(self, train_softmax_temp=True, softmax_temp_init=50.):
        super().__init__()
        if train_softmax_temp:
            self.softmax_temp_init = nn.Parameter(softmax_temp_init*torch.ones(1))
        else:
            self.softmax_temp_init = softmax_temp_init

    def forward(self, test_scores, train_labels, test_feat, train_feats):
        sim = compute_cosine_similarity(train_feats, test_feat)
        return self.fuse_scores(sim, test_scores, train_labels, test_feat, train_feats)

    def fuse_scores(self, sim, test_scores, train_labels, test_feat, train_feats):
        _, nseq, wl, hl = test_scores.shape
        nmem, _, _, wf, hf = train_feats.shape

        if len(sim.shape) == 3:
            sim = sim.reshape(nseq, wf*hf, nmem, wf*hf)
            sim = sim.permute(2, 0, 1, 3)

        scores_raw_down = F.interpolate(train_labels, size=(wf, hf), mode='bilinear')  # (22,22)

        pmt_down = proj_mem_to_test(sim, scores_raw_down, self.softmax_temp_init).view(nmem, nseq, wf, hf)  # (M, 1, W, H)
        pmt = F.interpolate(pmt_down, size=(wl, hl), mode='bilinear')  # (23,23)

        mean = torch.mean(pmt, dim=0).unsqueeze(0)
        std = torch.std(pmt, dim=0).unsqueeze(0)

        alpha = 20.
        certainty_scaling = torch.exp(alpha / (1. + std**2) - alpha)
        out = certainty_scaling*mean + test_scores
        return out


class AttentionAverageStdScalingRescaleModule(nn.Module):
    def __init__(self, train_softmax_temp=True, softmax_temp_init=50.):
        super().__init__()
        if train_softmax_temp:
            self.softmax_temp_init = nn.Parameter(softmax_temp_init * torch.ones(1))
        else:
            self.softmax_temp_init = softmax_temp_init

    def forward(self, test_scores, train_labels, test_feat, train_feats):
        sim = compute_cosine_similarity(train_feats, test_feat)
        return self.fuse_scores(sim, test_scores, train_labels, test_feat, train_feats)

    def fuse_scores(self, sim, test_scores, train_labels, test_feat, train_feats):
        _, nseq, wl, hl = test_scores.shape
        nmem, _, _, wf, hf = train_feats.shape

        if len(sim.shape) == 3:
            sim = sim.reshape(nseq, wf * hf, nmem, wf * hf)
            sim = sim.permute(2, 0, 1, 3)

        scores_raw_down = F.interpolate(train_labels, size=(wf, hf), mode='bilinear')  # (22,22)

        pmt_down = proj_mem_to_test(sim, scores_raw_down, self.softmax_temp_init).view(nmem, nseq, wf,
                                                                                       hf)  # (M, 1, W, H)
        pmt = F.interpolate(pmt_down, size=(wl, hl), mode='bilinear')  # (23,23)

        mean = torch.mean(pmt, dim=0).unsqueeze(0)
        std = torch.std(pmt, dim=0).unsqueeze(0)

        alpha = 20.
        certainty_scaling = torch.exp(alpha / (1. + std ** 2) - alpha)

        max_peak = torch.max(test_scores.max(), (certainty_scaling * mean).max())

        out = (certainty_scaling * mean + test_scores)
        out = out / out.max() * max_peak
        return out


class AttentionGradientDescentMergeModule(nn.Module):
    def __init__(self, train_softmax_temp=True, softmax_temp_init=50.):
        super().__init__()
        if train_softmax_temp:
            self.softmax_temp_init = nn.Parameter(softmax_temp_init*torch.ones(1))
        else:
            self.softmax_temp_init = softmax_temp_init

    def gradientdecent(self, x0, mu, sig, niter=1, gamma=0.1):
        xold = x0
        for i in range(0, niter):
            xnew = xold - gamma * (xold - mu) / sig ** 2
            xold = xnew
        return xnew

    def forward(self, test_scores, train_labels, test_feat, train_feats):
        sim = compute_cosine_similarity(train_feats, test_feat)
        return self.fuse_scores(sim, test_scores, train_labels, test_feat, train_feats)

    def fuse_scores(self, sim, test_scores, train_labels, test_feat, train_feats):
        _, nseq, wl, hl = test_scores.shape
        nmem, _, _, wf, hf = train_feats.shape

        if len(sim.shape) == 3:
            sim = sim.reshape(nseq, wf*hf, nmem, wf*hf)
            sim = sim.permute(2, 0, 1, 3)

        scores_raw_down = F.interpolate(train_labels, size=(wf, hf), mode='bilinear')  # (22,22)

        pmt_down = proj_mem_to_test(sim, scores_raw_down, self.softmax_temp_init).view(nmem, nseq, wf, hf)  # (M, N, W, H)
        pmt = F.interpolate(pmt_down, size=(wl, hl), mode='bilinear')  # (23,23)

        mean = torch.mean(pmt, dim=0).unsqueeze(0)
        std = torch.std(pmt, dim=0).unsqueeze(0)

        scale = mean.max() / test_scores.max()

        mask = (mean[0,0] > mean.mean()) & (test_scores[0, 0] > test_scores.mean())

        new_scores = test_scores[0, 0].clone()

        new_scores[mask] = self.gradientdecent(scale*new_scores[mask], mean[0,0][mask], std[0,0][mask].clamp_min(0.05),
                                               niter=150, gamma=1.0e-4)
        new_scores[mask] /= scale

        return new_scores.reshape(1, nseq, wl, hl)


# ---------------------------------------------------------------------------------------------------
#
# Learned Modules
#
# ---------------------------------------------------------------------------------------------------

class AttentionLearnFusionDirectModule(nn.Module):
    def __init__(self, train_softmax_temp=True, softmax_temp_init=50.):
        super().__init__()
        if train_softmax_temp:
            self.softmax_temp_init = nn.Parameter(softmax_temp_init * torch.ones(1))
        else:
            self.softmax_temp_init = softmax_temp_init

        self.kernel_size = 1
        self.scores_encoder = DistanceValueEncoder(num_bins=32, min_val=-1., max_val=1.)
        self.mean_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)
        self.std_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)

        self.layers = [
            nn.Conv2d(in_channels=32+2*16, out_channels=64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=self.kernel_size, padding=self.kernel_size//2),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, test_scores, train_labels, test_feat, train_feats):
        sim = compute_cosine_similarity(train_feats, test_feat)
        return self.fuse_scores(sim, test_scores, train_labels, test_feat, train_feats)

    def fuse_scores(self, sim, test_scores, train_labels, test_feat, train_feats):
        _, nseq, wl, hl = test_scores.shape
        nmem, _, _, wf, hf = train_feats.shape

        if len(sim.shape) == 3:
            sim = sim.reshape(nseq, wf * hf, nmem, wf * hf)
            sim = sim.permute(2, 0, 1, 3)

        scores_raw_down = F.interpolate(train_labels, size=(wf, hf), mode='bilinear')  # (22,22)

        pmt_down = proj_mem_to_test(sim, scores_raw_down, self.softmax_temp_init).view(nmem, nseq, wf, hf)  # (M, 1, W, H)
        pmt = F.interpolate(pmt_down, size=(wl, hl), mode='bilinear')  # (23,23)

        mean = torch.mean(pmt, dim=0).unsqueeze(1)
        std = torch.std(pmt, dim=0).unsqueeze(1)
        scores = test_scores.permute(1, 0, 2, 3)

        scores_enc = self.scores_encoder(F.tanh(scores))
        mean_enc = self.mean_encoder(F.tanh(mean))
        std_enc = self.std_encoder(F.tanh(std))

        inp = torch.cat([scores_enc, mean_enc, std_enc], dim=1)

        out = self.model(inp)

        return out.permute(1,0,2,3)



class AttentionLearnMeanScalingModule(nn.Module):
    def __init__(self, train_softmax_temp=True, softmax_temp_init=50.):
        super().__init__()
        if train_softmax_temp:
            self.softmax_temp_init = nn.Parameter(softmax_temp_init * torch.ones(1))
        else:
            self.softmax_temp_init = softmax_temp_init

        self.kernel_size = 1
        self.mean_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)
        self.std_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)

        self.layers = [
            nn.Conv2d(in_channels=2 * 16, out_channels=64, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, test_scores, train_labels, test_feat, train_feats):
        sim = compute_cosine_similarity(train_feats, test_feat)
        return self.fuse_scores(sim, test_scores, train_labels, test_feat, train_feats)

    def fuse_scores(self, sim, test_scores, train_labels, test_feat, train_feats):
        _, nseq, wl, hl = test_scores.shape
        nmem, _, _, wf, hf = train_feats.shape

        if len(sim.shape) == 3:
            sim = sim.reshape(nseq, wf * hf, nmem, wf * hf)
            sim = sim.permute(2, 0, 1, 3)

        scores_raw_down = F.interpolate(train_labels, size=(wf, hf), mode='bilinear')  # (22,22)

        pmt_down = proj_mem_to_test(sim, scores_raw_down, self.softmax_temp_init).view(nmem, nseq, wf, hf)  # (M, 1, W, H)
        pmt = F.interpolate(pmt_down, size=(wl, hl), mode='bilinear')  # (23,23)

        mean = torch.mean(pmt, dim=0).unsqueeze(1)
        std = torch.std(pmt, dim=0).unsqueeze(1)

        mean_enc = self.mean_encoder(F.tanh(mean))
        std_enc = self.std_encoder(F.tanh(std))

        inp = torch.cat([mean_enc, std_enc], dim=1)

        scaling = self.model(inp)
        scaled_mean = scaling*mean

        out = scaled_mean/scaled_mean.max()*mean.max()
        # out = scaled_mean

        max_peak = torch.max(test_scores.max(), mean.max())

        out = (out + test_scores.permute(1, 0, 2, 3))
        out = out / out.max() * max_peak

        return out.permute(1, 0, 2, 3)


class AttentionLearnScoreScalingModule(nn.Module):
    def __init__(self, train_softmax_temp=True, softmax_temp_init=50.):
        super().__init__()
        if train_softmax_temp:
            self.softmax_temp_init = nn.Parameter(softmax_temp_init * torch.ones(1))
        else:
            self.softmax_temp_init = softmax_temp_init

        self.kernel_size = 1
        # self.score_encoder = DistanceValueEncoder(num_bins=32, min_val=-1., max_val=1.)
        self.mean_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)
        self.std_encoder = DistanceValueEncoder(num_bins=16, min_val=0., max_val=1.)

        self.layers = [
            nn.Conv2d(in_channels=2 * 16, out_channels=64, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
        ]

        self.model = nn.Sequential(*self.layers)

        self.softplus = nn.Softplus()
        # self.bn = nn.BatchNorm2d(num_features=2)
        # self.merge = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0)

    def forward(self, test_scores, train_labels, test_feat, train_feats):
        sim = compute_cosine_similarity(train_feats, test_feat)
        return self.fuse_scores(sim, test_scores, train_labels, test_feat, train_feats)

    def fuse_scores(self, sim, test_scores, train_labels, test_feat, train_feats):
        _, nseq, wl, hl = test_scores.shape
        nmem, _, _, wf, hf = train_feats.shape

        if len(sim.shape) == 3:
            sim = sim.reshape(nseq, wf * hf, nmem, wf * hf)
            sim = sim.permute(2, 0, 1, 3)

        scores_raw_down = F.interpolate(train_labels, size=(wf, hf), mode='bilinear')  # (22,22)

        pmt_down = proj_mem_to_test(sim, scores_raw_down, self.softmax_temp_init).view(nmem, nseq, wf, hf)  # (M, 1, W, H)
        pmt = F.interpolate(pmt_down, size=(wl, hl), mode='bilinear')  # (23,23)

        mean = torch.mean(pmt, dim=0).unsqueeze(1)
        std = torch.std(pmt, dim=0).unsqueeze(1)

        # scores_enc = self.score_encoder(F.tanh(test_scores.permute(1, 0, 2, 3)))
        mean_enc = self.mean_encoder(F.tanh(mean))
        std_enc = self.std_encoder(F.tanh(std))

        inp = torch.cat([mean_enc, std_enc], dim=1)

        scaling = self.model(inp)
        scaled_mean = scaling * mean
        # inp_merge = torch.cat([scaled_mean, test_scores.permute(1, 0, 2, 3)], dim=1)

        # out = self.merge(self.bn(inp_merge))
        # out = scaling*test_scores.permute(1, 0, 2, 3)

        # out = out / out.max() * test_scores.max()

        out = test_scores.permute(1, 0, 2, 3) - self.softplus(scaled_mean)

        return out.permute(1, 0, 2, 3)


# class AttentionFusionModule(nn.Module):
#     def __init__(self, softmax_temp_init=50., feat_shape=None, kernel_size=1, hidden_dim=8):
#         super().__init__()
#
#         self.softmax_temp_mt = nn.Parameter(softmax_temp_init * torch.ones(1), requires_grad=True)
#
#
#     def forward(self, test_score, train_labels, test_cls_feat, train_cls_feat):
#         sim = compute_cosine_similarity(train_cls_feat, test_cls_feat)
#         scores = self.compute_scores(sim, test_score, train_labels)
#
#         return scores
#
#     def compute_scores(self, sim, test_score, train_labels):
#         _, nseq, w, h = test_score.shape
#         nmem, _, _, _ = train_labels.shape
#
#
#         if len(sim.shape) == 3:
#             sim = sim.reshape(nseq, 22*22, nmem, 22*22)
#             sim = sim.permute(2, 0, 1, 3)
#
#
#         scores_raw_down = F.interpolate(train_labels, size=(22, 22), mode='bilinear')  # (22,22)
#
#         pmt = proj_mem_to_test(sim, scores_raw_down).view(train_labels.shape[0], 1, 22, 22)  # (M, 1, W, H)
#         pmt = F.interpolate(pmt, size=(23, 23), mode='bilinear')  # (22,22)
#
#         mean = torch.mean(pmt, dim=0).unsqueeze(0)
#         std = torch.std(pmt, dim=0).unsqueeze(0)
#
#         max_mean = torch.max(mean)
#         mask = (mean >= 0.01) | (test_score >= 0.01)
#         mask &= (test_score >= 0)
#
#         certainty = torch.ones_like(std)
#         certainty[mask] = 1. / std[mask].clamp_min(0.01)
#
#         out1 = certainty * mean
#         out1 = out1 / torch.max(out1) * max_mean
#
#         max_score = torch.max(test_score)
#         out2 = certainty * test_score
#         out2 = out2 / torch.max(out2) * max_score
#
#
#         # out = 0.5*out1 + 0.5*test_score
#         out = 0.5*out1 + 0.5*out2
#
#         return out


# class ResBlock(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_hidden=8, kernel_size=1, enable_output_act=False):
#         super().__init__()
#         self.enable_output_act = enable_output_act
#         self.layers = [
#             nn.Conv2d(in_channels=dim_in, out_channels=dim_hidden, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.BatchNorm2d(dim_hidden),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=dim_hidden, out_channels=dim_out, kernel_size=kernel_size, padding=kernel_size//2),
#         ]
#
#         if self.enable_output_act:
#             self.layers.append(nn.BatchNorm2d(dim_out))
#
#         self.output_act = nn.ReLU()
#
#
#         self.residual = nn.Sequential(*self.layers)
#
#     def forward(self, x1, x2):
#         if self.enable_output_act:
#             h = self.output_act(x1 + self.residual(x2))
#         else:
#             h = x1 + self.residual(x2)
#
#         return h


# class ResidualProjTestToMemLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_hidden, softmax_temp, feat_shape=None, kernel_size=1,
#                  enable_output_act=False):
#         super().__init__()
#         self.softmax_temp = softmax_temp
#         self.feat_shape = feat_shape
#
#         self.resblock = ResBlock(dim_in=dim_in, dim_hidden=dim_hidden, dim_out=dim_out, kernel_size=kernel_size,
#                                  enable_output_act=enable_output_act)
#
#     def forward(self, sim, m, s):
#         '''
#         m: (M, Nseq, C1, W, H)
#         s: (1, Nseq, C2, W, H)
#         '''
#
#         expand = False
#         if len(s.shape) == 4:
#             s = s.unsqueeze(2)
#         if len(m.shape) == 4:
#             m = m.unsqueeze(2)
#             expand = True
#
#         _, nseq, cs, w, h = s.shape
#         nmem, _, cm, _, _ = m.shape
#
#         if len(sim.shape) == 3:
#             sim = sim.reshape(nseq, self.feat_shape[0]*self.feat_shape[1], nmem, self.feat_shape[0]*self.feat_shape[1])
#             sim = sim.permute(2, 0, 1, 3)
#
#         s_resize = F.interpolate(s.reshape(nseq, cs, w, h), size=self.feat_shape, mode='bilinear')
#         s_resize = s_resize.reshape(1, nseq, cs, s_resize.shape[2], s_resize.shape[3])
#
#         stm_resize = proj_test_to_mem(sim, s_resize, self.softmax_temp)  # (Nseq, M, C1, W, H)
#
#         stm = F.interpolate(stm_resize.reshape(nmem*nseq, cs, self.feat_shape[0], self.feat_shape[1]),
#                             size=(w, h), mode='bilinear')
#         stm = stm.reshape(nmem, nseq, cs, w, h)
#
#         inp = torch.cat([m, stm], dim=2)  # (Nseq, M, C1+C2, W, H)
#         inp_merge = inp.reshape(nmem*nseq, cm+cs, w, h) # (Nseq*M, C1+C2, W, H)
#         m_merge = m.reshape(nmem*nseq, cm, w, h)
#
#         z = self.resblock(m_merge, inp_merge)
#
#         if expand:
#             z = z.reshape(nmem, nseq, w, h)
#         else:
#             z = z.reshape(nmem, nseq, cm, w, h)
#
#         return z


# class ResidualProjMemToTestLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_hidden, softmax_temp, feat_shape=None, kernel_size=1,
#                  enable_output_act=False):
#         super().__init__()
#         self.softmax_temp = softmax_temp
#         self.feat_shape = (22, 22) if feat_shape is None else feat_shape
#
#         self.resblock = ResBlock(dim_in=dim_in, dim_hidden=dim_hidden, dim_out=dim_out, kernel_size=kernel_size,
#                                  enable_output_act=enable_output_act)
#
#     def forward(self, sim, m, s):
#         '''
#         m: (M, Nseq, C1, W, H)
#         s: (1, Nseq, C2, W, H)
#         '''
#         # s: torch.Size([1, 20, 23, 23])
#         # m torch.Size([5, 20, 23, 23])
#         expand = False
#         if len(s.shape) == 4:
#             s = s.unsqueeze(2)
#             expand = True
#         if len(m.shape) == 4:
#             m = m.unsqueeze(2)
#
#         _, nseq, cs, w, h = s.shape
#         nmem, _, cm, _, _ = m.shape
#
#         if len(sim.shape) == 3:
#             sim = sim.reshape(nseq, self.feat_shape[0]*self.feat_shape[1], nmem, self.feat_shape[0]*self.feat_shape[1])
#             sim = sim.permute(2, 0, 1, 3)
#
#         m_resize = F.interpolate(m.reshape(nmem*nseq, cm, w, h), size=self.feat_shape, mode='bilinear')
#         m_resize = m_resize.reshape(nmem, nseq, cm, m_resize.shape[2], m_resize.shape[3])
#
#         mts_resize = proj_mem_to_test(sim, m_resize, self.softmax_temp)  # (M, Nseq, C1, W, H)
#
#         mts = F.interpolate(mts_resize.reshape(nmem*nseq, cm, self.feat_shape[0], self.feat_shape[1]),
#                             size=(w,h), mode='bilinear')
#         mts = mts.reshape(nmem, nseq, cm, w, h)
#
#         mean = torch.mean(mts, dim=0).unsqueeze(0) # (1, Nseq, C1, W, H)
#         std = torch.mean(mts, dim=0).unsqueeze(0) # (1, Nseq, C1, W, H)
#
#         inp = torch.cat([s, mean, std], dim=2) # (1, Nseq, C1+C1+C1, W, H)
#         inp_merge = inp.reshape(nseq, cs+2*cm, w, h)
#
#         s_merge = s.reshape(nseq, cs, w, h)
#
#         z = self.resblock(s_merge, inp_merge)
#
#         if expand:
#             z = z.reshape(1, nseq, w, h)
#         else:
#             z = z.reshape(1, nseq, cs, w, h)
#
#         return z


# class AttentionFusionModule(nn.Module):
#     def __init__(self, softmax_temp_init=50., feat_shape=None, kernel_size=1, hidden_dim=8):
#         super().__init__()
#
#         self.softmax_temp_tm = nn.Parameter(softmax_temp_init * torch.ones(1), requires_grad=True)
#         self.softmax_temp_mt = nn.Parameter(softmax_temp_init * torch.ones(1), requires_grad=True)
#
#         self.res_tm = ResidualProjTestToMemLayer(dim_in=2, dim_hidden=hidden_dim, dim_out=1, feat_shape=feat_shape,
#                                                  softmax_temp=self.softmax_temp_tm, kernel_size=kernel_size)
#         self.res_mt = ResidualProjMemToTestLayer(dim_in=3, dim_hidden=hidden_dim, dim_out=1, feat_shape=feat_shape,
#                                                  softmax_temp=self.softmax_temp_mt, kernel_size=kernel_size)
#
#         # self.res_tm_2 = ResidualProjTestToMemLayer(dim_in=2, dim_hidden=hidden_dim, dim_out=1, feat_shape=feat_shape,
#         #                                            softmax_temp=self.softmax_temp_tm, kernel_size=kernel_size)
#         # self.res_mt_2 = ResidualProjMemToTestLayer(dim_in=3, dim_hidden=hidden_dim, dim_out=1, feat_shape=feat_shape,
#         #                                            softmax_temp=self.softmax_temp_mt, kernel_size=kernel_size)
#
#     def forward(self, test_score, train_labels, test_cls_feat, train_cls_feat):
#         sim = compute_cosine_similarity(train_cls_feat, test_cls_feat)
#         scores = self.compute_scores(sim, test_score, train_labels)
#
#         return scores
#
#     def compute_scores(self, sim, test_score, train_labels):
#         _, nseq, w, h = test_score.shape
#         nmem, _, _, _ = train_labels.shape
#
#
#         if len(sim.shape) == 3:
#             sim = sim.reshape(nseq, 22*22, nmem, 22*22)
#             sim = sim.permute(2, 0, 1, 3)
#
#
#         scores_raw_down = F.interpolate(train_labels, size=(22, 22), mode='bilinear')  # (22,22)
#
#         pmt = proj_mem_to_test(sim, scores_raw_down).view(train_labels.shape[0], 1, 22, 22)  # (M, 1, W, H)
#         pmt = F.interpolate(pmt, size=(23, 23), mode='bilinear')  # (22,22)
#
#         mean = torch.mean(pmt, dim=0).unsqueeze(0)
#         std = torch.std(pmt, dim=0).unsqueeze(0)
#
#         max_mean = torch.max(mean)
#         mask = (mean >= 0.01) | (test_score >= 0.01)
#         mask &= (test_score >= 0)
#
#         certainty = torch.ones_like(std)
#         certainty[mask] = 1. / std[mask].clamp_min(0.01)
#
#         out1 = certainty * mean
#         out1 = out1 / torch.max(out1) * max_mean
#
#         max_score = torch.max(test_score)
#         out2 = certainty * test_score
#         out2 = out2 / torch.max(out2) * max_score
#
#
#         # out = 0.5*out1 + 0.5*test_score
#         out = 0.5*out1 + 0.5*out2
#
#         return out

    # def compute_scores(self, sim, test_score, train_labels):
    #     m0 = train_labels
    #     s0 = test_score
    #
    #     m1 = self.res_tm(sim, m0, s0)
    #     s1 = self.res_mt(sim, m1, s0)
    #
    #     # m2 = self.res_tm_2(sim, m1, s1)
    #     # s2 = self.res_mt_2(sim, m2, s1)
    #
    #     return s1

