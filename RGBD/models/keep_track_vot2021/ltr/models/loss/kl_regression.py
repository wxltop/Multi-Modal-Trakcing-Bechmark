import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class KLRegression(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density, mc_dim=-1):
        """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""

        exp_val = scores - torch.log(sample_density + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - \
            torch.mean(scores * (gt_density / (sample_density + self.eps)), dim=mc_dim)

        return L.mean()


class MLRegression(nn.Module):
    """Maximum likelihood loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: probability density of the sample distribution
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""

        assert mc_dim == 1
        assert (sample_density[:,0,...] == -1).all()

        exp_val = scores[:, 1:, ...] - torch.log(sample_density[:, 1:, ...] + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim] - 1) - scores[:, 0, ...]
        loss = L.mean()
        return loss


class KLRegressionGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""

        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        L = torch.logsumexp(scores, dim=grid_dim) + math.log(grid_scale) - score_corr

        return L.mean()




class KLRegRegressionGrid(nn.Module):
    def __init__(self, exp_max=None, reg=None):
        super().__init__()
        self.exp_max = exp_max
        self.reg = reg

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):

        if self.exp_max is None:
            bias = scores.new_zeros(1)
            bias_squeeze = scores.new_zeros(1)
        else:
            if isinstance(grid_dim, int):
                max_score = torch.max(scores.detach(), dim=grid_dim, keepdim=True)[0]
            else:
                max_score = scores.detach()
                for d in grid_dim:
                    max_score = torch.max(max_score, dim=d, keepdim=True)[0]
                if self.reg is not None:
                    max_score.clamp_(min=self.reg)

            bias = (max_score - self.exp_max).clamp(min=0)
            bias_squeeze = bias.sum(dim=grid_dim)

        norm_const = grid_scale * torch.sum(torch.exp(scores - bias), dim=grid_dim)
        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        if self.reg is None:
            L = torch.log(norm_const) + bias_squeeze - score_corr
        else:
            L = torch.log(norm_const + torch.exp(self.reg - bias)) + bias_squeeze - score_corr

        return L.mean()



class MLklRegressionGrid(nn.Module):
    def __init__(self, exp_max=None, reg=None):
        super().__init__()
        self.exp_max = exp_max
        self.reg = reg

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """mc_dim is dimension of MC samples."""

        assert grid_dim == (-2, -1)

        gt_reshape = gt_density.view(-1, gt_density.shape[-2]*gt_density.shape[-1])
        one_hot = torch.zeros_like(gt_reshape)
        one_hot[torch.arange(gt_reshape.shape[0]), gt_reshape.argmax(dim=-1)] = 1.0
        gt_density = one_hot.view(gt_density.shape)

        if self.exp_max is None:
            bias = scores.new_zeros(1)
            bias_squeeze = scores.new_zeros(1)
        else:
            if isinstance(grid_dim, int):
                max_score = torch.max(scores.detach(), dim=grid_dim, keepdim=True)[0]
            else:
                max_score = scores.detach()
                for d in grid_dim:
                    max_score = torch.max(max_score, dim=d, keepdim=True)[0]
                if self.reg is not None:
                    max_score.clamp_(min=self.reg)

            bias = (max_score - self.exp_max).clamp(min=0)
            bias_squeeze = bias.sum(dim=grid_dim)

        norm_const = grid_scale * torch.sum(torch.exp(scores - bias), dim=grid_dim)
        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        if self.reg is None:
            L = torch.log(norm_const) + bias_squeeze - score_corr
        else:
            L = torch.log(norm_const + torch.exp(self.reg - bias)) + bias_squeeze - score_corr

        return L.mean()



class NCERegression(nn.Module):
    """ TODO! TODO! TODO! """

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: TODO! TODO!
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""

        assert mc_dim == 1

        # (scores has shape: (64, 1+128))
        # (sample_density has shape: (64, 1+128))

        f_samples = scores[:, 1:] # (shape: (64, 128))
        p_N_samples = sample_density[:, 1:] # (shape: (64, 128))

        f_0 = scores[:, 0] # (shape: (64))
        p_N_0 = sample_density[:, 0] # (shape: (64))

        exp_vals_0 = f_0-torch.log(p_N_0 + self.eps) # (shape: (64))

        exp_vals_samples = f_samples-torch.log(p_N_samples + self.eps) # (shape: (64, 128))

        exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1) # (shape: (64, 1+128))

        loss = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))

        return loss
