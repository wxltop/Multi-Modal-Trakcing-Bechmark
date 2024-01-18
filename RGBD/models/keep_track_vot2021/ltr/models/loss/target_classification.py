import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import ltr.data.processing_utils as prutils

def gauss_1d(sz, sigma, center):
    k = torch.arange(-(sz-1)/2, (sz+1)/2).reshape(1, -1)
    return torch.exp(-1.0/(2*sigma**2) * (k - center.reshape(-1, 1))**2)


def gauss_2d(sz, sigma, center):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0]).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1]).reshape(center.shape[0], -1, 1)


def get_gaussian_label(target, sigma_factor, kernel_sz, label_sz, image_sz):
    target_center = target[:, 0:2] + 0.5 * target[:, 2:4]
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = label_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * label_sz.prod().sqrt().item()

    gauss_label = gauss_2d(label_sz, sigma, center)
    return gauss_label


class TargetClassificationGaussL2(nn.Module):
    """
    """
    def __init__(self, sigma, kernel_sz, image_sz):
        super(TargetClassificationGaussL2, self).__init__()
        if isinstance(kernel_sz, int):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(image_sz, int):
            image_sz = (image_sz, image_sz)
        self.sigma = sigma
        self.kernel_sz = kernel_sz
        self.image_sz = torch.Tensor(image_sz)

    def forward(self, prediction, target_bb=None, is_distractor=None):
        # Construct gaussian label
        feature_sz = torch.Tensor([prediction.shape[-2], prediction.shape[-1]])
        gauss_label = get_gaussian_label(target_bb.view(-1,4).cpu(), self.sigma, self.kernel_sz, feature_sz,
                                         self.image_sz).to(prediction.device)

        if is_distractor is not None:
            gauss_label = gauss_label * (1 - is_distractor).view(-1, 1, 1).float()
        loss = F.mse_loss(prediction.view(-1, prediction.shape[-2], prediction.shape[-1]), gauss_label)

        return loss


class TargetClassificationGaussHinge(nn.Module):
    """
    """
    def __init__(self, sigma, kernel_sz, image_sz, error_metric=nn.MSELoss(), negative_threshold=-100):
        super(TargetClassificationGaussHinge, self).__init__()
        if isinstance(kernel_sz, int):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(image_sz, int):
            image_sz = (image_sz, image_sz)
        self.sigma = sigma
        self.kernel_sz = kernel_sz
        self.image_sz = torch.Tensor(image_sz)
        self.error_metric = error_metric
        self.negative_threshold = negative_threshold

    def forward(self, prediction, target_bb=None, is_distractor=None):
        # Construct gaussian label
        feature_sz = torch.Tensor([prediction.shape[-2], prediction.shape[-1]])
        gauss_label = get_gaussian_label(target_bb.view(-1,4).cpu(), self.sigma, self.kernel_sz, feature_sz,
                                         self.image_sz).to(prediction.device)
        if is_distractor is not None:
            gauss_label = gauss_label * (1 - is_distractor).view(-1, 1, 1).float()

        negative_mask = (gauss_label < self.negative_threshold).float()

        prediction = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])

        prediction = negative_mask * F.relu(prediction) + (1.0 - negative_mask) * prediction

        loss = self.error_metric(prediction, gauss_label)
        return loss


class TargetClassificationGaussHingeWeightScalar(nn.Module):
    """
    """
    def __init__(self, sigma, kernel_sz, image_sz, error_metric=nn.MSELoss(), negative_threshold=-100,
                 wt_pos=None, wt_neg=None):
        super(TargetClassificationGaussHingeWeightScalar, self).__init__()
        if isinstance(kernel_sz, int):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(image_sz, int):
            image_sz = (image_sz, image_sz)
        self.sigma = sigma
        self.kernel_sz = kernel_sz
        self.image_sz = torch.Tensor(image_sz)
        self.error_metric = error_metric
        self.negative_threshold = negative_threshold
        self.wt_pos = wt_pos
        self.wt_neg = wt_neg

    def forward(self, prediction, target_bb=None, is_distractor=None):
        if self.wt_pos is None:
            self.wt_pos = 1
        if self.wt_neg is None:
            self.wt_neg = 0.5
        # Construct gaussian label
        feature_sz = torch.Tensor([prediction.shape[-2], prediction.shape[-1]])
        gauss_label = get_gaussian_label(target_bb.view(-1,4).cpu(), self.sigma, self.kernel_sz, feature_sz,
                                         self.image_sz).to(prediction.device)
        if is_distractor is not None:
            gauss_label = gauss_label * (1 - is_distractor).view(-1, 1, 1).float()

        negative_mask = (gauss_label < self.negative_threshold).float()

        prediction = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])

        prediction = negative_mask * F.relu(prediction) + (1.0 - negative_mask) * prediction

        # weighting -> mind the sqrt
        if is_distractor is not None:
            flat_distractor = is_distractor.view(-1, is_distractor.shape[0]*is_distractor.shape[1])
            wt_flat_distractor = torch.sqrt(self.wt_pos * (1-flat_distractor) + self.wt_neg * flat_distractor)
            prediction = wt_flat_distractor.t().unsqueeze(2)*prediction
            gauss_label = wt_flat_distractor.t().unsqueeze(2)*gauss_label
        loss = self.error_metric(prediction, gauss_label)
        return loss


class TargetClassificationGaussHingeWeightGauss(nn.Module):
    """
    """
    def __init__(self, sigma, kernel_sz, image_sz, error_metric=nn.MSELoss(), negative_threshold=-100,
                 wt_pos=None, wt_neg_mx=None, wt_neg_off=None):
        super(TargetClassificationGaussHingeWeightGauss, self).__init__()
        if isinstance(kernel_sz, int):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(image_sz, int):
            image_sz = (image_sz, image_sz)
        self.sigma = sigma
        self.kernel_sz = kernel_sz
        self.image_sz = torch.Tensor(image_sz)
        self.error_metric = error_metric
        self.negative_threshold = negative_threshold
        self.wt_pos = wt_pos
        self.wt_neg_mx = wt_neg_mx
        self.wt_neg_off = wt_neg_off

    def forward(self, prediction, target_bb=None, is_distractor=None):
        # these would decide the peaks
        if self.wt_pos is None:
            self.wt_pos = 1
        if self.wt_neg_mx is None:
            self.wt_neg_mx = 1
        if self.wt_neg_off is None:
            self.wt_neg_off = 0.5

        # Construct gaussian label
        feature_sz = torch.Tensor([prediction.shape[-2], prediction.shape[-1]])
        gauss_label = get_gaussian_label(target_bb.view(-1,4).cpu(), self.sigma, self.kernel_sz, feature_sz,
                                         self.image_sz).to(prediction.device)
        if is_distractor is not None:
            gauss_label = gauss_label * (1 - is_distractor).view(-1, 1, 1).float()

        negative_mask = (gauss_label < self.negative_threshold).float()

        prediction = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])

        prediction = negative_mask * F.relu(prediction) + (1.0 - negative_mask) * prediction

        if is_distractor is not None:
            flat_distractor = is_distractor.view(-1, is_distractor.shape[0] * is_distractor.shape[1])
            flat_distractor_dash = 1-flat_distractor

            gauss_gt = gauss_label[flat_distractor_dash.byte().squeeze(0)]
            gauss_gt = gauss_gt.unsqueeze(1).repeat(1, is_distractor.shape[1], 1, 1)
            gauss_gt = (self.wt_neg_mx-self.wt_neg_off)*gauss_gt.view(-1, gauss_gt.shape[2], gauss_gt.shape[3]) + \
                self.wt_neg_off

            # gauss_gt_wt_
            prediction = flat_distractor.t().unsqueeze(2)*prediction*torch.sqrt(gauss_gt) + \
                         flat_distractor_dash.t().unsqueeze(2)*prediction*torch.sqrt(self.wt_pos*gauss_label)
            gauss_label = flat_distractor.t().unsqueeze(2)*gauss_label*torch.sqrt(gauss_gt) + \
                         flat_distractor_dash.t().unsqueeze(2)*gauss_label*torch.sqrt(self.wt_pos*gauss_label)

        loss = self.error_metric(prediction, gauss_label)
        return loss


class TargetClassificationGaussHingeWeightGaussOffset(nn.Module):
    """
    """
    def __init__(self, sigma, kernel_sz, image_sz, error_metric=nn.MSELoss(), negative_threshold=-100,
                 wt_pos_mx=None, wt_neg_mx=None, wt_neg_off=None, wt_pos_off=None):
        super(TargetClassificationGaussHingeWeightGaussOffset, self).__init__()
        if isinstance(kernel_sz, int):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(image_sz, int):
            image_sz = (image_sz, image_sz)
        self.sigma = sigma
        self.kernel_sz = kernel_sz
        self.image_sz = torch.Tensor(image_sz)
        self.error_metric = error_metric
        self.negative_threshold = negative_threshold
        self.wt_pos_mx = wt_pos_mx
        self.wt_neg_mx = wt_neg_mx
        self.wt_neg_off = wt_neg_off
        self.wt_pos_off = wt_pos_off

    def forward(self, prediction, target_bb=None, is_distractor=None):
        # these would decide the peaks
        if self.wt_pos_mx is None:
            self.wt_pos_mx = 1
        if self.wt_neg_mx is None:
            self.wt_neg_mx = 1
        if self.wt_neg_off is None:
            self.wt_neg_off = 0.5
        if self.wt_pos_off is None:
            self.wt_pos_off = 0.5

        # Construct gaussian label
        feature_sz = torch.Tensor([prediction.shape[-2], prediction.shape[-1]])
        gauss_label = get_gaussian_label(target_bb.view(-1,4).cpu(), self.sigma, self.kernel_sz, feature_sz,
                                         self.image_sz).to(prediction.device)
        if is_distractor is not None:
            gauss_label = gauss_label * (1 - is_distractor).view(-1, 1, 1).float()

        negative_mask = (gauss_label < self.negative_threshold).float()

        prediction = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])

        prediction = negative_mask * F.relu(prediction) + (1.0 - negative_mask) * prediction

        # TODO: check this
        if is_distractor is not None:
            flat_distractor = is_distractor.view(-1, is_distractor.shape[0] * is_distractor.shape[1])
            flat_distractor_dash = 1-flat_distractor

            gauss_gt = gauss_label[flat_distractor_dash.byte().squeeze(0)]
            gauss_gt = gauss_gt.unsqueeze(1).repeat(1, is_distractor.shape[1], 1, 1)
            gauss_gt = (self.wt_neg_mx-self.wt_neg_off)*gauss_gt.view(-1, gauss_gt.shape[2], gauss_gt.shape[3]) + \
                self.wt_neg_off

            gauss_gt_pos = (self.wt_pos_mx-self.wt_pos_off)*gauss_label.clone() + self.wt_pos_off

            # TODO: solve this - integrate the negative class kind off
            # gauss_gt_wt_
            prediction = flat_distractor.t().unsqueeze(2)*prediction*torch.sqrt(gauss_gt) + \
                         flat_distractor_dash.t().unsqueeze(2)*prediction*torch.sqrt(gauss_gt_pos)
            gauss_label = flat_distractor.t().unsqueeze(2)*gauss_label*torch.sqrt(gauss_gt) + \
                         flat_distractor_dash.t().unsqueeze(2)*gauss_label*torch.sqrt(gauss_gt_pos)

        loss = self.error_metric(prediction, gauss_label)
        return loss


class TargetClassificationGaussHingeWeightGaussFlat(nn.Module):
    """
    """
    def __init__(self, sigma, kernel_sz, image_sz, error_metric=nn.MSELoss(), negative_threshold=-100,
                 wt_pos=None, wt_neg=None):
        super(TargetClassificationGaussHingeWeightGaussFlat, self).__init__()
        if isinstance(kernel_sz, int):
            kernel_sz = (kernel_sz, kernel_sz)
        if isinstance(image_sz, int):
            image_sz = (image_sz, image_sz)
        self.sigma = sigma
        self.kernel_sz = kernel_sz
        self.image_sz = torch.Tensor(image_sz)
        self.error_metric = error_metric
        self.negative_threshold = negative_threshold
        self.wt_pos = wt_pos
        self.wt_neg = wt_neg

    def forward(self, prediction, target_bb=None, is_distractor=None):
        # these would decide the peaks
        if self.wt_pos is None:
            self.wt_pos = 1
        if self.wt_neg is None:
            self.wt_neg = 0.5

        # Construct gaussian label
        feature_sz = torch.Tensor([prediction.shape[-2], prediction.shape[-1]])
        gauss_label = get_gaussian_label(target_bb.view(-1,4).cpu(), self.sigma, self.kernel_sz, feature_sz,
                                         self.image_sz).to(prediction.device)
        if is_distractor is not None:
            gauss_label = gauss_label * (1 - is_distractor).view(-1, 1, 1).float()

        negative_mask = (gauss_label < self.negative_threshold).float()

        prediction = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])

        prediction = negative_mask * F.relu(prediction) + (1.0 - negative_mask) * prediction

        if is_distractor is not None:
            flat_distractor = is_distractor.view(-1, is_distractor.shape[0] * is_distractor.shape[1])
            flat_distractor_dash = 1-flat_distractor
            prediction[flat_distractor.byte().squeeze(dim=0)] = math.sqrt(self.wt_neg) * \
                prediction[flat_distractor.byte().squeeze(dim=0)]
            gauss_label[flat_distractor.byte().squeeze(dim=0)] = math.sqrt(self.wt_neg) * \
                gauss_label[flat_distractor.byte().squeeze(dim=0)]
            prediction = flat_distractor.t().unsqueeze(2)*prediction + \
                         flat_distractor_dash.t().unsqueeze(2)*prediction*torch.sqrt(self.wt_pos*gauss_label)
            gauss_label = flat_distractor.t().unsqueeze(2)*gauss_label + \
                         flat_distractor_dash.t().unsqueeze(2)*gauss_label*torch.sqrt(self.wt_pos*gauss_label)

        loss = self.error_metric(prediction, gauss_label)
        return loss


class TargetClassificationMarginLoss(nn.Module):
    """
    """
    def __init__(self, num_distractors=1, negative_threshold=0.3, pos_weight=1.0, mse_threshold=1, mse_weight=1):
        super(TargetClassificationMarginLoss, self).__init__()
        self.num_distractors = num_distractors
        self.negative_threshold = negative_threshold
        self.mse_loss = nn.MSELoss()
        self.mse_threshold = mse_threshold
        self.mse_weight = mse_weight
        self.pos_weight = pos_weight

    def forward(self, prediction, label, target_bb=None):
        prediction = prediction.view(-1, prediction.shape[-2]*prediction.shape[-1])
        label = label.view(-1, label.shape[-2]*label.shape[-1])

        negative_mask = (label < self.negative_threshold).float()
        prediction_masked = prediction * negative_mask

        distractor_val, _ = prediction_masked.topk(self.num_distractors, dim=1, sorted=False)
        label_val, label_ind = label.max(1)

        prediction_val = prediction[torch.arange(prediction.shape[0]), label_ind][label_val > self.negative_threshold]

        margin_loss = 1.0 - torch.min(prediction_val, torch.tensor([1.0], device=prediction.device)).mean() + \
                      torch.max(distractor_val, torch.tensor([0.0], device=prediction.device)).mean()/self.pos_weight

        residual = prediction - label
        residual_masked = residual * (residual.abs() > self.mse_threshold).float() * negative_mask
        mse_loss = self.mse_loss(residual_masked, torch.zeros_like(residual_masked))
        return margin_loss + mse_loss*self.mse_weight


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        # Mask invalid samples
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples

            loss = self.error_metric(prediction, positive_mask * label)

            loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


class LBHingeWeighted(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None, weights=(1.0,1.0)):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip
        self.weights = weights

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        negative_mask = math.sqrt(self.weights[1]) * (label < self.threshold).float()
        positive_mask = math.sqrt(self.weights[0]) * (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        norm_const = positive_mask.mean() + negative_mask.mean()

        # Mask invalid samples
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples

            loss = self.error_metric(prediction, positive_mask * label)

            loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)

        loss = loss / norm_const

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


class TargetClassificationDistractorLoss(nn.Module):
    """
    """
    def __init__(self, num_distractors=1, negative_threshold=0.01):
        super(TargetClassificationDistractorLoss, self).__init__()
        self.num_distractors = num_distractors
        self.negative_threshold = negative_threshold

    def forward(self, prediction, label, target_bb=None):
        prediction = prediction.view(-1, prediction.shape[-2]*prediction.shape[-1])
        label = label.view(-1, label.shape[-2]*label.shape[-1])

        negative_mask = (label < self.negative_threshold).float()
        prediction_masked = prediction * negative_mask

        distractor_val, _ = prediction_masked.topk(self.num_distractors, dim=1, sorted=False)

        margin_loss = torch.max(distractor_val, torch.tensor([0.0], device=prediction.device)).mean()

        return margin_loss


class CombinedLoss(nn.Module):
    """
    """
    def __init__(self, loss_functions, weights=None):
        super(CombinedLoss, self).__init__()
        self.loss_functions = loss_functions

        if weights is None:
            self.weights = [1.0 for _ in loss_functions]
        else:
            self.weights = weights

        assert len(self.loss_functions) == len(self.weights)

    def forward(self, prediction, label, target_bb=None):
        losses = [l(prediction, label) for l in self.loss_functions]

        weighted_losses = [l*w for l, w in zip(losses, self.weights)]
        return sum(weighted_losses)


class TargetClassificationBinaryLoss(nn.Module):
    """
    """
    def __init__(self, negative_threshold=0.01, positive_threshold=0.4, positive_weight=1.0, use_only_response_peak=False,
                 clip_positive=False):
        super(TargetClassificationBinaryLoss, self).__init__()
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        self.positive_weight = positive_weight
        self.use_only_response_peak = use_only_response_peak
        self.clip_positive = clip_positive

    def forward(self, prediction, label, target_bb=None):
        prediction = prediction.view(-1, prediction.shape[-2]*prediction.shape[-1])
        label = label.view(-1, label.shape[-2]*label.shape[-1])

        negative_mask = label < self.negative_threshold
        prediction_negative = F.relu(prediction[negative_mask])

        positive_mask = label > self.positive_threshold

        if not self.use_only_response_peak:
            prediction_positive = prediction[positive_mask]
        else:
            prediction_positive, _ = (prediction*positive_mask.float()).max(1)

        if self.clip_positive:
            prediction_positive = prediction_positive.clamp(max=1.0)

        negative_loss = F.mse_loss(prediction_negative, torch.zeros_like(prediction_negative), reduction='none')
        positive_loss = F.mse_loss(prediction_positive, torch.ones_like(prediction_positive), reduction='none')

        total_loss = (negative_loss.sum() + self.positive_weight*positive_loss.sum()) / \
                     (negative_loss.numel() + positive_loss.numel())
        return total_loss


class TargetClassificationBinaryLossIOU(nn.Module):
    """
    """
    def __init__(self, feat_stride, negative_threshold=0.25, positive_weight=1.0, clip_positive=False, normalize_size=False,
                 use_distance=False):
        super(TargetClassificationBinaryLossIOU, self).__init__()
        self.feat_stride = feat_stride
        self.negative_threshold = negative_threshold
        self.positive_weight = positive_weight
        self.clip_positive = clip_positive
        self.normalize_size = normalize_size
        self.use_distance = use_distance

    def forward(self, prediction, label, target_bb):
        target_center = target_bb[..., :2] + 0.5*target_bb[..., 2:]

        # Target center in feature scale
        target_center_feat = (target_center / self.feat_stride).view(-1, 2)
        target_sz_feat = (target_bb[..., 2:] / self.feat_stride).view(-1, 2)

        # Normalize if needed
        if self.normalize_size:
            target_sz_feat = (target_sz_feat[:, 0]*target_sz_feat[:, 1]).sqrt().unsqueeze(1).repeat(1, 2)

        # Intersection for each feature cell
        if not self.use_distance:
            del_x = target_sz_feat[:, 0:1] - (target_center_feat[:, 0:1] - torch.arange(0.0, prediction.shape[-1], device=prediction.device).view(1, -1)).abs()
            del_y = target_sz_feat[:, 1:2] - (target_center_feat[:, 1:2] - torch.arange(0.0, prediction.shape[-2],
                                                                                          device=prediction.device).view(1, -1)).abs()

            intersection = torch.max(del_x, torch.zeros(1, device=prediction.device)).view(del_x.shape[0], 1, -1)*\
                           torch.max(del_y, torch.zeros(1, device=prediction.device)).view(del_y.shape[0], -1, 1)

            iou = intersection / ((2*target_sz_feat[:, 0]*target_sz_feat[:, 1]).view(-1, 1, 1) - intersection)
            negative_mask = iou < self.negative_threshold
        else:
            del_x = (target_center_feat[:, 0:1] - torch.arange(0.0, prediction.shape[-1], device=prediction.device).view(1, -1)).view(target_center_feat.shape[0], 1, -1)
            del_y = (target_center_feat[:, 1:2] - torch.arange(0.0, prediction.shape[-2], device=prediction.device).view(1, -1)).view(target_center_feat.shape[0], -1, 1)

            dist = torch.sqrt(del_x * del_x + del_y * del_y)
            negative_mask = dist > self.negative_threshold

        prediction_negative = F.relu(prediction.view(-1, prediction.shape[-2], prediction.shape[-1])[negative_mask])

        prediction_positive = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])[torch.arange(target_center_feat.shape[0], device=target_center_feat.device), target_center_feat[:, 1].round().long(), target_center_feat[:, 0].round().long()]

        if self.clip_positive:
            prediction_positive = prediction_positive.clamp(max=1.0)

        negative_loss = F.mse_loss(prediction_negative, torch.zeros_like(prediction_negative), reduction='none')
        positive_loss = F.mse_loss(prediction_positive, torch.ones_like(prediction_positive), reduction='none')

        total_loss = (negative_loss.sum() + self.positive_weight*positive_loss.sum()) / \
                     (negative_loss.numel() + positive_loss.numel())
        return total_loss


class TargetClassificationLossIOU(nn.Module):
    """
    """
    def __init__(self, feat_stride, negative_threshold=0.25, positive_weight=1.0, use_hinge=False):
        super(TargetClassificationLossIOU, self).__init__()
        self.feat_stride = feat_stride
        self.negative_threshold = negative_threshold
        self.positive_weight = positive_weight
        self.use_hinge = use_hinge

    def forward(self, prediction, label, target_bb):
        target_center = target_bb[..., :2] + 0.5*target_bb[..., 2:]

        # Target center in feature scale
        target_center_feat = (target_center / self.feat_stride).view(-1, 2)
        target_sz_feat = (target_bb[..., 2:] / self.feat_stride).view(-1, 2)

        del_x = target_sz_feat[:, 0:1] - (target_center_feat[:, 0:1] - torch.arange(0.0, prediction.shape[-1], device=prediction.device).view(1, -1)).abs()
        del_y = target_sz_feat[:, 1:2] - (target_center_feat[:, 1:2] - torch.arange(0.0, prediction.shape[-2],
                                                                                      device=prediction.device).view(1, -1)).abs()

        intersection = torch.max(del_x, torch.zeros(1, device=prediction.device)).view(del_x.shape[0], 1, -1)*\
                       torch.max(del_y, torch.zeros(1, device=prediction.device)).view(del_y.shape[0], -1, 1)

        iou = intersection / ((2*target_sz_feat[:, 0]*target_sz_feat[:, 1]).view(-1, 1, 1) - intersection)
        negative_mask = (iou < self.negative_threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = prediction.view(-1, prediction.shape[-2], prediction.shape[-1])

        if self.use_hinge:
            prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        elem_loss = F.mse_loss(prediction, iou, reduction='none')

        weighted_loss = elem_loss*negative_mask + self.positive_weight*elem_loss*positive_mask

        total_loss = weighted_loss.mean()
        return total_loss


class LBHingev2(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=None, threshold=None, return_per_sequence=False):
        super().__init__()

        if error_metric is None:
            if return_per_sequence:
                reduction = 'none'
            else:
                reduction = 'mean'
            error_metric = nn.MSELoss(reduction=reduction)

        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100

        self.return_per_sequence = return_per_sequence

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        assert prediction.dim() == 4 and label.dim() == 4
        
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        # Mask invalid samples
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples

            loss = self.error_metric(prediction, positive_mask * label)

            if self.return_per_sequence:
                loss = loss.mean((-2, -1))
            else:
                loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)

            if self.return_per_sequence:
                loss = loss.mean((-2, -1))

        return loss


class LBHingeGen(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None, label_function_params=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip
        self.label_function_params = label_function_params

    def forward(self, prediction, label=None, target_bb=None, valid_samples=None):
        if label is None:
            label = prutils.gaussian_label_function(target_bb.view(-1, 4).cpu(),
                                                    self.label_function_params['sigma_factor'],
                                                    self.label_function_params['kernel_sz'],
                                                    self.label_function_params['feature_sz'],
                                                    self.label_function_params['image_sz'],
                                                    end_pad_if_even=self.label_function_params.get(
                                                        'end_pad_if_even', True))
            label = label.view(target_bb.shape[0], target_bb.shape[1], label.shape[-2], label.shape[-1]).to(target_bb.device)
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        # Mask invalid samples
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples

            loss = self.error_metric(prediction, positive_mask * label)

            loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss
