import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import ltr.models.loss.lovasz_loss as lovasz_loss


class BBSegBCE(nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, input, target):
        L = super().forward(input, target)
        L = (1 - target) * L
        return L.mean()


class BBEdgeLoss(nn.Module):
    def __init__(self, metric=nn.MSELoss(), kernel_size=10, ignore_threshold=None):
        super().__init__()
        self.metric = metric
        self.kernel_size = kernel_size
        self.ignore_threshold = ignore_threshold
        if ignore_threshold is None:
            self.metric.reduction = 'elementwise_mean'
        else:
            self.metric.reduction = 'none'

    def forward(self, response_x, response_y, target_bbox):
        response_y = response_y.view(-1, response_y.shape[-1])
        response_x = response_x.view(-1, response_x.shape[-1])

        target_bbox = target_bbox.view(-1, target_bbox.shape[-1])

        coord_y = torch.arange(response_y.shape[-1], dtype=response_y.dtype, device=response_y.device).view(1,-1)
        coord_x = torch.arange(response_x.shape[-1], dtype=response_x.dtype, device=response_x.device).view(1,-1)

        x1 = target_bbox[:,0:1]
        y1 = target_bbox[:,1:2]
        x2 = target_bbox[:,2:3]
        y2 = target_bbox[:,3:4]

        target_y = F.relu(1.0 - torch.abs(coord_y - y1) / self.kernel_size) - \
                   F.relu(1.0 - torch.abs(coord_y - y2) / self.kernel_size)

        target_x = F.relu(1.0 - torch.abs(coord_x - x1) / self.kernel_size) - \
                   F.relu(1.0 - torch.abs(coord_x - x2) / self.kernel_size)

        if self.ignore_threshold is None:
            return self.metric(response_y, target_y) + self.metric(response_x, target_x)

        mask_y = (target_y < -self.ignore_threshold) | (target_y > self.ignore_threshold) | (target_y.abs() < 1e-4)
        mask_x = (target_x < -self.ignore_threshold) | (target_x > self.ignore_threshold) | (target_x.abs() < 1e-4)

        loss = (mask_y.float() * self.metric(response_y, target_y)).mean() + (mask_x.float() * self.metric(response_x, target_x)).mean()

        return loss


class BBSegLoss(nn.Module):
    def __init__(self, min_thickness=1/5, object_pixel_prob=0.5, bce_weight=1.0, pos_bce_weight=0.0):
        super().__init__()
        self.target_score_offset = -math.log(1/object_pixel_prob - 1)
        self.min_thickness = min_thickness
        self.bce_weight = bce_weight
        self.pos_bce_weight = pos_bce_weight

    def forward(self, input, target):

        bce_loss = F.binary_cross_entropy_with_logits(input, target,  pos_weight=max(self.pos_bce_weight, 0))

        neg_bce_loss = 0
        if self.pos_bce_weight < 0:
            neg_bce_loss = F.binary_cross_entropy_with_logits(input, 1-target, pos_weight=0)

        seg = torch.sigmoid(input - self.target_score_offset) * target
        seg_y = seg.sum(dim=-1)
        seg_x = seg.sum(dim=-2)

        target_th_y = self.min_thickness * target.sum(dim=-1).float()
        target_th_x = self.min_thickness * target.sum(dim=-2).float()

        target_loss = F.relu(target_th_y - seg_y - 1e-6).mean() + F.relu(target_th_x - seg_x - 1e-6).mean()

        loss = target_loss + self.bce_weight * bce_loss - self.pos_bce_weight * neg_bce_loss

        return loss


class LovaszSegLoss(nn.Module):
    def __init__(self, classes=[1,], per_image=True):
        super().__init__()

        self.classes = classes
        self.per_image=per_image

    def forward(self, input, target):
        return lovasz_loss.lovasz_softmax(probas=torch.sigmoid(input), labels=target, per_image=self.per_image, classes=self.classes)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device = None,
            dtype = None,
            eps = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        #>>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        #>>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes, shape[1], shape[2])).to(device)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.
    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis bg_p = torch.prod(1 - segmentation_maps_t, dim=0).clamp(eps, 1.0 - eps)
    input_sig: torch.Tensor = torch.sigmoid(input)#F.softmax(input, dim=1) + eps
    input_soft = torch.cat([1.0-input_sig, input_sig], dim=1).clamp(eps, 1.0)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input_soft.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        #>>> N = 5  # num_classes
        #>>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        #>>> loss = kornia.losses.FocalLoss(**kwargs)
        #>>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        #>>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        #>>> output = loss(input, target)
        #>>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-7

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target.long().squeeze(dim=1), self.alpha, self.gamma, self.reduction, self.eps)

class LovaszSegLossMC(nn.Module):
    def __init__(self, classes='present', per_image=True):
        super().__init__()

        self.classes = classes
        self.per_image=per_image

    def forward(self, input, target):
        return lovasz_loss.lovasz_softmax(probas=torch.softmax(input, dim=1), labels=target, per_image=self.per_image, classes=self.classes)


class TargetClassificationSegLossLovasz(nn.Module):

    def __init__(self, feat_stride=16, upsample_residuals=False, per_image=True):
        super().__init__()

        self.feat_stride = feat_stride
        self.upsample_residuals = upsample_residuals
        self.per_image=per_image

    def resample(self, t, H, W):
        return F.interpolate(t, (H, W), mode='bilinear', align_corners=False)


    def forward(self, input, target):
        # Compute data residual
        if self.upsample_residuals:
            h,w = target.shape[-2:]
            input = F.interpolate(input, (h,w), mode='bilinear', align_corners=False)
        else:
            h, w = input.shape[-2:]
            target = F.interpolate(target, (h,w), mode='bilinear', align_corners=False)
            target = (target > 0.5).float()

        #scores_act = 2*input - 1#2*self.score_activation(input, target) - 1
        loss = lovasz_loss.lovasz_hinge(2*input-1.0, target, per_image=self.per_image)

        return loss