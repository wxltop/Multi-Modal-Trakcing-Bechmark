import numpy as np
import torch
import random
import math
from random import gauss


# !OUTDATED!
def calc_iou(box_a: np.array, box_b: np.array) -> float:
    """ Calculates IoU overlap of every box in box_a with the corresponding box in box_b.
    args:
        box_a - numpy array of shape [x1, x2, x3 .....,, xn, 4]
        box_b - numpy array of shape [x1, x2, x3 .....,, xn, 4]

    returns:
        np.array - array of shape [x1, x2, x3 .....,, xn], containing IoU overlaps
    """

    eps = 1e-10

    x1 = np.maximum(box_a[..., 0], box_b[..., 0])
    y1 = np.maximum(box_a[..., 1], box_b[..., 1])
    x2 = np.minimum(box_a[..., 0] + box_a[..., 2], box_b[..., 0] + box_b[..., 2])
    y2 = np.minimum(box_a[..., 1] + box_a[..., 3], box_b[..., 1] + box_b[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    area_intersect = w*h
    area_a = box_a[..., 2] * box_a[..., 3]
    area_b = box_b[..., 2] * box_b[..., 3]

    area_overlap = area_a + area_b - area_intersect + eps

    iou = area_intersect / area_overlap
    return iou


# !OUTDATED!
def get_gaussian_loss(box: np.array, box_per: np.array, sig=None):
    if sig is None:
        sig_x = 0.2
        sig_y = 0.2
        sig_w = 0.12
        sig_h = 0.12
    else:
        sig_x, sig_y, sig_w, sig_h = sig

    cx = box[0] + 0.5*box[2]
    cy = box[1] + 0.5*box[3]
    w = box[2]
    h = box[3]

    cx_p = box_per[0] + 0.5 * box_per[2]
    cy_p = box_per[1] + 0.5 * box_per[3]
    w_p = box_per[2]
    h_p = box_per[3]

    def _gauss(x, s):
        return math.exp(-0.5*(x/(s+1))**2)

    return _gauss(cx_p - cx, w*sig_x) * _gauss(cy_p - cy, h*sig_y) * _gauss(w_p - w, w*sig_w) * _gauss(h_p - h, h*sig_h)


# !OUTDATED!
def perturb_box(box: np.array, min_iou=0.5, sigma_factor=0.1, p_ar_jitter=None, p_scale_jitter=None,
                use_gaussian=False, sig=None):
    """ Clean this up!!!"""
    if isinstance(sigma_factor, list):
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, np.ndarray):
        c_sigma_factor = c_sigma_factor * np.ones(4)

    ar_jitter = False
    scale_jitter = False

    if p_ar_jitter is not None and random.uniform(0, 1) < p_ar_jitter:
        ar_jitter = True
    elif p_scale_jitter is not None and random.uniform(0, 1) < p_scale_jitter:
        scale_jitter = True

    perturb_factor = np.sqrt(box[2]*box[3])*c_sigma_factor

    if ar_jitter or scale_jitter:
        perturb_factor[0:2] = np.sqrt(box[2]*box[3])*0.05

    for i_ in range(100):
        c_x = box[0] + 0.5*box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = gauss(c_x, perturb_factor[0])
        c_y_per = gauss(c_y, perturb_factor[1])

        if p_scale_jitter:
            w_per = gauss(box[2], perturb_factor[2])
            h_per = box[3] * w_per / (box[2] + 1)
        else:
            w_per = gauss(box[2], perturb_factor[2])
            h_per = gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2]*np.random.uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3]*np.random.uniform(0.15, 0.5)

        box_per = np.array([c_x_per - 0.5*w_per, c_y_per - 0.5*h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2]*np.random.uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3]*np.random.uniform(0.15, 0.5)

        if use_gaussian:
            iou = get_gaussian_loss(box, box_per, sig)
        else:
            iou = calc_iou(box, box_per)

        if iou > min_iou:
            return box_per, iou, 0

        # Reduce perturb factor
        perturb_factor *= 0.9

    return box_per, iou, 1


def rect_to_rel(bb, sz_norm=None):
    """Convert standard rectangular parametrization of the bounding box [x, y, w, h]
    to relative parametrization [cx/sw, cy/sh, log(w), log(h)], where [cx, cy] is the center coordinate.
    args:
        bb  -  N x 4 tensor of boxes.
        sz_norm  -  [N] x 2 tensor of value of [sw, sh] (optional). sw=w and sh=h if not given.
    """

    c = bb[..., :2] + 0.5 * bb[..., 2:]
    if sz_norm is None:
        c_rel = c / bb[..., 2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[..., 2:])
    return torch.cat((c_rel, sz_rel), dim=-1)


def rel_to_rect(bb, sz_norm=None):
    """Inverts the effect of rect_to_rel. See above."""

    sz = torch.exp(bb[..., 2:])
    if sz_norm is None:
        c = bb[..., :2] * sz
    else:
        c = bb[..., :2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)


def masks_to_bboxes(mask, fmt='c'):

    """ Convert a mask tensor to one or more bounding boxes.
    Note: This function is a bit new, make sure it does what it says.  /Andreas
    :param mask: Tensor of masks, shape = (..., H, W)
    :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
                             't' => "top left + size" or (x_left, y_top, width, height)
                             'v' => "vertices" or (x_left, y_top, x_right, y_bottom)
    :return: tensor containing a batch of bounding boxes, shape = (..., 4)
    """
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    for m in mask:
        mx = m.sum(dim=-2).nonzero()
        my = m.sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)


def masks_to_bboxes_multi(mask, ids, fmt='c'):
    assert mask.dim() == 2
    bboxes = []

    for id in ids:
        mx = (mask == id).sum(dim=-2).nonzero()
        my = (mask == id).float().sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]

        bb = torch.tensor(bb, dtype=torch.float32, device=mask.device)

        x1 = bb[:2]
        s = bb[2:] - x1 + 1

        if fmt == 'v':
            pass
        elif fmt == 'c':
            bb = torch.cat((x1 + 0.5 * s, s), dim=-1)
        elif fmt == 't':
            bb = torch.cat((x1, s), dim=-1)
        else:
            raise ValueError("Undefined bounding box layout '%s'" % fmt)
        bboxes.append(bb)

    return bboxes
