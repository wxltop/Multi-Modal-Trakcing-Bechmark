import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seg2BoxSigmoid(nn.Module):
    def __init__(self, pixel_median, pixel_range=1.0, object_pixel_prob=0.5):
        super().__init__()
        self.pixel_median = pixel_median
        self.pixel_range = pixel_range
        self.score_offset = -math.log(1/object_pixel_prob - 1)

    def forward(self, seg_scores):

        y_sz = seg_scores.shape[-2]
        x_sz = seg_scores.shape[-1]

        # segmentation = segmentation - self.noise_threshold
        # mean_seg = segmentation.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).detach()
        # segmentation = segmentation + (2*self.pixel_median/(x_sz*y_sz) - mean_seg).clamp(min=0)

        # segmentation = F.relu(segmentation, inplace=True)

        segmentation = torch.sigmoid(seg_scores - self.score_offset).view(-1, 1, *seg_scores.shape[-2:])

        seg_y = segmentation.sum(dim=-1)
        seg_x = segmentation.sum(dim=-2)
        cum_y = seg_y.cumsum(dim=-1)
        cum_x = seg_x.cumsum(dim=-1)

        zero_bias = torch.sigmoid((segmentation.new_zeros(1) - self.pixel_median) / self.pixel_range)

        cum_min_y = torch.sigmoid((cum_y - self.pixel_median) / self.pixel_range) - zero_bias
        cum_min_x = torch.sigmoid((cum_x - self.pixel_median) / self.pixel_range) - zero_bias
        cum_max_y = torch.sigmoid(((seg_y - cum_y) + (cum_y[...,-1:] - self.pixel_median)) / self.pixel_range) - zero_bias
        cum_max_x = torch.sigmoid(((seg_x - cum_x) + (cum_x[...,-1:] - self.pixel_median)) / self.pixel_range) - zero_bias

        # Normalize cumulative distributions
        cum_min_y = cum_min_y / cum_min_y[...,-1:]
        cum_min_x = cum_min_x / cum_min_x[...,-1:]
        cum_max_y = cum_max_y / cum_max_y[...,:1]
        cum_max_x = cum_max_x / cum_max_x[...,:1]

        y1 = y_sz - cum_min_y.sum(dim=-1)
        x1 = x_sz - cum_min_x.sum(dim=-1)
        y2 = cum_max_y.sum(dim=-1)
        x2 = cum_max_x.sum(dim=-1)

        bbox = torch.stack((x1, y1, x2, y2), dim=-1)

        return bbox


class SegEdgeDetector(nn.Module):
    def __init__(self, object_pixel_prob=0.5, kernel_length=10, seg_thickness=5):
        super().__init__()
        self.score_offset = -math.log(1/object_pixel_prob - 1)
        self.kernel_length = kernel_length
        self.seg_thickness = seg_thickness

    def forward(self, seg_scores):
        kernel = seg_scores.new_zeros(1,1,2*self.kernel_length+1)
        kernel[0,0,:self.kernel_length] = -1/self.kernel_length
        kernel[0,0,self.kernel_length+1:] = 1/self.kernel_length

        object_pixels = torch.sigmoid(seg_scores - self.score_offset).view(-1, 1, *seg_scores.shape[-2:])

        sum_y = torch.tanh(object_pixels.sum(dim=-1) / self.seg_thickness)
        sum_x = torch.tanh(object_pixels.sum(dim=-2) / self.seg_thickness)

        response_y = F.conv1d(sum_y, kernel, padding=self.kernel_length).view(*seg_scores.shape[:2], -1)
        response_x = F.conv1d(sum_x, kernel, padding=self.kernel_length).view(*seg_scores.shape[:2], -1)

        return response_x, response_y


def bbox_from_edge_scores(response_x, response_y):
    x1 = torch.argmax(response_x, dim=-1)
    y1 = torch.argmax(response_y, dim=-1)
    x2 = torch.argmin(response_x, dim=-1)
    y2 = torch.argmin(response_y, dim=-1)

    bbox = torch.stack((x1, y1, x2, y2), dim=-1)
    return bbox.float()