import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


class SegAttention(nn.Module):
    """ Network module for IoU prediction. Refer to the paper for an illustration of the architecture."""
    def __init__(self, segmentation_net, feature_layers, refinement_dim):
        super().__init__()
        # Layers for modulation vector
        self.conv1_l2_r = conv(feature_layers['layer2'], 128, kernel_size=3, stride=1)
        self.prroi_pool_l2_r = PrRoIPool2D(3, 3, 1 / 8)
        self.fc_l2_r = conv(128, 256, kernel_size=3, stride=1, padding=0)

        self.conv1_l3_r = conv(feature_layers['layer3'], 256, kernel_size=3, stride=1)
        self.prroi_pool4r = PrRoIPool2D(1, 1, 1/16)

        self.modulation_predictor = nn.ModuleDict()
        for lname, dim in refinement_dim.items():
            self.modulation_predictor[lname] = conv(256 + 256, dim, kernel_size=1, stride=1, padding=0)

        self.refinement_feat = nn.ModuleDict()
        for lname, dim in refinement_dim.items():
            self.refinement_feat[lname] = conv(feature_layers[lname], dim, kernel_size=3, stride=1)

        self.segmentation_net = segmentation_net

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, ref_feat, test_feat, ref_bb):
        layer_names = list(test_feat.keys())

        assert test_feat[layer_names[0]].dim() == 5, 'Expect 5  dimensional feat1'

        # Extract first train sample
        ref_feat = OrderedDict({lname: f[0, ...] for lname, f in ref_feat.items()})
        ref_bb = ref_bb[0, ...]

        # Get modulation vector
        modulation_vec = self.get_modulation(ref_feat, ref_bb)

        # Predict segmentation
        return self.predict_segmentation(modulation_vec, test_feat)

    def predict_segmentation(self, modulation_vec, feat):
        layer_names = list(feat.keys())

        num_test_images = feat[layer_names[0]].shape[0]
        num_sequences = 1 if feat[layer_names[0]].dim() == 4 else feat[layer_names[0]].shape[1]

        # Extract features and do modulation
        seg_feat = OrderedDict()
        for lname, layer in self.refinement_feat.items():
            feat_l = layer(feat[lname].view(-1, *feat[lname].shape[-3:]))
            if num_sequences is None or num_sequences == 1:
                seg_feat[lname] = modulation_vec[lname] * feat_l
            else:
                feat_l = modulation_vec[lname].view(1, num_sequences, -1, 1, 1) * feat_l.view(num_test_images, num_sequences, *feat_l.shape[-3:])
                seg_feat[lname] = feat_l.view(-1, *feat_l.shape[-3:])

        seg = self.segmentation_net(seg_feat)

        seg = seg.view(num_test_images, num_sequences, *seg.shape[-2:])

        return seg


    def get_modulation(self, feat, bb):
        c3_r = self.conv1_l2_r(feat['layer2'])

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.Tensor([x for x in range(batch_size)]).view(batch_size, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool_l2_r(c3_r, roi1)

        c4_r = self.conv1_l3_r(feat['layer3'])
        roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc_l2_r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        modulation_vec = OrderedDict()
        for lname, predictor in self.modulation_predictor.items():
            modulation_vec[lname] = predictor(fc34_r)

        return modulation_vec