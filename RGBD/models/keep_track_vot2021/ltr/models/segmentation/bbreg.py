import torch
import torch.nn as nn
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.layers.blocks import LinearBlock
from ltr.models.bbreg.fpn_bbr_net import convert_bb_delta_to_abs_coord


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class BBRNet(nn.Module):
    def __init__(self, ref_input_dim=256, ref_feat_stride=16, ref_pool_sz=3, test_pool_sz=5,
                 test_input_dim=256, test_feat_stride=8):
        super().__init__()
        # _r for reference, _t for test
        self.conv1_r = conv(ref_input_dim, 256, kernel_size=3, stride=1)

        self.conv1_t = conv(test_input_dim, 256, kernel_size=3, stride=1)
        self.conv2_t = conv(256, 256, kernel_size=3, stride=1)

        self.prroi_pool_r = PrRoIPool2D(ref_pool_sz, ref_pool_sz, 1.0/ref_feat_stride)
        self.prroi_pool_t = PrRoIPool2D(test_pool_sz, test_pool_sz, 1.0/test_feat_stride)

        self.fc1_r = conv(256, 256, kernel_size=ref_pool_sz, stride=1, padding=0)
        # self.fc2_r = conv(1024, 256, kernel_size=1, stride=1, padding=0)

        self.fc1_rt = LinearBlock(256, 1024, test_pool_sz)

        # self.fc2_rt = LinearBlock(1024, 1024, 1)
        self.bb_predictor = nn.Linear(1024, 4, bias=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization,
                # which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1).
                # So we use the same initialization here.
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        # feat1 and feat2 should be tuples
        assert bb1.dim() == 3
        assert proposals2.dim() == 4

        num_images = proposals2.shape[0]
        num_sequences = proposals2.shape[1]

        # Extract first train sample
        feat1 = [f[0,...] if f.dim()==5 else f.view(-1, num_sequences, *f.shape[-3:])[0,...] for f in feat1]
        bb1 = bb1[0,...]

        # Get modulation vector
        modulation = self.get_modulation(feat1, bb1)

        iou_feat = self.get_iou_feat(feat2)

        modulation = [f.view(1, num_sequences, -1).repeat(num_images, 1, 1).view(num_sequences*num_images, -1) for f in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        bb_delta = self.predict_bb_delta(modulation, iou_feat, proposals2)
        return bb_delta.view(num_images, num_sequences, proposals2.shape[1], 4)

    def predict_bb(self, modulation, feat, proposals):
        bb_delta = self.predict_bb_delta(modulation, feat, proposals)

        # Convert to absolute co-ord
        return convert_bb_delta_to_abs_coord(proposals, bb_delta)

    def predict_bb_delta(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        modulation_vector, = modulation
        c2_t, = feat

        batch_size = c2_t.size()[0]

        # Modulation
        c2_t_att = c2_t * modulation_vector.view(batch_size, -1, 1, 1)

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(c2_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi_t = torch.cat((batch_index.view(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi_t = roi_t.view(-1, 5).to(proposals_xyxy.device)

        roi_feat_t = self.prroi_pool_t(c2_t_att, roi_t)

        fc1_rt = self.fc1_rt(roi_feat_t)

        bb_delta = self.bb_predictor(fc1_rt).view(batch_size, num_proposals_per_batch, 4)

        return bb_delta

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        feat_r, = feat

        c1_r = self.conv1_r(feat_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi1_feat_r = self.prroi_pool_r(c1_r, roi1)

        modulation_vector = self.fc1_r(roi1_feat_r)

        return modulation_vector,

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat_backbone, = feat2

        feat_backbone = feat_backbone.view(-1, *feat_backbone.shape[-3:]) if feat_backbone.dim()==5 else feat_backbone

        c2_t = self.conv2_t(self.conv1_t(feat_backbone))

        return c2_t,
