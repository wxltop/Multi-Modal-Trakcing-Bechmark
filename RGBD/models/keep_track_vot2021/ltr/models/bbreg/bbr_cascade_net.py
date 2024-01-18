import torch.nn as nn
import torch
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def convert_bb_delta_to_abs_coord(init_box, bb_delta):
    init_center = init_box[:, :, :2] + 0.5*init_box[:, :, 2:]

    new_center = init_center + bb_delta[:, :, :2] * init_box[:, :, 2:]
    new_sz = bb_delta[:, :, 2:].exp() * init_box[:, :, 2:]

    new_box = torch.cat((new_center - 0.5*new_sz, new_sz), dim=2)
    return new_box


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


class MODPredictorStageL23(nn.Module):
    def __init__(self, pred_input_dim, pred_inter_dim):
        super().__init__()
        self.prroi_pool3r = PrRoIPool2D(3, 3, 1 / 8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1 / 8)

        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)

        self.prroi_pool4r = PrRoIPool2D(1, 1, 1 / 16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)

        self.fc34_3r = conv(256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv(256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)

        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

        self.bb_predictor = nn.Linear(pred_inter_dim[0] + pred_inter_dim[1], 4, bias=True)

    def get_modulation(self, feat, bb):
        c3_r, c4_r = feat

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)

        return fc34_3_r, fc34_4_r

    def predict_bb_delta(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        fc34_3_r, fc34_4_r = modulation
        c3_t, c4_t = feat

        batch_size = c3_t.size()[0]

        # Modulation
        c3_t_att = c3_t * fc34_3_r.view(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.view(batch_size, -1, 1, 1)

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi2 = torch.cat((batch_index.view(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi2 = roi2.view(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)

        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)

        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)

        bb_delta = self.bb_predictor(fc34_rt_cat).view(batch_size, num_proposals_per_batch, 4)

        return bb_delta


class BBRNetModCascadeL23(nn.Module):
    def __init__(self, num_stages=3, input_dim=(128,256), pred_input_dim=(256,256), pred_inter_dim=(256,256)):
        super().__init__()
        self.num_stages = num_stages

        # _r for reference, _t for test
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)

        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)

        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)

        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)

        for i in range(num_stages):
            self.add_module('stage{}'.format(i), MODPredictorStageL23(pred_input_dim, pred_inter_dim))

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
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

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

        modulation = [[f.view(1, num_sequences, -1).repeat(num_images, 1, 1).view(num_sequences*num_images, -1) for f in f_t] for f_t in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        bb_delta, bb_proposals = self.predict_bb_delta(modulation, iou_feat, proposals2)
        bb_delta = [b.view(num_images, num_sequences, proposals2.shape[1], 4) for b in bb_delta]
        bb_proposals = [b.view(num_images, num_sequences, proposals2.shape[1], 4) for b in bb_proposals]
        return bb_delta, bb_proposals

    def predict_bb(self, modulation, feat, proposals):
        for i in range(self.num_stages):
            bb_delta = getattr(self, 'stage{}'.format(i)).predict_bb_delta(modulation[i], feat, proposals)

            new_proposals = convert_bb_delta_to_abs_coord(proposals, bb_delta.clone().detach())
            proposals = new_proposals.detach()

        return new_proposals

    def predict_bb_delta(self, modulation, feat, proposals):
        bb_delta_all = []
        proposals_all = []
        for i in range(self.num_stages):
            bb_delta = getattr(self, 'stage{}'.format(i)).predict_bb_delta(modulation[i], feat, proposals)

            bb_delta_all.append(bb_delta)
            proposals_all.append(proposals.clone())

            new_proposals = convert_bb_delta_to_abs_coord(proposals, bb_delta.clone().detach())

            proposals_sz = new_proposals[:, :, 2:].prod(-1).sqrt()
            min_xy = new_proposals[:, :, :2].min(-1)[0]
            max_xy = new_proposals[:, :, :2].max(-1)[0]
            bad_proposals = (proposals_sz > 1000) | (min_xy < -1000) | (max_xy > 1000)
            new_proposals[bad_proposals, :] *= 0
            proposals = new_proposals.detach()

        return bb_delta_all, proposals_all

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        feat3_r, feat4_r = feat
        c3_r = self.conv3_1r(feat3_r)
        c4_r = self.conv4_1r(feat4_r)
        c = (c3_r, c4_r)
        modulation_vector = [getattr(self, 'stage{}'.format(i)).get_modulation(c, bb) for i in range(self.num_stages)]

        return modulation_vector

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [f.view(-1, *f.shape[-3:]) if f.dim() == 5 else f for f in feat2]
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))

        return c3_t, c4_t


class FPNCatPredictorConvStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.prroi_pool_r = PrRoIPool2D(7, 7, 1 / 8)
        self.prroi_pool_t = PrRoIPool2D(7, 7, 1 / 8)

        self.proj = conv(256 + 256, 256, kernel_size=1, stride=1, padding=0)
        self.conv1 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv(256, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = conv(256, 1024, kernel_size=7, stride=1, padding=0)

        self.bb_predictor = nn.Linear(1024, 4, bias=True)

    def get_modulation(self, feat, bb):
        c3_r, = feat

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool_r(c3_r, roi1)

        return roi3r,

    def predict_bb_delta(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        roi3r, = modulation
        c3_t, = feat

        batch_size = c3_t.size()[0]

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi2 = torch.cat((batch_index.view(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi2 = roi2.view(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool_t(c3_t, roi2)

        roi3t = roi3t.view(batch_size, num_proposals_per_batch, *roi3t.shape[-3:])
        roi3r = roi3r.view(batch_size, 1, *roi3r.shape[-3:]).repeat(1, num_proposals_per_batch, 1, 1, 1)

        roi_cat = torch.cat((roi3t, roi3r), dim=2)
        roi_cat = roi_cat.view(-1, *roi_cat.shape[-3:])

        roi_cat_proj = self.proj(roi_cat)
        c4 = self.conv4(self.conv3(self.conv2(self.conv1(roi_cat_proj))))

        fc1 = self.fc1(c4)
        fc1 = fc1.view(*fc1.shape[:2])
        bb_delta = self.bb_predictor(fc1).view(batch_size, num_proposals_per_batch, 4)

        return bb_delta


class FPNCatPredictorConv(nn.Module):
    def __init__(self, num_stages=3):
        super().__init__()
        self.num_stages = num_stages

        for i in range(num_stages):
            self.add_module('stage{}'.format(i), FPNCatPredictorConvStage())

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
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

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

        modulation = [[f.view(1, num_sequences, *f.shape[-3:]).repeat(num_images, 1, 1, 1, 1).
                           view(num_sequences*num_images, *f.shape[-3:]) for f in f_t] for f_t in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        bb_delta, bb_proposals = self.predict_bb_delta(modulation, iou_feat, proposals2)
        bb_delta = [b.view(num_images, num_sequences, proposals2.shape[1], 4) for b in bb_delta]
        bb_proposals = [b.view(num_images, num_sequences, proposals2.shape[1], 4) for b in bb_proposals]
        return bb_delta, bb_proposals

    def predict_bb(self, modulation, feat, proposals):
        for i in range(self.num_stages):
            bb_delta = getattr(self, 'stage{}'.format(i)).predict_bb_delta(modulation[i], feat, proposals)

            new_proposals = convert_bb_delta_to_abs_coord(proposals, bb_delta.clone().detach())
            proposals = new_proposals.detach()

        return new_proposals

    def predict_bb_delta(self, modulation, feat, proposals):
        bb_delta_all = []
        proposals_all = []
        for i in range(self.num_stages):
            bb_delta = getattr(self, 'stage{}'.format(i)).predict_bb_delta(modulation[i], feat, proposals)

            bb_delta_all.append(bb_delta)
            proposals_all.append(proposals.clone())

            new_proposals = convert_bb_delta_to_abs_coord(proposals, bb_delta.clone().detach())

            proposals_sz = new_proposals[:, :, 2:].prod(-1).sqrt()
            min_xy = new_proposals[:, :, :2].min(-1)[0]
            max_xy = new_proposals[:, :, :2].max(-1)[0]
            bad_proposals = (proposals_sz > 1000) | (min_xy < -1000) | (max_xy > 1000)
            new_proposals[bad_proposals, :] *= 0
            proposals = new_proposals.detach()

        return bb_delta_all, proposals_all

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        c = feat
        modulation_vector = [getattr(self, 'stage{}'.format(i)).get_modulation(c, bb) for i in range(self.num_stages)]

        return modulation_vector

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [f.view(-1, *f.shape[-3:]) if f.dim() == 5 else f for f in feat2]

        return feat2


class CatPredictorConvStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.prroi_pool_3 = PrRoIPool2D(7, 7, 1 / 8)
        self.prroi_pool_4 = PrRoIPool2D(7, 7, 1 / 16)

        self.proj1 = conv(256 + 256, 256, kernel_size=1, stride=1, padding=0)
        self.proj2 = conv(256 + 256, 256, kernel_size=1, stride=1, padding=0)
        self.conv1 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv(256, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = conv(256, 1024, kernel_size=7, stride=1, padding=0)

        self.bb_predictor = nn.Linear(1024, 4, bias=True)

    def get_modulation(self, feat, bb):
        c3_r, c4_r = feat

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool_3(c3_r, roi1)
        roi4r = self.prroi_pool_4(c4_r, roi1)

        return roi3r, roi4r

    def predict_bb_delta(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        roi3r, roi4r = modulation
        c3_t, c4_t = feat

        batch_size = c3_t.size()[0]

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi2 = torch.cat((batch_index.view(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi2 = roi2.view(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool_3(c3_t, roi2)
        roi4t = self.prroi_pool_4(c4_t, roi2)

        roi3t = roi3t.view(batch_size, num_proposals_per_batch, *roi3t.shape[-3:])
        roi3r = roi3r.view(batch_size, 1, *roi3r.shape[-3:]).repeat(1, num_proposals_per_batch, 1, 1, 1)

        roi4t = roi4t.view(batch_size, num_proposals_per_batch, *roi4t.shape[-3:])
        roi4r = roi4r.view(batch_size, 1, *roi4r.shape[-3:]).repeat(1, num_proposals_per_batch, 1, 1, 1)

        roi_r_cat = torch.cat((roi3r, roi4r), dim=2)
        roi_t_cat = torch.cat((roi3t, roi4t), dim=2)

        roi_r_cat = roi_r_cat.view(-1, *roi_r_cat.shape[-3:])
        roi_t_cat = roi_t_cat.view(-1, *roi_t_cat.shape[-3:])

        roi_r_cat_proj = self.proj1(roi_r_cat)
        roi_t_cat_proj = self.proj2(roi_t_cat)

        roi_cat = torch.cat((roi_r_cat_proj, roi_t_cat_proj), dim=1)
        roi_cat = roi_cat.view(-1, *roi_cat.shape[-3:])

        roi_cat_proj = self.proj2(roi_cat)
        c4 = self.conv4(self.conv3(self.conv2(self.conv1(roi_cat_proj))))

        fc1 = self.fc1(c4)
        fc1 = fc1.view(*fc1.shape[:2])
        bb_delta = self.bb_predictor(fc1).view(batch_size, num_proposals_per_batch, 4)

        return bb_delta


class CatPredictorConv(nn.Module):
    def __init__(self, input_dim=(128, 256), num_stages=3):
        super().__init__()
        self.feat_conv3 = conv(input_dim[0], 256, kernel_size=3, stride=1)
        self.feat_conv4 = conv(input_dim[1], 256, kernel_size=3, stride=1)

        self.num_stages = num_stages

        for i in range(num_stages):
            self.add_module('stage{}'.format(i), CatPredictorConvStage())

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
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

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

        modulation = [[f.view(1, num_sequences, *f.shape[-3:]).repeat(num_images, 1, 1, 1, 1).
                           view(num_sequences*num_images, *f.shape[-3:]) for f in f_t] for f_t in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        bb_delta, bb_proposals = self.predict_bb_delta(modulation, iou_feat, proposals2)
        bb_delta = [b.view(num_images, num_sequences, proposals2.shape[1], 4) for b in bb_delta]
        bb_proposals = [b.view(num_images, num_sequences, proposals2.shape[1], 4) for b in bb_proposals]
        return bb_delta, bb_proposals

    def predict_bb(self, modulation, feat, proposals):
        for i in range(self.num_stages):
            bb_delta = getattr(self, 'stage{}'.format(i)).predict_bb_delta(modulation[i], feat, proposals)

            new_proposals = convert_bb_delta_to_abs_coord(proposals, bb_delta.clone().detach())
            proposals = new_proposals.detach()

        return new_proposals

    def predict_bb_delta(self, modulation, feat, proposals):
        bb_delta_all = []
        proposals_all = []
        for i in range(self.num_stages):
            bb_delta = getattr(self, 'stage{}'.format(i)).predict_bb_delta(modulation[i], feat, proposals)

            bb_delta_all.append(bb_delta)
            proposals_all.append(proposals.clone())

            new_proposals = convert_bb_delta_to_abs_coord(proposals, bb_delta.clone().detach())

            proposals_sz = new_proposals[:, :, 2:].prod(-1).sqrt()
            min_xy = new_proposals[:, :, :2].min(-1)[0]
            max_xy = new_proposals[:, :, :2].max(-1)[0]
            bad_proposals = (proposals_sz > 1000) | (min_xy < -1000) | (max_xy > 1000)
            new_proposals[bad_proposals, :] *= 0
            proposals = new_proposals.detach()

        return bb_delta_all, proposals_all

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        feat3_r, feat4_r = feat
        c3_r = self.feat_conv3(feat3_r)
        c4_r = self.feat_conv4(feat4_r)
        c = (c3_r, c4_r)

        modulation_vector = [getattr(self, 'stage{}'.format(i)).get_modulation(c, bb) for i in range(self.num_stages)]

        return modulation_vector

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [f.view(-1, *f.shape[-3:]) if f.dim() == 5 else f for f in feat2]

        feat3_t, feat4_t = feat2
        c3_t = self.feat_conv3(feat3_t)
        c4_t = self.feat_conv4(feat4_t)

        return c3_t, c4_t
