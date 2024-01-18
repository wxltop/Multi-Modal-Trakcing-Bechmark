import torch.nn as nn
import torch
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


class FPNIoUNetHR(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=256, pred_input_dim=256, pool_stride=8, pool_r=5, pool_t=7):
        super().__init__()
        # _r for reference, _t for test
        self.conv1_r = conv(input_dim, 256, kernel_size=3, stride=1)
        self.conv1_t = conv(input_dim, 256, kernel_size=3, stride=1)

        self.conv2_t = conv(256, pred_input_dim, kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(pool_r, pool_r, 1/pool_stride)
        self.prroi_pool3t = PrRoIPool2D(pool_t, pool_t, 1/pool_stride)

        self.fc1_r = conv(pred_input_dim, 1024, kernel_size=pool_r, stride=1, padding=0)
        self.fc2_r = conv(1024, pred_input_dim, kernel_size=1, stride=1, padding=0)

        self.fc1_rt = LinearBlock(pred_input_dim, 1024, pool_t)
        self.fc2_rt = LinearBlock(1024, 1024, 1)

        self.iou_predictor = nn.Linear(1024, 1, bias=True)

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

        if not isinstance(feat1, (list, tuple)):
            feat1 = [feat1, ]
        if not isinstance(feat2, (list, tuple)):
            feat2 = [feat2, ]
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
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.view(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        modulation_vector, = modulation
        c3_t, = feat

        batch_size = c3_t.size()[0]

        # Modulation
        c3_t_att = c3_t * modulation_vector.view(batch_size, -1, 1, 1)

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

        fc3_rt = self.fc2_rt(self.fc1_rt(roi3t))

        iou_pred = self.iou_predictor(fc3_rt).view(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        feat3_r, = feat

        c3_r = self.conv1_r(feat3_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        modulation_vector = self.fc2_r(self.fc1_r(roi3r))

        return modulation_vector,

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat_backbone, = feat2

        feat_backbone = feat_backbone.view(-1, *feat_backbone.shape[-3:]) if feat_backbone.dim()==5 else feat_backbone

        c3_t = self.conv2_t(self.conv1_t(feat_backbone))

        return c3_t,


class FPNIoUNetHRCorr(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=256, pred_input_dim=256):
        super().__init__()
        # _r for reference, _t for test
        self.conv1_r = conv(input_dim, 256, kernel_size=3, stride=1)
        self.conv1_t = conv(input_dim, 256, kernel_size=3, stride=1)

        self.conv2_t = conv(256, pred_input_dim, kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(5, 5, 1/8)
        self.prroi_pool3t = PrRoIPool2D(7, 7, 1/8)

        self.fc1_r = conv(pred_input_dim, 1024, kernel_size=5, stride=1, padding=0)
        self.fc1_t = conv(pred_input_dim, 1024, kernel_size=7, stride=1, padding=0)

        self.fc1_rt = LinearBlock(1024, 1024, 1)

        self.iou_predictor = nn.Linear(1024, 1, bias=True)

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

        modulation = [f.view(1, num_sequences, -1).repeat(num_images, 1, 1).view(num_sequences*num_images, -1) for f in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.view(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        modulation_vector, = modulation
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

        roi3t = self.prroi_pool3t(c3_t, roi2)

        fc3_t = self.fc1_t(roi3t).view(batch_size, num_proposals_per_batch, -1)
        modulation_vector = modulation_vector.view(batch_size, 1, -1).repeat(1, num_proposals_per_batch, 1)
        fc3_rt = modulation_vector * fc3_t
        fc3_rt2 = self.fc1_rt(fc3_rt.view(batch_size * num_proposals_per_batch, -1))
        iou_pred = self.iou_predictor(fc3_rt2).view(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        feat3_r, = feat

        c3_r = self.conv1_r(feat3_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        modulation_vector = self.fc1_r(roi3r)

        return modulation_vector,

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat_backbone, = feat2

        feat_backbone = feat_backbone.view(-1, *feat_backbone.shape[-3:]) if feat_backbone.dim()==5 else feat_backbone

        c3_t = self.conv2_t(self.conv1_t(feat_backbone))

        return c3_t,


class FPNIoUNetHRCat(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=256, pred_input_dim=256):
        super().__init__()
        # _r for reference, _t for test
        self.conv1_r = conv(input_dim, pred_input_dim, kernel_size=3, stride=1)
        self.conv1_t = conv(input_dim, 256, kernel_size=3, stride=1)

        self.conv2_t = conv(256, pred_input_dim, kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(7, 7, 1/8)
        self.prroi_pool3t = PrRoIPool2D(7, 7, 1/8)

        self.conv1_rt = conv(2*pred_input_dim, pred_input_dim, 1, stride=1, padding=0)

        self.fc1_rt = conv(pred_input_dim, 1024, kernel_size=7, stride=1, padding=0)
        self.fc2_rt = LinearBlock(1024, 1024, 1)

        self.iou_predictor = nn.Linear(1024, 1, bias=True)

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

        modulation = [f.view(1, num_sequences, -1).repeat(num_images, 1, 1).view(num_sequences*num_images, -1) for f in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.view(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        modulation_vector, = modulation
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

        roi3t = self.prroi_pool3t(c3_t, roi2)
        roi3t = roi3t.view(batch_size, num_proposals_per_batch, *roi3t.shape[1:])

        modulation_vector = modulation_vector.view(batch_size, 1, *roi3t.shape[2:]).repeat(1, num_proposals_per_batch, 1, 1, 1)

        roi3_rt = torch.cat((modulation_vector, roi3t), dim=2).view(batch_size*num_proposals_per_batch, -1, modulation_vector.shape[-2], modulation_vector.shape[-1])
        roi3_rt_dim = self.conv1_rt(roi3_rt)

        fc3_rt = self.fc2_rt(self.fc1_rt(roi3_rt_dim))

        iou_pred = self.iou_predictor(fc3_rt).view(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image img_coords in the reference samples. Dims (batch, 4)."""

        feat3_r, = feat

        c3_r = self.conv1_r(feat3_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        modulation_vector = roi3r

        return modulation_vector,

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat_backbone, = feat2

        feat_backbone = feat_backbone.view(-1, *feat_backbone.shape[-3:]) if feat_backbone.dim()==5 else feat_backbone

        c3_t = self.conv2_t(self.conv1_t(feat_backbone))

        return c3_t,


class FPNCatIoUPredictorConv(nn.Module):
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

        self.iou_predictor = nn.Linear(1024, 1, bias=True)

    def forward(self, feat1, feat2, bb1, proposals2):
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

        modulation = [f.view(1, num_sequences, *f.shape[-3:]).repeat(num_images, 1, 1, 1, 1).
                           view(num_sequences * num_images, *f.shape[-3:]) for f in modulation]

        proposals2 = proposals2.view(num_sequences*num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.view(num_images, num_sequences, -1)

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

    def predict_iou(self, modulation, feat, proposals):
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

        iou_pred = self.iou_predictor(fc1).view(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat_backbone, = feat2

        feat_backbone = feat_backbone.view(-1, *feat_backbone.shape[-3:]) if feat_backbone.dim()==5 else feat_backbone

        return feat_backbone,