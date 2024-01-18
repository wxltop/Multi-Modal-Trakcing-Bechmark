import torch


def bbox_to_roi(bb):
    assert bb.dim() == 2

    batch_size = bb.shape[0]
    batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb.device)

    # input bb is in format xywh, convert it to x0y0x1y1 format
    bb = bb.clone()
    bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
    roi = torch.cat((batch_index, bb), dim=1)

    return roi