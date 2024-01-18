import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv4d(nn.Module):
    def __init__(self, kernel_size=3, input_dim=1, inter_dim=1, output_dim=1, bias=True, padding=None,
                 permute_back_output=True):
        super().__init__()

        assert input_dim==1

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.weight1 = nn.Parameter(torch.zeros(inter_dim, input_dim, *kernel_size))
        self.weight2 = nn.Parameter(torch.zeros(output_dim, inter_dim, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

        self.padding = [k//2 for k in kernel_size] if padding is None else padding

        self.permute_back_output = permute_back_output


    def forward(self, x, transpose=False):
        input_dim = 1
        output_dim, inter_dim = self.weight2.shape[:2]

        if transpose:
            # Expect 6D input
            assert x.dim() == 6

            if self.permute_back_output:
                x = x.permute(0,4,5,3,1,2)

            batch_size = x.shape[0]
            sz1 = x.shape[1:3]
            sz2 = x.shape[-2:]

            x2 = F.conv_transpose2d(x.reshape(-1, output_dim, *sz2), self.weight2, padding=self.padding)
            x2 = x2.reshape(batch_size, sz1[0]*sz1[1], inter_dim, sz2[0]*sz2[1]).permute(0,3,2,1)

            x3 = F.conv_transpose2d(x2.reshape(-1, inter_dim, *sz1), self.weight1, padding=self.padding)

            return x3.reshape(batch_size, *sz2, *sz1)


        # Expect 5D input
        assert x.dim() == 5

        batch_size = x.shape[0]
        sz2 = x.shape[1:3]
        sz1 = x.shape[-2:]

        x2 = F.conv2d(x.reshape(-1, input_dim, *sz1), self.weight1, padding=self.padding)
        x2 = x2.reshape(batch_size, sz2[0]*sz2[1], inter_dim, sz1[0]*sz1[1]).permute(0,3,2,1)

        x3 = F.conv2d(x2.reshape(-1, inter_dim, *sz2), self.weight2, bias=self.bias, padding=self.padding)
        x3 = x3.reshape(batch_size, *sz1, output_dim, *sz2)

        if self.permute_back_output:
            x3 = x3.permute(0,4,5,3,1,2).contiguous()

        return x3
