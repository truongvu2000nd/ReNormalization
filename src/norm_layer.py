"""
Comparison of manual BatchNorm2d layer implementation in Python and
nn.BatchNorm2d
@author: ptrblck
"""

import torch
from torch.functional import norm
import torch.nn as nn


class LogBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, input_type="2d", eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(LogBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        # These buffers track layer state before and after normalization
        self.register_buffer('before_mean', None)
        self.register_buffer('before_var', None)
        self.register_buffer('after_mean', None)
        self.register_buffer('after_var', None)
        self.register_buffer('after_affine_mean', None)
        self.register_buffer('after_affine_var', None)

        self.input_type = input_type
        if input_type == "1d":
            self.norm_dims = [0]
        elif input_type == "2d":
            self.norm_dims = [0, 2, 3]
        else:
            raise NotImplementedError

    def forward(self, input):
        # self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        self.before_mean = input.mean(self.norm_dims)
        self.before_var = input.var(self.norm_dims, unbiased=False)

        # calculate running estimates
        if bn_training:
            mean = self.before_mean
            var = self.before_var

            n = input.numel() / input.size(1)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        if self.input_type == "1d":
            input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        else:
            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        self.after_mean = input.mean(self.norm_dims)
        self.after_var = input.var(self.norm_dims, unbiased=False)

        if self.affine:
            if self.input_type == "1d":
                input = input * self.weight[None, :] + self.bias[None, :]
            else:
                input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        self.after_affine_mean = input.mean(self.norm_dims)
        self.after_affine_var = input.var(self.norm_dims, unbiased=False)

        return input


class LogGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(LogGroupNorm, self).__init__(
            num_groups, num_channels, eps, affine)
        self.register_buffer('before_mean', None)
        self.register_buffer('before_var', None)
        self.register_buffer('after_mean', None)
        self.register_buffer('after_var', None)
        self.register_buffer('after_affine_mean', None)
        self.register_buffer('after_affine_var', None)

    def forward(self, input):
        b = input.size(0)
        init_size = input.size()
        input = input.view(b, self.num_groups, -1)
        self.before_mean = input.mean(2)
        self.before_var = input.var(2, unbiased=False)

        mean = self.before_mean
        var = self.before_var

        input = (input - mean[:, :, None]) / (torch.sqrt(var[:, :, None] + self.eps))

        self.after_mean = input.mean(2)
        self.after_var = input.var(2, unbiased=False)

        input = input.view(init_size)
        if self.affine:
            if len(init_size) == 2:
                input = input * self.weight[None, :] + self.bias[None, :]
            elif len(init_size) == 4:
                input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            else:
                raise NotImplementedError("Only 1D and 2D groupnorm with affine")

        input_ = input.view(b, self.num_groups, -1)
        self.after_affine_mean = input_.mean(2)
        self.after_affine_var = input_.var(2, unbiased=False)

        return input


if __name__ == '__main__':
    torch.manual_seed(0)
    x1 = torch.rand(3, 4, 2, 2)
    x2 = torch.rand(3, 4, 2, 2)
    x3 = torch.rand(3, 4, 2, 2)
    
    norm1 = nn.BatchNorm2d(4)
    norm2 = LogBatchNorm(4, input_type="2d")
    print(torch.allclose(norm1(x1), norm2(x1)))
    print(torch.allclose(norm1(x2), norm2(x2)))
    track_buffers = ["before_mean", "before_var", "after_mean", "after_var", "after_affine_mean", "after_affine_var"]
    track_buffers += ["running_mean", "running_var"]
    for name, p in norm2.named_buffers():
        for n in track_buffers:
            if n in name:
                flattened_p = torch.flatten(p).clamp(max=100).float().detach()

    norm1.eval()
    norm2.eval()

    print(torch.allclose(norm1(x3), norm2(x3)))
    print(torch.allclose(norm1(x1), norm2(x1)))
    print(torch.allclose(norm1(x2), norm2(x2)))

    norm1.train()
    norm2.train()
    print(torch.allclose(norm1(x3), norm2(x3)))
    print(torch.allclose(norm1(x1), norm2(x1)))
    print(torch.allclose(norm1(x2), norm2(x2)))
