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


# class LogBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=True, track_running_stats=True):
#         super(LogBatchNorm2d, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
#         self.register_buffer('before_mean', torch.zeros(num_features))
#         self.register_buffer('before_var', torch.ones(num_features))
#         self.register_buffer('after_mean', torch.zeros(num_features))
#         self.register_buffer('after_var', torch.ones(num_features))

#     def forward(self, input):
#         self.before_mean = input.mean([0, 2, 3])
#         self.before_var = input.var([0, 2, 3], unbiased=False)

#         output = super().forward(input)

#         self.after_mean = output.mean([0, 2, 3])
#         self.after_var = output.var([0, 2, 3], unbiased=False)
#         return output


# class LogGroupNorm2d(nn.GroupNorm):
#     def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
#         super(LogGroupNorm2d, self).__init__(
#             num_groups, num_channels, eps, affine)

#         self.register_buffer('before_mean', None)
#         self.register_buffer('before_var', None)
#         self.register_buffer('after_mean', None)
#         self.register_buffer('after_var', None)

#     def forward(self, input):
#         b, c, h, w = input.shape
#         input_ = input.view(b, self.num_groups, -1, h, w)
#         self.before_mean = input_.mean([2, 3, 4])
#         self.before_var = input_.var([2, 3, 4], unbiased=False)

#         output = super().forward(input)

#         output_ = output.view(b, self.num_groups, -1, h, w)
#         self.after_mean = output_.mean([2, 3, 4])
#         self.after_var = output_.var([2, 3, 4], unbiased=False)
#         return output


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.rand(3, 8, 2, 2)

    
    norm1 = nn.BatchNorm2d(8)
    norm2 = LogBatchNorm(8, input_type="2d")
    print(torch.allclose(norm1(x), norm2(x)))
    print(norm1(x), norm1(x).size())
    print(norm2(x), norm2(x).size())
    # print(norm2.before_mean.size(), norm2.before_var.size(), norm2.after_mean.size(), norm2.after_var.size())
    # print(list(norm2.named_buffers()))
    print(norm2.after_var, norm2.after_mean)
    print(norm2.after_affine_mean, norm2.after_affine_var)
