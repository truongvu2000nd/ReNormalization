"""
@author: ptrblck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, input_type="2d", eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        # These buffers track layer state before and after normalization
        self.register_buffer('before_mean', None)
        self.register_buffer('before_var', None)
        self.register_buffer('after_mean', None)
        self.register_buffer('after_var', None)

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

        if not self.training:
            self.after_mean = input.mean(self.norm_dims)
            self.after_var = input.var(self.norm_dims, unbiased=False)

        if self.affine:
            if self.input_type == "1d":
                input = input * self.weight[None, :] + self.bias[None, :]
            else:
                input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(GroupNorm, self).__init__(
            num_groups, num_channels, eps, affine)
        self.register_buffer('before_mean', None)
        self.register_buffer('before_var', None)
        self.register_buffer('after_mean', None)
        self.register_buffer('after_var', None)

    def forward(self, input):
        b = input.size(0)
        init_size = input.size()
        input = input.view(b, self.num_groups, -1)
        mean = input.mean(2)
        var = input.var(2, unbiased=False)

        input = (input - mean[:, :, None]) / (torch.sqrt(var[:, :, None] + self.eps))

        if not self.training:
            self.before_mean = mean
            self.before_var = var
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

        return input


class ReBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, input_type="2d", r=1., eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(ReBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.r = r

        # These buffers track layer state before and after normalization
        self.register_buffer('before_mean', None)
        self.register_buffer('before_var', None)
        self.register_buffer('after_mean', None)
        self.register_buffer('after_var', None)

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
            input = (input - mean[None, :]) / torch.sqrt(var[None, :]).clamp(min=self.r)
        else:
            input = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None]).clamp(min=self.r)

        if not self.training:
            self.after_mean = input.mean(self.norm_dims)
            self.after_var = input.var(self.norm_dims, unbiased=False)

        if self.affine:
            if self.input_type == "1d":
                input = input * self.weight[None, :] + self.bias[None, :]
            else:
                input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class ReGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, r=1., affine=True, modified=False):
        super(ReGroupNorm, self).__init__(
            num_groups, num_channels, affine)
        self.r = r
        self.modified = modified
        self.register_buffer('before_mean', None)
        self.register_buffer('before_var', None)
        self.register_buffer('after_mean', None)
        self.register_buffer('after_var', None)

    def forward(self, input):
        b = input.size(0)
        init_size = input.size()
        input = input.view(b, self.num_groups, -1)
        s = input.size(2)
        mean = input.mean(2)
        var = input.var(2, unbiased=False)

        input = (input - mean[:, :, None]) / torch.sqrt(var[:, :, None]).clamp(min=self.r)

        if not self.training:
            self.before_mean = mean
            self.before_var = var
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

        if self.modified:
            input = input * s / (s - 1)
        return input


def get_norm_layer(norm_layer=None, **kwargs):
    if norm_layer == "bn" or norm_layer is None:
        norm_layer = BatchNorm
    elif norm_layer == "ln":
        norm_layer = partial(GroupNorm, 1)
    elif norm_layer == "gn":
        norm_layer = partial(GroupNorm, 32)
    elif norm_layer == "bn-torch":
        norm_layer = nn.BatchNorm2d
    elif norm_layer == "ln-torch":
        norm_layer = partial(nn.GroupNorm, 1)
    elif norm_layer == "gn-torch":
        norm_layer = partial(nn.GroupNorm, 32)
    elif norm_layer == "rebn":
        norm_layer = partial(ReBatchNorm, **kwargs)
    elif norm_layer == "reln":
        norm_layer = partial(ReGroupNorm, 1, **kwargs)
    elif norm_layer == "regn":
        norm_layer = partial(ReGroupNorm, 32, **kwargs)
    else:
        raise NotImplementedError

    return norm_layer


if __name__ == '__main__':
    import time
