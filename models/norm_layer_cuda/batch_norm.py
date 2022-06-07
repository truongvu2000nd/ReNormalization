import math
from torch import nn
from torch.autograd import Function
import torch
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)
batch_norm_cpp = load(name="batch_norm_cpp", sources=[
                      os.path.join(module_path, "batch_norm.cpp"), 
                      os.path.join(module_path, "batch_norm.cu")])


class BatchNormFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, training, momentum, eps):
        output, mean, invstd = batch_norm_cpp.forward(
            input, weight, bias, running_mean, running_var, training, momentum, eps)
        ctx.save_for_backward(input, mean, invstd, weight)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        grad_input, grad_weight, grad_bias = batch_norm_cpp.backward(
            grad_out, *ctx.saved_tensors)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class BatchNormCPP(_NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNormCPP, self).__init__(
            num_features, eps, momentum, True, True
        )

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        return BatchNormFunction.apply(
            input,
            self.weight,
            self.bias,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


if __name__ == '__main__':
    import sys
    import time
    sys.path.append("..")
    from torch.autograd import gradcheck

    bn = BatchNormCPP(4).cuda()
    bn2 = nn.BatchNorm2d(4)

    print("Test forward..............")
    x = torch.randn(1, 4, 2, 2).cuda()
    print(bn(x))
    print(bn2(x))

    print("Test backward.............")
    x = torch.randn(4, 4, 4, 4, dtype=torch.double, requires_grad=True).cuda()

    weight, bias, running_mean, running_var, training, momentum, eps = \
        bn.weight, bn.bias, bn.running_mean, bn.running_var, True, bn.momentum, bn.eps
    if gradcheck(BatchNormFunction.apply, [x, weight.double(), bias.double(), running_mean.double(), running_var.double(), training, momentum, eps]):
        print('Ok')
