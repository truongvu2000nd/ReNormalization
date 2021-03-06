import math
from torch import nn
from torch.autograd import Function
import torch
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)
rebn_cpp = load(name="rebn_cpp", sources=[
                      os.path.join(module_path, "rebn.cpp"), 
                      os.path.join(module_path, "rebn.cu")])


class ReBNFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, training, momentum, r, straight_through):
        output, mean, invstd = rebn_cpp.forward(
            input, weight, bias, running_mean, running_var, training, momentum, r)
        ctx.save_for_backward(input, mean, invstd, weight)
        ctx.r = r
        ctx.straight_through = straight_through
        return output

    @staticmethod
    def backward(ctx, grad_out):
        grad_input, grad_weight, grad_bias = rebn_cpp.backward(grad_out, *ctx.saved_tensors, ctx.r, ctx.straight_through)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class ReBNCPP(_NormBase):
    def __init__(self, num_features, r=1., momentum=0.1,
                 affine=True, track_running_stats=True, straight_through=False):
        dummy_eps = 1e-5
        super(ReBNCPP, self).__init__(
            num_features, dummy_eps, momentum, affine, track_running_stats
        )
        self.r = r
        self.straight_through = straight_through

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

        return ReBNFunction.apply(
            input,
            self.weight,
            self.bias,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            bn_training,
            exponential_average_factor,
            self.r,
            self.straight_through
        )


if __name__ == '__main__':
    import sys
    import time
    from torch.autograd import gradcheck

    bn = ReBNCPP(2).cuda()
    x = torch.randn(1, 2, 3, 3, dtype=torch.double, requires_grad=True).cuda() * 5

    weight, bias, running_mean, running_var, training, momentum, r = \
        bn.weight, bn.bias, bn.running_mean, bn.running_var, True, bn.momentum, bn.r
    if gradcheck(ReBNFunction.apply, [x, weight.double(), bias.double(), running_mean.double(), running_var.double(), training, momentum, r, False]):
        print('Test 1 Ok')

    bn = ReBNCPP(16, r=0.5).cuda()
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).cuda() * 0.1

    weight, bias, running_mean, running_var, training, momentum, r = \
        bn.weight, bn.bias, bn.running_mean, bn.running_var, True, bn.momentum, bn.r

    if gradcheck(ReBNFunction.apply, [x, weight.double(), bias.double(), running_mean.double(), running_var.double(), training, momentum, r, False]):
        print('Test 2 Ok')

    bn = ReBNCPP(16, r=0.1).cuda()
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).cuda() * 5

    weight, bias, running_mean, running_var, training, momentum, r = \
        bn.weight, bn.bias, bn.running_mean, bn.running_var, True, bn.momentum, bn.r

    if gradcheck(ReBNFunction.apply, [x, weight.double(), bias.double(), running_mean.double(), running_var.double(), training, momentum, r, False]):
        print('Test 3 Ok')
    
    bn = ReBNCPP(16, r=0.1).cuda()
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).cuda() * 5

    weight, bias, running_mean, running_var, training, momentum, r = \
        bn.weight, bn.bias, bn.running_mean, bn.running_var, True, bn.momentum, bn.r

    if gradcheck(ReBNFunction.apply, [x, weight.double(), bias.double(), running_mean.double(), running_var.double(), training, momentum, r, True]):
        print('Test 4 Ok')
