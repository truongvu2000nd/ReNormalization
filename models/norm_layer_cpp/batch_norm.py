import math
from torch import nn
from torch.autograd import Function
import torch
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.cpp_extension import load

batch_norm_cpp = load(name="batch_norm_cpp", sources=["batch_norm.cpp"])
# import batch_norm_cpp

torch.manual_seed(42)


class BatchNormFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, training, momentum, eps):
        output, mean, inv_std, x_hat = batch_norm_cpp.forward(input, weight, bias, running_mean, running_var, training, momentum, eps)
        ctx.save_for_backward(mean, inv_std, x_hat, weight)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        # print(grad_out)
        grad_input, grad_weight, grad_bias = batch_norm_cpp.backward(grad_out, *ctx.saved_tensors)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class BatchNormCPP(_NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BatchNormCPP, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
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
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

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
    from norm_layer import BN

    def print_grad():
        def hook(module, grad_input, grad_output):
            print(grad_input)
            print(grad_output[0].mean(), grad_output[0].size())
        return hook

    bn = BatchNormCPP(8)
    bn2 = nn.BatchNorm2d(8)
    bn3 = BN(8)

    conv = nn.Conv2d(8, 8, 3, 1, 1)
    conv2 = nn.Conv2d(8, 8, 3, 1, 1)
    # conv3 = nn.Conv2d(8, 8, 3, 1, 1)

    # bn.register_full_backward_hook(print_grad())
    # bn2.register_full_backward_hook(print_grad())
    # bn3.register_full_backward_hook(print_grad())

    for i in range(5):
        x = torch.randn(1, 8, 4, 4) * 10 + 10

        out1 = conv2(bn(conv(x)))
        # out1.sum().backward()
        # print(bn.weight.grad)
        # print(conv.weight.grad.mean())
        # print(conv2.weight.grad.mean())
        # conv.weight.grad.zero_()
        # conv2.weight.grad.zero_()
        print(bn.running_mean)
        # print(bn.bias.grad)

        out2 = conv2(bn2(conv(x)))
        # out2.sum().backward()
        # print(bn2.weight.grad)
        # print(conv.weight.grad.mean())
        # print(conv2.weight.grad.mean())
        # conv.weight.grad.zero_()
        # conv2.weight.grad.zero_()
        print(bn2.running_mean)

        out3 = conv2(bn3(conv(x)))
        # out3.sum().backward()
        # print(bn3.weight.grad)
        # print(conv.weight.grad.mean())
        # print(conv2.weight.grad.mean())
        # conv.weight.grad.zero_()
        # conv2.weight.grad.zero_()
        print(bn3.running_mean)

    # print(bn3.bias.grad)
    print(torch.allclose(out1, out2))
    # print(torch.allclose(out3, out2))
    # print(torch.allclose(out1, out3))
