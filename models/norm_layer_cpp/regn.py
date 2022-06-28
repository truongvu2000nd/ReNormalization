import math
from torch import nn
from torch.autograd import Function
import torch
from torch.utils.cpp_extension import load
import os

# module_path = os.path.dirname(__file__)
# regn_cpp = load(name="regn_cpp", sources=[
#                       os.path.join(module_path, "regn.cpp"), 
#                       os.path.join(module_path, "regn.cu")])


# class ReGNFunction(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, r, straight_through):
#         output, mean, invstd = regn_cpp.forward(input, weight, bias, r)
#         ctx.save_for_backward(input, mean, invstd, weight)
#         ctx.r = r
#         ctx.straight_through = straight_through
#         return output

#     @staticmethod
#     def backward(ctx, grad_out):
#         grad_input, grad_weight, grad_bias = regn_cpp.backward(grad_out, *ctx.saved_tensors, ctx.r, ctx.straight_through)
#         return grad_input, grad_weight, grad_bias, None, None

class ReGNFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, num_groups, r, straight_through):
        dummy_eps = 1e-5
        b = input.size(0)
        init_size = input.size()
        input = input.reshape(b, num_groups, -1)
        mean = input.mean(2)
        var = input.var(2, unbiased=False)
        rstd = 1 / (torch.sqrt(var[:, :, None] + dummy_eps))

        input = (input - mean[:, :, None]) / rstd

        input = input.reshape(init_size)
        if len(init_size) == 2:
            input = input * weight[None, :] + bias[None, :]
        elif len(init_size) == 4:
            input = input * weight[None, :, None, None] + bias[None, :, None, None]

        ctx.save_for_backward(input, mean, rstd, weight)
        ctx.num_groups = num_groups

        return input

    @staticmethod
    def backward(ctx, grad_out):
        input, mean, rstd, weight = ctx.saved_tensors

        bs = grad_out.size(0)
        input_reshaped = input.reshape(bs, ctx.num_groups, -1)

        xhat = (input_reshaped - mean[:, :, None]) / rstd

        grad_out_reshaped = grad_out.reshape(bs, ctx.num_groups, -1)
        grad_bias = grad_out_reshaped.sum(dim=2)
        grad_weight = (grad_out_reshaped * xhat).sum(dim=2)
        grad_xhat = (grad_out * weight[None, :, None, None]).reshape(bs, ctx.num_groups, -1).sum(dim=2, keepdim=True)

        N = grad_out_reshaped.size(2)

        grad_input = (1. / N) * rstd * \
            (N * grad_xhat - grad_xhat.sum(dim=2, keepdim=True) - xhat * (grad_xhat * xhat).sum(dim=2, keepdim=True))
        
        grad_input = grad_input.view(input.size())
        print(grad_weight.size())
        return grad_input, grad_weight, grad_bias, None, None, None


class ReGNCPP(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, r=1., straight_through=False):
        dummy_eps = 1e-5
        super(ReGNCPP, self).__init__(num_groups, num_channels, dummy_eps)
        self.r = r
        self.straight_through = straight_through

    def forward(self, input):
        return ReGNFunction.apply(
            input,
            self.weight,
            self.bias,
            self.num_groups,
            self.r,
            self.straight_through
        )


if __name__ == '__main__':
    import sys
    import time
    from torch.autograd import gradcheck

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gn = ReGNCPP(6, 2).to(device)
    x = torch.randn(1, 6, 3, 3, dtype=torch.double, requires_grad=True).to(device) * 5

    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, False]):
        print('Test 1 Ok')

    gn = ReGNCPP(16, 2).to(device)
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).to(device) * 0.1
    
    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, False]):
        print('Test 2 Ok')

    gn = ReGNCPP(16, 2).to(device)
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).to(device) * 5

    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, False]):
        print('Test 3 Ok')
    
    gn = ReGNCPP(16, 2).to(device)
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).to(device) * 5

    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, True]):
        print('Test 4 Ok')
