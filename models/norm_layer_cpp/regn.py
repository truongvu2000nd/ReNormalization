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
    def forward(ctx, input_, weight, bias, num_groups, r, straight_through):
        dummy_eps = 1e-5
        b = input_.size(0)
        init_size = input_.size()
        input_reshaped = input_.reshape(b, num_groups, -1)
        s = input_reshaped.size(2)

        var, mean = torch.var_mean(input_reshaped, 2, unbiased=False)
        rstd = torch.rsqrt(var[:, :, None] + dummy_eps)
        re_rstd = 1. / torch.sqrt(var[:, :, None] + dummy_eps).clamp(min=r)

        xhat = (input_reshaped - mean[:, :, None]) * re_rstd
        # return xhat

        xhat = xhat.reshape(init_size)
        if len(init_size) == 2:
            output = xhat * weight[None, :] + bias[None, :]
        elif len(init_size) == 4:
            output = xhat * weight[None, :, None, None] + bias[None, :, None, None]

        output = output * s / (s - 1)

        ctx.save_for_backward(input_, mean, re_rstd, weight)
        ctx.num_groups = num_groups
        ctx.r = r
        ctx.straight_through = straight_through
        ctx.s = s

        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, mean, rstd, weight = ctx.saved_tensors

        grad_out = grad_out * ctx.s / (ctx.s-1)

        bs = grad_out.size(0)
        input_reshaped = input.reshape(bs, ctx.num_groups, -1)
        xhat = (input_reshaped - mean[:, :, None]) * rstd

        grad_bias = grad_out.sum(dim=(2, 3))
        grad_weight = (grad_out * xhat.view(grad_out.size())).sum(dim=(2, 3))

        grad_out_reshaped = grad_out.reshape(bs, ctx.num_groups, -1)
        grad_xhat = (grad_out * weight[None, :, None, None]).reshape(bs, ctx.num_groups, -1)

        N = grad_out_reshaped.size(2)

        grad_input = (1. / N) * rstd * \
            (N * grad_xhat - grad_xhat.sum(dim=2, keepdim=True) - xhat * (grad_xhat * xhat).sum(dim=2, keepdim=True))
        
        grad_input = grad_input.view(input.size())
        return grad_input, grad_weight, grad_bias, None, None, None


class ReGN(nn.GroupNorm):
    def __init__(self, num_channels, group_size, r=1., straight_through=False):
        dummy_eps = 1e-5
        num_groups = num_channels // group_size
        super(ReGN, self).__init__(num_groups, num_channels, dummy_eps)
        self.r = r
        self.straight_through = straight_through
        self.group_size = group_size

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
    sys.path.append("..")

    from norm_layer import GN, ReGN as ReGN2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gn = ReGN(6, 2).to(device)
    gn2 = ReGN2(6, 2).to(device)
    x = torch.randn(1, 6, 3, 3, requires_grad=True).to(device) * 5

    # print(gn(x))
    # print(gn2(x))
    # print(torch.allclose(gn(x), gn2(x)))

    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, False]):
        print('Test 1 Ok')

    gn = ReGN(16, 2).to(device)
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).to(device) * 0.1
    
    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, False]):
        print('Test 2 Ok')

    gn = ReGN(16, 2).to(device)
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).to(device) * 5

    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, False]):
        print('Test 3 Ok')
    
    gn = ReGN(16, 2).to(device)
    x = torch.randn(4, 16, 4, 4, dtype=torch.double, requires_grad=True).to(device) * 5

    weight, bias, num_groups, r = gn.weight, gn.bias, gn.num_groups, gn.r
    if gradcheck(ReGNFunction.apply, [x, weight.double(), bias.double(), num_groups, r, True]):
        print('Test 4 Ok')
