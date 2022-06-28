#include <torch/extension.h>

#include <vector>

// input, weight, bias, eps
std::vector<torch::Tensor> group_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps)
{
    torch::Tensor mean, var;
    mean = input.mean({0, 2, 3}, true);
    var = input.var({0, 2, 3}, false, true);
    auto n = input.numel() / input.size(1);

    auto inv_std = torch::rsqrt(var + eps);
    auto x_hat = (input - mean) * inv_std;

    auto output = x_hat * _unsqueeze_023(weight) + _unsqueeze_023(bias);

    return {output,
            mean,
            inv_std,
            x_hat};
}

std::vector<torch::Tensor> group_norm_backward(
    torch::Tensor grad_out,
    torch::Tensor mean,
    torch::Tensor inv_var,
    torch::Tensor x_hat,
    torch::Tensor gamma)
{
    auto N = grad_out.size(0) * grad_out.size(2) * grad_out.size(3);

    auto grad_xhat = grad_out * _unsqueeze_023(gamma);

    auto grad_input = (1. / N) * inv_var * (N * grad_xhat - grad_xhat.sum({0, 2, 3}, true) - x_hat * (grad_xhat * x_hat).sum({0, 2, 3}, true));

    auto grad_weight = (x_hat * grad_out).sum({0, 2, 3});
    auto grad_bias = grad_out.sum({0, 2, 3});

    return {grad_input,
            grad_weight,
            grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &group_norm_forward, "GroupNorm forward");
    m.def("backward", &group_norm_backward, "GroupNorm backward");
}