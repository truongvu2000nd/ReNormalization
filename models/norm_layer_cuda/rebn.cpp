#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> batch_norm_cuda_forward(torch::Tensor input,
                                                   torch::Tensor weight,
                                                   torch::Tensor bias,
                                                   torch::Tensor running_mean,
                                                   torch::Tensor running_var,
                                                   bool training,
                                                   float momentum,
                                                   float r);

std::vector<torch::Tensor> batch_norm_cuda_backward(torch::Tensor grad_out,
                                                    torch::Tensor input,
                                                    torch::Tensor inv_var,
                                                    torch::Tensor x_hat,
                                                    torch::Tensor gamma,
                                                    float r);

// C++ interface

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> batch_norm_forward(torch::Tensor input,
                                              torch::Tensor weight,
                                              torch::Tensor bias,
                                              torch::Tensor running_mean,
                                              torch::Tensor running_var,
                                              bool training,
                                              float momentum,
                                              float r)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  CHECK_INPUT(running_mean);
  CHECK_INPUT(running_var);

  return batch_norm_cuda_forward(
      input, weight, bias, running_mean, running_var, training, momentum, r);
}

std::vector<torch::Tensor> batch_norm_backward(torch::Tensor grad_out,
                                               torch::Tensor input,
                                               torch::Tensor inv_var,
                                               torch::Tensor x_hat,
                                               torch::Tensor gamma,
                                               float r)
{
  CHECK_INPUT(input);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(inv_var);
  CHECK_INPUT(x_hat);
  CHECK_INPUT(gamma);

  return batch_norm_cuda_backward(grad_out, input, inv_var, x_hat, gamma, r);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &batch_norm_forward, "BatchNorm forward");
  m.def("backward", &batch_norm_backward, "BatchNorm backward");
}