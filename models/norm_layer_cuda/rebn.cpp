#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> rebn_cuda_forward(torch::Tensor input,
                                             torch::Tensor weight,
                                             torch::Tensor bias,
                                             torch::Tensor running_mean,
                                             torch::Tensor running_var,
                                             bool training,
                                             float momentum,
                                             float r);

std::vector<torch::Tensor> rebn_cuda_backward(torch::Tensor grad_out,
                                              torch::Tensor input,
                                              torch::Tensor mean,
                                              torch::Tensor invstd,
                                              torch::Tensor weight,
                                              float r,
                                              bool straight_through);

// C++ interface

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> rebn_forward(torch::Tensor input,
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

  return rebn_cuda_forward(
      input, weight, bias, running_mean, running_var, training, momentum, r);
}

std::vector<torch::Tensor> rebn_backward(torch::Tensor grad_out,
                                         torch::Tensor input,
                                         torch::Tensor mean,
                                         torch::Tensor invstd,
                                         torch::Tensor weight,
                                         float r,
                                         bool straight_through)
{
  CHECK_INPUT(input);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(mean);
  CHECK_INPUT(invstd);
  CHECK_INPUT(weight);

  return rebn_cuda_backward(grad_out, input, mean, invstd, weight, r, straight_through);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &rebn_forward, "ReBN forward");
  m.def("backward", &rebn_backward, "ReBN backward");
}