#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor>
batch_norm_forward(torch::Tensor& input_,
                   torch::Tensor& weight,
                   torch::Tensor& bias,
                   torch::Tensor& running_mean,
                   torch::Tensor& running_var,
                   bool training,
                   float momentum,
                   float eps)
{
  auto input_reshaped = input_.reshape({ input_.size(0), input_.size(1), -1 });
  input_reshaped.data_ptr;
  torch::Tensor mean, var;

  if (training) {
    mean = input_reshaped.mean({ 0, 2 });
    var = input_reshaped.var({ 0, 2 }, false);
    auto n = input_reshaped.numel() / input_reshaped.size(1);
    running_mean.mul_(1. - momentum).add_(momentum * mean.detach());
    running_var.mul_(1. - momentum)
      .add_(momentum * var.detach())
      .mul_(n)
      .div_(n - 1);
  } else {
    mean = running_mean;
    var = running_var;
  }

  auto inv_std = torch::rsqrt(var + eps);

  int tf = std::max<int>(getNumThreads(input.size(2) / 4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64 / tf, 1);
  dim3 blocks_trans(
    input.size(1),
    std::max<int>(1,
                  std::min<int>((256 * 1024) / input.size(1),
                                (input.size(0) + tb - 1) / tb)));
  blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);

  AT_DISPATCH_FLOATING_TYPES(
    input_reshaped.type(), "batch_norm_forward_cuda", ([&] {
      batch_norm_cuda_forward_kernel<scalar_t>
        <<<blocks_trans, threads_trans>>>(input_reshaped.data_ptr<scalar_t>(),
                                          mean.data_ptr<scalar_t>(),
                                          invstd.data_ptr<scalar_t>(),
                                          weight.data_ptr<scalar_t>(),
                                          bias.data_ptr<scalar_t>(),
                                          x_hat.data_ptr<scalar_t>(),
                                          output_reshaped.data_ptr<scalar_t>(),
                                          eps);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));
  return { output, mean, inv_std, x_hat };
}

template<typename scalar_t>
__global__ void
batch_norm_cuda_forward_kernel(const scalar_t* input,
                               const scalar_t* mean,
                               const scalar_t* invstd,
                               const scalar_t* weight,
                               const scalar_t* bias,
                               scalar_t* x_hat,
                               scalar_t* output,
                               size_t eps)
{
  int plane = blockIdx.x;
  if (plane >= input.size(1)) {
    return;
  }

  int bs = input.size(0);
  int fs = input.size(2);

  int bstep = blockDim.y * gridDim.y;
  for (int batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs;
       batch += bstep) {
    auto x_h = x_hat[batch][plane];
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (int feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      x_h[feature] = (i[feature] - mean) * invstd o[feature] =
                       weight * x_h[feature] + bias;
    }
  }
}

std::vector<torch::Tensor>
batch_norm_backward(torch::Tensor grad_out,
                    torch::Tensor mean,
                    torch::Tensor inv_var,
                    torch::Tensor x_hat,
                    torch::Tensor gamma)
{
  auto N = grad_out.size(0) * grad_out.size(2) * grad_out.size(3);

  auto grad_xhat = grad_out * _unsqueeze_023(gamma);

  auto grad_input = (1. / N) * inv_var *
                    (N * grad_xhat - grad_xhat.sum({ 0, 2, 3 }, true) -
                     x_hat * (grad_xhat * x_hat).sum({ 0, 2, 3 }, true));

  auto grad_weight = (x_hat * grad_out).sum({ 0, 2, 3 });
  auto grad_bias = grad_out.sum({ 0, 2, 3 });

  return { grad_input, grad_weight, grad_bias };
}
