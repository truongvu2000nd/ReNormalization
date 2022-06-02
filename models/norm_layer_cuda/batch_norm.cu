#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

constexpr int MAX_BLOCK_SIZE = 512;

constexpr unsigned MAX_GRID_SIZE = 65535u;

static int getNumThreads(int nElem)
{
#if defined(USE_ROCM)
  int threadSizes[5] = {16, 32, 64, 128, MAX_BLOCK_SIZE};
#else
  int threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
#endif
  for (int i = 0; i != 5; ++i)
  {
    if (nElem <= threadSizes[i])
    {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

template <typename scalar_t>
__global__ void batch_norm_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean_,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> invstd_,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> x_hat,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,
    float eps)
{
  int plane = blockIdx.x;
  if (plane >= input.size(1))
  {
    return;
  }

  int bs = input.size(0);
  int fs = input.size(2);

  int bstep = blockDim.y * gridDim.y;

  scalar_t gamma = weight.size(0) > 0 ? static_cast<scalar_t>(weight[plane]) : static_cast<scalar_t>(1);
  scalar_t beta = bias.size(0) > 0 ? static_cast<scalar_t>(bias[plane]) : static_cast<scalar_t>(0);
  scalar_t mean = static_cast<scalar_t>(mean_[plane]);
  scalar_t invstd = static_cast<scalar_t>(invstd_[plane]);
  for (int batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep)
  {
    auto x_h = x_hat[batch][plane];
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (int feature = threadIdx.x; feature < fs; feature += blockDim.x)
    {
      x_h[feature] = (i[feature] - mean) * invstd;
      o[feature] = gamma * x_h[feature] + beta;
    }
  }
}

std::vector<torch::Tensor> batch_norm_cuda_forward(torch::Tensor input,
                                                   torch::Tensor weight,
                                                   torch::Tensor bias,
                                                   torch::Tensor running_mean,
                                                   torch::Tensor running_var,
                                                   bool training,
                                                   float momentum,
                                                   float eps)
{
  auto input_reshaped = input.reshape({input.size(0), input.size(1), -1});
  torch::Tensor mean, var;

  if (training)
  {
    c10::IntArrayRef norm_dim = {0, 2};
    mean = input_reshaped.mean(norm_dim, false);
    var = input_reshaped.var(norm_dim, false, false);
    auto n = input_reshaped.numel() / input_reshaped.size(1);
    running_mean.mul_(1. - momentum).add_(momentum * mean.detach());
    running_var.mul_(1. - momentum).add_(momentum * var.detach()).mul_(n).div_(n - 1);
  }
  else
  {
    mean = running_mean;
    var = running_var;
  }

  auto inv_std = torch::rsqrt(var + eps);

  auto x_hat = torch::zeros_like(input_reshaped);
  auto output_reshaped = torch::zeros_like(input_reshaped);

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

  AT_DISPATCH_FLOATING_TYPES(input_reshaped.scalar_type(), "batch_norm_forward_cuda", [&]
                             {
      batch_norm_cuda_forward_kernel<scalar_t><<<blocks_trans, threads_trans>>>(
        input_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        mean.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        inv_std.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        x_hat.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        output_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        eps);
      C10_CUDA_KERNEL_LAUNCH_CHECK(); });

  auto output = output_reshaped.view(input.sizes());
  return {output, mean, inv_std, x_hat};
}

std::vector<torch::Tensor> batch_norm_backward(torch::Tensor grad_out,
                                               torch::Tensor mean,
                                               torch::Tensor inv_var,
                                               torch::Tensor x_hat,
                                               torch::Tensor gamma)
{
  auto N = grad_out.size(0) * grad_out.size(2) * grad_out.size(3);

  auto grad_xhat = grad_out * gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3);

  auto grad_input = (1. / N) * inv_var *
                    (N * grad_xhat - grad_xhat.sum({0, 2, 3}, true) -
                     x_hat * (grad_xhat * x_hat).sum({0, 2, 3}, true));

  auto grad_weight = (x_hat * grad_out).sum({0, 2, 3});
  auto grad_bias = grad_out.sum({0, 2, 3});

  return {grad_input, grad_weight, grad_bias};
}
