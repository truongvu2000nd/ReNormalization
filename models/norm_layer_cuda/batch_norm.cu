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

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[C10_WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % C10_WARP_SIZE == 0) {
    shared[tid / C10_WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / C10_WARP_SIZE && tid < C10_WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / C10_WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t>
__global__ void batch_norm_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean_,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> invstd_,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
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

  scalar_t gamma = weight[plane];
  scalar_t beta = bias[plane];
  scalar_t mean = mean_[plane];
  scalar_t invstd = invstd_[plane];
  for (int batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep)
  {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (int feature = threadIdx.x; feature < fs; feature += blockDim.x)
    {
      o[feature] = gamma * (i[feature] - mean) * invstd + beta;
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

  auto invstd = torch::rsqrt(var + eps);

  auto output = torch::zeros_like(input);
  auto output_reshaped = output.view(input_reshaped.sizes());

  int tf = std::max<int>(getNumThreads(input.size(2) / 4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64 / tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256 * 1024) / input.size(1),
                                                                  (input.size(0) + tb - 1) / tb)));
  blocks_trans.y = std::min(blocks_trans.y, MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);

  AT_DISPATCH_FLOATING_TYPES(input_reshaped.scalar_type(), "batch_norm_forward_cuda", [&]
                             {
    batch_norm_cuda_forward_kernel<scalar_t><<<blocks_trans, threads_trans>>>(
      input_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      mean.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      invstd.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      weight.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      output_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); });

  return {output, mean, invstd};
}

template <typename scalar_t>
__global__ void batch_norm_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_weight,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_bias,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> mean_,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> invstd_,
    int N)
{
  int plane = blockIdx.x;
  scalar_t mean = mean_[plane];
  scalar_t invstd = invstd_[plane];
  scalar_t weight_val = weight[plane];
  scalar_t norm = scalar_t(1) / N;

  GradOp<scalar_t, scalar_t, torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>> g(mean, input, grad_output);
  auto res = reduce<Float2<scalar_t, scalar_t>>(g, grad_output, plane);

  scalar_t grad_output_sum = res.v1;
  scalar_t dot_p = res.v2;

  scalar_t grad_mean = grad_output_sum * norm;
  scalar_t proj_scale = dot_p * norm * invstd * invstd;
  scalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        scalar_t go = grad_output[batch][plane][x];
        scalar_t inp = input[batch][plane][x];
        scalar_t proj = (inp - mean) * proj_scale;
        grad_input[batch][plane][x] = (go - proj - grad_mean) * grad_scale;
      }
    }
  }

  if (threadIdx.x == 0) {
    grad_weight[plane] = dot_p;
  }

  if (threadIdx.x == 0) {
    grad_bias[plane] = grad_output_sum;
  }
}

std::vector<torch::Tensor> batch_norm_cuda_backward(torch::Tensor grad_out_,
                                                    torch::Tensor input_,
                                                    torch::Tensor weight_,
                                                    torch::Tensor mean_,
                                                    torch::Tensor invstd_,)
{
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  auto grad_input = torch::zeros_like(input);
  auto grad_input_reshaped = grad_input_.view(input_reshaped.sizes());

  auto grad_weight = torch::zeros_like(weight_);
  auto grad_bias = torch::zeros_like(weight_);
  
  auto N = grad_out_reshaped.size(0) * grad_out_reshaped.size(2);

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

  AT_DISPATCH_FLOATING_TYPES(grad_out_reshaped.scalar_type(), "batch_norm_backward_cuda", [&]
                             {
    batch_norm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
      input_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      grad_out_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      grad_input_reshaped.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      grad_weight.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      grad_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      weight_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      mean_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      invstd_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK(); });

  return {grad_input, grad_weight, grad_bias};
}
