#include "qcuda.h"
// Block tree reduction in shared memory; thread 0 adds the block's sum to *out.
// blockDim.x must be a power of two <= 32 (the array is sized to fit qgpu's 128-word memory).
extern "C" __global__ void reduce(const int *x, int *out, int n) {
  __shared__ int s[32];
  unsigned t = threadIdx.x;
  unsigned i = blockIdx.x * blockDim.x + t;
  s[t] = i < n ? x[i] : 0;
  __syncthreads();
  for (unsigned k = blockDim.x / 2; k > 0; k >>= 1) {
    if (t < k) s[t] += s[t + k];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}
