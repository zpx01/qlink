#include "qcuda.h"

// Dot product: each block multiplies its elements and tree-reduces them in shared memory;
// thread 0 adds the block's sum to *out. blockDim.x must be a power of two <= 32.
extern "C" __global__ void dot(const int *a, const int *b, int *out, int n) {
  __shared__ int s[32];
  unsigned t = threadIdx.x;
  unsigned i = blockIdx.x * blockDim.x + t;
  s[t] = i < n ? a[i] * b[i] : 0;
  __syncthreads();
  for (unsigned k = blockDim.x / 2; k > 0; k >>= 1) {
    if (t < k) s[t] += s[t + k];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}
