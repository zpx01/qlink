#include "qcuda.h"

// Largest element of x (signed): one block, a shared-memory max tree; thread 0 writes *out.
// blockDim.x must be a power of two <= 32 and >= n.
extern "C" __global__ void vmax(const int *x, int *out, int n) {
  __shared__ int s[32];
  unsigned t = threadIdx.x;
  s[t] = x[t < n ? t : 0];  // padding lanes repeat x[0]
  __syncthreads();
  for (unsigned k = blockDim.x / 2; k > 0; k >>= 1) {
    if (t < k) { int o = s[t + k]; if (o > s[t]) s[t] = o; }
    __syncthreads();
  }
  if (t == 0) *out = s[0];
}
