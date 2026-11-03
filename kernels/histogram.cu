#include "qcuda.h"
// 16-bin histogram of x[i] & 15 with global atomics.
extern "C" __global__ void histogram(const unsigned *x, unsigned *bins, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[x[i] & 15], 1u);
}
