#include "qcuda.h"
// A loop whose trip count differs per thread: a divergent branch (not supported yet).
extern "C" __global__ void divloop(const int *x, int *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int acc = 0;
#pragma unroll 1
  for (int k = 0; k < i; k++) acc += x[k];
  y[i] = acc;
}
