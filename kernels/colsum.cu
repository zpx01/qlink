#include "qcuda.h"
// y[i] = sum over k < m of x[k * n + i]: a loop with a uniform trip count.
extern "C" __global__ void colsum(const int *x, int *y, int n, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int acc = 0;
#pragma unroll 1
  for (int k = 0; k < m; k++) acc += x[k * n + i];
  y[i] = acc;
}
