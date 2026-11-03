#include "qcuda.h"

// y[i] = x[i] * k: every element multiplied by one number
extern "C" __global__ void mulk(const int *x, int *y, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * k;
}
