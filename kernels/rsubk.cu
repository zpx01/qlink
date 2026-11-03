#include "qcuda.h"

// y[i] = k - x[i]
extern "C" __global__ void rsubk(const int *x, int *y, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = k - x[i];
}
