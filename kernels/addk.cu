#include "qcuda.h"

// y[i] = x[i] + k: one number added to every element
extern "C" __global__ void addk(const int *x, int *y, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + k;
}
