#include "qcuda.h"
// y = a * x + y (integer)
extern "C" __global__ void saxpy(int a, const int *x, int *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
