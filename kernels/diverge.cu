#include "qcuda.h"
// Lanes of one warp take different branches.
extern "C" __global__ void diverge(const int *x, int *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int v = x[i];
  if (v % 3 == 0) v = v / 3;
  else if (v % 3 == 1) v = v * 2 - 7;
  else v = -v ^ 0x55;
  y[i] = v;
}
