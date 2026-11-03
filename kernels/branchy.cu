#include "qcuda.h"
// Lanes of one warp take different paths through an if / else-if / else chain.
extern "C" __global__ void branchy(const int *x, int *y, int *z, int n, int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int v = x[i];
  if (v > t) {
    y[i] = v - t;
  } else if (v == t) {
    z[i] = v + i;
    y[i] = 0;
  } else {
    int s = 0;
    for (int k = 0; k < 3; k++) s += x[(i + k) % 1 + i];
    y[i] = s + t;
  }
}
