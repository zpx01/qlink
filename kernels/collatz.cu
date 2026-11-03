#include "qcuda.h"
// Loop with a lane-dependent trip count: Collatz steps to reach 1.
extern "C" __global__ void collatz(const unsigned *x, unsigned *steps, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned v = x[i], s = 0;
  while (v > 1) {
    v = (v & 1) ? 3 * v + 1 : v >> 1;
    s++;
  }
  steps[i] = s;
}
