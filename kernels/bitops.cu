#include "qcuda.h"
// Per-thread bit operations, shifts by constants, high multiplies and a select.
extern "C" __global__ void bitops(const int *x, int *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int v = x[i];
  unsigned u = (unsigned)v;
  int a = ((u >> 3) ^ (u << 5)) & 0x0F0F0F0F;
  int b = (v >> 7) | (int)(u >> 29);
  int c = __umulhi(u, 0x9E3779B9u);
  int d = __mulhi(v, -12345);
  y[i] = (v < 0 ? a : b) + c - d;
}
