#include "qcuda.h"
// Warp-shuffle reduction: lane 0 of each warp writes its warp's sum.
extern "C" __global__ void warpsum(const int *x, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v = i < n ? x[i] : 0;
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
  if ((threadIdx.x & 31) == 0) out[i >> 5] = v;
}
