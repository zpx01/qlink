#include "qcuda.h"
// Inclusive prefix sum within each block (Hillis-Steele, double-buffered shared memory).
// blockDim.x <= 16 (the buffers are sized to fit qgpu's 128-word memory).
extern "C" __global__ void scan(const int *x, int *y, int n) {
  __shared__ int buf[2][16];
  unsigned t = threadIdx.x;
  unsigned i = blockIdx.x * blockDim.x + t;
  int cur = 0;
  buf[0][t] = i < n ? x[i] : 0;
  __syncthreads();
  for (unsigned d = 1; d < blockDim.x; d <<= 1) {
    int v = buf[cur][t];
    if (t >= d) v += buf[cur][t - d];
    buf[1 - cur][t] = v;
    cur = 1 - cur;
    __syncthreads();
  }
  if (i < n) y[i] = buf[cur][t];
}
