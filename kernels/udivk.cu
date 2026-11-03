#include "qcuda.h"

// The unsigned flavour of divk (div.u32 / rem.u32), for the tests
extern "C" __global__ void udivk(const unsigned *x, unsigned *q, unsigned *r, unsigned k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    q[i] = x[i] / k;
    r[i] = x[i] % k;
  }
}
