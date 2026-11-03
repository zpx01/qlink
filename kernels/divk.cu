#include "qcuda.h"

// q[i] = x[i] / k and r[i] = x[i] % k (C division: toward zero, r has the sign of x[i])
extern "C" __global__ void divk(const int *x, int *q, int *r, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    q[i] = x[i] / k;
    r[i] = x[i] % k;
  }
}
