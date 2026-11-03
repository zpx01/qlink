#include "qcuda.h"

// Python's floor division by one number: q[i] = x[i] // k, r[i] = x[i] % k (r has k's sign)
extern "C" __global__ void fdivk(const int *x, int *q, int *r, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int a = x[i], qq = a / k, rr = a - qq * k;
    int adj = (rr != 0) && ((rr < 0) != (k < 0));
    q[i] = qq - adj;
    r[i] = rr + (adj ? k : 0);
  }
}
