#include "qcuda.h"
// Every lane's shuffle results: lanes past the end (or past their width-8 segment) keep
// their own value.
extern "C" __global__ void shfl(const int *x, int *y, int *z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v = i < n ? x[i] : i;
  int a = __shfl_down_sync(0xffffffffu, v, 5);
  int b = __shfl_down_sync(0xffffffffu, v, 3, 8);
  if (i < n) {
    y[i] = a;
    z[i] = b;
  }
}
