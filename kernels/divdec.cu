#include "qcuda.h"

// x[i] / k to two decimals: the whole part w[i] (toward zero) and the hundredths h[i]
// (with the quotient's sign), as mean.cu does for an average
extern "C" __global__ void divdec(const int *x, int *w, int *h, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int a = x[i];
    w[i] = a / k;
    h[i] = a % k * 100 / k;
  }
}
