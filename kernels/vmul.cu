#include "qcuda.h"

// c = a * b, elementwise (int32, low 32 bits)
extern "C" __global__ void vmul(const int *a, const int *b, int *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];
}
