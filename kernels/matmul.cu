#include "qcuda.h"
// C = A * B for n x n int matrices, one thread per output element (2D grid).
extern "C" __global__ void matmul(const int *A, const int *B, int *C, int n) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < n && c < n) {
    int acc = 0;
    for (int k = 0; k < n; k++) acc += A[r * n + k] * B[k * n + c];
    C[r * n + c] = acc;
  }
}
