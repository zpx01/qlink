#include "qcuda.h"

// Average of x (n <= 32, one block): a shared-memory sum, then thread 0 writes the whole part
// out[0] = sum / n and the hundredths out[1] = (sum % n) * 100 / n (C division, toward zero).
extern "C" __global__ void mean(const int *x, int *out, int n) {
  __shared__ int s[32];
  unsigned t = threadIdx.x;
  s[t] = t < n ? x[t] : 0;
  __syncthreads();
  for (unsigned k = blockDim.x / 2; k > 0; k >>= 1) {
    if (t < k) s[t] += s[t + k];
    __syncthreads();
  }
  if (t == 0) {
    int sum = s[0];
    out[0] = sum / n;
    out[1] = sum % n * 100 / n;
  }
}
