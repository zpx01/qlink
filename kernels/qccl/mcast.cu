#include "../qcuda.h"
// Multicast (NVLS-style) helpers. A multicast window is memory at the same address on every
// rank whose stores land locally and at every peer (NVLink SHARP's multimem.st on a multicast
// object): one store from a GPU's weights reaches every replica.

// dst[i] = src[i] for one thread block (dst usually in a multicast window), then post `epoch`
// to *flag (also in a window): every rank that sees the flag has the data, links being FIFO.
extern "C" __global__ void qccl_mcast(const int *src, int *dst, int *flag, int epoch) {
  int i = threadIdx.x;
  dst[i] = src[i];
  __threadfence_system();
  __syncthreads();
  if (i != 0) return;
  *flag = epoch;
}

// Wait until flags[0..n) are all >= value (each written by a different rank).
extern "C" __global__ void qccl_wait(const volatile int *flags, int n, int value) {
#pragma unroll 1
  for (int k = 0; k < n; k++)
    while (flags[k] < value) {
    }
}
