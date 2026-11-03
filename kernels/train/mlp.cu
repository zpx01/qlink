#include "../qcuda.h"
// A 2-8-1 ReLU MLP in Q8 fixed point (1.0 = 256), trained with mean-squared error.
// Parameters (32 words): W1[k][j] at p[2k + j], b1[k] at p[16 + k], W2[k] at p[24 + k].
// Gradients use the same layout. One thread per (sample s, hidden unit k): t = 8 s + k.

#define HID 8

// o[s] += W2[k] * relu(W1[k] . x[s] + b1[k]), summed over k by atomicAdd (o starts at 0)
extern "C" __global__ void mlp_forward(const int *p, const int *x, int *o) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = t >> 3, k = t & (HID - 1);
  int z = ((p[2 * k] * x[2 * s] + p[2 * k + 1] * x[2 * s + 1]) >> 8) + p[16 + k];
  int a = z > 0 ? z : 0;
  atomicAdd(&o[s], (p[24 + k] * a) >> 8);
}

// dL/do = o - y; backpropagate through unit k (recomputing its pre-activation) and add this
// sample's gradients into g; loss += (o - y)^2 / 256 once per sample
extern "C" __global__ void mlp_backward(const int *p, const int *x, const int *y,
                                        const int *o, int *g, int *loss) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = t >> 3, k = t & (HID - 1);
  int e = o[s] - y[s];
  int x0 = x[2 * s], x1 = x[2 * s + 1];
  int w2 = p[24 + k];  // loaded unconditionally: the select below needs no branch
  int z = ((p[2 * k] * x0 + p[2 * k + 1] * x1) >> 8) + p[16 + k];
  int a = z > 0 ? z : 0;
  int dz = z > 0 ? (e * w2) >> 8 : 0;
  atomicAdd(&g[24 + k], (e * a) >> 8);
  atomicAdd(&g[2 * k], (dz * x0) >> 8);
  atomicAdd(&g[2 * k + 1], (dz * x1) >> 8);
  atomicAdd(&g[16 + k], dz);
  atomicAdd(loss, k == 0 ? (e * e) >> 8 : 0);
}

// SGD: p -= round(g / 2^shift); then g = 0 for the next step. One thread per parameter.
extern "C" __global__ void sgd(int *p, int *g, int shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  p[i] -= (g[i] + (1 << (shift - 1))) >> shift;
  g[i] = 0;
}

// ZeRO-1 SGD on this rank's shard: p, g point at the shard (p in a multicast window, so the
// update lands in every replica); then post `epoch` to this rank's flag (also multicast).
extern "C" __global__ void sgd_shard(int *p, const int *g, int *flag, int shift, int epoch) {
  int i = threadIdx.x;
  p[i] -= (g[i] + (1 << (shift - 1))) >> shift;
  __threadfence_system();
  __syncthreads();
  if (i != 0) return;
  *flag = epoch;
}
