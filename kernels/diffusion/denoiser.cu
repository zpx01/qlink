#include "../qcuda.h"
// A diffusion model's denoiser on QwenGPUs: eps_hat(x_t, t), an MLP 3 -> 32 -> 2 (ReLU), tensor
// parallel as Megatron's MLP: rank r holds 8 of the 32 hidden units (W1 columns, b1, W2 rows);
// a qccl AllReduce sums the ranks' partial outputs. Q8 fixed point (1.0 = 256).
// Layouts: w1[i * 8 + k] (input i: x, y, t), b1[k], w2[k * 2 + o], b2[o] (replicated), a sample
// s: in[s * 3 + i], z[s * 8 + k], out[s * 2 + o].
#define HR 8

// x / 2^s rounded to nearest: tiny updates do not drift the way a floor shift would
#define RS(x, s) (((x) + (1 << ((s) - 1))) >> (s))

// z = W1 . in + b1 (stored for the backward pass), out[s] += W2 . relu(z) (+ b2 on rank 0, whose
// out starts at -eps: after the AllReduce, out is eps_hat - eps, the loss gradient)
extern "C" __global__ void dn_forward(const int *w1, const int *b1, const int *w2,
                                      const int *b2, const int *in, int *z, int *out,
                                      int bias) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = t >> 3, k = t & (HR - 1);
  int v = ((in[s * 3] * w1[k] + in[s * 3 + 1] * w1[HR + k] + in[s * 3 + 2] * w1[2 * HR + k]) >> 8)
          + b1[k];
  z[t] = v;
  int a = v > 0 ? v : 0;
  int c0 = b2[0], c1 = b2[1];  // loaded unconditionally: the selects need no branch
  int add = bias && k == 0;
  atomicAdd(&out[s * 2], ((a * w2[k * 2]) >> 8) + (add ? c0 : 0));
  atomicAdd(&out[s * 2 + 1], ((a * w2[k * 2 + 1]) >> 8) + (add ? c1 : 0));
}

// eps_hat for a batch of points, nothing kept (generation): out[s] += W2 . relu(W1 . in + b1)
extern "C" __global__ void dn_eps(const int *w1, const int *b1, const int *w2, const int *b2,
                                  const int *in, int *out, int bias) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = t >> 3, k = t & (HR - 1);
  int v = ((in[s * 3] * w1[k] + in[s * 3 + 1] * w1[HR + k] + in[s * 3 + 2] * w1[2 * HR + k]) >> 8)
          + b1[k];
  int a = v > 0 ? v : 0;
  int c0 = b2[0], c1 = b2[1];
  int add = bias && k == 0;
  atomicAdd(&out[s * 2], ((a * w2[k * 2]) >> 8) + (add ? c0 : 0));
  atomicAdd(&out[s * 2 + 1], ((a * w2[k * 2 + 1]) >> 8) + (add ? c1 : 0));
}

// The first layer's update: dz = relu'(z) (W2 . e), W1 -= in dz, b1 -= dz (summed over the batch
// by atomicAdd; the second layer is updated after this, from the same W2)
extern "C" __global__ void dn_backward1(int *w1, int *b1, const int *w2, const int *in,
                                        const int *z, const int *e, int shift) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int s = t >> 3, k = t & (HR - 1);
  int w20 = w2[k * 2], w21 = w2[k * 2 + 1], e0 = e[s * 2], e1 = e[s * 2 + 1];
  int g = (e0 * w20 + e1 * w21) >> 8;
  int dz = z[t] > 0 ? g : 0;
  atomicAdd(&w1[k], -RS(dz * in[s * 3], shift + 8));
  atomicAdd(&w1[HR + k], -RS(dz * in[s * 3 + 1], shift + 8));
  atomicAdd(&w1[2 * HR + k], -RS(dz * in[s * 3 + 2], shift + 8));
  atomicAdd(&b1[k], -RS(dz, shift));
}

// The second layer's update for a batch of 4: thread (k, o): W2[k][o] -= sum_s relu(z) e
extern "C" __global__ void dn_backward2(int *w2, const int *z, const int *e, int shift) {
  int t = threadIdx.x;
  int k = t >> 1, o = t & 1;
  int g = 0;
  for (int s = 0; s < 4; s++) {
    int v = z[s * HR + k];
    g += ((v > 0 ? v : 0) * e[s * 2 + o]) >> 8;
  }
  w2[t] -= RS(g, shift);
}

// The output bias, replicated on every rank: b2[o] -= sum_s e[s][o]
extern "C" __global__ void dn_bias(int *b2, const int *e, int shift) {
  int o = threadIdx.x;
  if (o >= 2) return;
  b2[o] -= RS(e[o] + e[2 + o] + e[4 + o] + e[6 + o], shift);
}

// One DDIM step for a batch of points: x <- (A x + B eps_hat) / 4096, and the next time feature
extern "C" __global__ void dn_ddim(int *in, const int *eps, int A, int B, int tnext) {
  int s = threadIdx.x;
  int x = in[s * 3], y = in[s * 3 + 1];
  in[s * 3] = RS(A * x + B * eps[s * 2], 12);
  in[s * 3 + 1] = RS(A * y + B * eps[s * 2 + 1], 12);
  in[s * 3 + 2] = tnext;
}
