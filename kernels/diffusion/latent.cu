#include "../qcuda.h"
// Latent image diffusion: the denoiser eps_hat(z_t, t) is an MLP 9 -> 16 -> 8 (ReLU) split over
// 4 QwenGPUs as Megatron splits an MLP: this rank holds 4 hidden units. Q8 fixed point.
// Layouts: w1[i * 4 + k] (input i: 8 latent dims, then time), b1[k], w2[k * 8 + o], b2[o]
// (replicated), in[9], z[4], out[8 + padding]; one image per launch.
#define D 8
#define HR 4
#define NIN 9
#define RS(x, s) (((x) + (1 << ((s) - 1))) >> (s))

// z[k] = (W1[:, k] . in) >> 8 + b1[k]
extern "C" __global__ void lt_hidden(const int *w1, const int *b1, const int *in, int *z) {
  int k = threadIdx.x;
  int s = 0;
  for (int i = 0; i < NIN; i++) s += in[i] * w1[i * HR + k];
  z[k] = (s >> 8) + b1[k];
}

// out[o] += sum_k (relu(z[k]) W2[k][o]) >> 8, plus b2[o] on rank 0 (whose out starts at -eps,
// so the AllReduce gives eps_hat - eps)
extern "C" __global__ void lt_out(const int *w2, const int *b2, const int *z, int *out, int bias) {
  int o = threadIdx.x;
  int s = out[o];
  for (int k = 0; k < HR; k++) {
    int v = z[k];
    s += ((v > 0 ? v : 0) * w2[k * D + o]) >> 8;
  }
  int c = b2[o];
  out[o] = s + (bias ? c : 0);
}

// The backward pass, after the AllReduce has left e = eps_hat - eps in out[0..8):
//   lt_dz   dz[k] = relu'(z[k]) (W2[k] . e) >> 8, into out[8 + k] (the AllReduce's padding)
//   lt_w1   W1 -= in dz, b1 -= dz: b1 sits right after W1 as its row 9 with input 1.0 (256),
//           and RS(dz * 256, s + 8) == RS(dz, s)
//   lt_w2   W2 -= relu(z) e, b2 -= e: b2 sits right after W2 as its row 4 with relu(z) = 1.0
// dz reads the old W2, so lt_dz runs before lt_w2. Rows past the end (in[9], z[4]) are loaded
// but replaced by the constant: the selects need no branch.
extern "C" __global__ void lt_dz(const int *w2, const int *z, int *e) {
  int k = threadIdx.x;
  int g = 0;
  for (int o = 0; o < D; o++) g += w2[k * D + o] * e[o];
  int zz = z[k];
  e[D + k] = zz > 0 ? g >> 8 : 0;
}

extern "C" __global__ void lt_w1(int *w1, const int *in, const int *e, int shift) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;  // (i, k) = (t / 4, t % 4), i in [0, 10)
  int i = t >> 2, k = t & 3;
  int dz = e[D + k], xi = in[i];
  int x = i < NIN ? xi : 256;
  w1[t] -= RS(dz * x, shift + 8);
}

extern "C" __global__ void lt_w2(int *w2, const int *z, const int *e, int shift) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;  // (k, o) = (t / 8, t % 8), k in [0, 5)
  int k = t >> 3, o = t & 7;
  int v = z[k];
  int a = k < HR ? (v > 0 ? v : 0) : 256;
  w2[t] -= RS((a * e[o]) >> 8, shift);
}

// One DDIM step: z <- (A z + B eps_hat) / 4096, and the next time feature
extern "C" __global__ void lt_ddim(int *in, const int *eps, int A, int B, int tnext) {
  int o = threadIdx.x;
  in[o] = RS(A * in[o] + B * eps[o], 12);
  if (o != 0) return;
  in[D] = tnext;
}
