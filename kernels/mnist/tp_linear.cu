#include "../qcuda.h"
// MNIST on QwenGPUs: a linear classifier, tensor-parallel as Megatron's RowParallelLinear.
// Rank r holds features [8 r, 8 r + 8) of every input (32 Q8 features: 31 whitened principal
// components and a bias, or 25 pixels of a 5x5 image, a bias and padding) and the 10 x 8 slice of W for them (Q16, w[c * 8 + j]). The ranks'
// partial logits are summed by a qccl AllReduce; the update needs only local data.
#define F 8
#define NCLS 10

// z[c] = (sum_j w[c][j] x[j]) >> 8 for the 10 classes, 0 for the padding up to 16 threads
extern "C" __global__ void tp_forward(const int *w, const int *x, int *z) {
  int c = threadIdx.x;
  int s = 0;
  for (int j = 0; j < F; j++) s += w[c * F + j] * x[j];
  z[c] = c < NCLS ? s >> 8 : 0;
}

// The error of the full logits, in place: e[c] = z[c] - (c == label ? 1.0 : 0) (Q16)
extern "C" __global__ void tp_error(int *z, const int *label) {
  int c = threadIdx.x;
  z[c] -= c == label[0] ? 65536 : 0;
}

// One-vs-rest squared hinge, in place: the target is +1 for the label and -1 for the other
// classes (Q16), and a class already past its margin contributes nothing:
// e[c] = s z[c] < 1 ? z[c] - s : 0. The update below is the same as for least squares.
extern "C" __global__ void tp_hinge(int *z, const int *label) {
  int c = threadIdx.x;
  int v = z[c], y = label[0];
  int s = c == y ? 65536 : -65536;
  int past = c == y ? v >= 65536 : v <= -65536;
  z[c] = past || c >= NCLS ? 0 : v - s;
}

// Least-mean-squares step on classes [c0, c0 + 5) (two launches cover the 10; ten unrolled
// classes would not fit the 32-block ROM): thread j updates feature j,
// w[c][j] -= (e[c] x[j]) >> shift
extern "C" __global__ void tp_update(int *w, const int *x, const int *e, int c0, int shift) {
  int j = threadIdx.x;
  int xj = x[j];
  for (int c = c0; c < c0 + 5; c++) w[c * F + j] -= (e[c] * xj) >> shift;
}
