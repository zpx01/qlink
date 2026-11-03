// Minimal CUDA device shim for clang without the CUDA SDK (-nocudainc -nocudalib).
// Gives kernels the usual spellings: __global__, __shared__, threadIdx, ... (__syncthreads is
// a clang builtin).
#pragma once
#ifdef __NVCC__
// nvcc includes the real CUDA headers, which define all of this.
#else
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#include <__clang_cuda_builtin_vars.h>

static __device__ __forceinline__ int atomicAdd(int *p, int v) {
  return __nvvm_atom_add_gen_i(p, v);
}
static __device__ __forceinline__ unsigned atomicAdd(unsigned *p, unsigned v) {
  return (unsigned)__nvvm_atom_add_gen_i((int *)p, (int)v);
}
static __device__ __forceinline__ unsigned __umulhi(unsigned a, unsigned b) {
  return __nvvm_mulhi_ui(a, b);
}
static __device__ __forceinline__ int __mulhi(int a, int b) { return __nvvm_mulhi_i(a, b); }
// fences: membar.sys / membar.gl (no-ops on qgpu: one warp at a time, stores leave in order)
static __device__ __forceinline__ void __threadfence_system() { __nvvm_membar_sys(); }
static __device__ __forceinline__ void __threadfence() { __nvvm_membar_gl(); }
static __device__ __forceinline__ int __shfl_down_sync(unsigned mask, int v, unsigned delta,
                                                       int width = 32) {
  // the clamp operand as CUDA's header builds it: segment mask in bits 8-12, lane limit 31
  return __nvvm_shfl_sync_down_i32(mask, v, delta, ((32 - width) << 8) | 0x1f);
}
#endif
