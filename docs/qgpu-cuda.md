# qgpu CUDA Reference

qgpu is a SIMT GPU built into the weights of a Qwen3.5 language model. It runs CUDA C
kernels exactly: the model itself writes every step of a launch, and the result is read back
from the model's state. This reference describes the GPU, the subset of CUDA C it runs, and
how kernels are launched through the `cuda` tool. The last section has a kernel for each
common vector and matrix operation.

## 1. Architecture

| Property | Value |
|---|---|
| Data type | 32-bit integers (`int`, `unsigned`), two's complement, wrapping on overflow |
| Lanes per warp | 4 |
| Threads per launch | at most 128, in thread blocks of up to 32 |
| Registers per thread | 15 |
| Program size | 32 machine blocks (one short kernel) |
| Global memory | 126 words, shared by all the arrays of a launch |
| Floating point | none |

Warps run one after another; the lanes of a warp run together. Memory never faults: a read
past the end of an array returns a word of another array, so every kernel guards its index.

## 2. Execution model

A launch starts one thread per item. Every thread runs the whole kernel, at the same time as
the others. A thread finds its item from the built-in variables:

    int i = blockIdx.x * blockDim.x + threadIdx.x;

The launch is rounded up to whole warps, so some threads have no item: `if (i < n)` keeps
them out. Work written outside such a guard, or a loop over all the items, is done by every
thread: a loop that adds all the items in each thread counts every item once per thread.

Threads that take different branches of an `if` are supported (each lane runs under a mask).

## 3. Memory

- **Global memory** holds the kernel's arrays. Input arrays are filled by the host before the
  launch; output arrays start at zero.
- **Shared memory**: `__shared__ int` arrays are shared by the threads of a block;
  `__syncthreads()` waits for all of them.
- **Atomics**: `atomicAdd(int *address, int value)` adds to a global word without losing
  updates from other threads. Its return value cannot be used. A plain `x[0] += v` from many
  threads loses updates: the threads overwrite each other.

## 4. Kernels

A program is exactly one kernel:

    extern "C" __global__ void name(const int* in1, const int* in2, int* out, int n)

- **Input arrays** are `const int*`.
- **Output arrays** are `int*`. Only output arrays are returned.
- **The count** `int n` is the number of items.
- There is no host code: no `main`, `#include`, `printf`, `malloc`, `new`, other functions,
  global variables or recursion. The GPU launches the kernel itself.

### Operations

| Supported | Notes |
|---|---|
| `+ - *`, unary `-` | on `int` and `unsigned` |
| `/ %` | only by a constant or a kernel parameter (such as `n`), not by a per-thread value |
| `& \| ^ ~`, `<< >>` | shifts by a constant |
| `== != < <= > >=`, `&& \|\| !`, `?:` | `min` and `max` are written with `?:` |
| `if` / `else` | also when threads disagree |
| `for` / `while` | one exit condition; loops cost registers |
| `return` | at most one early return |
| `__shared__`, `__syncthreads()` | integer arrays, within a block |
| `__shfl_down_sync(0xffffffff, v, d)` | across whole warps, outside `if` branches |
| `atomicAdd` | global memory, return value unused |

Not supported: `float` and `double`, 64-bit integer arithmetic, division by a per-thread
value, atomics on shared memory, other atomic functions.

## 5. Launching with the cuda tool

A launch is one call to the `cuda` tool:

- `source`: the kernel.
- `args`: a JSON object with each input array, keyed by its parameter name, as a list of
  integers. Element `i` of every input array belongs to item `i`.

The host compiles the source, writes the inputs to global memory, creates each output array
(zeros, as long as the longest input), sets `n` to the number of items (the length of the
longest input) and launches one thread per item. The tool returns every output array by name.

## 6. Errors

| Error | Cause and fix |
|---|---|
| `does not compile` | a C error; for `undeclared identifier x`, declare `x` as a parameter |
| `needs more than 15 registers` | the kernel is too long or loops too much: shorten it |
| `machine blocks, the ROM holds 32` | the kernel is too long |
| `by a per-thread value` | divide only by a constant or a parameter |
| `an atomic whose old value is used` | do not use `atomicAdd`'s return value |
| `more than one early exit` | keep one `if (i < n)` guard and no other `return` |
| `the arrays hold more than 126 numbers` | the launch is too big for the GPU |
| `give the input array ...` | `args` must give each input array as a list of integers |
| `has no output array` | add an `int*` parameter for the results |

## 7. Vector and matrix operations

Each kernel below runs as shown. Vectors are arrays with one element per item; a matrix is
stored row by row in one array. A result with one value per item is written to `out[i]`; a
single value from all the items is accumulated in `out[0]` with `atomicAdd`.

### Vector addition

```cuda
extern "C" __global__ void add(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
```
Call: `{"a": [2, 3, 4], "b": [5, 6, 7]}` returns `{"out": [7, 9, 11]}`

### Vector subtraction

```cuda
extern "C" __global__ void subtract(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] - b[i];
}
```
Call: `{"a": [5, 6, 7], "b": [2, 3, 4]}` returns `{"out": [3, 3, 3]}`

### Scalar multiplication

```cuda
extern "C" __global__ void scale(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 5 * a[i];
}
```
Call: `{"a": [3, -1, 4]}` returns `{"out": [15, -5, 20]}`

### Linear combination

```cuda
extern "C" __global__ void combine(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 5 * a[i] + b[i];
}
```
Call: `{"a": [3, -1, 4], "b": [1, 1, 1]}` returns `{"out": [16, -4, 21]}`

### Element-wise product

```cuda
extern "C" __global__ void product(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}
```
Call: `{"a": [1, 2, 3], "b": [4, 5, 6]}` returns `{"out": [4, 10, 18]}`

### Dot product

```cuda
extern "C" __global__ void dot(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&out[0], a[i] * b[i]);
}
```
Call: `{"a": [1, 2, 3], "b": [4, 5, 6]}` returns `{"out": [32, 0, 0]}`: the dot product is
`out[0]`.

### Squared length

```cuda
extern "C" __global__ void length2(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&out[0], a[i] * a[i]);
}
```
Call: `{"a": [3, 4, 12]}` returns `{"out": [169, 0, 0]}`

### Sum of the elements

```cuda
extern "C" __global__ void sum(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&out[0], a[i]);
}
```
Call: `{"a": [12, 7, 30, 5]}` returns `{"out": [54, 0, 0, 0]}`

### Squared distance

```cuda
extern "C" __global__ void distance2(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int d = a[i] - b[i];
    atomicAdd(&out[0], d * d);
  }
}
```
Call: `{"a": [1, 2, 3], "b": [4, 6, 3]}` returns `{"out": [25, 0, 0]}`

### Count of the elements above a value

```cuda
extern "C" __global__ void count_above(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&out[0], a[i] > 10 ? 1 : 0);
}
```
Call: `{"a": [12, 7, 30, 5]}` returns `{"out": [2, 0, 0, 0]}`

### Cross product

```cuda
extern "C" __global__ void cross(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3) {
    int j = (i + 1) % 3, k = (i + 2) % 3;
    out[i] = a[j] * b[k] - a[k] * b[j];
  }
}
```
Call: `{"a": [1, 2, 3], "b": [4, 5, 6]}` returns `{"out": [-3, 6, -3]}`

### Matrix-vector product

For a D×D matrix `m`, stored row by row, and a vector `v` of D elements; thread `i` computes
row `i`.

```cuda
extern "C" __global__ void matvec(const int* m, const int* v, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int D = 2;
  if (i < D) {
    int s = 0;
    for (int k = 0; k < D; k++) s += m[i * D + k] * v[k];
    out[i] = s;
  }
}
```
Call: `{"m": [1, 2, 3, 4], "v": [5, 6]}` returns `{"out": [17, 39, 0, 0]}`: the product is
the first D elements.

### Matrix product

For D×D matrices `a` and `b`, stored row by row; thread `i` computes element `i` of the
product, in row `i / D` and column `i % D`.

```cuda
extern "C" __global__ void matmul(const int* a, const int* b, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int D = 2;
  if (i < D * D) {
    int r = i / D, c = i % D, s = 0;
    for (int k = 0; k < D; k++) s += a[r * D + k] * b[k * D + c];
    out[i] = s;
  }
}
```
Call: `{"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}` returns `{"out": [19, 22, 43, 50]}`, the
matrix `[[19, 22], [43, 50]]`.

### Maximum

One thread scans the array; the others do nothing.

```cuda
extern "C" __global__ void maximum(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    int m = a[0];
    for (int k = 1; k < n; k++) m = a[k] > m ? a[k] : m;
    out[0] = m;
  }
}
```
Call: `{"a": [71, 88, 64, 93, 79]}` returns `{"out": [93, 0, 0, 0, 0]}`

### Running sum

Element `i` is the sum of elements `0` to `i`. Every thread runs the same number of rounds
and adds only the elements up to its own.

```cuda
extern "C" __global__ void running_sum(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int s = 0;
    for (int k = 0; k < n; k++) s += a[k] * (k <= i);
    out[i] = s;
  }
}
```
Call: `{"a": [1, 2, 3, 4]}` returns `{"out": [1, 3, 6, 10]}`
