"""CUDA as Qwen writes it: compiled to PTX on the host, laid out, lowered, run (reference and
circuit), and the errors Qwen gets back."""

import pytest

from qgpu.cuda import CUDAError, compile_cuda, plan
from qgpu.runtime import host_tokens, read, reference

ELEMENTWISE = """#include <cuda_runtime.h>
extern "C" __global__ void f(const int* a, const int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i] - 4;
}"""

COPY = """extern "C" __global__ void copy(const int *x, int *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}"""

SUM = """__global__ void total(int *x, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, x[i]);
}"""

RUNNING = """extern "C" __global__ void running(const int *x, int *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int s = 0;
    for (int k = 0; k < n; k++)  // the same trip count in every thread, no branch inside
      s += x[k] * (k <= i);
    y[i] = s;
  }
}"""

RUNNING_DIVERGENT = RUNNING.replace("k < n; k++)  // the same trip count in every thread, no "
                                    "branch inside\n      s += x[k] * (k <= i);",
                                    "k <= i; k++) s += x[k];")

SHARED = """extern "C" __global__ void block_max(const int *x, int *out, int n) {
  __shared__ int s[32];
  int t = threadIdx.x;
  s[t] = t < n ? x[t] : -2147483647;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    if (t < k && s[t + k] > s[t]) s[t] = s[t + k];
    __syncthreads();
  }
  if (t == 0) out[0] = s[0];
}"""

FLOORDIV = """extern "C" __global__ void floordiv(const int *x, int *q, int *r, int k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int a = x[i], qq = a / k, rr = a - qq * k;
    int adj = (rr != 0) && ((rr < 0) != (k < 0));  // two predicates: and.pred
    q[i] = qq - adj;
    r[i] = rr + (adj ? k : 0);
  }
}"""


@pytest.mark.parametrize("source,args,threads,want", [
    (ELEMENTWISE, {"a": [2, 3, 4, 5], "b": [2, 3, 4, 6], "c": {"zeros": 4}, "n": 4}, 4,
     {"c": [0, 5, 12, 26]}),
    (ELEMENTWISE, {"a": [2, 3, 4, 5], "b": [2, 3, 4, 6], "n": 4}, None,  # c, threads: defaults
     {"c": [0, 5, 12, 26]}),
    (COPY, {"x": [4200, 8100, 6500, 9900, 7300], "n": 5}, None,  # a load, then a write-back
     {"y": [4200, 8100, 6500, 9900, 7300]}),
    (SUM, {"x": ["12", "7", "30", "5"], "out": {"zeros": 1}, "n": "4"}, None,  # strings: numbers
     {"out": [54]}),
    (RUNNING, {"x": [4200, 8100, 6500, 9900, 7300], "n": 5}, None,
     {"y": [4200, 12300, 18800, 28700, 36000]}),
    (SHARED, {"x": [71, 88, 64, 93, 79], "out": {"zeros": 1}, "n": 5}, 8, {"out": [93]}),
    (FLOORDIV, {"x": [-7, 7, 0, -1, 5, -6, 9, -9], "k": -3, "n": 8}, None,
     {"q": [x // -3 for x in [-7, 7, 0, -1, 5, -6, 9, -9]],
      "r": [x % -3 for x in [-7, 7, 0, -1, 5, -6, 9, -9]]}),
])
def test_run(source, args, threads, want):
    """Compiled, laid out and run on the reference: the arrays the kernel writes."""
    p = plan(compile_cuda(source), args, threads, 4, 4)
    assert reference(p) == want


def test_through_circuit():
    """A kernel Qwen-style source through the circuit (merged config): every owned token is
    right and memory, read as the host reads the model's state, holds the outputs."""
    from chipformer.q35.ir import evaluate as ir

    from qgpu.circuit.kernel import Kernel, memory, predicted, run_blocks
    from qgpu.merge import STAGES

    mul, logic, atom = STAGES[4]
    c = Kernel(lanes=4, stages=4, mul_stages=mul, logic_stages=logic, atom_stages=atom)
    for source, args, threads in ((ELEMENTWISE, {"a": [2, 3, 4, 5], "b": [2, 3, 4, 6], "n": 4}, 4),
                                  (FLOORDIV, {"x": [-7, 7, -1, 5], "k": 2, "n": 4}, 4)):
        p = plan(compile_cuda(source), args, threads, 4, 4)
        lw = p.launch
        tr = run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
        assert tr.tokens[:len(host_tokens(lw))] == host_tokens(lw)
        carry = {}
        m = ir(c, tr.tokens[:-1], carry=carry)
        assert all(predicted(c, m, t) == tr.tokens[t + 1]
                   for t in tr.owned if t + 1 < len(tr.tokens))
        assert read(p, memory(c, carry)) == read(p, tr.mem)


@pytest.mark.parametrize("source,args,threads,says", [
    ("__global__ void f(float *x, int n) {}", {"x": [1], "n": 1}, 1, "integers only"),
    ("int main() { return 0; }", {}, 1, "one __global__ kernel"),
    ("__global__ void f(int *x, int n) { x[0] = y; }", {"x": [1], "n": 1}, 1, "does not compile"),
    (ELEMENTWISE, [[1, 2], [3, 4]], 2, "by parameter name"),
    (ELEMENTWISE, {"a": [1, 2], "b": [3, 4], "m": 2}, 2, "no parameter m"),
    (ELEMENTWISE, {"a": [1, 2], "n": 2}, 2, "give the input array b"),
    (ELEMENTWISE, {"a": [1, 2], "b": 5, "n": 2}, 2, "give the input array b"),
    (ELEMENTWISE, {"a": [1, 2], "b": [3, 4], "n": 2}, 500, "threads must be"),
    (ELEMENTWISE, {"a": list(range(60)), "b": list(range(60)), "n": 60}, 64, "numbers"),
    (RUNNING_DIVERGENT, {"x": [1, 2, 3], "n": 3}, 3, "registers"),  # a GPU limit
])
def test_refused(source, args, threads, says):
    with pytest.raises(CUDAError, match=says):
        plan(compile_cuda(source), args, threads, 4, 4)


def test_host_makes_outputs_and_count():
    """args carries the inputs; the outputs and n are the host's, so a wrong n or a short
    output array Qwen sends cannot shrink the result."""
    p = plan(compile_cuda(ELEMENTWISE), {"a": [2, 3, 4], "b": [5, 6, 7], "c": [0, 0], "n": 2},
             None, 4, 4)
    assert reference(p) == {"c": [6, 14, 24]}  # a * b - 4 over all three items


def test_reference_doc_matches_the_machine():
    """docs/qgpu-cuda.md is in Qwen's system prompt: its numbers are the GPU's."""
    from qgpu.circuit import kernel
    from qgpu.circuit.kernel import BBITS, NREG
    from qgpu.cuda import DOC, MAX_THREADS

    for fact in (f"at most {MAX_THREADS}", f"{kernel.SINK - 1} words",
                 f"| Registers per thread | {NREG - 1} |", f"{1 << BBITS} machine blocks",
                 "| Lanes per warp | 4 |"):
        assert fact in DOC, fact


def doc_examples():
    """Each worked example of docs/qgpu-cuda.md: (title, kernel, args, the stated result)."""
    import json
    import re

    from qgpu.cuda import DOC

    found = re.findall(r"### (.+?)\n.*?```cuda\n(.*?)```\nCall: `(\{.*?\})` returns `(\{.*?\})`",
                       DOC, re.DOTALL)
    return [(t, src, json.loads(a), json.loads(r)) for t, src, a, r in found]


def test_doc_has_every_example():
    assert len(doc_examples()) == 15


@pytest.mark.parametrize("title, source, args, want", doc_examples(),
                         ids=[e[0] for e in doc_examples()])
def test_doc_example_runs_as_shown(title, source, args, want):
    assert reference(plan(compile_cuda(source), args, None, 4, 4)) == want


def k(body: str, params: str = "const int* a, int* out, int n") -> str:
    return (f'extern "C" __global__ void f({params}) {{\n'
            f"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n{body}\n}}")


A = {"a": [7, -3, 12, 5]}
DOC_RUNS = [  # each claim of docs/qgpu-cuda.md that a kernel does: (body, args, result)
    ("if (i < n) out[i] = a[i] % 4 + a[i] / n;", A, [4, -3, 3, 2]),       # / % by n, a constant
    ("if (i < n) out[i] = (a[i] << 2) ^ (a[i] >> 1) & ~1 | 1;", A, [31, 11, 55, 23]),
    ("if (i < n) out[i] = a[i] > 4 ? a[i] : -a[i];", A, [7, 3, 12, 5]),  # ?: (min / max)
    ("if (i < n) { if (a[i] > 6) out[i] = 1; else out[i] = 2; }", A, [1, 2, 1, 2]),  # divergent
    ("if (i < n) { int s = 0; for (int k = 0; k < 3; k++) s += a[i]; out[i] = s; }", A,
     [21, -9, 36, 15]),                                                    # a loop
    ("if (i >= n) return; out[i] = a[i] + 1;", A, [8, -2, 13, 6]),       # one early return
    ("if (i < n) atomicAdd(&out[0], a[i] * a[i]);", A, [227, 0, 0, 0]),  # one value from all
    ("if (i < n) out[i] = (unsigned)a[i] >> 28;", A, [0, 15, 0, 0]),       # unsigned
]


@pytest.mark.parametrize("body, args, want", DOC_RUNS)
def test_doc_operations_run(body, args, want):
    assert reference(plan(compile_cuda(k(body)), args, None, 4, 4)) == {"out": want}


def test_doc_plain_increment_loses_updates():
    """The doc says out[0] += part loses updates: the threads overwrite each other."""
    got = reference(plan(compile_cuda(k("if (i < n) out[0] += a[i];")), A, None, 4, 4))
    assert got["out"][0] != sum(A["a"])


@pytest.mark.parametrize("body, says", [
    ("if (i < n) out[i] = 100 / a[i];", "by a per-thread value"),
    ("if (i < n) out[i] = atomicAdd(&out[0], a[i]);", "old value is used"),
])
def test_doc_refusals(body, says):
    with pytest.raises(CUDAError, match=says):
        plan(compile_cuda(k(body)), A, None, 4, 4)
