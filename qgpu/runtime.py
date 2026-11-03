"""The GPU runtime: the driver's side of a launch on the GPU in the model's weights.

`prepare` lowers a library kernel for a launch (qgpu.lower, as NVIDIA's driver JIT-compiles
PTX at launch; qgpu/cuda.py does the same for CUDA that Qwen writes), `host_tokens` is what the
host feeds the model for it, and `read` / `Device` read the outputs back from the model's
state (cudaMemcpy device -> host). The library kernels (KERNELS) are the building blocks the
tests and the GPU-panel demos use.
"""

from dataclasses import dataclass
from pathlib import Path

from qgpu import ptx
from qgpu.circuit import kernel
from qgpu.circuit.kernel import EXEC, LIVE, MEM, PROG, PROGI, Kernel, encode_live
from qgpu.lower import Unsupported, lower
from qgpu.merge import STAGES

ROOT = Path(__file__).resolve().parent.parent


class GPUError(Exception):
    """An expression the runtime cannot run; its message goes back to Qwen."""


# --- kernels and the driver -----------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    ptx: str
    params: tuple  # ("in", name) | ("out", name, size) | ("scalar", name) | ("n",); an out's
    #               size is "n", "nn" (n * n) or an int
    signed: bool = True
    max_block: int = 32  # threads per block (a barrier holds at most WPB_MAX warps)
    max_n: int | None = None  # one-block kernels: at most this many elements (or rows)
    matrix: bool = False  # a 2D launch over n x n
    doc: str = ""  # what it computes, for display


VV = (("in", "a"), ("in", "b"), ("out", "c", "n"), ("n",))  # c = a op b
VK = (("in", "x"), ("out", "y", "n"), ("scalar", "k"), ("n",))  # y = x op k
KERNELS = {
    "vadd": Spec("kernels/vadd.ptx", VV, doc="c = a + b"),
    "vsub": Spec("kernels/vsub.ptx", VV, doc="c = a - b"),
    "vmul": Spec("kernels/vmul.ptx", VV, doc="c = a * b"),
    "addk": Spec("kernels/addk.ptx", VK, doc="y = x + k"),
    "subk": Spec("kernels/subk.ptx", VK, doc="y = x - k"),
    "rsubk": Spec("kernels/rsubk.ptx", VK, doc="y = k - x"),
    "mulk": Spec("kernels/mulk.ptx", VK, doc="y = x * k"),
    "fdivk": Spec("kernels/fdivk.ptx",
                  (("in", "x"), ("out", "q", "n"), ("out", "r", "n"), ("scalar", "k"), ("n",)),
                  doc="q = x // k, r = x % k (floor)"),
    "divdec": Spec("kernels/divdec.ptx",
                   (("in", "x"), ("out", "w", "n"), ("out", "h", "n"), ("scalar", "k"), ("n",)),
                   doc="x / k to two decimals"),
    "divk": Spec("kernels/divk.ptx",
                 (("in", "x"), ("out", "q", "n"), ("out", "r", "n"), ("scalar", "k"), ("n",)),
                 doc="q = x / k, r = x % k"),
    "reduce": Spec("kernels/nvcc/reduce.ptx", (("in", "x"), ("out", "out", 1), ("n",)),
                   doc="out = sum of x"),
    "mean": Spec("kernels/mean.ptx", (("in", "x"), ("out", "out", 2), ("n",)), max_n=32,
                 doc="out = sum / n, hundredths"),
    "vmax": Spec("kernels/vmax.ptx", (("in", "x"), ("out", "out", 1), ("n",)), max_n=32,
                 doc="out = the largest x"),
    "scan": Spec("kernels/scan.ptx", (("in", "x"), ("out", "y", "n"), ("n",)), max_block=16,
                 max_n=16, doc="y = running totals of x"),
    "dot": Spec("kernels/dot.ptx", (("in", "a"), ("in", "b"), ("out", "out", 1), ("n",)),
                doc="out = sum of a * b"),
    "histogram": Spec("kernels/nvcc/histogram.ptx", (("in", "x"), ("out", "bins", 16), ("n",)),
                      signed=False, doc="bins[v] = how often v occurs"),
    "matmul": Spec("kernels/matmul.ptx", (("in", "A"), ("in", "B"), ("out", "C", "nn"), ("n",)),
                   max_n=4, matrix=True, doc="C = A @ B"),
}


@dataclass
class Prepared:
    """One launch, as the driver prepares it."""
    name: str
    launch: object  # qgpu.lower.Launch
    outputs: dict  # output name -> (buffer id, size)
    signed: bool
    inputs: dict  # what it was given, by parameter name (for display)


def prepare(name: str, data: dict, n: int, lanes: int, stages: int) -> Prepared:
    """Lower kernel `name` for this launch: `data` holds its inputs (lists of int) and scalars
    by parameter name, `n` is the element count (rows, for a matrix)."""
    spec = KERNELS[name]
    if spec.max_n and n > spec.max_n:
        raise GPUError(f"{name} takes at most {spec.max_n} {'rows' if spec.matrix else 'numbers'}")
    args, bufs, outs = [], {}, {}
    for p in spec.params:
        if p[0] == "n":
            args.append(n)
        elif p[0] == "scalar":
            args.append(data[p[1]] & 0xFFFFFFFF)
        else:
            bid = len(bufs)
            if p[0] == "out":
                size = {"n": n, "nn": n * n}.get(p[2], p[2])
                bufs[bid] = [0] * size
                outs[p[1]] = (bid, size)
            else:
                bufs[bid] = [v & 0xFFFFFFFF for v in data[p[1]]]
            args.append(bid)
    if sum(len(b) for b in bufs.values()) >= kernel.SINK:
        raise GPUError(f"too many numbers for one launch: the GPU has {kernel.SINK} words of "
                       "memory; split the work")
    side = 1 << (n - 1).bit_length()  # powers of two
    if spec.matrix:
        grid, block = (1, 1), (side, side)
    else:
        threads = max(side, lanes)
        bx = min(threads, spec.max_block, lanes * kernel.WPB_MAX)
        grid, block = (threads // bx,), (bx,)
    mul, logic, atom = STAGES[stages]
    try:
        lw = lower(ptx.load(ROOT / spec.ptx)[name], grid, block, args, bufs, lanes=lanes,
                   stages=stages, mul_stages=mul, logic_stages=logic, atom_stages=atom,
                   write_through=True)
    except (Unsupported, AssertionError) as e:
        raise GPUError(f"{name} on {n} numbers does not fit the GPU ({e}); use fewer") from e
    if max(lw.mem, default=0) >= kernel.SINK:
        raise GPUError("the launch needs more memory than the GPU has; use fewer numbers")
    return Prepared(name, lw, outs, spec.signed, data)


def host_tokens(lw, clear: bool = False) -> list[int]:
    """What the driver feeds: the program image, the initial memory and the first EXEC token
    (as `run_blocks` lays them out). With `clear` (a context that has run a launch before),
    every other word is zeroed too, like a cudaMemset: the reference starts from zeroed
    memory, and inactive lanes' unguarded loads read past the buffers."""
    toks = []
    for b, blk in enumerate(lw.blocks):
        for s, ins in enumerate(blk.instrs):
            f0, f1, i0, i1 = ins.encode()
            toks += [PROG(b, s), f0, f1, PROGI(b, s), i0, i1]
        toks += [LIVE(b), *encode_live(blk)]
    mem = {**({a: 0 for a in range(kernel.SINK)} if clear else {}), **lw.mem}
    for a, v in mem.items():
        toks += [MEM(a), v & 0xFFFF, v >> 16 & 0xFFFF]
    return toks + [EXEC(0, 0, True)]


def read(p: Prepared, mem: dict[int, int]) -> dict[str, list[int]]:
    """The output buffers from global memory (cudaMemcpy device -> host)."""
    out = {}
    for name, (bid, size) in p.outputs.items():
        vals = [mem.get(p.launch.buffers[bid] + i, 0) for i in range(size)]
        out[name] = [v - (1 << 32) if p.signed and v >> 31 else v for v in vals]
    return out


class Device:
    """Where global memory lives in a merged checkpoint: (layer, value head) of the `in` head
    (the circuit's layers come first; emit_layers puts key head j's first value head at
    j * r)."""

    def __init__(self, meta: dict, tc: dict):
        from chipformer.q35.ir import Head, Target

        mul, logic, atom = STAGES[meta["stages"]]
        nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
        c = Kernel(Target(nk, nv, tc["intermediate_size"]), lanes=meta["lanes"],
                   stages=meta["stages"], mul_stages=mul, logic_stages=logic, atom_stages=atom)
        assert c.n_layers() == meta["chip_layers"]
        heads = [m for m in c.nodes if isinstance(m, Head) and m.level == c.mem_head.level]
        self.mem = ((c.mem_head.level - 1) // 2, heads.index(c.mem_head) * (nv // nk))

    def read(self, cache) -> dict[int, int]:
        """Global memory: the `in` head's written rows, 32 sign bits each (kernel.memory)."""
        layer, head = self.mem
        st = cache.get_recurrent_state(layer, state_idx=0)[0, head].float().cpu()
        return {row: sum(int(st[row, 2 + i] > 0) << i for i in range(32))
                for row in range(st.shape[0]) if st[row, 0] != 0}


def reference(p: Prepared, lanes: int = 4) -> dict[str, list[int]]:
    """A launch's outputs on the reference interpreter (no model): for tests and checks."""
    lw = p.launch
    return read(p, kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=lanes).mem)
