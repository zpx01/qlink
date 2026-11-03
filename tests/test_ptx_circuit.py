"""Real PTX through the driver (lower.py) into the kernel circuit, exact in the IR evaluator,
against the reference interpreter (qgpu/sim.py, itself 49/49 identical to a Tesla T4)."""

from functools import cache

import numpy as np
import pytest
from chipformer.q35.ir import evaluate

from qgpu import ptx
from qgpu.circuit.kernel import CMPS, Kernel, MBlock, memory, predicted, run_blocks, stored, trace
from qgpu.circuit.kernel import KInstr as K
from qgpu.lower import Unsupported, lower
from qgpu.sim import GLOBAL_BASE, Memory, launch

STAGES = 6
MUL = frozenset({4})  # the multiply stage
LOGIC = frozenset({2})  # the logic stage
ATOM = frozenset({5})  # the atomic stage


# the configuration proposed for merging into Qwen3.5-4B (fit study): 4 lanes, 4 stages
SMALL = {"lanes": 4, "stages": 4, "mul_stages": frozenset({2}), "logic_stages": frozenset({1}),
         "atom_stages": frozenset({3})}


@cache
def circuit(small: bool = False) -> Kernel:
    if small:
        return Kernel(**SMALL)
    return Kernel(stages=STAGES, mul_stages=MUL, logic_stages=LOGIC, atom_stages=ATOM)


def run_circuit(program, mem, warps, with_memory=False):
    return run_trace(trace(program, mem, warps, stages=STAGES, mul_stages=MUL,
                           logic_stages=LOGIC, atom_stages=ATOM), with_memory)


def run_launch(lw, with_memory=False, small=False):
    """A lowered launch's machine blocks through the circuit."""
    lanes = SMALL["lanes"] if small else 8
    return run_trace(run_blocks(lw.blocks, lw.mem, lw.warps, lanes=lanes), with_memory, small)


def run_trace(tr, with_memory=False, small=False):
    """The circuit on a trace: its stores (and, with_memory, the final global memory too, for
    words no store wrote: atomics' results)."""
    carry, c = {}, circuit(small)
    m = evaluate(c, tr.tokens[:-1], carry=carry)
    bad = [t for t in tr.owned if t + 1 < len(tr.tokens) and predicted(c, m, t) != tr.tokens[t + 1]]
    assert not bad, f"{len(bad)} wrong predictions"
    got = stored(c, carry)
    assert got == tr.out  # the circuit's stores equal the lowered program's reference
    mem = memory(c, carry)  # global memory (atomics write it) equals the reference's
    assert {a: mem.get(a) for a in tr.mem} == tr.mem
    return got | {a: v for a, v in tr.mem.items() if a not in got} if with_memory else got


def reference_vadd(k, a, b, n, grid, block):
    mem = Memory(1 << 16, GLOBAL_BASE)
    pa, pb, pc = (mem.alloc(4 * len(x)) for x in (a, b, a))
    mem.write_i32(pa, a)
    mem.write_i32(pb, b)
    launch(k, mem, grid, block, [pa, pb, pc, n])
    return mem.read_i32(pc, n, signed_=False)


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block,grid", [(1, 32, 1), (31, 32, 1), (40, 8, 5), (17, 16, 2)])
def test_vadd_ptx(src, n, block, grid):
    k = ptx.load(f"{src}/vadd.ptx")["vadd"]
    rng = np.random.default_rng(n)
    a, b = [[int(x) for x in rng.integers(0, 2**32, n)] for _ in range(2)]
    launch_ = lower(k, (grid,), (block,), [0, 1, 2, n], {0: a, 1: b, 2: [0] * n})
    got = run_launch(launch_)
    c0 = launch_.buffers[2]
    assert [got.get(c0 + i) for i in range(n)] == reference_vadd(k, a, b, n, (grid,), (block,))
    assert set(got) == set(range(c0, c0 + n)), "threads past n must not store"


EDGES = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE]


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("useimm", [False, True])
def test_setp_every_compare(signed, useimm):
    """Each compare, on lane pairs around 0, the sign boundary and equality; the predicate
    picks which of two stores each lane makes."""
    pairs = [(x, y) for x in EDGES for y in EDGES]
    for cmp in CMPS:
        for start in range(0, len(pairs), 8):
            chunk = pairs[start:start + 8]
            if useimm:  # one immediate per launch: the rhs is uniform
                chunk = [(x, chunk[0][1]) for x, _ in chunk]
            mem = {i: x for i, (x, _) in enumerate(chunk)} | {8 + i: y for i, (_, y) in
                                                               enumerate(chunk)}
            prog = [K("tid", rd=0), K("ld", rd=1, ra=0), K("ld", rd=2, ra=0, imm=8),
                    K("setp", ra=1, rb=2, imm=chunk[0][1], useimm=useimm, pd=3, cmp=cmp,
                      signed=signed),
                    K("st", ra=0, rb=1, imm=16, guard=3),
                    K("st", ra=0, rb=2, imm=24, guard=3, gneg=True)]
            got = run_circuit(prog, mem, 1)
            assert len(got) == 8  # every lane stores exactly once


def reference_saxpy(k, a, x, y, n, grid, block):
    mem = Memory(1 << 16, GLOBAL_BASE)
    px, py = mem.alloc(4 * len(x)), mem.alloc(4 * len(y))
    mem.write_i32(px, x)
    mem.write_i32(py, y)
    launch(k, mem, grid, block, [a, px, py, n])
    return mem.read_i32(py, len(y), signed_=False)


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block,grid,a", [(1, 32, 1, 22), (31, 32, 1, -7), (40, 8, 5, 65537),
                                            (17, 16, 2, -1)])
def test_saxpy_ptx(src, n, block, grid, a):
    k = ptx.load(f"{src}/saxpy.ptx")["saxpy"]
    rng = np.random.default_rng(n)
    x, y = [[int(v) for v in rng.integers(0, 2**32, n)] for _ in range(2)]
    lw = lower(k, (grid,), (block,), [a, 0, 1, n], {0: x, 1: y})
    assert [i.op for b in lw.blocks for i in b.instrs].count("mad") == 1
    got = run_launch(lw)
    y0 = lw.buffers[1]
    after = [got.get(y0 + i, lw.mem[y0 + i]) for i in range(n)]
    assert after == reference_saxpy(k, a, x, y, n, (grid,), (block,))
    assert set(got) == set(range(y0, y0 + n))


MUL_EDGES = [0, 1, 15, 16, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]


@pytest.mark.parametrize("useimm", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_mad_low_32_bits(useimm, seed):
    """rd = ra * (rb | imm) + rc on random words and on carry / sign edge values."""
    rng = np.random.default_rng(seed)
    pick = (lambda: [int(v) for v in rng.choice(MUL_EDGES, 8)]) if seed == 0 else \
        (lambda: [int(v) for v in rng.integers(0, 2**32, 8)])
    x, y, z = pick(), pick(), pick()
    imm = int(rng.choice(MUL_EDGES[1:])) if useimm else 0
    mem = ({i: x[i] for i in range(8)} | {8 + i: y[i] for i in range(8)}
           | {16 + i: z[i] for i in range(8)})
    prog = [K("tid", rd=0), K("ld", rd=1, ra=0), K("ld", rd=2, ra=0, imm=8),
            K("ld", rd=3, ra=0, imm=16), K("mad", rd=4, ra=1, rb=2, rc=3, imm=imm, useimm=useimm),
            K("st", ra=0, rb=4, imm=24)]
    got = run_circuit(prog, mem, 1)
    assert got == {24 + i: (x[i] * (imm if useimm else y[i]) + z[i]) & 0xFFFFFFFF
                   for i in range(8)}


def test_mad_is_placed_in_the_multiply_stage():
    """A mad at block position 2 is padded with no-ops so it runs in stage 4."""
    prog = [K("tid", rd=0), K("ld", rd=1, ra=0), K("mad", rd=2, ra=1, rb=1, rc=1),
            K("st", ra=0, rb=2, imm=8)]
    tr = trace(prog, {i: i + 3 for i in range(8)}, 1, stages=STAGES, mul_stages=MUL)
    assert [i.op for i in tr.blocks[0].instrs] == ["tid", "ld", "nop", "nop", "mad", "st"]
    got = run_circuit(prog, {i: i + 3 for i in range(8)}, 1)
    assert got == {8 + i: (i + 3) * (i + 3) + (i + 3) for i in range(8)}


def reference(k, bufs, args_of, n_out, out_buf, grid, block):
    mem = Memory(1 << 16, GLOBAL_BASE)
    ptrs = []
    for words in bufs:
        ptrs.append(mem.alloc(4 * max(len(words), 1)))
        mem.write_i32(ptrs[-1], words)
    launch(k, mem, grid, block, args_of(ptrs))
    return mem.read_i32(ptrs[out_buf], n_out, signed_=False)


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,m,block,grid", [(8, 5, 8, 1), (13, 4, 16, 1), (20, 3, 8, 3),
                                            (7, 1, 8, 1), (9, 0, 16, 1), (20, 3, 24, 1)])
def test_colsum_loop_ptx(src, n, m, block, grid):
    """A loop with a uniform trip count: y[i] = sum over k < m of x[k * n + i]."""
    k = ptx.load(f"{src}/colsum.ptx")["colsum"]
    rng = np.random.default_rng(10 * n + m)
    x = [int(v) for v in rng.integers(0, 2**32, max(n * m, 1))]
    lw = lower(k, (grid,), (block,), [0, 1, n, m], {0: x, 1: [0] * n})
    assert any(b.branch[0] == "cond" for b in lw.blocks)  # a real loop, not unrolled
    got = run_launch(lw)
    y0 = lw.buffers[1]
    assert [got.get(y0 + i, 0) for i in range(n)] == reference(
        k, [x, [0] * n], lambda p: [p[0], p[1], n, m], n, 1, (grid,), (block,))
    assert set(got) == set(range(y0, y0 + n))


def signed32(v):
    return v - (1 << 32) if v >> 31 else v


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block,grid", [(1, (4, 4), (1, 1)), (5, (4, 4), (2, 2)),
                                          (3, (4, 2), (1, 2)), (5, (8, 2), (1, 3)),
                                          (6, (2, 4), (4, 2))])
def test_matmul_ptx(src, n, block, grid):
    """2D thread ids, loops (unrolled by 2 in clang's PTX, by 4 in nvcc's) and two multiplies
    per iteration: C = A B for n x n int matrices."""
    k = ptx.load(f"{src}/matmul.ptx")["matmul"]
    rng = np.random.default_rng(n)
    A, B = ([int(v) & 0xFFFFFFFF for v in rng.integers(-2**15, 2**15, n * n)] for _ in range(2))
    lw = lower(k, grid, block, [0, 1, 2, n], {0: A, 1: B, 2: [0] * (n * n)})
    got = run_launch(lw)
    c0 = lw.buffers[2]
    want = [sum(signed32(A[r * n + j]) * signed32(B[j * n + c]) for j in range(n)) & 0xFFFFFFFF
            for r in range(n) for c in range(n)]
    assert [got.get(c0 + i) for i in range(n * n)] == want
    assert want == reference(k, [A, B, [0] * (n * n)], lambda p: [p[0], p[1], p[2], n], n * n, 2,
                             grid, block)
    assert set(got) == set(range(c0, c0 + n * n))


@pytest.mark.parametrize("op", ["max", "min", "sub"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("useimm", [False, True])
def test_max_min_sub(op, signed, useimm):
    x, y = EDGES, EDGES[::-1]
    mem = {i: x[i] for i in range(8)} | {8 + i: y[i] for i in range(8)}
    prog = [K("tid", rd=0), K("ld", rd=1, ra=0), K("ld", rd=2, ra=0, imm=8),
            K(op, rd=3, ra=1, rb=2, imm=0x80000001, useimm=useimm, signed=signed),
            K("st", ra=0, rb=3, imm=16)]
    run_circuit(prog, mem, 1)  # checks the circuit's stores against the reference


@pytest.mark.parametrize("maps", [[0, 1, 2, 7, 7, 7, 7], [3, 7, 7, 7, 7, 7, 7],
                                  [0, 1, 2, 4, 5, 6, 7], [6, 5, 4, 3, 2, 1, 0]])
def test_sreg_gathers_thread_index_bits(maps):
    imm = sum(v << 3 * i for i, v in enumerate(maps))
    got = run_circuit([K("tid", rd=0), K("sreg", rd=1, imm=imm), K("st", ra=0, rb=1)], {}, 15)
    assert got == {t: sum((t >> m & 1) << i for i, m in enumerate(maps) if m != 7)
                   for t in range(120)}


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block", [(12, 16), (20, 8), (5, 32)])
def test_divergent_loop_ptx(src, n, block):
    """Each lane loops a different number of times (i iterations): run under a per-lane mask,
    the back-edge taken while any lane still loops."""
    k = ptx.load(f"{src}/divloop.ptx")["divloop"]
    rng = np.random.default_rng(n)
    x = [int(v) for v in rng.integers(0, 2**32, n)]
    grid = ((n + block - 1) // block,)
    lw = lower(k, grid, (block,), [0, 1, n], {0: x, 1: [0] * n})
    assert any(len(b.branch) > 4 and b.branch[4] for b in lw.blocks)  # an any-lane loop
    got = run_launch(lw)
    y0 = lw.buffers[1]
    assert [got.get(y0 + i, 0) for i in range(n)] == reference(
        k, [x, [0] * n], lambda p: [p[0], p[1], n], n, 1, grid, (block,))
    assert set(got) == set(range(y0, y0 + n))


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,t,block", [(16, 1, 8), (13, 0, 8), (24, -2, 8), (30, 3, 32)])
def test_if_else_chain_ptx(src, n, t, block):
    """Lanes split three ways; one arm stores to z, every arm to y."""
    k = ptx.load(f"{src}/branchy.ptx")["branchy"]
    rng = np.random.default_rng(n + t)
    x = [int(v) & 0xFFFFFFFF for v in rng.integers(-4, 5, n)]
    grid = ((n + block - 1) // block,)
    lw = lower(k, grid, (block,), [0, 1, 2, n, t], {0: x, 1: [0] * n, 2: [0] * n})
    got = run_launch(lw)
    y0, z0 = lw.buffers[1], lw.buffers[2]
    want = [reference(k, [x, [0] * n, [0] * n], lambda p: [p[0], p[1], p[2], n, t], n, i, grid,
                      (block,)) for i in (1, 2)]
    assert [got.get(y0 + i, 0) for i in range(n)] == want[0]
    assert [got.get(z0 + i, 0) for i in range(n)] == want[1]
    assert {a for a in got if z0 <= a < z0 + n} == {z0 + i for i in range(n)
                                                     if x[i] == t & 0xFFFFFFFF}


def run_kernel(src, name, args, bufs, grid, block, small=False):
    """A kernel through the driver and the circuit; every buffer afterwards must equal the
    reference interpreter's (itself checked against a Tesla T4). small: the 4-lane, 4-stage
    configuration."""
    k = ptx.load(f"{src}/{name}.ptx")[name]
    lw = lower(k, grid, block, args, dict(enumerate(bufs)), **(SMALL if small else {}))
    got = run_launch(lw, with_memory=True, small=small)
    want = [reference(k, bufs, lambda p: [p[a["buf"]] if isinstance(a, dict) else a
                                          for a in args_spec(args, bufs)], len(b), j, grid,
                      block) for j, b in enumerate(bufs)]
    for j, b in enumerate(bufs):
        assert [got.get(lw.buffers[j] + i, lw.mem.get(lw.buffers[j] + i))
                for i in range(len(b))] == want[j], f"buffer {j}"
    return lw


def args_spec(args, bufs):
    """Pointer arguments are buffer ids: mark them so `reference` passes device pointers."""
    return [{"buf": a} if i < len(bufs) else a for i, a in enumerate(args)]


EDGE_WORDS = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF, 0x0F0F0F0F]


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block", [(16, 8), (7, 8), (24, 8)])
def test_bitops_ptx(src, n, block):
    """and / or / xor, shifts by constants (signed and unsigned), __umulhi / __mulhi, a select."""
    rng = np.random.default_rng(n)
    x = (EDGE_WORDS + [int(v) for v in rng.integers(0, 2**32, n)])[:n]
    run_kernel(src, "bitops", [0, 1, n], [x, [0] * n], ((n + block - 1) // block,), (block,))


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block", [(16, 8), (13, 8), (8, 16)])
def test_diverge_ptx(src, n, block):
    """x % 3 picks one of three arms; clang computes x / 3 with mul.hi by a magic number."""
    rng = np.random.default_rng(n)
    x = (EDGE_WORDS + [int(v) & 0xFFFFFFFF for v in rng.integers(-2**31, 2**31, n)])[:n]
    run_kernel(src, "diverge", [0, 1, n], [x, [0] * n], ((n + block - 1) // block,), (block,))


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block,top", [(8, 8, 12), (11, 8, 20)])
def test_collatz_ptx(src, n, block, top):
    """A divergent loop (Collatz steps per lane) with and, shr, mad and selp in its body."""
    rng = np.random.default_rng(n)
    x = [int(v) for v in rng.integers(0, top, n)]
    run_kernel(src, "collatz", [0, 1, n], [x, [0] * n], ((n + block - 1) // block,), (block,))


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block", [(31, 32), (70, 8), (9, 16), (40, 40)])
def test_histogram_atomics_ptx(src, n, block):
    """atomicAdd(&bins[x[i] & 15], 1): lanes of a warp hit the same bin."""
    rng = np.random.default_rng(n)
    x = [int(v) for v in rng.integers(0, 2**32, n)]
    bins = [int(v) for v in rng.integers(0, 9, 16)]
    run_kernel(src, "histogram", [0, 1, n], [x, bins], ((n + block - 1) // block,), (block,))


@pytest.mark.parametrize("seed,spread", [(0, 1), (1, 4), (2, 16), (3, 128)])
def test_atom_combines_lanes(seed, spread):
    """atom [x + 64], tid: lanes with the same address add up; the first one writes back."""
    rng = np.random.default_rng(seed)
    x = [int(v) % 63 for v in rng.integers(0, spread, 24)]  # word 127 is the atomics' sink
    old = {64 + b: int(rng.integers(0, 2**32)) for b in range(63)}
    mem = {i: x[i] for i in range(24)} | old
    prog = [K("tid", rd=0), K("ld", rd=1, ra=0), K("atom", ra=1, rb=0, imm=64)]
    got = run_circuit(prog, mem, 3, with_memory=True)
    for a, v in old.items():
        assert got[a] == (v + sum(i for i in range(24) if x[i] + 64 == a)) & 0xFFFFFFFF


def barrier_program(wpb):
    """Thread t writes t to word 64 + t (stw), passes a barrier, then reads the word of thread
    t ^ 8, which belongs to the other warp of its 16-thread block, and stores it to word t."""
    nop = K("nop")
    return [MBlock([K("tid", rd=0)], [(0, 0)], wpb=wpb),
            MBlock([nop] * 5 + [K("stw", ra=0, rb=0, imm=64)], [], ("bar",), wpb=wpb),
            MBlock([nop, nop, K("xor", rd=1, ra=0, imm=8, useimm=True),
                    K("ld", rd=2, ra=1, imm=64), K("st", ra=0, rb=2)], [], ("end",), wpb=wpb)]


@pytest.mark.parametrize("blocks_", [1, 2])
def test_barrier_waits_for_every_warp(blocks_):
    """Two warps per thread block: after the barrier each lane reads the other warp's write.
    The warps take turns phase by phase, so the reads see every write of the phase before."""
    tr = run_blocks(barrier_program(2), {}, 2 * blocks_)
    assert tr.out == {t: t ^ 8 for t in range(16 * blocks_)}
    assert run_trace(tr) == tr.out
    # without the barrier's rotation (one warp per block), warp 0 reads before warp 1 writes
    assert run_blocks(barrier_program(1), {}, 2 * blocks_).out != tr.out


def test_stw_guard_writes_only_active_lanes():
    """@!p stw: inactive lanes write nothing (their write-back goes to the sink word)."""
    nop = K("nop")
    prog = [MBlock([K("tid", rd=0)], [(0, 0)]),
            MBlock([K("setp", ra=0, imm=5, useimm=True, pd=1, cmp="lt"), nop, nop, nop, nop,
                    K("stw", ra=0, rb=0, imm=32, guard=1, gneg=True)], []),
            MBlock([K("ld", rd=1, ra=0, imm=32), K("st", ra=0, rb=1)], [], ("end",))]
    mem = {32 + t: 1000 + t for t in range(16)}
    tr = run_blocks(prog, mem, 2)
    assert tr.out == {t: t if t >= 5 else 1000 + t for t in range(16)}
    assert run_trace(tr) == tr.out


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block", [(37, 8), (20, 16), (60, 16)])
def test_reduce_shared_ptx(src, n, block):
    """A tree reduction in shared memory with __syncthreads(), then one atomicAdd per block."""
    rng = np.random.default_rng(n)
    x = [int(v) for v in rng.integers(0, 2**32, n)]
    run_kernel(src, "reduce", [0, 1, n], [x, [5]], ((n + block - 1) // block,), (block,))


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,block", [(24, 8), (13, 16), (30, 8)])
def test_scan_shared_ptx(src, n, block):
    """Hillis-Steele inclusive scan, double-buffered in shared memory, a barrier per step."""
    rng = np.random.default_rng(n)
    x = [int(v) for v in rng.integers(0, 2**32, n)]
    run_kernel(src, "scan", [0, 1, n], [x, [0] * n], ((n + block - 1) // block,), (block,))


def test_text_around_launches():
    """In a merged checkpoint Qwen's text surrounds launches (every text token reaches the
    circuit as TEXT): two launches with text before, between and after each stay exact, the
    output code names the next token wherever the circuit predicts and is 0 on text, and tau
    is 0 on chip tokens and TAU_TEXT on text after the exit."""
    from qgpu.circuit.kernel import KAPPA, NCODE, TAU_TEXT, TEXT, dim
    rng = np.random.default_rng(7)
    x = [int(v) for v in rng.integers(0, 2**32, 8)]
    runs = []
    for name, args, bufs in (("vadd", [0, 1, 2, 4], {0: x[:4], 1: x[4:], 2: [0] * 4}),
                             ("histogram", [0, 1, 8], {0: x, 1: [0] * 16})):
        k = ptx.load(f"kernels/{name}.ptx")[name]
        lw = lower(k, (1,), (8,), args, bufs, **SMALL)
        runs.append(run_blocks(lw.blocks, lw.mem, lw.warps, lanes=SMALL["lanes"]))
    seq, owned = [TEXT] * 3, []
    for i, tr in enumerate(runs):
        owned += [len(seq) + t for t in tr.owned]
        seq += tr.tokens + [TEXT] * (4 if i == 0 else 2)
    c = circuit(True)
    m = evaluate(c, seq[:-1])
    code = np.stack([m[:, dim(e)] for e in c.code], 1)
    for t in range(len(seq) - 1):
        if seq[t] == TEXT:
            assert np.all(code[t] == 0) and m[t, dim(c.tau)] == TAU_TEXT
            continue
        assert m[t, dim(c.tau)] == 0
        if t in owned and seq[t + 1] != TEXT:
            assert set(np.round(code[t] / KAPPA).astype(int).tolist()) <= {-1, 1}
            assert sum(int(code[t, i] > 0) << i for i in range(NCODE)) == seq[t + 1], t


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
def test_sixteen_warps(src):
    """128 threads in two blocks of 64: 16 warps, past the old 15-warp limit (4 warp bits)."""
    rng = np.random.default_rng(100)
    x = [int(v) for v in rng.integers(0, 2**32, 100)]
    bins = [int(v) for v in rng.integers(0, 9, 16)]
    lw = run_kernel(src, "histogram", [0, 1, 100], [x, bins], (2,), (64,))
    assert lw.warps == 16


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("name,n,block", [("scan", 13, 16), ("reduce", 20, 16), ("warpsum", 20, 32),
                                          ("divloop", 9, 8), ("histogram", 31, 32)])
def test_small_configuration(src, name, n, block):
    """4 lanes, 4 stages (the merge configuration): more warps per launch and more machine
    blocks per kernel than the old limits (scan and reduce need 18-20 blocks)."""
    rng = np.random.default_rng(n)
    x = [int(v) % 16 if name == "divloop" else int(v) for v in rng.integers(0, 2**32, n)]
    grid = (n + block - 1) // block
    out = {"scan": [0] * n, "reduce": [3], "warpsum": [0] * grid, "divloop": [0] * n,
           "histogram": [0] * 16}[name]
    lw = run_kernel(src, name, [0, 1, n], [x, out], (grid,), (block,), small=True)
    if name in ("scan", "reduce"):
        assert len(lw.blocks) > 16


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n,grid", [(40, 2), (17, 1)])
def test_warpsum_shuffles_ptx(src, n, grid):
    """__shfl_down_sync by 16, 8, 4, 2, 1 across a 32-thread warp: four of our 8-lane warps,
    so each shuffle goes through a scratch array and a barrier."""
    rng = np.random.default_rng(n)
    x = [int(v) & 0xFFFFFFFF for v in rng.integers(-2**24, 2**24, n)]
    run_kernel(src, "warpsum", [0, 1, n], [x, [0] * grid], (grid,), (32,))


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
@pytest.mark.parametrize("n", [20, 29])
def test_shuffle_every_lane_ptx(src, n):
    """Every lane's shfl.down by 5, and by 3 within 8-lane segments (width 8): lanes past the
    end keep their own value. n = 29 leaves room for one scratch array only (two barriers)."""
    rng = np.random.default_rng(n)
    x = [int(v) & 0xFFFFFFFF for v in rng.integers(-2**31, 2**31, n)]
    lw = run_kernel(src, "shfl", [0, 1, 2, n], [x, [0] * n, [0] * n], (1,), (32,))
    bars = sum(b.branch == ("bar",) for b in lw.blocks)
    assert bars == (4 if n == 29 else 2)


@pytest.mark.parametrize("src", ["kernels", "kernels/nvcc"])
def test_shuffle_in_a_partial_warp_is_refused(src):
    """A 16-thread block is half a CUDA warp: the driver must refuse, never guess."""
    k = ptx.load(f"{src}/warpsum.ptx")["warpsum"]
    with pytest.raises(Unsupported):
        lower(k, (1,), (16,), [0, 1, 16], {0: [0] * 16, 1: [0]})


def test_allocator_knows_every_op():
    """Each op either writes rd or does not: a new op must be classified for the allocator."""
    from qgpu.circuit.kernel import OPS
    from qgpu.lower import NO_WRITE, WRITES
    assert set(OPS) == WRITES | NO_WRITE and not WRITES & NO_WRITE
