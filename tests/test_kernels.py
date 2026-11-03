"""Every kernel in kernels/ on the reference SIMT interpreter, checked against numpy."""

from pathlib import Path

import numpy as np
import pytest

from qgpu import driver, ptx
from qgpu.sim import GLOBAL_BASE, Memory, launch

KERNELS = Path(__file__).parent.parent / "kernels"


def kernel(name: str) -> ptx.Kernel:
    return ptx.load(KERNELS / f"{name}.ptx")[name]


def i32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64).astype(np.int32)


class Dev:
    """A device memory with numpy-style buffers."""

    def __init__(self) -> None:
        self.mem = Memory(1 << 20, GLOBAL_BASE)

    def buf(self, values) -> int:
        values = list(np.asarray(values, dtype=np.int64))
        addr = self.mem.alloc(4 * max(len(values), 1))
        self.mem.write_i32(addr, values)
        return addr

    def zeros(self, n: int) -> int:
        return self.buf([0] * n)

    def get(self, addr: int, n: int, signed: bool = True) -> np.ndarray:
        return np.array(self.mem.read_i32(addr, n, signed), dtype=np.int64)


rng = np.random.default_rng(0)
# (n, block): partial warps and partial last blocks included
SIZES = [(1, 32), (31, 32), (100, 64), (257, 128), (70, 48)]


@pytest.mark.parametrize("n,block", SIZES)
def test_vadd(n, block):
    d = Dev()
    a, b = rng.integers(-2**31, 2**31, n), rng.integers(-2**31, 2**31, n)
    pa, pb, pc = d.buf(a), d.buf(b), d.zeros(n)
    launch(kernel("vadd"), d.mem, ((n + block - 1) // block,), (block,), [pa, pb, pc, n])
    assert (d.get(pc, n) == i32(a + b)).all()


@pytest.mark.parametrize("n,block", SIZES)
def test_saxpy(n, block):
    d = Dev()
    a = int(rng.integers(-1000, 1000))
    x, y = rng.integers(-2**31, 2**31, n), rng.integers(-2**31, 2**31, n)
    px, py = d.buf(x), d.buf(y)
    launch(kernel("saxpy"), d.mem, ((n + block - 1) // block,), (block,), [a, px, py, n])
    assert (d.get(py, n) == i32(a * x + y)).all()


@pytest.mark.parametrize("n,block", [(1, 32), (100, 32), (1000, 32), (513, 16), (37, 8)])
def test_reduce(n, block):
    d = Dev()
    x = rng.integers(-2**20, 2**20, n)
    px, pout = d.buf(x), d.zeros(1)
    stats = launch(kernel("reduce"), d.mem, ((n + block - 1) // block,), (block,), [px, pout, n])
    assert d.get(pout, 1)[0] == i32(x.sum())
    assert stats.divergent_branches > 0


@pytest.mark.parametrize("n,block", [(1, 16), (64, 16), (100, 8), (300, 16), (40, 16)])
def test_scan(n, block):
    d = Dev()
    x = rng.integers(-1000, 1000, n)
    px, py = d.buf(x), d.zeros(n)
    launch(kernel("scan"), d.mem, ((n + block - 1) // block,), (block,), [px, py, n])
    want = np.concatenate([np.cumsum(x[s:s + block]) for s in range(0, n, block)])
    assert (d.get(py, n) == want).all()


@pytest.mark.parametrize("n,block", SIZES)
def test_histogram(n, block):
    d = Dev()
    x = rng.integers(0, 2**32, n)
    px, pbins = d.buf(x), d.zeros(16)
    launch(kernel("histogram"), d.mem, ((n + block - 1) // block,), (block,), [px, pbins, n])
    assert (d.get(pbins, 16) == np.bincount(x & 15, minlength=16)).all()


@pytest.mark.parametrize("n,tile", [(1, 4), (5, 4), (8, 8), (13, 4)])
def test_matmul(n, tile):
    d = Dev()
    A, B = rng.integers(-1000, 1000, (n, n)), rng.integers(-1000, 1000, (n, n))
    pa, pb, pc = d.buf(A.ravel()), d.buf(B.ravel()), d.zeros(n * n)
    g = (n + tile - 1) // tile
    launch(kernel("matmul"), d.mem, (g, g), (tile, tile), [pa, pb, pc, n])
    assert (d.get(pc, n * n).reshape(n, n) == i32(A @ B)).all()


def c_diverge(v: int) -> int:
    q = abs(v) // 3 * (1 if v >= 0 else -1)  # C division truncates toward zero
    r = v - 3 * q
    if r == 0:
        return q
    if r == 1:
        return v * 2 - 7
    return (-v) ^ 0x55


@pytest.mark.parametrize("n,block", SIZES)
def test_diverge(n, block):
    d = Dev()
    x = rng.integers(-10**6, 10**6, n)
    px, py = d.buf(x), d.zeros(n)
    stats = launch(kernel("diverge"), d.mem, ((n + block - 1) // block,), (block,), [px, py, n])
    assert (d.get(py, n) == i32([c_diverge(int(v)) for v in x])).all()
    if n > 8:
        assert stats.divergent_branches > 0


def collatz_steps(v: int) -> int:
    s = 0
    while v > 1:
        v = (3 * v + 1) & 0xFFFFFFFF if v & 1 else v >> 1
        s += 1
    return s


@pytest.mark.parametrize("n,block", [(1, 32), (40, 32), (100, 64)])
def test_collatz(n, block):
    d = Dev()
    x = rng.integers(0, 5000, n)
    px, ps = d.buf(x), d.zeros(n)
    launch(kernel("collatz"), d.mem, ((n + block - 1) // block,), (block,), [px, ps, n])
    assert (d.get(ps, n) == [collatz_steps(int(v)) for v in x]).all()


@pytest.mark.parametrize("n,block", [(32, 32), (64, 64), (100, 128), (256, 96)])
def test_warpsum(n, block):
    d = Dev()
    x = rng.integers(-2**20, 2**20, n)
    nb = (n + block - 1) // block
    nw = nb * block // 32
    px, pout = d.buf(x), d.zeros(nw)
    launch(kernel("warpsum"), d.mem, (nb,), (block,), [px, pout, n])
    xs = np.concatenate([x, np.zeros(nb * block - n, dtype=np.int64)])
    assert (d.get(pout, nw) == xs.reshape(-1, 32).sum(1)).all()


def test_reconvergence_points():
    k = kernel("diverge")
    lab = k.labels
    # the bounds check joins at ret; both arms of the if/else chain join at the store block
    assert driver.reconvergence(k) == {10: lab["$L__BB0_6"], 21: lab["$L__BB0_5"],
                                       23: lab["$L__BB0_5"]}


def run_ptx(text: str, name: str, block: int, args_fn) -> list[int]:
    d = Dev()
    k = ptx.parse(text)[name]
    out = d.zeros(block)
    launch(k, d.mem, (1,), (block,), args_fn(d) + [out])
    return list(d.get(out, block))


def test_predicate_logic():
    """or/and/xor/not on .pred (nvcc joins bounds checks with or.pred)."""
    text = """
.visible .entry preds(
	.param .u64 preds_param_0
)
{
	.reg .pred 	%p<8>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<4>;
	ld.param.u64 	%rd1, [preds_param_0];
	mov.u32 	%r1, %tid.x;
	and.b32 	%r2, %r1, 1;
	setp.ne.s32 	%p1, %r2, 0;
	and.b32 	%r3, %r1, 2;
	setp.ne.s32 	%p2, %r3, 0;
	or.pred 	%p3, %p1, %p2;
	and.pred 	%p4, %p1, %p2;
	xor.pred 	%p5, %p1, %p2;
	not.pred 	%p6, %p1;
	selp.b32 	%r4, 1, 0, %p3;
	selp.b32 	%r5, 2, 0, %p4;
	add.s32 	%r4, %r4, %r5;
	selp.b32 	%r5, 4, 0, %p5;
	add.s32 	%r4, %r4, %r5;
	selp.b32 	%r5, 8, 0, %p6;
	add.s32 	%r4, %r4, %r5;
	mul.wide.u32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	st.global.u32 	[%rd3], %r4;
	ret;
}
"""
    got = run_ptx(text, "preds", 4, lambda d: [])
    want = []
    for t in range(4):
        a, b = bool(t & 1), bool(t & 2)
        want.append((a or b) * 1 + (a and b) * 2 + (a != b) * 4 + (not a) * 8)
    assert got == want


def test_every_kernel_parses():
    names = sorted(p.stem for p in KERNELS.glob("*.ptx"))
    assert names == ["addk", "bitops", "branchy", "collatz", "colsum", "divdec", "diverge",
                     "divk", "divloop", "dot", "fdivk", "histogram", "matmul", "mean", "mulk",
                     "reduce", "rsubk", "saxpy", "scan", "shfl", "subk", "udivk", "vadd", "vmax",
                     "vmul", "vsub", "warpsum"]
    for nm in names:
        k = kernel(nm)
        assert k.instrs and k.params
