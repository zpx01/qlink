"""The kernel circuit: global loads, token-free stores, a warp loop. Exact in the IR
evaluator, and the stored words read back from the `out` heads' recurrent state."""

from functools import cache

import numpy as np
import pytest
from chipformer.q35.ir import evaluate

from qgpu.circuit.kernel import LANES, Kernel, predicted, stored, trace
from qgpu.circuit.kernel import KInstr as K


@cache
def circuit() -> Kernel:
    return Kernel()


def run(program, mem, warps, outputs=frozenset(), **kw):
    tr = trace(program, mem, warps, outputs, **kw)
    carry = {}
    # the last token is the circuit's own "next warp" prediction past the launch: the host
    # stops there and never feeds it back
    m = evaluate(circuit(), tr.tokens[:-1], carry=carry)
    bad = [t for t in tr.owned if predicted(circuit(), m, t) != tr.tokens[t + 1]]
    assert not bad, f"{len(bad)} wrong predictions, first at {bad[0]}"
    assert stored(circuit(), carry) == tr.out
    return tr


def words(rng, n):
    return [int(x) for x in rng.integers(0, 2**32, n)]


VADD = [K("tid", rd=1), K("ld", rd=2, ra=1, imm=0), K("ld", rd=3, ra=1, imm=32),
        K("add", rd=4, ra=2, rb=3), K("st", ra=1, rb=4, imm=64)]


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_vadd_one_token_per_warp(warps):
    rng = np.random.default_rng(warps)
    n = LANES * warps
    a, b = words(rng, n), words(rng, n)
    mem = {i: a[i] for i in range(n)} | {32 + i: b[i] for i in range(n)}
    tr = run(VADD, mem, warps)
    assert tr.out == {64 + i: (a[i] + b[i]) & 0xFFFFFFFF for i in range(n)}
    assert len(tr.owned) == warps  # one EXEC per warp, no VALs


def test_saxpy_three_x_plus_y():
    """y = 3x + y with adds; the stored value is forwarded through three stages."""
    rng = np.random.default_rng(7)
    x, y = words(rng, 16), words(rng, 16)
    prog = [K("tid", rd=1), K("ld", rd=2, ra=1, imm=0), K("add", rd=3, ra=2, rb=2),
            K("add", rd=3, ra=3, rb=2), K("ld", rd=4, ra=1, imm=16),
            K("add", rd=4, ra=4, rb=3), K("st", ra=1, rb=4, imm=96)]
    mem = {i: x[i] for i in range(16)} | {16 + i: y[i] for i in range(16)}
    tr = run(prog, mem, 2)
    assert tr.out == {96 + i: (3 * x[i] + y[i]) & 0xFFFFFFFF for i in range(16)}
    assert len(tr.blocks) == 2  # 7 instructions over 5-stage blocks, with live-outs between


def test_store_reads_a_register_a_live_out_overwrites():
    """A block that stores r2 and also writes r2 back as a live-out: the store must use the
    old r2 even though later tokens of the record see r2 committed."""
    rng = np.random.default_rng(9)
    x = words(rng, 8)
    prog = [K("tid", rd=1), K("ld", rd=2, ra=1, imm=0), K("add", rd=5, ra=1, rb=1),
            K("add", rd=5, ra=5, rb=5), K("add", rd=5, ra=5, rb=5),
            K("st", ra=1, rb=2, imm=64), K("add", rd=2, ra=2, rb=2), K("st", ra=5, rb=2, imm=72),
            K("add", rd=2, ra=2, rb=1), K("st", ra=1, rb=2, imm=96)]
    tr = run(prog, {i: x[i] for i in range(8)}, 1, lmax=2)  # blocks as with 2 live-outs
    # the hazard is really there: some block stores a register it also writes back
    assert any(any(i.op == "st" and i.rb == rd for i in blk.instrs[:s])
               for blk in tr.blocks for s, rd in blk.live)
    assert tr.out[64] == x[0] and tr.out[96 + 3] == (2 * x[3] + 3) & 0xFFFFFFFF


def test_address_wraps_and_immediates():
    rng = np.random.default_rng(3)
    x = words(rng, 128)
    prog = [K("tid", rd=1), K("ld", rd=2, ra=1, imm=125), K("st", ra=1, rb=2, imm=120)]
    tr = run(prog, dict(enumerate(x)), 1)
    assert tr.out == {(120 + i) % 128: x[(125 + i) % 128] for i in range(LANES)}
