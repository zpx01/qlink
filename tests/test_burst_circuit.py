"""The burst circuit (blocks of up to S instructions per pass) and its block driver."""

from functools import cache

import numpy as np
import pytest
from chipformer.q35.ir import evaluate

from qgpu.circuit import blocks
from qgpu.circuit.burst import LANES, LMAX, Burst, S, predicted, trace


@cache
def circuit() -> Burst:
    return Burst()


def lanes(rng, edge=False):
    if edge:
        return [int(v) for v in rng.choice([0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF], LANES)]
    return [int(v) for v in rng.integers(0, 2**32, LANES)]


def check(program, init, outputs):
    tr = trace(program, init, outputs)
    m = evaluate(circuit(), tr.tokens)
    bad = [t for t in tr.owned if predicted(circuit(), m, t) != tr.tokens[t + 1]]
    assert not bad, f"{len(bad)} wrong predictions, first at {bad[0]}"
    return tr


def sequential(program, init):
    regs = {r: list(v) for r, v in init.items()}
    zero = [0] * LANES
    for rd, ra, rb in program:
        regs[rd] = [(x + y) & 0xFFFFFFFF for x, y in zip(regs.get(ra, zero), regs.get(rb, zero))]
    return regs


@pytest.mark.parametrize("seed", range(8))
def test_random_programs(seed):
    rng = np.random.default_rng(seed)
    nreg = int(rng.integers(3, 8))
    program = [tuple(int(x) for x in rng.integers(0, nreg, 3))
               for _ in range(int(rng.integers(1, 30)))]
    init = {r: lanes(rng, edge=seed % 2 == 1) for r in range(nreg)}
    outputs = {int(x) for x in rng.choice(nreg, 2, replace=False)}
    tr = check(program, init, outputs)
    want = sequential(program, init)
    for r in outputs:
        assert tr.regs[r] == want[r]


def test_forwarding_chain_in_one_block():
    """Each stage reads the previous stage's result: r3 = 9 x through 2x, 4x, 8x."""
    rng = np.random.default_rng(1)
    x = lanes(rng)
    program = [(2, 1, 1), (2, 2, 2), (2, 2, 2), (3, 2, 1)]  # r3 = 9x
    tr = check(program, {1: x}, {3})
    assert len(tr.blocks) == 1 and tr.blocks[0].live == [(3, 3)]
    assert tr.regs[3] == [(9 * v) & 0xFFFFFFFF for v in x]


def test_live_out_order_respects_inputs():
    """r1 = r1 + r2; r3 = r1 + r2: r3's chain reads the input r1, which live-out r1 overwrites,
    so r3 is emitted first."""
    rng = np.random.default_rng(2)
    program = [(1, 1, 2), (3, 1, 2)]
    blk = blocks.split(program, {1, 3}, S, LMAX)
    assert len(blk) == 1 and blk[0].live == [(1, 3), (0, 1)]
    check(program, {1: lanes(rng), 2: lanes(rng)}, {1, 3})


def test_cyclic_live_outs_split_the_block():
    """r1 <- f(r2) and r2 <- f(r1) in one block cannot be ordered: the driver splits it."""
    rng = np.random.default_rng(3)
    program = [(5, 1, 1), (1, 2, 2), (2, 5, 5)]
    assert blocks.plan(program, {1, 2}, LMAX) is None
    tr = check(program, {1: lanes(rng), 2: lanes(rng)}, {1, 2})
    assert len(tr.blocks) == 2


def test_dead_results_cost_no_tokens():
    rng = np.random.default_rng(4)
    program = [(3, 1, 2), (4, 3, 3), (5, 4, 4), (6, 5, 1)]
    tr = check(program, {1: lanes(rng), 2: lanes(rng)}, {6})
    assert [len(b.live) for b in tr.blocks] == [1]
    assert len(tr.owned) == 1 + 16  # one EXEC, one live-out
