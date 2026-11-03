"""The GPU runtime: every library kernel through the circuit, several launches in one
context, division at the edges of the range."""

import pytest

from qgpu.circuit.kernel import run_blocks
from qgpu.runtime import KERNELS, host_tokens, prepare, read

CIRCUIT = {
    "vadd": ({"a": [1, -2, 3], "b": [10, 20, 30]}, 3),
    "vsub": ({"a": [1, -2, 3], "b": [10, 20, -30]}, 3),
    "vmul": ({"a": [7, -3, 100000], "b": [6, 5, 70000]}, 3),
    "addk": ({"x": [7, 1, -9], "k": 10}, 3),
    "subk": ({"x": [7, 1, -9], "k": 3}, 3),
    "rsubk": ({"x": [7, 1, -9], "k": 3}, 3),
    "mulk": ({"x": [2, 3, -4, 5, 6], "k": -4}, 5),
    "divk": ({"x": [50, -17, 9, 2147483647, -2147483648], "k": -7}, 5),
    "fdivk": ({"x": [50, -17, 9, -2147483648, 7], "k": -7}, 5),
    "divdec": ({"x": [50, -17, 9, 2147483647, -1], "k": 6}, 5),
    "reduce": ({"x": [5, 6, 7, -1, 2]}, 5),
    "mean": ({"x": [-12, 7, 30, 5, 1]}, 5),
    "vmax": ({"x": [-4, 9, 27, -30, 5]}, 5),
    "scan": ({"x": [3, 1, 4, 1, 5]}, 5),
    "dot": ({"a": [3, 2, 9], "b": [12, 45, -1]}, 3),
    "histogram": ({"x": [1, 3, 3, 7, 19]}, 5),
    "matmul": ({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, 2),
}


def circuit():
    from qgpu.circuit.kernel import Kernel
    from qgpu.merge import STAGES

    mul, logic, atom = STAGES[4]
    return Kernel(lanes=4, stages=4, mul_stages=mul, logic_stages=logic, atom_stages=atom)


@pytest.mark.parametrize("name", list(KERNELS))
def test_kernel_through_circuit(name):
    """Every runtime kernel through the circuit (merged config): every owned token is right and
    global memory, read as the host reads the model's state, holds the reference outputs."""
    from chipformer.q35.ir import evaluate as ir

    from qgpu.circuit.kernel import memory, predicted

    data, n = CIRCUIT[name]
    p = prepare(name, data, n, 4, 4)
    lw = p.launch
    tr = run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
    assert not tr.out  # write-through: every result is in global memory
    host = host_tokens(lw)
    assert tr.tokens[:len(host)] == host and tr.owned[0] == len(host) - 1
    c, carry = circuit(), {}
    m = ir(c, tr.tokens[:-1], carry=carry)
    bad = [t for t in tr.owned if t + 1 < len(tr.tokens)
           and predicted(c, m, t) != tr.tokens[t + 1]]
    assert not bad, f"{len(bad)} wrong tokens of {len(tr.tokens)}"
    assert read(p, memory(c, carry)) == read(p, tr.mem)


def test_several_launches_in_one_context():
    """vadd, then histogram over the addresses vadd used, then subk, with text between them,
    through the circuit: each launch exact, memory right after each. From the second launch on
    the host zeroes the rest of memory (inactive lanes' loads read past the buffers)."""
    from chipformer.q35.ir import evaluate as ir

    from qgpu.circuit.kernel import TEXT, memory, predicted, stored

    c = circuit()
    calls = [("vadd", {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}, 5),
             ("histogram", {"x": [1, 17, 2, 3, 3, 3]}, 6),
             ("subk", {"x": [1, 2], "k": 5}, 2)]
    seq = [TEXT] * 3
    for i, (name, data, n) in enumerate(calls):
        p = prepare(name, data, n, 4, 4)
        lw = p.launch
        tr = run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
        host, plain = host_tokens(lw, clear=i > 0), len(host_tokens(lw))
        tokens, owned = host + tr.tokens[plain:], [t + len(host) - plain for t in tr.owned]
        start = len(seq)
        seq += tokens
        m = ir(c, seq[:-1])
        bad = [t for t in owned if t + 1 < len(tokens)
               and predicted(c, m, start + t) != tokens[t + 1]]
        assert not bad, (name, len(bad))
        carry = {}
        ir(c, seq, carry=carry)
        assert not stored(c, carry)
        assert read(p, memory(c, carry)) == read(p, tr.mem), name
        seq += [TEXT] * 4


@pytest.mark.parametrize("k", [1, 2, 3, 7, 10, 641, 1 << 31, 0xFFFFFFFF, 0x9E3779B9])
def test_unsigned_division_through_circuit(k):
    """div.u32 / rem.u32 by a value known at launch (udivk.ptx), against Python, through the
    circuit: the driver's multiply-high sequence is exact at the edges of the range."""
    from chipformer.q35.ir import evaluate as ir

    from qgpu import ptx
    from qgpu.circuit.kernel import memory, predicted
    from qgpu.lower import lower
    from qgpu.merge import STAGES

    mul, logic, atom = STAGES[4]
    xs = [0, 1, k - 1 & 0xFFFFFFFF, k, 0xFFFFFFFF, 0x80000000, 123456789, 0xDEADBEEF]
    lw = lower(ptx.load("kernels/udivk.ptx")["udivk"], (1,), (8,), [0, 1, 2, k, 8],
               {0: xs, 1: [0] * 8, 2: [0] * 8}, lanes=4, stages=4, mul_stages=mul,
               logic_stages=logic, atom_stages=atom, write_through=True)
    tr = run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
    q = [tr.mem[lw.buffers[1] + i] for i in range(8)]
    r = [tr.mem[lw.buffers[2] + i] for i in range(8)]
    assert q == [x // k for x in xs] and r == [x % k for x in xs]
    c, carry = circuit(), {}
    m = ir(c, tr.tokens[:-1], carry=carry)
    assert all(predicted(c, m, t) == tr.tokens[t + 1] for t in tr.owned if t + 1 < len(tr.tokens))
    assert {a: memory(c, carry).get(a) for a in tr.mem} == tr.mem
