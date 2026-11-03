"""The GPU monitor follows the model's chip tokens: decoded against the reference."""

import pytest

from qgpu.circuit.kernel import EXEC0, LOAD0, run_blocks
from qgpu.monitor import Monitor, Program
from qgpu.runtime import host_tokens, prepare


@pytest.mark.parametrize("name,data,n", [
    ("vadd", {"a": [1, -2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}, 5),
    ("divk", {"x": [50, -17, 9], "k": 4}, 3),
    ("reduce", {"x": list(range(20))}, 20),
    ("histogram", {"x": [1, 3, 3, 7, 19]}, 5),
    ("matmul", {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, 2),
])
def test_monitor_decodes_the_stream(name, data, n):
    """Every record is seen, and the memory write-backs decoded from the stream are the
    launch's outputs (write-through: every store is a write-back)."""
    p = prepare(name, data, n, 4, 4)
    lw = p.launch
    tr = run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
    mon = Monitor()
    mon.load(Program.of(lw, 4))
    for t in tr.tokens[len(host_tokens(lw)):]:  # only what the model writes, as the app sees it
        mon.feed(t)
        mon.tick()
    assert mon.done
    assert mon.records == sum(EXEC0 <= t < LOAD0 for t in tr.tokens) - 1  # but the end token
    # block 0 starts at the host's EXEC: its values are decoded too
    assert all((0, rd, L) in mon.values for _, rd in Program.of(lw, 4).live[0] for L in range(4))
    for bid, size in p.outputs.values():
        for a in range(lw.buffers[bid], lw.buffers[bid] + size):
            if a in mon.written or tr.mem[a] != lw.mem.get(a, 0):
                assert mon.written[a] == tr.mem[a], a
    assert all(v < 1 for v in mon.warps)  # lights fade
