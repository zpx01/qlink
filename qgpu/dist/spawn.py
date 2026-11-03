"""Run an SPMD program on a simulated multi-QwenGPU node: one thread per rank, each with its own
device and backend, connected by a ThreadTransport. (For one process per physical GPU, run the
same program under torchrun with a TorchTransport: scripts/qlink_run.py.)"""

import threading

from qgpu.dist.device import QGPU
from qgpu.dist.transport import ThreadTransport


def spawn(world: int, program, backend_factory, **dev_kw):
    """program(rank, dev, ep) -> result; returns [result per rank] (re-raises a rank's error)."""
    fabric = ThreadTransport(world)
    results, errors = [None] * world, [None] * world

    def run(rank):
        try:
            dev = QGPU(rank, backend_factory(rank), **dev_kw)
            results[rank] = program(rank, dev, fabric.endpoint(rank))
        except BaseException as e:  # noqa: BLE001 - surfaced below
            errors[rank] = e
            fabric.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None and not isinstance(e, threading.BrokenBarrierError):
            raise e
    for e in errors:
        if e is not None:
            raise e
    return results
