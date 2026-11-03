"""QLink and qccl: several QwenGPUs, their collectives, against numpy.

Reference-level tests run every collective on threads (fast). The circuit test runs one
AllReduce on the compiled GPU layers, evaluated exactly, and checks that both ranks' token
streams equal the reference's: the weights themselves accept a peer's store at a record
boundary and the sink header for a store sent away.
"""

from pathlib import Path

import numpy as np
import pytest

from qgpu.dist.backends import CircuitBackend, RefBackend
from qgpu.dist.device import OWN, PEER, LinkError
from qgpu.dist.qccl import Comm
from qgpu.dist.spawn import spawn
from qgpu.dist.transport import synchronize


def i32(a):
    return ((np.asarray(a, dtype=np.int64) + 2**31) % 2**32 - 2**31).tolist()


def words(rng, *shape):
    return rng.integers(-2**31, 2**31, shape).tolist()


def run(world, body, backend=lambda r: RefBackend(), chunk=4, **kw):
    def program(rank, dev, ep):
        comm = Comm(dev, rank, world, chunk=chunk)
        out = body(rank, dev, comm)
        synchronize(dev, ep)
        return {k: dev.read(a, n) for k, (a, n) in out.items()}, dev
    return spawn(world, program, backend, **kw)


@pytest.mark.parametrize("world,n,chunk", [(2, 8, 4), (2, 32, 16), (3, 12, 4), (4, 16, 4),
                                           (4, 32, 8), (4, 64, 8), (2, 64, 16)])
def test_all_reduce(world, n, chunk):
    rng = np.random.default_rng(n + world)
    data = words(rng, world, n)

    def body(rank, dev, comm):
        x = dev.malloc(n)
        dev.memcpy_h2d(x, data[rank])
        comm.all_reduce(x, n)
        return {"x": (x, n)}

    res = run(world, body, chunk=chunk)
    want = i32(np.sum(data, axis=0))
    assert all(r["x"] == want for r, _ in res)


@pytest.mark.parametrize("world,c", [(2, 4), (3, 4), (4, 8)])
def test_reduce_scatter_and_all_gather(world, c):
    rng = np.random.default_rng(c * world)
    data = words(rng, world, world * c)
    mine = words(rng, world, c)

    def body(rank, dev, comm):
        x, y = dev.malloc(world * c), dev.malloc(c)
        g, z = dev.malloc(c), dev.malloc(world * c)
        dev.memcpy_h2d(x, data[rank])
        dev.memcpy_h2d(g, mine[rank])
        comm.reduce_scatter(x, y, c)
        comm.all_gather(g, z, c)
        return {"rs": (y, c), "ag": (z, world * c)}

    res = run(world, body, chunk=c)
    total = i32(np.sum(data, axis=0))
    for rank, (r, _) in enumerate(res):
        assert r["rs"] == total[rank * c:(rank + 1) * c]
        assert r["ag"] == [v for m in mine for v in m]


@pytest.mark.parametrize("world,root", [(2, 1), (3, 0), (4, 2)])
def test_broadcast_reduce(world, root):
    rng = np.random.default_rng(root + 10 * world)
    n = 12
    data = words(rng, world, n)

    def body(rank, dev, comm):
        b, r = dev.malloc(16), dev.malloc(16)
        dev.memcpy_h2d(b, data[rank] + [0] * 4)
        dev.memcpy_h2d(r, data[rank] + [0] * 4)
        comm.broadcast(b, 16, root=root)
        comm.reduce(r, 16, root=root)
        return {"b": (b, n), "r": (r, n)}

    res = run(world, body)
    for rank, (r, _) in enumerate(res):
        assert r["b"] == data[root]
        assert r["r"] == (i32(np.sum(data, axis=0)) if rank == root else data[rank])


def test_sendrecv_ring_shift():
    """Every rank sends to the next and receives from the previous (a pipeline's step), more
    chunks than the FIFO holds: grouped, as ncclGroupStart/End would."""
    world, n = 4, 24
    rng = np.random.default_rng(5)
    data = words(rng, world, n)

    def body(rank, dev, comm):
        x, y = dev.malloc(n), dev.malloc(n)
        dev.memcpy_h2d(x, data[rank])
        comm.sendrecv(x, y, n)
        return {"y": (y, n)}

    res = run(world, body)
    for rank, (r, _) in enumerate(res):
        assert r["y"] == data[(rank - 1) % world]


def test_flow_control_one_way():
    """Rank 0 streams 8 chunks through a 2-slot FIFO to rank 1: it must wait for credits."""
    n = 32
    rng = np.random.default_rng(9)
    data = words(rng, n)

    def body(rank, dev, comm):
        x = dev.malloc(n)
        if rank == 0:
            dev.memcpy_h2d(x, data)
            comm.send(x, n)
        else:
            comm.recv(x, n)
        return {"x": (x, n)}

    res = run(2, body)
    assert res[1][0]["x"] == data


def test_unpaired_sends_deadlock():
    """Both ranks send everything before receiving: with a bounded FIFO each waits for the
    other's credits forever (NCCL needs a group for this too). The reference interpreter
    reports a launch that never ends."""
    n = 32
    rng = np.random.default_rng(9)
    data = words(rng, 2, n)

    def body(rank, dev, comm):
        x, y = dev.malloc(n), dev.malloc(n)
        dev.memcpy_h2d(x, data[rank])
        comm.send(x, n)
        comm.recv(y, n)
        return {}

    with pytest.raises(AssertionError, match="max_records"):
        run(2, body)


def test_remote_atomic_refused():
    from qgpu import ptx
    from qgpu.lower import DevPtr

    k = ptx.load("kernels/nvcc/reduce.ptx")["reduce"]

    def body(rank, dev, comm):
        x = dev.malloc(4)
        dev.memcpy_h2d(x, [1, 2, 3, 4])
        if rank == 0:  # atomicAdd into the channel's FIFO: a peer aperture
            dev.launch(k, 4, [DevPtr(x), DevPtr(comm.fifo), 4])
        return {}

    with pytest.raises(LinkError, match="atomic"):
        run(2, body)


def test_stream_provenance():
    """Every word a rank receives arrives as PEER tokens; its own stores to the peer are fed
    locally with the sink header."""
    rng = np.random.default_rng(3)
    data = words(rng, 2, 8)

    def body(rank, dev, comm):
        x = dev.malloc(8)
        dev.memcpy_h2d(x, data[rank])
        comm.all_reduce(x, 8)
        return {"x": (x, 8)}

    res = run(2, body)
    for _, dev in res:
        peer = [t for t, s in zip(dev.log, dev.src) if s == PEER]
        own = [t for t, s in zip(dev.log, dev.src) if s == OWN]
        assert len(peer) == 3 * sum(h.sent for h in dev.history)  # symmetric traffic
        assert own


@pytest.mark.slow
def test_all_reduce_on_the_circuit():
    world, n = 2, 8
    rng = np.random.default_rng(0)
    data = words(rng, world, n)

    def body(rank, dev, comm):
        x = dev.malloc(n)
        dev.memcpy_h2d(x, data[rank])
        comm.all_reduce(x, n)
        return {"x": (x, n)}

    ref = run(world, body)
    c = CircuitBackend().c
    got = run(world, body, backend=lambda r: CircuitBackend(c), mode="draft")
    for (rr, rd), (gr, gd) in zip(ref, got):
        assert gr["x"] == i32(np.sum(data, axis=0)) == rr["x"]
        assert gd.log == rd.log


@pytest.mark.parametrize("world", [2, 3, 4])
def test_multicast_all_gather_and_nvls_all_reduce(world):
    """NVLS-style: a multicast window replicates each store to every rank."""
    c = 4
    rng = np.random.default_rng(world)
    mine, data = words(rng, world, c), words(rng, world, world * c)

    def body(rank, dev, comm):
        flags = comm.window(world, "flags")
        g, y = dev.malloc(c), comm.window(world * c, "ag")
        x, z = dev.malloc(world * c), comm.window(world * c, "ar")
        dev.memcpy_h2d(g, mine[rank])
        dev.memcpy_h2d(x, data[rank])
        comm.all_gather_mc(g, y, c, flags, epoch=1)
        comm.all_reduce_nvls(x, z, world * c, flags, epoch=2)
        return {"ag": (y, world * c), "ar": (z, world * c)}

    res = run(world, body, chunk=c)
    for r, _ in res:
        assert r["ag"] == [v for m in mine for v in m]
        assert r["ar"] == i32(np.sum(data, axis=0))


@pytest.mark.slow
def test_multicast_and_rebase_on_the_circuit():
    """Multicast windows (the local write kept, two routes) and a context rebase between
    collectives, on the compiled circuit in verify mode: streams equal the reference's."""
    world, c = 2, 4
    rng = np.random.default_rng(3)
    mine, data = words(rng, world, c), words(rng, world, world * c)

    def body(rank, dev, comm):
        flags = comm.window(world, "f")
        g, y = dev.malloc(c), comm.window(world * c, "ag")
        x, z = dev.malloc(world * c), comm.window(world * c, "ar")
        dev.memcpy_h2d(g, mine[rank])
        dev.memcpy_h2d(x, data[rank])
        comm.all_gather_mc(g, y, c, flags, 1)
        return {"y": (y, world * c), "z": (z, world * c), "x": (x, world * c)}

    def program(rank, dev, ep):
        comm = Comm(dev, rank, world, chunk=c)
        bufs = body(rank, dev, comm)
        synchronize(dev, ep)
        dev.rebase()
        comm.all_reduce_nvls(bufs["x"][0], bufs["z"][0], world * c, dev.allocs["f"][0], 2)
        synchronize(dev, ep)
        return dev.read(bufs["y"][0], world * c), dev.read(bufs["z"][0], world * c), dev.log

    ref = spawn(world, program, lambda r: RefBackend())
    circ = CircuitBackend().c
    got = spawn(world, program, lambda r: CircuitBackend(circ), mode="verify")
    for (ry, rz, rl), (gy, gz, gl) in zip(ref, got):
        assert gy == ry == [v for m in mine for v in m]
        assert gz == rz == i32(np.sum(data, axis=0))
        assert gl == rl


MNIST = Path(__file__).resolve().parent.parent / "data" / "mnist28.npz"
needs_mnist = pytest.mark.skipif(not MNIST.exists(), reason="python scripts/get_mnist.py")


@needs_mnist
@pytest.mark.parametrize("recipe", ["default", "first run"])
def test_mnist_tensor_parallel_matches_numpy(recipe):
    from qgpu.dist import mnist
    cfg = mnist.Config(data=str(MNIST), checkpoint_every=10,
                       **(mnist.FIRST_RUN if recipe == "first run" else {}))
    cfg.steps = 30
    res = spawn(4, lambda r, dev, ep: mnist.program(r, dev, ep, cfg, log=lambda s: None),
                lambda r: RefBackend(), keep_log=False)
    X, y, _, _ = mnist.load(cfg)
    ref = mnist.reference(cfg, X, y)
    assert mnist.assemble([r.W for r in res]).tolist() == ref["W"]
    for s, W in ref["checkpoints"].items():
        assert mnist.assemble([r.checkpoints[s] for r in res]).tolist() == W


@needs_mnist
def test_latent_diffusion_matches_numpy():
    from qgpu.dist import latent
    cfg = latent.Config(steps=20, samples=2, checkpoint_every=10, data=str(MNIST))
    res = spawn(4, lambda r, dev, ep: latent.program(r, dev, ep, cfg, log=lambda s: None),
                lambda r: RefBackend(), keep_log=False)
    ref = latent.reference(cfg)
    assert latent.assemble([r.params for r in res]) == ref["P"]
    for s, P in ref["checkpoints"].items():
        assert latent.assemble([r.checkpoints[s] for r in res]) == P
    assert all(r.trajectory == ref["samples"] for r in res)  # every DDIM step of every sample
