"""Data-parallel training on QwenGPUs: a 2-8-1 ReLU MLP learns XOR (the sign of x0 * x1).

Every rank holds a replica of the parameters in device memory and a shard of the training set.
One step on each rank, all on the GPUs in the models' weights:

    H2D        the rank's minibatch (its data loader), o = 0, loss = 0
    forward    kernels/train/mlp.cu mlp_forward: o[s] = sum_k W2[k] relu(W1[k] x[s] + b1[k])
    backward   mlp_backward: per-sample gradients, summed into g by atomicAdd; the loss
    allreduce  qccl ring AllReduce of g over QLink: the global gradient on every rank
    sgd        p -= round(g / 2^shift); g = 0

Before step 0, rank 0 initialises the parameters and a qccl Broadcast gives every rank the
same replica (what DDP does at construction). All arithmetic is int32 fixed point (Q8, 1.0 =
256), so `reference` replays the same training exactly in numpy: the QwenGPUs' parameters must
equal it after every step, on every rank.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qgpu import ptx
from qgpu.dist.qccl import Comm
from qgpu.dist.transport import synchronize
from qgpu.lower import DevPtr

ROOT = Path(__file__).resolve().parent.parent.parent
HID, NP = 8, 32  # hidden units; parameters (W1 16, b1 8, W2 8)
Q = 256


@dataclass
class Config:
    world: int = 4
    batch: int = 8  # samples per rank per step
    steps: int = 40
    shift: int = 6  # learning rate: p -= g / 2^shift (g summed over world x batch samples)
    init_seed: int = 0
    data_seed: int = 1
    test_seed: int = 2
    sample_seed: int = 100
    n_train: int = 256
    chunk: int = 8
    zero: bool = False  # ZeRO-1 over multicast instead of AllReduce + a replicated update


def dataset(n: int, seed: int):
    """Points in [-1, 1]^2 away from the axes; label +1 if x0 x1 > 0 else -1. Q8."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (3 * n, 2))
    x = x[np.abs(x).min(1) > 0.1][:n]
    y = np.where(x[:, 0] * x[:, 1] > 0, 1, -1)
    return np.round(x * Q).astype(np.int64), (y * Q).astype(np.int64)


def init_params(seed: int) -> np.ndarray:
    """Hidden units along random unit directions (Q8), zero biases, output weights +-0.5."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, HID)
    w1 = np.round(np.stack([np.cos(ang), np.sin(ang)], 1).ravel() * Q).astype(np.int64)
    return np.concatenate([w1, np.zeros(HID, np.int64), rng.choice([-128, 128], HID)])


def batches(cfg: Config):
    """[step][rank] -> sample indices: rank r draws from its shard (every world-th sample)."""
    rng = np.random.default_rng(cfg.sample_seed)
    shards = [np.arange(r, cfg.n_train, cfg.world) for r in range(cfg.world)]
    return [[rng.choice(shards[r], cfg.batch, replace=False) for r in range(cfg.world)]
            for _ in range(cfg.steps)]


def forward(p, x):
    w1, b1, w2 = p[:2 * HID].reshape(HID, 2), p[2 * HID:3 * HID], p[3 * HID:]
    z = ((w1[None, :, 0] * x[:, :1] + w1[None, :, 1] * x[:, 1:]) >> 8) + b1[None]
    return z, ((w2[None] * np.maximum(z, 0)) >> 8).sum(1)


def grads(p, x, y):
    z, o = forward(p, x)
    e, w2, a = o - y, p[3 * HID:], np.maximum(z, 0)
    dz = np.where(z > 0, (e[:, None] * w2[None]) >> 8, 0)
    g = np.zeros(NP, np.int64)
    g[3 * HID:] = ((e[:, None] * a) >> 8).sum(0)
    g[0:2 * HID:2] = ((dz * x[:, :1]) >> 8).sum(0)
    g[1:2 * HID:2] = ((dz * x[:, 1:]) >> 8).sum(0)
    g[2 * HID:3 * HID] = dz.sum(0)
    return g, int(((e * e) >> 8).sum())


def accuracy(p, x, y) -> float:
    return float(np.mean(np.sign(forward(np.asarray(p, np.int64), x)[1]) == np.sign(y)))


def reference(cfg: Config) -> dict:
    """The same training in numpy int64 (no value leaves int32 range): params after each step
    and each rank's loss."""
    xtr, ytr = dataset(cfg.n_train, cfg.data_seed)
    p = init_params(cfg.init_seed)
    params, losses = [], []
    for idx in batches(cfg):
        gs, ls = np.zeros(NP, np.int64), []
        for r in range(cfg.world):
            g, loss = grads(p, xtr[idx[r]], ytr[idx[r]])
            gs += g
            ls.append(loss)
        p = p - ((gs + (1 << (cfg.shift - 1))) >> cfg.shift)
        params.append(p.tolist())
        losses.append(ls)
    return {"params": params, "losses": losses}


@dataclass
class StepLog:
    step: int
    loss: int  # this rank's batch loss, sum of (o - y)^2 / 256
    params: list
    seconds: float
    ticks: int
    generated: int
    records: int
    launches: dict = field(default_factory=dict)  # kernel -> generated tokens


def program(rank: int, dev, ep, cfg: Config, log=print) -> dict:
    """One rank's training run (SPMD). Returns its per-step logs."""
    ks = ptx.load(ROOT / "kernels" / "train" / "mlp.ptx")
    comm = Comm(dev, rank, cfg.world, chunk=cfg.chunk)
    B = cfg.batch
    if cfg.zero:  # parameters and epoch flags in multicast windows
        assert NP % cfg.world == 0 and NP // cfg.world <= cfg.chunk
        P, E, C = comm.window(NP, "params"), comm.window(cfg.world, "epochs"), NP // cfg.world
    else:
        P = dev.malloc(NP, "params")
    G = dev.malloc(NP, "grads")
    X, T = dev.malloc(2 * B, "batch.x"), dev.malloc(B, "batch.y")
    O, L = dev.malloc(B, "out"), dev.malloc(1, "loss")
    xtr, ytr = dataset(cfg.n_train, cfg.data_seed)
    plan = batches(cfg)
    t0 = time.time()
    dev.memcpy_h2d(P, init_params(cfg.init_seed).tolist() if rank == 0 else [0] * NP)
    dev.memcpy_h2d(G, [0] * NP)
    if cfg.zero:  # rank 0's stores into the window reach every replica
        if rank == 0:
            comm.mcast(P, P, NP, E, epoch=1)
        comm.wait(E, 1, 1)
    else:
        comm.broadcast(P, NP, root=0)  # every replica starts from rank 0's parameters
    ticks = synchronize(dev, ep)
    init = dev.read(P, NP)
    log(f"rank {rank}: initial parameters on every replica: {ticks} ticks, "
        f"{time.time() - t0:.1f}s")
    steps = []
    for step, idx in enumerate(plan):
        t0, h0 = time.time(), len(dev.history)
        xb, yb = xtr[idx[rank]], ytr[idx[rank]]
        dev.memcpy_h2d(X, xb.ravel().tolist())
        dev.memcpy_h2d(T, yb.tolist())
        dev.memcpy_h2d(O, [0] * B)
        dev.memcpy_h2d(L, [0])
        if cfg.zero:
            dev.memcpy_h2d(G, [0] * NP)  # zero_grad (a memset)
            if step:
                comm.wait(E, cfg.world, step + 1)  # every shard of the last step has landed
        threads = HID * B
        dev.launch(ks["mlp_forward"], threads, [DevPtr(P), DevPtr(X), DevPtr(O)],
                   block=min(threads, 32))
        dev.launch(ks["mlp_backward"], threads,
                   [DevPtr(P), DevPtr(X), DevPtr(T), DevPtr(O), DevPtr(G), DevPtr(L)],
                   block=min(threads, 32))
        if cfg.zero:
            comm.reduce_scatter_inplace(G, C)
            dev.launch(ks["sgd_shard"], C, [DevPtr(P + rank * C), DevPtr(G + rank * C),
                                            DevPtr(E + rank), cfg.shift, step + 2])
            if step == cfg.steps - 1:
                comm.wait(E, cfg.world, step + 2)  # the final parameters are everywhere
        else:
            comm.all_reduce(G, NP)
            dev.launch(ks["sgd"], NP, [DevPtr(P), DevPtr(G), cfg.shift], block=32)
        ticks = synchronize(dev, ep)
        hs = dev.history[h0:]
        per = {}
        for h in hs:
            k = h.name.split(".")[0] if h.name.startswith(("ar.", "rs.", "mc.")) else h.name
            per[k] = per.get(k, 0) + h.generated
        s = StepLog(step, dev.read(L, 1)[0], dev.read(P, NP), round(time.time() - t0, 2),
                    ticks, sum(h.generated for h in hs), sum(h.records for h in hs), per)
        steps.append(s)
        log(f"rank {rank} step {step}: loss {s.loss} ({s.ticks} ticks, {s.generated} tokens, "
            f"{s.seconds}s)")
    return {"init": init, "steps": steps}
