"""A diffusion model on QwenGPUs: four of them learn to draw a 7.

The data are points in the plane sampled from the glyph "7" (DejaVu Sans Bold). The model is a
DDPM (20 noise levels, epsilon prediction) whose denoiser eps_hat(x_t, t) is an MLP 3 -> 32 -> 2,
tensor-parallel over 4 QwenGPUs as Megatron's MLP: rank r holds hidden units [8 r, 8 r + 8),
i.e. its columns of W1 and b1 and its rows of W2 (50 words with the replicated output bias), and
one qccl AllReduce sums the ranks' partial outputs. Q8 fixed point throughout
(kernels/diffusion/denoiser.cu).

One training step (a batch of 4) on every rank:

    H2D       the data loader's batch: (x_t, t) for 4 samples; rank 0's output starts at -eps
    forward   dn_forward    z = W1 in + b1 (kept), out += W2 relu(z) (+ b2 on rank 0)
    allreduce qccl AllReduce(out): eps_hat - eps, the loss gradient, on every rank
    backward  dn_backward1  W1 -= in dz, b1 -= dz       (dz = relu'(z) W2 e; atomics)
              dn_backward2  W2 -= relu(z) e
              dn_bias       b2 -= e

Generation runs on the QwenGPUs too: DDIM (deterministic) from Gaussian noise, 20 steps of
dn_eps + AllReduce + dn_ddim for 8 points at a time. Everything is int32, so `reference` replays
training and sampling exactly in numpy.
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
T, HR, B, BS = 20, 8, 4, 8  # noise levels, hidden units per rank, batch, points per sampling pass


@dataclass
class Config:
    world: int = 4
    steps: int = 6000
    schedule: tuple = ((0, 6), (3000, 7), (5000, 8))  # (from step, k): learning rate 2^-k per sample
    glyph: str = "7"
    seed: int = 1
    samples: int = 256  # points generated at the end
    checkpoint_every: int = 500
    rebase_tokens: int = 200_000


BETAS = np.linspace(1e-3, 0.25, T)
ABAR = np.cumprod(1 - BETAS)


def q8(v):
    return np.round(np.asarray(v, np.float64) * 256).astype(np.int64)


def rs(x, s):
    """x / 2^s rounded to nearest, as the kernels' RS()."""
    return (x + (1 << (s - 1))) >> s


def glyph_points(ch: str, n: int = 4000, seed: int = 0) -> np.ndarray:
    """Points uniformly on the glyph, centred, scaled into [-1, 1]."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
    im = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(im).text((64, 64), ch, fill=255, font=font, anchor="mm")
    ys, xs = np.nonzero(np.array(im) > 128)
    rng = np.random.default_rng(seed)
    i = rng.integers(len(xs), size=n)
    p = np.c_[xs[i] + rng.uniform(0, 1, n), 128 - ys[i] - rng.uniform(0, 1, n)]
    p -= p.mean(0)
    return p / np.abs(p).max()


def init_params(cfg: Config) -> dict:
    H = cfg.world * HR
    rng = np.random.default_rng(cfg.seed)
    return {"W1": q8(rng.normal(0, 1.5, (3, H))), "b1": q8(rng.normal(0, 0.5, H)),
            "W2": q8(rng.normal(0, 1 / np.sqrt(H), (H, 2))), "b2": np.zeros(2, np.int64)}


def shard(P: dict, r: int) -> dict:
    sl = slice(r * HR, (r + 1) * HR)
    return {"W1": P["W1"][:, sl], "b1": P["b1"][sl], "W2": P["W2"][sl], "b2": P["b2"]}


def batches(cfg: Config):
    """The data loader: (in Q8: x_t, y_t, t feature; eps Q8) per step, the same on every rank."""
    data = glyph_points(cfg.glyph)
    rng = np.random.default_rng(cfg.seed + 100)
    for _ in range(cfg.steps):
        x0 = data[rng.integers(len(data), size=B)]
        t = rng.integers(T, size=B)
        eps = rng.normal(size=(B, 2))
        xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * eps
        yield np.c_[q8(xt), q8(t / T * 2 - 1)], q8(eps)


def shift_at(cfg: Config, step: int) -> int:
    return [k for t, k in cfg.schedule if step >= t][-1]


def ddim(t: int) -> tuple[int, int, int]:
    """x_{t-1} = (A x_t + B eps_hat) / 4096 (DDIM, eta = 0), and the next time feature."""
    ap = ABAR[t - 1] if t else 1.0
    a = np.sqrt(ap / ABAR[t])
    b = np.sqrt(1 - ap) - np.sqrt(ap * (1 - ABAR[t]) / ABAR[t])
    return int(round(a * 4096)), int(round(b * 4096)), int(q8((t - 1) / T * 2 - 1)) if t else 0


# --- the exact numpy replay -------------------------------------------------------------------

def _eps(P, inp, world):
    z = ((inp @ P["W1"]) >> 8) + P["b1"]
    a = np.maximum(z, 0)
    part = sum(((a[:, r * HR:(r + 1) * HR, None] * P["W2"][None, r * HR:(r + 1) * HR, :]) >> 8).sum(1)
               for r in range(world))
    return z, part + P["b2"]


def reference(cfg: Config) -> dict:
    P = init_params(cfg)
    checkpoints, losses = {}, []
    for i, (inp, eps) in enumerate(batches(cfg)):
        k = shift_at(cfg, i)
        z, out = _eps(P, inp, cfg.world)
        e = out - eps
        losses.append(int((e * e).sum()))
        a = np.maximum(z, 0)
        dz = np.where(z > 0, (e @ P["W2"].T) >> 8, 0)
        P["W1"] -= rs(dz[:, None, :] * inp[:, :, None], 8 + k).sum(0)
        P["b1"] -= rs(dz, k).sum(0)
        P["W2"] -= rs(((a[:, :, None] * e[:, None, :]) >> 8).sum(0), k)
        P["b2"] -= rs(e.sum(0), k)
        if (i + 1) % cfg.checkpoint_every == 0:
            checkpoints[i + 1] = {n: v.tolist() for n, v in P.items()}
    return {"P": {n: v.tolist() for n, v in P.items()}, "checkpoints": checkpoints,
            "losses": losses, "samples": sample_reference(P, cfg)}


def noise(cfg: Config) -> np.ndarray:
    return q8(np.random.default_rng(cfg.seed + 7).normal(size=(cfg.samples, 2)))


def sample_reference(P, cfg: Config) -> list:
    """DDIM trajectories of every point: [step][point][x, y] (Q8)."""
    x = noise(cfg)
    traj = [x.tolist()]
    tf = int(q8((T - 1) / T * 2 - 1))
    for t in reversed(range(T)):
        _, eh = _eps(P, np.c_[x, np.full(len(x), tf)], cfg.world)
        A, Bc, tf = ddim(t)
        x = rs(A * x + Bc * eh, 12)
        traj.append(x.tolist())
    return traj


# --- on the QwenGPUs ----------------------------------------------------------------------------

@dataclass
class Result:
    rank: int
    params: dict  # this rank's shard at the end
    checkpoints: dict
    trajectory: list  # [step][point][x, y], read back from the device
    seconds: float
    generated: int
    per_kernel: dict = field(default_factory=dict)
    rebases: int = 0


def program(rank: int, dev, ep, cfg: Config, log=print) -> Result:
    ks = ptx.load(ROOT / "kernels" / "diffusion" / "denoiser.ptx")
    comm = Comm(dev, rank, cfg.world, chunk=4)
    sh = shard(init_params(cfg), rank)
    W1, b1 = dev.malloc(3 * HR, "W1"), dev.malloc(HR, "b1")
    W2, b2 = dev.malloc(2 * HR, "W2"), dev.malloc(2, "b2")
    for a, v in ((W1, sh["W1"]), (b1, sh["b1"]), (W2, sh["W2"]), (b2, sh["b2"])):
        dev.memcpy_h2d(a, np.asarray(v).ravel().tolist())
    mark = dev.heap
    IN, Z, OUT = dev.malloc(3 * B, "in"), dev.malloc(HR * B, "z"), dev.malloc(16, "out")

    def read_params():
        return {"W1": np.asarray(dev.read(W1, 3 * HR)).reshape(3, HR).tolist(),
                "b1": dev.read(b1, HR), "W2": np.asarray(dev.read(W2, 2 * HR)).reshape(HR, 2).tolist(),
                "b2": dev.read(b2, 2)}

    t0, h0 = time.time(), len(dev.history)
    checkpoints = {}
    bias = int(rank == 0)
    for i, (inp, eps) in enumerate(batches(cfg)):
        k = shift_at(cfg, i)
        dev.memcpy_h2d(IN, inp.ravel().tolist())
        dev.memcpy_h2d(OUT, (-eps).ravel().tolist() + [0] * 8 if rank == 0 else [0] * 16)
        dev.launch(ks["dn_forward"], HR * B, [DevPtr(W1), DevPtr(b1), DevPtr(W2), DevPtr(b2),
                                              DevPtr(IN), DevPtr(Z), DevPtr(OUT), bias])
        comm.all_reduce(OUT, 16)
        dev.launch(ks["dn_backward1"], HR * B, [DevPtr(W1), DevPtr(b1), DevPtr(W2), DevPtr(IN),
                                                DevPtr(Z), DevPtr(OUT), k])
        dev.launch(ks["dn_backward2"], 2 * HR, [DevPtr(W2), DevPtr(Z), DevPtr(OUT), k])
        dev.launch(ks["dn_bias"], 4, [DevPtr(b2), DevPtr(OUT), k])
        synchronize(dev, ep)
        if (i + 1) % cfg.checkpoint_every == 0:
            checkpoints[i + 1] = read_params()
            log(f"rank {rank} step {i + 1}: {time.time() - t0:.0f}s")
        if dev.fed > cfg.rebase_tokens:
            dev.rebase()
    params = read_params()

    # generation: DDIM from noise, BS points at a time, every step read back for the animation
    dev.heap = mark  # the training buffers are free again
    IN, OUT = dev.malloc(3 * BS, "points"), dev.malloc(16, "eps")
    x0 = noise(cfg)
    traj = [[[0, 0] for _ in range(cfg.samples)] for _ in range(T + 1)]
    tf0 = int(q8((T - 1) / T * 2 - 1))
    for p0 in range(0, cfg.samples, BS):
        pts = x0[p0:p0 + BS]
        dev.memcpy_h2d(IN, np.c_[pts, np.full(BS, tf0)].ravel().tolist())
        for j, t in enumerate(reversed(range(T))):
            dev.memcpy_h2d(OUT, [0] * 16)
            dev.launch(ks["dn_eps"], HR * BS, [DevPtr(W1), DevPtr(b1), DevPtr(W2), DevPtr(b2),
                                               DevPtr(IN), DevPtr(OUT), bias], block=32)
            comm.all_reduce(OUT, 16)
            A, Bc, tf = ddim(t)
            dev.launch(ks["dn_ddim"], BS, [DevPtr(IN), DevPtr(OUT), A, Bc, tf])
            synchronize(dev, ep)
            cur = np.asarray(dev.read(IN, 3 * BS)).reshape(BS, 3)[:, :2]
            for q in range(BS):
                traj[j + 1][p0 + q] = cur[q].tolist()
            if dev.fed > cfg.rebase_tokens:
                dev.rebase()
        for q in range(BS):
            traj[0][p0 + q] = pts[q].tolist()
        log(f"rank {rank}: generated points {p0 + BS}/{cfg.samples}")
    hs = dev.history[h0:]
    per = {}
    for h in hs:
        name = h.name.split(".")[0] if h.name.startswith("ar.") else h.name
        per[name] = per.get(name, 0) + h.generated
    return Result(rank, params, checkpoints, traj, round(time.time() - t0, 1),
                  sum(h.generated for h in hs), per, dev.rebases)


def assemble(shards: list) -> dict:
    return {"W1": np.concatenate([np.asarray(s["W1"]) for s in shards], 1).tolist(),
            "b1": np.concatenate([np.asarray(s["b1"]) for s in shards]).tolist(),
            "W2": np.concatenate([np.asarray(s["W2"]) for s in shards], 0).tolist(),
            "b2": shards[0]["b2"]}
