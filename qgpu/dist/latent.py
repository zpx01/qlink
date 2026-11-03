"""Latent image diffusion on QwenGPUs: four of them learn to generate handwritten digits.

A 28 x 28 MNIST image does not fit in 127 words, so the diffusion runs in a latent space, as
Stable Diffusion does: a fixed linear autoencoder (the mean image and the top 8 principal
components of the training set, whitened) maps an image to 8 numbers and back. It is computed
once from the data, not learned on the QwenGPUs. The QwenGPUs learn the diffusion model: a DDPM
(20 noise levels, epsilon prediction) whose denoiser, an MLP 9 -> 16 -> 8 (the latent and the
time), is split over 4 QwenGPUs as Megatron splits an MLP: rank r holds hidden units
[4 r, 4 r + 4) (its columns of W1 and b1, its rows of W2; 80 words with the replicated output
bias), and a qccl AllReduce sums the partial outputs. Q8 fixed point (kernels/diffusion/latent.cu).

One training step (one image) on every rank:

    H2D        the data loader's (z_t, t); rank 0's output starts at -eps
    forward    lt_hidden  z = W1 in + b1 (kept)       lt_out  out += W2 relu(z) (+ b2 on rank 0)
    allreduce  qccl AllReduce(out): eps_hat - eps on every rank
    backward   lt_dz      dz = relu'(z) (W2 . e)       lt_w1  W1 -= in dz, b1 -= dz
               lt_w2      W2 -= relu(z) e, b2 -= e  (each bias is its matrix's extra row)

Generation: DDIM from Gaussian noise, 20 steps of lt_hidden + lt_out + AllReduce + lt_ddim, on
the QwenGPUs; the decoder turns the 8 numbers into an image. Everything the QwenGPUs compute is
int32, so `reference` replays it exactly in numpy.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qgpu import ptx
from qgpu.dist.diffusion import ABAR, T, ddim, q8, rs
from qgpu.dist.qccl import Comm
from qgpu.dist.transport import synchronize
from qgpu.lower import DevPtr

ROOT = Path(__file__).resolve().parent.parent.parent
D, HR = 8, 4  # latent dimensions, hidden units per rank
NIN = D + 1  # the latent and the time feature


@dataclass
class Config:
    world: int = 4
    steps: int = 6000
    schedule: tuple = ((0, 5), (3000, 6), (5000, 7))  # (from step, k): learning rate 2^-k
    seed: int = 1
    data: str = "data/mnist28.npz"
    samples: int = 64  # images generated at the end
    checkpoint_every: int = 1000
    rebase_tokens: int = 200_000


def autoencoder(cfg: Config):
    """The fixed linear autoencoder: mean (784), basis (D x 784), latent scale (D)."""
    d = np.load(cfg.data)
    X = d["Xtr"].reshape(-1, 784).astype(np.float64) / 255
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X[:20000] - mu, full_matrices=False)
    V = Vt[:D]
    V *= np.sign(V[np.arange(D), np.abs(V).argmax(1)])[:, None]  # signs fixed: any LAPACK agrees
    sd = ((X - mu) @ V.T).std(0)
    return mu, V, sd


def latents(cfg: Config) -> np.ndarray:
    mu, V, sd = autoencoder(cfg)
    X = np.load(cfg.data)["Xtr"].reshape(-1, 784).astype(np.float64) / 255
    return ((X - mu) @ V.T) / sd


def decode(z_q8, cfg: Config) -> np.ndarray:
    mu, V, sd = autoencoder(cfg)
    return np.clip((np.asarray(z_q8, np.float64) / 256 * sd) @ V + mu, 0, 1).reshape(-1, 28, 28)


def init_params(cfg: Config) -> dict:
    H = cfg.world * HR
    rng = np.random.default_rng(cfg.seed)
    return {"W1": q8(rng.normal(0, 1.5 / np.sqrt(NIN), (NIN, H))), "b1": q8(rng.normal(0, 0.5, H)),
            "W2": q8(rng.normal(0, 1 / np.sqrt(H), (H, D))), "b2": np.zeros(D, np.int64)}


def shard(P: dict, r: int) -> dict:
    sl = slice(r * HR, (r + 1) * HR)
    return {"W1": P["W1"][:, sl], "b1": P["b1"][sl], "W2": P["W2"][sl], "b2": P["b2"]}


def batches(cfg: Config):
    Z = latents(cfg)
    rng = np.random.default_rng(cfg.seed + 100)
    for _ in range(cfg.steps):
        z0 = Z[rng.integers(len(Z))]
        t = int(rng.integers(T))
        eps = rng.normal(size=D)
        zt = np.sqrt(ABAR[t]) * z0 + np.sqrt(1 - ABAR[t]) * eps
        yield np.r_[q8(zt), q8(t / T * 2 - 1)], q8(eps)


def shift_at(cfg: Config, step: int) -> int:
    return [k for t, k in cfg.schedule if step >= t][-1]


def _eps(P, inp, world):
    z = ((inp @ P["W1"]) >> 8) + P["b1"]
    a = np.maximum(z, 0)
    part = sum(((a[..., r * HR:(r + 1) * HR, None] * P["W2"][r * HR:(r + 1) * HR, :]) >> 8).sum(-2)
               for r in range(world))
    return z, part + P["b2"]


def noise(cfg: Config) -> np.ndarray:
    return q8(np.random.default_rng(cfg.seed + 7).normal(size=(cfg.samples, D)))


def sample_reference(P, cfg: Config, x=None) -> list:
    x = noise(cfg) if x is None else x
    traj = [x.tolist()]
    tf = int(q8((T - 1) / T * 2 - 1))
    for t in reversed(range(T)):
        _, eh = _eps(P, np.c_[x, np.full(len(x), tf)], cfg.world)
        A, Bc, tf = ddim(t)
        x = rs(A * x + Bc * eh, 12)
        traj.append(x.tolist())
    return traj


def reference(cfg: Config) -> dict:
    P = init_params(cfg)
    checkpoints, losses = {}, []
    for i, (inp, eps) in enumerate(batches(cfg)):
        k = shift_at(cfg, i)
        z, out = _eps(P, inp, cfg.world)
        e = out - eps
        losses.append(int((e * e).sum()))
        a = np.maximum(z, 0)
        dz = np.where(z > 0, (P["W2"] @ e) >> 8, 0)
        P["W1"] -= rs(inp[:, None] * dz[None, :], 8 + k)
        P["b1"] -= rs(dz, k)
        P["W2"] -= rs((a[:, None] * e[None, :]) >> 8, k)
        P["b2"] -= rs(e, k)
        if (i + 1) % cfg.checkpoint_every == 0:
            checkpoints[i + 1] = {n: v.tolist() for n, v in P.items()}
    return {"P": {n: v.tolist() for n, v in P.items()}, "checkpoints": checkpoints,
            "losses": losses, "samples": sample_reference(P, cfg)}


@dataclass
class Result:
    rank: int
    params: dict
    checkpoints: dict
    trajectory: list  # [step][sample][D] (Q8), read back from the device
    seconds: float
    generated: int
    per_kernel: dict = field(default_factory=dict)
    rebases: int = 0


def program(rank: int, dev, ep, cfg: Config, log=print) -> Result:
    ks = ptx.load(ROOT / "kernels" / "diffusion" / "latent.ptx")
    comm = Comm(dev, rank, cfg.world, chunk=4)
    sh = shard(init_params(cfg), rank)
    W1, b1 = dev.malloc(NIN * HR, "W1"), dev.malloc(HR, "b1")  # b1 right after W1: its row 9
    W2, b2 = dev.malloc(HR * D, "W2"), dev.malloc(D, "b2")  # b2 right after W2: its row 4
    for a, v in ((W1, sh["W1"]), (b1, sh["b1"]), (W2, sh["W2"]), (b2, sh["b2"])):
        dev.memcpy_h2d(a, np.asarray(v).ravel().tolist())
    IN, Z, OUT = dev.malloc(NIN, "in"), dev.malloc(HR, "z"), dev.malloc(16, "out")
    bias = int(rank == 0)
    P = [DevPtr(W1), DevPtr(b1), DevPtr(W2), DevPtr(b2)]

    def read_params():
        return {"W1": np.asarray(dev.read(W1, NIN * HR)).reshape(NIN, HR).tolist(),
                "b1": dev.read(b1, HR), "W2": np.asarray(dev.read(W2, HR * D)).reshape(HR, D).tolist(),
                "b2": dev.read(b2, D)}

    def eps_hat():  # out (+)= W2 relu(W1 in + b1) (+ b2 on rank 0), summed over the ranks
        dev.launch(ks["lt_hidden"], 4, P[:2] + [DevPtr(IN), DevPtr(Z)])
        dev.launch(ks["lt_out"], 8, [DevPtr(W2), DevPtr(b2), DevPtr(Z), DevPtr(OUT), bias])
        comm.all_reduce(OUT, 16)

    t0, h0 = time.time(), len(dev.history)
    checkpoints = {}
    for i, (inp, eps) in enumerate(batches(cfg)):
        k = shift_at(cfg, i)
        dev.memcpy_h2d(IN, inp.tolist())
        dev.memcpy_h2d(OUT, (-eps).tolist() + [0] * 8 if rank == 0 else [0] * 16)
        eps_hat()
        dev.launch(ks["lt_dz"], HR, [DevPtr(W2), DevPtr(Z), DevPtr(OUT)])
        dev.launch(ks["lt_w1"], (NIN + 1) * HR, [DevPtr(W1), DevPtr(IN), DevPtr(OUT), k], block=8)
        dev.launch(ks["lt_w2"], (HR + 1) * D, [DevPtr(W2), DevPtr(Z), DevPtr(OUT), k], block=8)
        synchronize(dev, ep)
        if (i + 1) % cfg.checkpoint_every == 0:
            checkpoints[i + 1] = read_params()
            log(f"rank {rank} step {i + 1}: {time.time() - t0:.0f}s")
        if dev.fed > cfg.rebase_tokens:
            dev.rebase()
    params = read_params()
    # generation: DDIM from noise, one image at a time, every step read back
    z0 = noise(cfg)
    traj = [[None] * cfg.samples for _ in range(T + 1)]
    tf0 = int(q8((T - 1) / T * 2 - 1))
    for s in range(cfg.samples):
        dev.memcpy_h2d(IN, z0[s].tolist() + [tf0])
        traj[0][s] = z0[s].tolist()
        for j, t in enumerate(reversed(range(T))):
            dev.memcpy_h2d(OUT, [0] * 16)
            eps_hat()
            A, Bc, tf = ddim(t)
            dev.launch(ks["lt_ddim"], 8, [DevPtr(IN), DevPtr(OUT), A, Bc, tf])
            synchronize(dev, ep)
            traj[j + 1][s] = dev.read(IN, D)
            if dev.fed > cfg.rebase_tokens:
                dev.rebase()
        if (s + 1) % 8 == 0:
            log(f"rank {rank}: generated {s + 1}/{cfg.samples} images")
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
