"""MNIST on QwenGPUs: a linear classifier trained tensor-parallel (Megatron's RowParallelLinear).

Each image becomes 32 Q8 features (1.0 = 256) on the host, the data loader's fixed encoder: its
top 31 principal components, whitened, and a bias (features="pca"; or, as in the first run,
features="pool": the central 20 x 20 pixels average-pooled to 5 x 5, a bias and padding). Rank r
of 4 holds features [8 r, 8 r + 8) of every image and the 10 x 8 slice of the weights for them
(Q16). A QwenGPU has 127 words, so no single one could hold the 320-word layer; four hold it as
80 words each.

One step (one image) on every rank, all on the QwenGPUs (kernels/mnist/tp_linear.cu):

    H2D        the rank's 8 features of the image, and its label (each rank's data loader)
    forward    tp_forward   z_r[c] = (sum_j W_r[c][j] x_r[j]) >> 8       partial logits
    allreduce  qccl AllReduce(z): the full logits z = sum_r z_r on every rank
    error      tp_hinge     e = z - s where s z < 1 (s = +-1), else 0    (one-vs-rest squared hinge)
               tp_error     e = z - onehot(y)                           (or least squares)
    update     tp_update    W_r[c][j] -= (e[c] x_r[j]) >> (8 + k)

The update only needs the full logits and the rank's own features, so the forward AllReduce is
the layer's only communication. The learning rate 2^-k decays on a fixed schedule. Everything is
int32, so `reference` replays the training exactly in numpy.
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
F, NCLS, NF = 8, 10, 32  # features per rank, classes, features (25 pixels, bias, padding)


@dataclass
class Config:
    world: int = 4
    steps: int = 6000
    schedule: tuple = ((0, 6), (2000, 7), (4000, 8))  # (from step, k): lr = 2^-k
    features: str = "pca"  # or "pool"
    loss: str = "hinge"  # or "lsq"
    seed: int = 0
    data: str = "data/mnist28.npz"
    checkpoint_every: int = 100
    rebase_tokens: int = 200_000  # start a fresh context when one grows past this


# the first run (M1): 5 x 5 pixels, least squares, 3,000 images
FIRST_RUN = {"steps": 3000, "schedule": ((0, 5), (800, 6), (1600, 7), (2400, 8)),
             "features": "pool", "loss": "lsq"}


def pca(train: np.ndarray, k: int = NF - 1):
    """The encoder: mean, top-k principal directions (signs fixed: largest entry positive) and
    their scales, from the first 20,000 training images."""
    X = train.reshape(len(train), -1).astype(np.float64) / 255
    mu = X.mean(0)
    V = np.linalg.svd(X[:20000] - mu, full_matrices=False)[2][:k]
    V *= np.sign(V[np.arange(k), np.abs(V).argmax(1)])[:, None]
    return mu, V, ((X - mu) @ V.T).std(0)


def features(images: np.ndarray, enc=None) -> np.ndarray:
    """28 x 28 uint8 images -> 32 Q8 features. With an encoder (mu, V, sd): the whitened
    principal components, clipped to +-4, and a bias (1.0). Without: the central 20 x 20
    average-pooled to 5 x 5, a bias and 6 zeros."""
    if enc is not None:
        mu, V, sd = enc
        Z = ((images.reshape(len(images), -1) / 255 - mu) @ V.T) / sd
        return np.c_[np.round(np.clip(Z, -4, 4) * 256).astype(np.int64), np.full(len(images), 256)]
    p = images[:, 4:24, 4:24].astype(np.float64).reshape(-1, 5, 4, 5, 4).mean((2, 4))
    p = np.round(p.reshape(len(images), 25) / 255 * 256).astype(np.int64)
    return np.c_[p, np.full(len(images), 256), np.zeros((len(images), NF - 26), np.int64)]


def load(cfg: Config):
    d = np.load(cfg.data)
    order = np.random.default_rng(cfg.seed).permutation(len(d["ytr"]))[:cfg.steps]
    enc = pca(d["Xtr"]) if cfg.features == "pca" else None
    return (features(d["Xtr"][order], enc), d["ytr"][order].astype(np.int64),
            features(d["Xte"], enc), d["yte"].astype(np.int64))


def error(cfg: Config, z: np.ndarray, label: int) -> np.ndarray:
    """What tp_hinge / tp_error leave in z (Q16)."""
    s = np.where(np.arange(NCLS) == label, 65536, -65536)
    if cfg.loss == "hinge":
        return np.where(s * z < 65536 * 65536, z - s, 0)
    return z - np.where(s > 0, 65536, 0)


def shift_at(cfg: Config, step: int) -> int:
    return 8 + [k for t, k in cfg.schedule if step >= t][-1]


def logits(W: np.ndarray, X: np.ndarray, world: int) -> np.ndarray:
    """The ranks' partial logits, each >> 8, summed: what the kernels and the AllReduce do."""
    return sum((W[None, :, r * F:(r + 1) * F] * X[:, None, r * F:(r + 1) * F]).sum(2) >> 8
               for r in range(world))


def accuracy(W, X, y, world: int) -> float:
    return float((logits(np.asarray(W, np.int64), X, world).argmax(1) == y).mean())


def reference(cfg: Config, X, y) -> dict:
    W = np.zeros((NCLS, NF), np.int64)
    hits, checkpoints = [], {}
    for i in range(cfg.steps):
        z = logits(W, X[i:i + 1], cfg.world)[0]
        hits.append(int(z.argmax() == y[i]))
        W -= (error(cfg, z, y[i])[:, None] * X[i][None, :]) >> shift_at(cfg, i)
        if (i + 1) % cfg.checkpoint_every == 0:
            checkpoints[i + 1] = W.tolist()
    return {"W": W.tolist(), "hits": hits, "checkpoints": checkpoints}


@dataclass
class Result:
    rank: int
    W: list  # this rank's 10 x 8 slice at the end
    checkpoints: dict  # step -> slice
    hits: list  # was the logits' argmax (before the update) the label, per step
    seconds: float
    generated: int  # tokens the QwenGPU wrote
    per_kernel: dict = field(default_factory=dict)
    rebases: int = 0


def program(rank: int, dev, ep, cfg: Config, log=print) -> Result:
    ks = ptx.load(ROOT / "kernels" / "mnist" / "tp_linear.ptx")
    X, y, _, _ = load(cfg)
    comm = Comm(dev, rank, cfg.world, chunk=4)
    W = dev.malloc(NCLS * F, "W")
    x, z, lab = dev.malloc(F, "x"), dev.malloc(16, "z"), dev.malloc(1, "label")
    dev.memcpy_h2d(W, [0] * (NCLS * F))
    synchronize(dev, ep)
    hits, checkpoints, per = [], {}, {}
    t0, h0 = time.time(), len(dev.history)
    for i in range(cfg.steps):
        dev.memcpy_h2d(x, X[i, rank * F:(rank + 1) * F].tolist())
        dev.memcpy_h2d(lab, [int(y[i])])
        dev.launch(ks["tp_forward"], 16, [DevPtr(W), DevPtr(x), DevPtr(z)])
        comm.all_reduce(z, 16)
        dev.launch(ks["tp_hinge" if cfg.loss == "hinge" else "tp_error"], 16, [DevPtr(z), DevPtr(lab)])
        for c0 in (0, 5):
            dev.launch(ks["tp_update"], F, [DevPtr(W), DevPtr(x), DevPtr(z), c0, shift_at(cfg, i)])
        synchronize(dev, ep)
        if (i + 1) % cfg.checkpoint_every == 0:  # reads are rare: each one checks the model first
            Wr = dev.read(W, NCLS * F)
            checkpoints[i + 1] = [Wr[c * F:(c + 1) * F] for c in range(NCLS)]
            log(f"rank {rank} step {i + 1}: {time.time() - t0:.0f}s")
        if dev.fed > cfg.rebase_tokens:
            dev.rebase()
    hs = dev.history[h0:]
    for h in hs:
        name = h.name.split(".")[0] if h.name.startswith("ar.") else h.name
        per[name] = per.get(name, 0) + h.generated
    Wr = dev.read(W, NCLS * F)
    return Result(rank, [Wr[c * F:(c + 1) * F] for c in range(NCLS)], checkpoints, hits,
                  round(time.time() - t0, 1), sum(h.generated for h in hs), per, dev.rebases)


def assemble(slices: list) -> np.ndarray:
    """Every rank's 10 x 8 slice -> the 10 x 32 weights."""
    return np.concatenate([np.asarray(s, np.int64) for s in slices], axis=1)
