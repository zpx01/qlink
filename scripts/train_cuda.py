"""The data-parallel training of qgpu/dist/train.py on real GPUs: the same CUDA kernels (the same
PTX files), the same qccl schedule, real NVLink P2P stores between the GPUs of one node. Every
step's parameters are compared with the exact numpy replay, and with a QwenGPU run's summary
(scripts/train_dp.py --out DIR) if given.

  python scripts/train_cuda.py --world 4 --compare runs/t1-verify
"""

import argparse
import json
import time
from pathlib import Path

import torch

from qgpu import ptx
from qgpu.dist import train
from qgpu.dist.qccl import Comm
from qgpu.dist.realgpu import CudaGPU
from qgpu.lower import DevPtr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--compare")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = train.Config(world=a.world, steps=a.steps)
    N, B, NP, HID = cfg.world, cfg.batch, train.NP, train.HID
    ks = ptx.load(train.ROOT / "kernels" / "train" / "mlp.ptx")
    CudaGPU.enable_peer_access(range(N))
    devs = [CudaGPU(r, r) for r in range(N)]
    for dev in devs:
        dev.preload()
    comms = [Comm(devs[r], r, N, chunk=cfg.chunk) for r in range(N)]
    lay = []
    for dev in devs:  # the same allocation order as train.program
        P, G = dev.malloc(NP), dev.malloc(NP)
        X, T, O, L = dev.malloc(2 * B), dev.malloc(B), dev.malloc(B), dev.malloc(1)
        lay.append((P, G, X, T, O, L))
    xtr, ytr = train.dataset(cfg.n_train, cfg.data_seed)
    plan = train.batches(cfg)
    ref = train.reference(cfg)
    for r, dev in enumerate(devs):
        P, G = lay[r][:2]
        dev.memcpy_h2d(P, train.init_params(cfg.init_seed).tolist() if r == 0 else [0] * NP)
        dev.memcpy_h2d(G, [0] * NP)
        comms[r].broadcast(P, NP, root=0)
    for dev in devs:
        dev.synchronize()
    t0 = time.time()
    params, exact, same = [], 0, 0
    for step, idx in enumerate(plan):
        # the data loader's copies first: pinning host memory synchronizes, and a kernel already
        # spinning on a peer's flag would wait for a launch stuck behind it
        for r, dev in enumerate(devs):
            P, G, X, T, O, L = lay[r]
            dev.memcpy_h2d(X, xtr[idx[r]].ravel().tolist())
            dev.memcpy_h2d(T, ytr[idx[r]].tolist())
            dev.memcpy_h2d(O, [0] * B)
            dev.memcpy_h2d(L, [0])
        for r, dev in enumerate(devs):  # every rank's whole step is enqueued; no host waits
            P, G, X, T, O, L = lay[r]
            dev.launch(ks["mlp_forward"], HID * B, [DevPtr(P), DevPtr(X), DevPtr(O)], block=32)
            dev.launch(ks["mlp_backward"], HID * B,
                       [DevPtr(P), DevPtr(X), DevPtr(T), DevPtr(O), DevPtr(G), DevPtr(L)],
                       block=32)
            comms[r].all_reduce(G, NP)
            dev.launch(ks["sgd"], NP, [DevPtr(P), DevPtr(G), cfg.shift], block=32)
        ps = [dev.read(lay[r][0], NP) for r, dev in enumerate(devs)]
        same += all(p == ps[0] for p in ps)
        exact += ps[0] == ref["params"][step]
        params.append(ps[0])
    dt = time.time() - t0
    xte, yte = train.dataset(256, cfg.test_seed)
    out = {"world": N, "steps": cfg.steps, "replicas_identical_steps": same,
           "exact_vs_numpy_steps": exact, "seconds": round(dt, 3),
           "launches": sum(d.launches for d in devs),
           "test_acc": train.accuracy(params[-1], xte, yte), "final_params": params[-1]}
    if a.compare:
        s = json.loads((Path(a.compare) / "summary.json").read_text())
        out["same_final_params_as_qwengpu_run"] = s["final_params"] == params[-1]
    print(json.dumps({k: v for k, v in out.items() if k != "final_params"}))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
