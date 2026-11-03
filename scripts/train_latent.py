"""Latent image diffusion trained and sampled on QwenGPUs (qgpu/dist/latent.py), one process per
GPU: training, checkpoints and every generated latent are compared with the exact numpy replay,
and the samples are decoded to 28 x 28 images.

  python scripts/get_mnist.py
  torchrun --nproc-per-node 4 scripts/train_latent.py build/merged-2b --mode verify --burst 8 --out runs/latent
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qgpu.dist import latent as lt


def main():
    import torch
    import torch.distributed as dist

    from qgpu.dist.backends import ModelBackend, RefBackend
    from qgpu.dist.device import QGPU
    from qgpu.dist.transport import TorchTransport

    ap = argparse.ArgumentParser()
    ap.add_argument("merged", nargs="?", default="build/merged-2b")
    ap.add_argument("--backend", choices=("model", "ref"), default="model")
    ap.add_argument("--mode", choices=("greedy", "draft", "verify"), default="verify")
    ap.add_argument("--steps", type=int, default=lt.Config.steps)
    ap.add_argument("--samples", type=int, default=lt.Config.samples)
    ap.add_argument("--data", default="data/mnist28.npz")
    ap.add_argument("--burst", type=int, default=1, help="records per exchange")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = lt.Config(world=world, steps=a.steps, samples=a.samples, data=a.data)
    backend = RefBackend() if a.backend == "ref" else ModelBackend(a.merged, device=device)
    mode = "greedy" if a.backend == "ref" else a.mode
    dev = QGPU(rank, backend, mode=mode, keep_log=False)
    dev.burst = a.burst
    ep = TorchTransport(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"rank{rank}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    dist.barrier()
    res = lt.program(rank, dev, ep, cfg, log=log)
    mine = asdict(res) | {"checked": dev.checked, "forwards": getattr(backend, "forwards", None)}
    allr = [None] * world
    dist.all_gather_object(allr, mine)
    if rank == 0:
        ref = lt.reference(cfg)
        P = lt.assemble([r["params"] for r in allr])
        ck = {s: lt.assemble([r["checkpoints"][s] for r in allr]) for s in allr[0]["checkpoints"]}
        traj = allr[0]["trajectory"]
        images = lt.decode(np.asarray(traj[-1]), cfg)
        np.save(out / "images.npy", (images * 255).round().astype(np.uint8))
        summary = {"config": asdict(cfg), "mode": mode, "backend": a.backend, "world": world,
                   "exact_params": P == ref["P"],
                   "exact_checkpoints": sum(ck[s] == ref["checkpoints"][s] for s in ck),
                   "exact_samples": all(r["trajectory"] == ref["samples"] for r in allr),
                   "tokens_per_rank": [r["generated"] for r in allr],
                   "per_kernel": allr[0]["per_kernel"], "checked_positions": [r["checked"] for r in allr],
                   "forwards": [r["forwards"] for r in allr], "rebases": [r["rebases"] for r in allr],
                   "wall_seconds": max(r["seconds"] for r in allr), "params": P, "checkpoints": ck,
                   "losses": ref["losses"], "trajectory": traj}
        (out / "summary.json").write_text(json.dumps(summary))
        log(f"DONE: params exact {summary['exact_params']} ({summary['exact_checkpoints']}/"
            f"{len(ck)} checkpoints); generated latents exact {summary['exact_samples']}; "
            f"{summary['wall_seconds']}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
