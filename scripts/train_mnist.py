"""MNIST on QwenGPUs: a linear classifier trained tensor-parallel (qgpu/dist/mnist.py), one
process per GPU. The weights every rank holds are compared with the exact numpy replay, and
the test accuracy is measured at every checkpoint from the weights read back from the ranks.

  python scripts/get_mnist.py                     # -> data/mnist28.npz
  torchrun --nproc-per-node 4 scripts/train_mnist.py build/merged-2b --mode verify --burst 8 --out runs/mnist
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from qgpu.dist import mnist


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
    ap.add_argument("--steps", type=int)
    ap.add_argument("--first-run", action="store_true", help="the first run's recipe (M1)")
    ap.add_argument("--data", default="data/mnist28.npz")
    ap.add_argument("--burst", type=int, default=1, help="records per exchange")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = mnist.Config(world=world, data=a.data, **(mnist.FIRST_RUN if a.first_run else {}))
    cfg.steps = a.steps or cfg.steps
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
    res = mnist.program(rank, dev, ep, cfg, log=log)
    mine = asdict(res) | {"checked": dev.checked, "forwards": getattr(backend, "forwards", None),
                          "positions": getattr(backend, "positions", None)}
    allr = [None] * world
    dist.all_gather_object(allr, mine)
    if rank == 0:
        X, y, Xte, yte = mnist.load(cfg)
        ref = mnist.reference(cfg, X, y)
        W = mnist.assemble([r["W"] for r in allr])
        curve = []
        for step in sorted(allr[0]["checkpoints"]):
            Wc = mnist.assemble([r["checkpoints"][step] for r in allr])
            curve.append({"step": int(step), "test_acc": mnist.accuracy(Wc, Xte, yte, world),
                          "exact": Wc.tolist() == ref["checkpoints"][int(step)]})
        summary = {"config": asdict(cfg), "mode": mode, "backend": a.backend, "world": world,
                   "exact_weights": W.tolist() == ref["W"],
                   "exact_checkpoints": sum(c["exact"] for c in curve),
                   "test_acc": mnist.accuracy(W, Xte, yte, world), "curve": curve,
                   "tokens_per_step": [r["generated"] / cfg.steps for r in allr],
                   "per_kernel_per_step": {k: v / cfg.steps for k, v in allr[0]["per_kernel"].items()},
                   "checked_positions": [r["checked"] for r in allr],
                   "forwards": [r["forwards"] for r in allr], "rebases": [r["rebases"] for r in allr],
                   "wall_seconds": max(r["seconds"] for r in allr), "W": W.tolist(),
                   "checkpoints": {c["step"]: mnist.assemble([r["checkpoints"][c["step"]] for r in allr]).tolist() for c in curve}}
        (out / "summary.json").write_text(json.dumps(summary))
        log(f"DONE: weights exact vs numpy {summary['exact_weights']} "
            f"({summary['exact_checkpoints']}/{len(curve)} checkpoints); test accuracy "
            f"{summary['test_acc']:.4f}; {summary['wall_seconds']}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
