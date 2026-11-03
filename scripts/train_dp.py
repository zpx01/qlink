"""Data-parallel training of an MLP on several QwenGPUs (qgpu/dist/train.py), one process per
physical GPU. Every rank's parameters are compared with the exact numpy replay after every step,
and with each other; rank 0 writes a JSON summary (loss, test accuracy, tokens, timing).

  torchrun --nproc-per-node 4 scripts/train_dp.py build/merged-2b --mode verify --out runs/t1
  torchrun --nproc-per-node 4 scripts/train_dp.py build/merged-2b --mode greedy --steps 1 ...
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from qgpu.dist import train


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
    ap.add_argument("--check", action="store_true", help="greedy: compare every token as written")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--zero", action="store_true", help="ZeRO-1 over multicast windows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = train.Config(world=world, steps=a.steps, batch=a.batch, zero=a.zero)
    backend = RefBackend() if a.backend == "ref" else ModelBackend(a.merged, device=device)
    mode = "greedy" if a.backend == "ref" else a.mode  # the reference checks nothing
    dev = QGPU(rank, backend, mode=mode, check=a.check)
    ep = TorchTransport(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = train.reference(cfg)
    xte, yte = train.dataset(256, cfg.test_seed)
    logf = open(out / f"rank{rank}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    dist.barrier()
    t0 = time.time()
    res = train.program(rank, dev, ep, cfg, log=log)
    wall = time.time() - t0
    steps = res["steps"]
    exact = [s.params == ref["params"][i] for i, s in enumerate(steps)]
    loss_ok = [s.loss == ref["losses"][i][rank] for i, s in enumerate(steps)]
    mine = {"rank": rank, "init": res["init"], "exact_steps": sum(exact), "loss_ok": sum(loss_ok),
            "steps": [asdict(s) for s in steps], "wall_seconds": round(wall, 1),
            "checked_positions": dev.checked, "forwards": getattr(backend, "forwards", None),
            "positions": getattr(backend, "positions", None), "context_tokens": len(dev.log)}
    allr = [None] * world
    dist.all_gather_object(allr, mine)
    if rank == 0:
        replicas = all(allr[r]["steps"][i]["params"] == allr[0]["steps"][i]["params"]
                       for r in range(world) for i in range(cfg.steps))
        curve = []
        for i in range(cfg.steps):
            p = allr[0]["steps"][i]["params"]
            curve.append({"step": i, "loss": sum(allr[r]["steps"][i]["loss"] for r in range(world)),
                          "test_acc": train.accuracy(p, xte, yte),
                          "tokens_per_rank": [allr[r]["steps"][i]["generated"] for r in range(world)],
                          "seconds": max(allr[r]["steps"][i]["seconds"] for r in range(world)),
                          "per_kernel": allr[0]["steps"][i]["launches"]})
        summary = {"config": asdict(cfg), "mode": a.mode, "backend": a.backend, "world": world,
                   "init_test_acc": train.accuracy(allr[0]["init"], xte, yte),
                   "replicas_identical_every_step": replicas,
                   "exact_vs_numpy_steps": [allr[r]["exact_steps"] for r in range(world)],
                   "loss_vs_numpy_steps": [allr[r]["loss_ok"] for r in range(world)],
                   "checked_positions": [allr[r]["checked_positions"] for r in range(world)],
                   "forwards": [allr[r]["forwards"] for r in range(world)],
                   "context_tokens": [allr[r]["context_tokens"] for r in range(world)],
                   "wall_seconds": max(allr[r]["wall_seconds"] for r in range(world)),
                   "final_params": allr[0]["steps"][-1]["params"], "curve": curve}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        log(f"DONE: replicas identical {replicas}; exact vs numpy {summary['exact_vs_numpy_steps']}"
            f"/{cfg.steps}; test acc {summary['init_test_acc']:.3f} -> {curve[-1]['test_acc']:.3f}; "
            f"{summary['wall_seconds']}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
