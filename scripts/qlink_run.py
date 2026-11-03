"""Run qccl's collectives on several QwenGPUs and check them against real NCCL.

One process per physical GPU (torchrun): each process loads the merged checkpoint on its own
GPU, the QwenGPUs exchange QLink packets over torch.distributed (NCCL on the real GPUs), and
every collective's result is compared, bit for bit, with the same collective that real NCCL
computes on the same GPUs from the same int32 inputs. With --check each model token is also
compared with a reference shadow as it is written; --dump saves every rank's token stream, and
`--replay DIR` re-runs the whole system on the reference interpreter (threads) and checks that
every rank's stream is identical.

  torchrun --nproc-per-node 4 scripts/qlink_run.py build/merged-2b --check --dump runs/e4
  python scripts/qlink_run.py --replay runs/e4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from qgpu.dist.qccl import Comm
from qgpu.dist.transport import synchronize


def inputs(world: int, seed: int = 2026):
    rng = np.random.default_rng(seed)
    w = lambda *s: rng.integers(-2**31, 2**31, s).tolist()  # noqa: E731
    return {"ar": w(world, 4 * world), "rs": w(world, 4 * world), "ag": w(world, 4),
            "bc": w(world, 8), "rd": w(world, 8), "sr": w(world, 8)}


def program(rank: int, world: int, dev, ep, log=print):
    """The collective suite on one rank: returns {op: (result, stats)}."""
    d = inputs(world)
    comm = Comm(dev, rank, world, chunk=4)
    n = 4 * world
    out = {}

    base = dev.heap  # each collective's buffers are freed after it (8 ranks fit in 127 words)

    def run(name, fn, read):
        h0 = len(dev.history)
        t0 = time.time()
        fn()
        ticks = synchronize(dev, ep)
        hs = dev.history[h0:]
        out[name] = (read(), {"ticks": ticks, "seconds": round(time.time() - t0, 1),
                              "launches": len(hs), "generated": sum(h.generated for h in hs),
                              "records": sum(h.records for h in hs),
                              "sent_packets": sum(h.sent for h in hs),
                              "received_tokens": sum(h.injected for h in hs)})
        log(f"rank {rank}: {name} done: {out[name][1]}")
        dev.heap = base

    x = dev.malloc(n, "ar.x")
    dev.memcpy_h2d(x, d["ar"][rank])
    run("all_reduce", lambda: comm.all_reduce(x, n), lambda: dev.read(x, n))
    xs, ys = dev.malloc(n, "rs.x"), dev.malloc(4, "rs.y")
    dev.memcpy_h2d(xs, d["rs"][rank])
    run("reduce_scatter", lambda: comm.reduce_scatter(xs, ys, 4), lambda: dev.read(ys, 4))
    g, z = dev.malloc(4, "ag.x"), dev.malloc(n, "ag.y")
    dev.memcpy_h2d(g, d["ag"][rank])
    run("all_gather", lambda: comm.all_gather(g, z, 4), lambda: dev.read(z, n))
    b = dev.malloc(8, "bc.x")
    dev.memcpy_h2d(b, d["bc"][rank])
    run("broadcast", lambda: comm.broadcast(b, 8, root=1), lambda: dev.read(b, 8))
    r = dev.malloc(8, "rd.x")
    dev.memcpy_h2d(r, d["rd"][rank])
    run("reduce", lambda: comm.reduce(r, 8, root=2 % world), lambda: dev.read(r, 8))
    sx, sy = dev.malloc(8, "sr.x"), dev.malloc(8, "sr.y")
    dev.memcpy_h2d(sx, d["sr"][rank])
    run("sendrecv", lambda: comm.sendrecv(sx, sy, 8), lambda: dev.read(sy, 8))
    return out


def numpy_reference(rank: int, world: int) -> dict:
    """The same collectives in numpy (int32, wrapping): for more ranks than GPUs, where NCCL
    cannot run (it takes one rank per GPU)."""
    import numpy as np

    d = inputs(world)
    i32 = lambda a: ((np.asarray(a, np.int64) + 2**31) % 2**32 - 2**31).tolist()  # noqa: E731
    s = lambda k: i32(np.sum(d[k], axis=0))  # noqa: E731
    return {"all_reduce": s("ar"), "reduce_scatter": s("rs")[4 * rank:4 * rank + 4],
            "all_gather": [v for row in d["ag"] for v in row], "broadcast": d["bc"][1],
            "reduce": s("rd") if rank == 2 % world else d["rd"][rank],
            "sendrecv": d["sr"][(rank - 1) % world]}


def nccl_reference(rank: int, world: int, device: str) -> dict:
    """The same collectives, computed by real NCCL on the real GPUs."""
    import torch
    import torch.distributed as dist

    d = inputs(world)
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    out = {}
    a = t(d["ar"][rank])
    dist.all_reduce(a)
    out["all_reduce"] = a.tolist()
    rs = torch.empty(4, dtype=torch.int32, device=device)
    dist.reduce_scatter_tensor(rs, t(d["rs"][rank]))
    out["reduce_scatter"] = rs.tolist()
    ag = torch.empty(4 * world, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(ag, t(d["ag"][rank]))
    out["all_gather"] = ag.tolist()
    bc = t(d["bc"][rank])
    dist.broadcast(bc, src=1)
    out["broadcast"] = bc.tolist()
    rd = t(d["rd"][rank])
    dist.reduce(rd, dst=2 % world)
    out["reduce"] = rd.tolist()
    sr = torch.empty(8, dtype=torch.int32, device=device)
    ops = [dist.P2POp(dist.isend, t(d["sr"][rank]), (rank + 1) % world),
           dist.P2POp(dist.irecv, sr, (rank - 1) % world)]
    for w in dist.batch_isend_irecv(ops):
        w.wait()
    out["sendrecv"] = sr.tolist()
    torch.cuda.synchronize()
    return out


def main_torchrun(a):
    import torch
    import torch.distributed as dist

    from qgpu.dist.backends import ModelBackend, RefBackend
    from qgpu.dist.device import QGPU
    from qgpu.dist.transport import TorchTransport

    local = int(os.environ["LOCAL_RANK"])
    ngpu = torch.cuda.device_count()
    device = f"cuda:{local % ngpu}"
    torch.cuda.set_device(local % ngpu)
    shared = int(os.environ["LOCAL_WORLD_SIZE"]) > ngpu  # several QwenGPUs per physical GPU
    if shared:
        dist.init_process_group("gloo")
    else:
        dist.init_process_group("nccl", device_id=torch.device(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    t0 = time.time()
    backend = RefBackend() if a.backend == "ref" else ModelBackend(a.merged, device=device)
    print(f"rank {rank}: {a.backend} backend on {device} ready in {time.time() - t0:.0f}s",
          flush=True)
    dev = QGPU(rank, backend, mode=a.mode, check=a.check)
    dev.burst = a.burst
    ep = TorchTransport(device)
    dist.barrier()
    res = program(rank, world, dev, ep, log=lambda s: print(s, flush=True))
    nccl = numpy_reference(rank, world) if shared else nccl_reference(rank, world, device)
    same = {k: res[k][0] == nccl[k] for k in nccl}
    print(f"rank {rank}: qccl == {'numpy' if shared else 'NCCL'}: {same}", flush=True)
    if a.dump:
        out = Path(a.dump)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank{rank}.json").write_text(json.dumps({
            "rank": rank, "world": world, "backend": a.backend, "mode": a.mode,
            "reference": "numpy" if shared else "nccl",
            "check": a.check, "results": {k: v[0] for k, v in res.items()},
            "stats": {k: v[1] for k, v in res.items()}, "nccl": nccl, "same_as_nccl": same,
            "forwards": getattr(backend, "forwards", None),
            "positions": getattr(backend, "positions", None),
            "log": dev.log, "src": dev.src}))
    dist.barrier()
    dist.destroy_process_group()


def replay(a):
    """The whole system again, on the reference interpreter in threads: same streams?"""
    from qgpu.dist.backends import RefBackend
    from qgpu.dist.spawn import spawn

    runs = sorted(Path(a.replay).glob("rank*.json"))
    dumps = [json.loads(p.read_text()) for p in runs]
    world = dumps[0]["world"]
    devs = [None] * world

    def prog(rank, dev, ep):
        devs[rank] = dev
        return program(rank, world, dev, ep, log=lambda s: None)

    res = spawn(world, prog, lambda r: RefBackend())
    for d in dumps:
        r = d["rank"]
        same_log = devs[r].log == d["log"] and devs[r].src == d["src"]
        same_res = {k: res[r][k][0] == v for k, v in d["results"].items()}
        print(f"rank {r}: token stream identical to the reference system: {same_log} "
              f"({len(d['log'])} tokens); results: {same_res}; vs NCCL: {d['same_as_nccl']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("merged", nargs="?", default="build/merged-2b")
    ap.add_argument("--backend", choices=("model", "ref"), default="model")
    ap.add_argument("--mode", choices=("greedy", "draft", "verify"), default="greedy")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dump")
    ap.add_argument("--burst", type=int, default=1, help="records per exchange")
    ap.add_argument("--replay")
    a = ap.parse_args()
    replay(a) if a.replay else main_torchrun(a)
