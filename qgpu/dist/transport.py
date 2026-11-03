"""The fabric between QwenGPUs: how packets get from one device's context to another's.

Devices advance in lockstep, one record per tick: every busy device writes one record, then
the packets those records sent are exchanged, and each device feeds what it received at its
next record boundary. The schedule depends only on what the devices write, so a run is
deterministic and every backend (reference, circuit, model) produces the same token streams.

- ThreadTransport: ranks are threads of one process (a single-node simulation);
- TorchTransport: ranks are processes (torchrun, one per physical GPU), and the exchange is a
  torch.distributed all_to_all: QLink packets ride on the real GPUs' NCCL / NVLink.
"""

import threading


class ThreadTransport:
    """One shared fabric for `world` threads; `endpoint(rank)` is each thread's handle."""

    def __init__(self, world: int):
        self.world = world
        self.box = [[] for _ in range(world)]
        self.busy = [False] * world
        self.sent = [0] * world
        self.result = None
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(world, action=self._settle)
        self.ticks = 0

    def _settle(self):
        self.result = (any(self.busy), sum(self.sent))
        self.ticks += 1

    def endpoint(self, rank: int) -> "Endpoint":
        return Endpoint(self, rank)


class Endpoint:
    def __init__(self, fabric: ThreadTransport, rank: int):
        self.fabric, self.rank, self.world = fabric, rank, fabric.world

    def exchange(self, outgoing: list, busy: bool) -> tuple[list, bool, int]:
        """outgoing: [(peer, packet)]. Returns (packets for this rank, any rank busy, packets
        sent in this tick by all ranks)."""
        f = self.fabric
        with f.lock:
            for peer, pkt in outgoing:
                f.box[peer].append((self.rank, pkt))
            f.busy[self.rank] = busy
            f.sent[self.rank] = len(outgoing)
        f.barrier.wait()  # every rank has posted
        with f.lock:
            mine = sorted(f.box[self.rank], key=lambda x: x[0])  # by sender: deterministic
            f.box[self.rank] = []
        any_busy, sent = f.result
        f.barrier.wait()  # every rank has read (only then may anyone post the next tick)
        return [p for _, p in mine], any_busy, sent


class TorchTransport:
    """Each process is one rank; packets cross with torch.distributed. Per tick: an all_gather of
    [busy, packets to each peer], then, if any packet moves, one all_to_all of the packets (3
    tokens each). By default on a CPU (gloo) group: a tick then never waits behind the model's
    work on the GPU. backend="nccl" carries the ticks over the real GPUs' NVLink instead."""

    def __init__(self, device: str = "cuda", backend: str = "gloo"):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.group = dist.new_group(backend=backend) if backend != dist.get_backend() else None
        self.device = "cpu" if backend == "gloo" else device
        self.ticks = 0

    def exchange(self, outgoing: list, busy: bool) -> tuple[list, bool, int]:
        torch, dist, g = self.torch, self.dist, self.group
        per = [[] for _ in range(self.world)]
        for peer, pkt in outgoing:
            per[peer].append(pkt)
        meta = torch.tensor([int(busy)] + [len(p) for p in per], dtype=torch.int64,
                            device=self.device)
        allm = torch.empty(self.world * (self.world + 1), dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(allm, meta, group=g)
        allm = allm.view(self.world, self.world + 1).cpu().tolist()
        self.ticks += 1
        any_busy = any(row[0] for row in allm)
        sent = sum(sum(row[1:]) for row in allm)
        if not sent:
            return [], any_busy, 0
        send_counts = [3 * len(p) for p in per]
        recv_counts = [3 * allm[src][1 + self.rank] for src in range(self.world)]
        flat = [t for p in per for pkt in p for t in pkt]
        inp = torch.tensor(flat or [0], dtype=torch.int32, device=self.device)[:sum(send_counts)]
        out = torch.empty(sum(recv_counts), dtype=torch.int32, device=self.device)
        dist.all_to_all_single(out, inp, recv_counts, send_counts, group=g)
        toks = out.cpu().tolist()
        return [toks[i:i + 3] for i in range(0, len(toks), 3)], any_busy, sent


def synchronize(dev, transport, max_ticks: int = 1_000_000, burst: int | None = None) -> int:
    """Run the system until every device is idle and nothing is in flight (cudaDeviceSynchronize
    on every rank, collectively). Returns the number of exchanges.

    burst: records each device writes between two exchanges (default dev.burst, else 1). 1 is
    strict lockstep; more trades a little latency (a peer's packet waits for the next exchange)
    for far fewer collectives. The schedule stays deterministic for a given burst, and the
    protocols do not depend on it: they wait on flags, not on timing."""
    burst = burst or getattr(dev, "burst", 1)
    ticks = 0
    while True:
        out = []
        for _ in range(burst):
            if not dev.busy():
                break
            out += dev.tick()
        incoming, any_busy, sent = transport.exchange(out, dev.busy())
        dev.deliver(incoming)
        ticks += 1
        if not any_busy and not sent:  # the same on every rank: all stop together
            if dev.mode == "verify" and dev.backend.pending and \
                    len(dev.backend.pending) >= dev.verify_every:
                dev.verify()  # here every rank checks at once, instead of stalling the others
            return ticks
        if ticks >= max_ticks:
            raise TimeoutError(f"rank {dev.rank}: not idle after {ticks} ticks (deadlock?)")
