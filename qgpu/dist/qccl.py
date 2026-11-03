"""qccl: NCCL's collectives on QwenGPUs, over QLink.

A communicator (Comm, like ncclCommInitRank) gives every rank a ring channel to the next rank,
laid out the same on every rank (a symmetric heap, as NVSHMEM and NCCL's registered buffers
are):

    fifo   S slots of C words   the chunks the previous rank sent us   } stores -> next rank
    tail   1 word               how many chunks it has sent            }   (same addresses)
    head   1 word               how many of our chunks the next rank has consumed;
                                stores -> previous rank

QLink apertures route stores only (a load always reads local memory), so a channel's send side
sits on the same addresses as its receive side: a kernel loads what the previous rank sent and
its stores to those addresses land at the next rank. The flow control is NCCL's (the Simple
protocol): the sender waits for a free slot (head), writes the chunk into the receiver's slot,
fences and posts tail; the receiver waits for tail, consumes the slot and posts head. All of it
is done by one CUDA kernel, kernels/qccl/prim.cu, which every step of every collective launches
with a different mode (send, recvReduceSend, recvReduceCopySend, recvCopySend, recv), as
NCCL's ring algorithms do (src/device/all_reduce.h, runRing). The host only enqueues the
launches and keeps the step counters; it never touches the data, and no host code makes one
rank wait for another: the kernels do, by polling the flags their peers' stores write.
"""

from pathlib import Path

from qgpu import ptx
from qgpu.dist.device import Remote
from qgpu.lower import DevPtr

ROOT = Path(__file__).resolve().parent.parent.parent
RECV, LOCAL, COPY, SEND, SPLIT = 1, 2, 4, 8, 16
_K = {}


def kernel(name: str):
    if name not in _K:
        _K.update(ptx.load(ROOT / "kernels" / "qccl" / "prim.ptx"))
        _K.update(ptx.load(ROOT / "kernels" / "qccl" / "mcast.ptx"))
    return _K[name]


def prim_kernel():
    return kernel("qccl_prim")


def pow2_at_least(n: int, lo: int = 4) -> int:
    return max(lo, 1 << (n - 1).bit_length())


class Comm:
    """A ring communicator. Every rank calls everything in the same order (SPMD), with its own
    device and transport endpoint; buffers are word addresses in device memory."""

    def __init__(self, dev, rank: int, world: int, chunk: int = 8, slots: int = 2):
        assert chunk in (4, 8, 16, 32), "a chunk is one thread block: 4 to 32 threads"
        assert world >= 2
        self.dev, self.rank, self.world, self.C, self.S = dev, rank, world, chunk, slots
        self.next, self.prev = (rank + 1) % world, (rank - 1) % world
        self.fifo = dev.malloc(slots * chunk, "qccl.fifo")
        self.tail = dev.malloc(1, "qccl.tail")
        self.head = dev.malloc(1, "qccl.head")
        dev.memcpy_h2d(self.fifo, [0] * (slots * chunk + 2))
        # the send side of the channel: the next rank's fifo and tail, the previous one's head
        dev.map(self.fifo, slots * chunk + 1, [(self.next, self.fifo)])
        dev.map(self.head, 1, [(self.prev, self.head)])
        self.sent = 0  # chunks sent to the next rank on this channel (NCCL's step counter)
        self.recvd = 0  # chunks received from the previous rank
        self.launches = 0

    # --- the primitive ----------------------------------------------------------------------

    def prim(self, mode: int, x: int | None = None, y: int | None = None,
             n: int | None = None, name: str = "") -> None:
        """One step: v = (recv ? slot : 0) + (local ? x : 0); copy: y = v; send: next's slot = v.
        n elements (<= C); the launch has a power-of-two number of threads >= n."""
        n = n or self.C
        assert n == pow2_at_least(n) and n <= self.C, \
            f"a step moves a power-of-two chunk of 4..{self.C} words, not {n} (pad the buffer)"
        C, S = self.C, self.S
        rslot = self.fifo + (self.recvd % S) * C
        sslot = self.fifo + (self.sent % S) * C
        # the send side (sslot, stail, shead) is peer memory: on a QwenGPU the same addresses
        # as the receive side, routed by the channel's apertures
        args = [DevPtr(x if x is not None else 0), DevPtr(y if y is not None else 0),
                DevPtr(rslot), Remote(sslot), DevPtr(self.tail), Remote(self.tail),
                DevPtr(self.head), Remote(self.head),
                self.recvd + 1,  # rwait: the chunk we want has been posted
                self.sent + 1 - S,  # swait: the next rank has consumed chunk sent - S
                self.sent + 1,  # spost
                self.recvd + 1,  # rpost
                mode]
        self.dev.launch(prim_kernel(), pow2_at_least(n), args, name=name or mode_name(mode))
        self.launches += 1
        if mode & RECV:
            self.recvd += 1
        if mode & SEND:
            self.sent += 1

    # --- collectives (NCCL's ring algorithms) -----------------------------------------------

    def _rounds(self, n: int, per_rank: bool):
        """Split n elements into rounds of world x C (allreduce) or C (per-rank) elements."""
        step = self.C * (1 if per_rank else self.world)
        assert n % (self.world if not per_rank else 1) == 0 or per_rank
        for off in range(0, n, step):
            yield off, min(step, n - off)

    def all_reduce(self, x: int, n: int) -> None:
        """In place: x <- sum over ranks of x (int32, wrapping). n must split into world equal
        chunks of at most C."""
        N, r = self.world, self.rank
        for off, m in self._rounds(n, per_rank=False):
            assert m % N == 0, f"allreduce of {m} elements over {N} ranks"
            c = m // N

            def at(ch):
                return x + off + (ch % N) * c

            self.prim(LOCAL | SEND, x=at(r - 1), n=c, name="ar.send")
            for j in range(2, N):
                self.prim(RECV | LOCAL | SEND, x=at(r - j), n=c, name="ar.recvReduceSend")
            self.prim(RECV | LOCAL | COPY | SEND, x=at(r), y=at(r), n=c,
                      name="ar.recvReduceCopySend")
            for j in range(1, N - 1):
                self.prim(RECV | COPY | SEND, y=at(r - j), n=c, name="ar.recvCopySend")
            self.prim(RECV | COPY, y=at(r + 1), n=c, name="ar.recv")

    def reduce_scatter(self, x: int, y: int, c: int) -> None:
        """y (c words) <- chunk `rank` of the sum over ranks of x (world x c words)."""
        N, r = self.world, self.rank
        assert c <= self.C

        def at(ch):
            return x + (ch % N) * c

        self.prim(LOCAL | SEND, x=at(r - 1), n=c, name="rs.send")
        for j in range(2, N):
            self.prim(RECV | LOCAL | SEND, x=at(r - j), n=c, name="rs.recvReduceSend")
        self.prim(RECV | LOCAL | COPY, x=at(r), y=y, n=c, name="rs.recvReduceCopy")

    def all_gather(self, x: int, y: int, c: int) -> None:
        """y (world x c words) <- every rank's x (c words), in rank order."""
        N, r = self.world, self.rank
        assert c <= self.C

        def at(ch):
            return y + (ch % N) * c

        self.prim(LOCAL | COPY | SEND, x=x, y=at(r), n=c, name="ag.copySend")
        for j in range(1, N - 1):
            self.prim(RECV | COPY | SEND, y=at(r - j), n=c, name="ag.recvCopySend")
        self.prim(RECV | COPY, y=at(r + 1), n=c, name="ag.recv")

    def broadcast(self, x: int, n: int, root: int = 0) -> None:
        """x on every rank <- x on root (pipelined in chunks of C along the ring)."""
        pos = (self.rank - root) % self.world  # position along the ring from the root
        for off in range(0, n, self.C):
            c = min(self.C, n - off)
            if pos == 0:
                self.prim(LOCAL | SEND, x=x + off, n=c, name="bc.send")
            elif pos == self.world - 1:
                self.prim(RECV | COPY, y=x + off, n=c, name="bc.recv")
            else:
                self.prim(RECV | COPY | SEND, y=x + off, n=c, name="bc.recvCopySend")

    def reduce(self, x: int, n: int, root: int = 0) -> None:
        """x on root <- sum over ranks of x (the chain starts after the root, ends at it)."""
        pos = (self.rank - root - 1) % self.world
        for off in range(0, n, self.C):
            c = min(self.C, n - off)
            if pos == 0:
                self.prim(LOCAL | SEND, x=x + off, n=c, name="rd.send")
            elif pos == self.world - 1:
                self.prim(RECV | LOCAL | COPY, x=x + off, y=x + off, n=c, name="rd.recvReduceCopy")
            else:
                self.prim(RECV | LOCAL | SEND, x=x + off, n=c, name="rd.recvReduceSend")

    def sendrecv(self, x: int, y: int, n: int) -> None:
        """Grouped send to the next rank and receive from the previous (ncclGroupStart, ncclSend,
        ncclRecv, ncclGroupEnd), software-pipelined as NCCL's ring steps are: send chunk 0,
        then each step receives chunk k - 1 while it sends chunk k, then receive the last.
        Every rank sends before it waits, so no rank waits on another's credits forever."""
        C = self.C
        offs = list(range(0, n, C))
        size = [min(C, n - o) for o in offs]
        self.prim(LOCAL | SEND, x=x, n=size[0], name="p2p.send")
        for k in range(1, len(offs)):
            assert size[k] == size[k - 1], "sendrecv: equal chunks"
            self.prim(RECV | LOCAL | COPY | SEND | SPLIT, x=x + offs[k], y=y + offs[k - 1],
                      n=size[k], name="p2p.sendrecv")
        self.prim(RECV | COPY, y=y + offs[-1], n=size[-1], name="p2p.recv")

    # --- multicast (NVLS-style) -------------------------------------------------------------

    def window(self, n: int, name: str = "") -> int:
        """A multicast window: n words at the same address on every rank; a store into it lands
        locally and at every peer (a multicast object, NVLink SHARP's multimem.st). Like every
        allocation, all ranks must make it in the same order."""
        a = self.dev.malloc(n, name)
        self.dev.memcpy_h2d(a, [0] * n)
        peers = [(p, a) for p in range(self.world) if p != self.rank]
        self.dev.map(a, n, peers, local=True)
        return a

    def mcast(self, src: int, dst: int, n: int, flag: int, epoch: int) -> None:
        """dst (in a window) <- src, n <= 32 words, then flag (in a window) <- epoch."""
        assert n == pow2_at_least(n) and n <= 32
        self.dev.launch(kernel("qccl_mcast"), n, [DevPtr(src), DevPtr(dst), DevPtr(flag), epoch],
                        name="mc.copy")
        self.launches += 1

    def wait(self, flags: int, n: int, value: int) -> None:
        """Wait until flags[0..n) >= value (each written by another rank's multicast)."""
        self.dev.launch(kernel("qccl_wait"), 4, [DevPtr(flags), n, value], name="mc.wait")
        self.launches += 1

    def all_gather_mc(self, x: int, y: int, c: int, flags: int, epoch: int) -> None:
        """y (a world x c window) <- every rank's x: each rank multicasts its own c words once,
        posts flags[rank] = epoch, and waits for every rank's flag. One store per word."""
        self.mcast(x, y + self.rank * c, c, flags + self.rank, epoch)
        self.wait(flags, self.world, epoch)

    def all_reduce_nvls(self, x: int, y: int, n: int, flags: int, epoch: int) -> None:
        """y (a window of n words) <- sum over ranks of x: ring ReduceScatter in place, then each
        rank multicasts its reduced chunk (NVLS two-shot AllReduce)."""
        c = n // self.world
        assert c * self.world == n and c <= self.C
        self.reduce_scatter_inplace(x, c)
        self.all_gather_mc(x + self.rank * c, y, c, flags, epoch)

    def reduce_scatter_inplace(self, x: int, c: int) -> None:
        """Chunk `rank` of x (world x c words) <- its sum over ranks (the others are scratch)."""
        self.reduce_scatter(x, x + self.rank * c, c)

    def send(self, x: int, n: int) -> None:
        """Point to point, to the next rank on the ring (pipeline parallelism's send)."""
        for off in range(0, n, self.C):
            self.prim(LOCAL | SEND, x=x + off, n=min(self.C, n - off), name="p2p.send")

    def recv(self, y: int, n: int) -> None:
        """Point to point, from the previous rank on the ring."""
        for off in range(0, n, self.C):
            self.prim(RECV | COPY, y=y + off, n=min(self.C, n - off), name="p2p.recv")


def mode_name(mode: int) -> str:
    names = {LOCAL | SEND: "send", RECV | LOCAL | SEND: "recvReduceSend",
             RECV | LOCAL | COPY | SEND: "recvReduceCopySend", RECV | COPY | SEND: "recvCopySend",
             RECV | COPY: "recv", RECV | LOCAL | COPY: "recvReduceCopy",
             LOCAL | COPY | SEND: "copySend", RECV | LOCAL | COPY | SEND | SPLIT: "sendrecv"}
    return names.get(mode, f"mode{mode}")
