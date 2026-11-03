"""A QwenGPU as one device of a multi-GPU system.

Device memory. The GPU's 127 words of global memory persist across launches, like HBM: the
host allocates from a heap (cudaMalloc), copies in (MEM records, cudaMemcpy H2D) and reads back
from the model's state (D2H). A launch takes DevPtr arguments into that heap; its `.shared`
arrays go above the heap.

Streams. Work is queued in order (launches, H2D copies) and runs when the device gets to it;
`tick` advances the device by one *record*: a warp running one machine block, the EXEC token
and the tokens the model writes after it. Between two records the device is at a record
boundary: the model has predicted the next EXEC token, which is not fed yet. That is where
packets from peers are fed (they write memory words; nothing else in the circuit reads them),
so a kernel that polls a flag in memory sees a peer's store at its next record, as a GPU sees
an NVLink store arrive while its kernel runs.

Apertures (QLink). A store whose word address falls in an aperture is routed: the model writes
it as MEM(a) VAL VAL (a write-back record), and the device sends those three tokens to every
peer the aperture names (the address mapped to the peer's range); unless the aperture keeps a
local copy, the device feeds MEM(SINK) for the header instead of MEM(a), so locally the word
goes to the sink. Loads are never routed: an aperture address reads local memory. Atomics into
an aperture are refused (a remote atomic needs the peer to add; QLink only carries stores).

Two ways to write a record, with the same result:
- greedy: the backend predicts one token at a time and each prediction is fed back
  (autoregressive decoding: one forward pass per token);
- draft: a reference shadow drafts the whole record, the backend runs it in one forward pass
  and every drafted token must be the backend's own argmax at its position (speculative
  decoding with an exact drafter; any mismatch is an error, not a fallback).
"""

import time
from collections import deque
from dataclasses import dataclass, field

from qgpu import ptx
from qgpu.circuit import kernel
from qgpu.circuit.kernel import (EXEC, LIVE, MEM, MEM0, PROG, PROGI, SINK, encode_live, kind,
                                 program_tokens)
from qgpu.dist.backends import RefBackend, is_exec
from qgpu.lower import DevPtr, lower  # noqa: F401 - DevPtr: kernel pointer arguments
from qgpu.merge import STAGES


HOST, OWN, PEER = 0, 1, 2  # who wrote a token of a device's context


class LinkError(Exception):
    pass


class Divergence(Exception):
    """The backend's argmax differs from the drafted (reference) token."""


@dataclass(frozen=True)
class Remote:
    """A pointer argument a kernel only stores through, into a peer's memory: an address in one
    of the device's apertures. A QwenGPU passes the address itself (its aperture routes the
    stores); a real GPU (realgpu.py) passes the peer's own pointer, as NVLink P2P does."""
    word: int


@dataclass(frozen=True)
class Aperture:
    lo: int  # first word address
    n: int  # words
    routes: tuple  # ((peer rank, the peer's first word address), ...)
    local: bool = False  # the store also lands in local memory (a multicast including self)


@dataclass
class LaunchStats:
    name: str
    warps: int
    blocks: int
    host_tokens: int = 0  # program image, initial words: prefill
    records: int = 0
    generated: int = 0  # tokens the backend wrote (EXEC tokens of later records included)
    injected: int = 0  # packet tokens fed from peers during the launch
    sent: int = 0  # packets this launch sent over QLink
    ticks: int = 0
    seconds: float = 0.0


@dataclass
class Op:
    kind: str  # "launch" | "h2d"
    name: str = ""
    launch: object = None
    addr: int = 0
    words: list = field(default_factory=list)


class QGPU:
    def __init__(self, rank: int, backend, lanes: int = 4, stages: int = 4,
                 mode: str = "greedy", check: bool = False, heap_limit: int = SINK,
                 keep_log: bool = True):
        """mode: "greedy" or "draft" (see the module doc); check: in greedy mode, compare every
        token with a reference shadow as it is written."""
        assert mode in ("greedy", "draft", "verify")
        self.rank, self.backend, self.lanes, self.stages = rank, backend, lanes, stages
        self.mode = mode
        shadow = mode != "greedy" or (check and not isinstance(backend, RefBackend))
        self.drafter = RefBackend(lanes) if shadow else None
        self.heap = 0
        self.heap_limit = heap_limit
        self.allocs: dict[str, tuple[int, int]] = {}
        self.ap: list[Aperture | None] = [None] * (SINK + 1)
        self.stream: deque[Op] = deque()
        self.cur: LaunchStats | None = None  # the running launch
        self.cur_launch = None
        self.next_exec: int | None = None  # predicted at a record boundary, not fed yet
        self.inbox: list[list[int]] = []
        self.history: list[LaunchStats] = []
        self.keep_log = keep_log  # long runs keep only counts
        self.log: list[int] = []  # every token fed, in order (for cross-backend checks)
        self.src: list[int] = []  # who wrote each: HOST, OWN (this GPU's weights) or PEER
        self.last = None  # the last token fed
        self.fed = 0  # tokens fed in this context
        self.rebases = 0
        # what the context's program ROM holds: (kind, block, stage) -> payload. A launch only
        # writes the rows it changes (the ROM is model state and persists between launches; a
        # row past a block's last instruction is never read: only declared live-outs and
        # write-backs leave a block)
        self.rom: dict = {}
        self.rom_cache = True
        self.burst = 1  # records per exchange in synchronize() (1: strict lockstep)
        self.injected_idle = 0
        self.expect: list[tuple[int, int]] = []  # verify mode: (position, token) to check
        self.checked = 0  # positions the backend has been checked at (draft / verify)
        self.verify_every = 16384  # verify mode: check every this many context tokens

    # --- device memory --------------------------------------------------------------------

    def malloc(self, n: int, name: str = "") -> int:
        """n words of device memory; returns the first word address."""
        if self.heap + n > self.heap_limit:
            raise MemoryError(f"rank {self.rank}: {name or 'buffer'} needs {n} words, "
                              f"{self.heap_limit - self.heap} of {self.heap_limit} are free")
        a = self.heap
        self.heap += n
        if name:
            self.allocs[name] = (a, n)
        return a

    def map(self, lo: int, n: int, routes, local: bool = False) -> None:
        """Route stores into [lo, lo + n) to peers: routes = [(peer rank, peer address)]."""
        ap = Aperture(lo, n, tuple(routes), local)
        for a in range(lo, lo + n):
            assert self.ap[a] is None, f"word {a} is in two apertures"
            self.ap[a] = ap

    def memcpy_h2d(self, addr: int, words) -> None:
        self.stream.append(Op("h2d", addr=addr, words=[w & 0xFFFFFFFF for w in words]))

    def read(self, addr: int, n: int, signed: bool = True) -> list[int]:
        """Device -> host copy; the stream must be drained (synchronize first)."""
        assert self.idle(), "read with work in flight: synchronize first"
        if self.mode == "verify":
            self.verify()  # nothing goes through the backend unchecked
        mem = self.backend.memory()
        vals = [mem.get(addr + i, 0) for i in range(n)]
        return [v - (1 << 32) if signed and v >> 31 else v for v in vals]

    # --- launches ---------------------------------------------------------------------------

    def lower(self, k: ptx.Kernel, threads: int, args: list, block: int | None = None):
        """Lower a kernel for this device (the driver's JIT): one-dimensional launch of
        `threads` threads in blocks of `block`; pointer arguments as DevPtr."""
        mul, logic, atom = STAGES[self.stages]
        block = block or threads
        return lower(k, (threads // block,), (block,), args, {}, lanes=self.lanes,
                     stages=self.stages, mul_stages=mul, logic_stages=logic, atom_stages=atom,
                     write_through=True, scratch_base=self.heap)

    def launch(self, k: ptx.Kernel, threads: int, args: list, block: int | None = None,
               name: str | None = None) -> None:
        args = [DevPtr(a.word) if isinstance(a, Remote) else a for a in args]
        lw = self.lower(k, threads, args, block)
        self.stream.append(Op("launch", name=name or k.name, launch=lw))

    def idle(self) -> bool:
        return self.cur is None and not self.stream

    def busy(self) -> bool:
        return not self.idle() or bool(self.inbox)

    # --- the token interface ------------------------------------------------------------------

    def _feed(self, toks: list[int], src: int = 0) -> None:
        if not toks:
            return
        self.backend.feed(toks)
        if self.drafter is not None:
            self.drafter.feed(toks)
        if self.keep_log:
            self.log += toks
            self.src += [src] * len(toks)
        self.last = toks[-1]
        self.fed += len(toks)

    def deliver(self, packets: list[list[int]]) -> None:
        self.inbox += packets

    def _inject(self) -> int:
        """Feed the packets that arrived, at this record boundary."""
        n = 0
        for p in self.inbox:
            self._feed(p, PEER)
            n += len(p)
        self.inbox = []
        return n

    def _begin(self, op: Op) -> None:
        lw = op.launch
        if self.next_exec is not None:  # the previous launch's end token
            self._feed([self.next_exec], OWN)
            self.next_exec = None
        self.backend.begin(lw)
        if self.drafter is not None:
            self.drafter.begin(lw)
        host = self._program(lw.blocks)
        for a, v in lw.mem.items():
            host += [MEM(a), v & 0xFFFF, v >> 16 & 0xFFFF]
        host.append(EXEC(0, 0, True))
        self._feed(host)
        self.cur = LaunchStats(op.name, lw.warps, len(lw.blocks), host_tokens=len(host))
        self.cur_launch = lw

    def _program(self, blocks) -> list[int]:
        """The PROG / PROGI / LIVE records of the rows this program changes."""
        if not self.rom_cache:
            return program_tokens(blocks)
        toks = []
        for b, blk in enumerate(blocks):
            for s, ins in enumerate(blk.instrs):
                f0, f1, i0, i1 = ins.encode()
                for key, rec in ((("P", b, s), [PROG(b, s), f0, f1]), (("I", b, s), [PROGI(b, s), i0, i1])):
                    if self.rom.get(key) != rec:
                        self.rom[key] = rec
                        toks += rec
            rec = [LIVE(b), *encode_live(blk)]
            if self.rom.get(("L", b)) != rec:
                self.rom[("L", b)] = rec
                toks += rec
        return toks

    def tick(self) -> list[tuple[int, list[int]]]:
        """Advance by one record; returns the packets it sent: (peer rank, tokens)."""
        if self.cur is None:
            n = self._inject()
            self.injected_idle += n
            while self.stream and self.stream[0].kind == "h2d":
                op = self.stream.popleft()
                self._feed([t for i, w in enumerate(op.words)
                            for t in (MEM(op.addr + i), w & 0xFFFF, w >> 16)])
            if not self.stream:
                return []
            self._begin(self.stream.popleft())
        else:
            self.cur.injected += self._inject()
            self._feed([self.next_exec], OWN)
            self.next_exec = None
        t0 = time.time()
        packets = {"greedy": self._record, "draft": self._record_draft,
                   "verify": self._record_verify}[self.mode]()
        st = self.cur
        st.records += 1
        st.ticks += 1
        st.sent += len(packets)
        st.seconds += time.time() - t0
        done = self.next_exec in (EXEC(0, st.warps, False), EXEC(0, st.warps, True))
        if self.mode == "verify" and len(self.backend.pending) >= 4 * self.verify_every:
            self.verify()  # a cap on memory; normally synchronize() checks (and every read)
        if done:
            self.history.append(st)  # the end token: the launch is done
            self.cur, self.cur_launch = None, None
        return packets

    def rebase(self) -> None:
        """Start a fresh context with the same device memory, between launches: read the 127
        words back, drop the context, and write them in again (a state save and restore; the
        host copies words, it computes nothing). Registers and the program are per launch, so
        nothing else carries over. Keeps contexts short in long runs."""
        assert self.idle() and not self.inbox
        if self.mode == "verify":
            self.verify()
        mem = self.backend.memory()
        self.backend.reset()
        if self.drafter is not None:
            self.drafter.reset()
        self.next_exec = None  # the previous launch's end token belongs to the old context
        self.fed = 0
        self.rom = {}
        words = [(a, v) for a, v in sorted(mem.items()) if v and a != SINK]
        self._feed([t for a, v in words for t in (MEM(a), v & 0xFFFF, v >> 16)], HOST)
        self.rebases += 1

    def verify(self) -> None:
        """Verify mode: run the context written since the last check through the backend in one
        forward pass; its argmax at every position the GPU owns must be the token there."""
        n, bad = self.backend.check(self.expect)
        self.checked += n
        self.expect = []
        if bad is not None:
            p, got, want = bad
            raise Divergence(f"rank {self.rank}: at context position {p} the backend predicts "
                             f"{got}, the stream has {want}")

    def _route(self, t: int, state: dict, packets: list) -> int:
        """A token the backend wrote inside a record -> the token to feed; collects packets."""
        k = kind(t)
        if k == "mem":
            a = t - MEM0
            ap = self.ap[a]
            state["wb"] = None
            if ap is None:
                return t
            blk = self.cur_launch.blocks[state["block"]]
            if any(ins.op == "atom" for ins in blk.instrs):
                raise LinkError(f"rank {self.rank}: an atomic into peer aperture word {a} "
                                "(QLink carries stores only)")
            state["wb"] = (ap, a, [])
            return t if ap.local else MEM(SINK)
        if k == "val":
            wb = state.get("wb")
            if wb is not None:
                wb[2].append(t)
                if len(wb[2]) == 2:
                    ap, a, (lo, hi) = wb
                    for peer, base in ap.routes:
                        packets.append((peer, [MEM(base + a - ap.lo), lo, hi]))
                    state["wb"] = None
            return t
        raise LinkError(f"rank {self.rank}: a {k} token inside a record")

    def _record(self) -> list:
        """Greedy: one forward pass per token, each prediction fed back (and, with a shadow,
        compared with the reference's as it is written)."""
        packets, state = [], {"block": (self.last - kernel.EXEC0) & ((1 << kernel.BBITS) - 1)}
        n = 0
        while True:
            (t,) = self.backend.predict(1)
            self.cur.generated += 1
            if self.drafter is not None:
                (want,) = self.drafter.predict(1)
                if t != want:
                    raise Divergence(f"rank {self.rank}, {self.cur.name}, record "
                                     f"{self.cur.records}: the backend wrote {t} where the "
                                     f"reference has {want} (token {n})")
            n += 1
            if is_exec(t):
                self.next_exec = t
                return packets
            self._feed([self._route(t, state, packets)], OWN)

    def _record_verify(self) -> list:
        """The reference shadow writes the record; every token it predicts is queued as a check
        on the backend at that position (run by verify(), in one pass for many records)."""
        packets, state = [], {"block": (self.last - kernel.EXEC0) & ((1 << kernel.BBITS) - 1)}
        base = self.backend.n_fed - 1  # position of the last token fed (the record's EXEC)
        k = 0
        while True:
            (t,) = self.drafter.predict(1)
            self.expect.append((base + k, t))
            self.cur.generated += 1
            if is_exec(t):
                self.next_exec = t
                return packets
            f = self._route(t, state, packets)
            self._feed([f], OWN)
            k += 1

    def _record_draft(self) -> list:
        """Draft the record on the reference shadow, verify it in one forward pass."""
        packets, state = [], {"block": (self.last - kernel.EXEC0) & ((1 << kernel.BBITS) - 1)}
        want, fed = [], []
        while True:
            (t,) = self.drafter.predict(1)
            want.append(t)
            if is_exec(t):
                break
            f = self._route(t, state, packets)
            self.drafter.feed([f])
            fed.append(f)
        self.backend.feed(fed)
        if self.keep_log:
            self.log += fed
            self.src += [OWN] * len(fed)
        if fed:
            self.last = fed[-1]
        self.fed += len(fed)
        got = self.backend.predict(len(want))
        self.cur.generated += len(want)
        self.checked += len(want)
        if got != want:
            i = next(j for j, (a, b) in enumerate(zip(got, want)) if a != b)
            raise Divergence(f"rank {self.rank}, {self.cur.name}, record {self.cur.records}: "
                             f"the backend wrote {got[i]} where the reference has {want[i]} "
                             f"(token {i} of {len(want)})")
        self.next_exec = want[-1]
        return packets

