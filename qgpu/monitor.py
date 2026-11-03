"""What the GPU is doing, decoded from the chip tokens the model writes.

Each record the model writes is EXEC(b, w) (machine block b runs for warp w), then each of the
block's live-outs as two 16-bit chunks per lane (low first), then, for a block with a
write-back (atom / stw), MEM(a), low, high per lane; EXEC(0, warps) ends the launch
(circuit/kernel.py, run_blocks). The monitor follows that stream with the program the host
loaded, and keeps a light per unit: set to 1 when the unit does something, fading each frame.
"""

from dataclasses import dataclass, field

from qgpu.circuit import kernel
from qgpu.circuit.kernel import ATOM_OPS, EXEC0, LOAD0, MEM0, NREG

FADE = 0.55  # a light keeps this much of its brightness each frame


@dataclass
class Program:
    """The launch's shape, as the driver loaded it."""
    lanes: int
    warps: int
    stages: list[list[str]]  # per machine block: the op in each stage slot
    live: list[list[tuple[int, int]]]  # per block: (stage, register) live-outs
    writeback: list[bool]  # per block: does it write memory back (atom / stw)
    memory: list[int]  # word addresses the launch's buffers hold

    @classmethod
    def of(cls, launch, lanes: int) -> "Program":
        return cls(lanes, launch.warps,
                   [[i.op for i in b.instrs] for b in launch.blocks],
                   [list(b.live) for b in launch.blocks],
                   [any(i.op in ATOM_OPS for i in b.instrs) for b in launch.blocks],
                   sorted(launch.mem))


@dataclass
class Monitor:
    prog: Program | None = None
    block: int | None = None
    warp: int | None = None
    done: bool = False
    tokens: int = 0
    records: int = 0
    reg_writes: int = 0
    mem_writes: int = 0
    last: str = ""  # the last thing that happened, in words
    # lights (0..1)
    warps: list[float] = field(default_factory=list)
    blocks: list[float] = field(default_factory=list)
    stages: list[float] = field(default_factory=list)
    regs: list[list[float]] = field(default_factory=list)  # [register][lane]
    mem: list[float] = field(default_factory=lambda: [0.0] * kernel.SINK)
    values: dict = field(default_factory=dict)  # (warp, register, lane) -> last value written
    written: dict = field(default_factory=dict)  # memory word -> last value written back
    _expect: list = field(default_factory=list)  # what the next tokens of the record are
    _addr: int = 0
    _lo: int = 0

    def load(self, prog: Program) -> None:
        """A new launch: the program image and memory are in."""
        self.prog, self.block, self.warp, self.done = prog, None, None, False
        self.tokens = self.records = self.reg_writes = self.mem_writes = 0
        self.warps = [0.0] * prog.warps
        self.blocks = [0.0] * len(prog.stages)
        self.stages = [0.0] * max(len(s) for s in prog.stages)
        self.regs = [[0.0] * prog.lanes for _ in range(NREG)]
        self.mem = [0.0] * kernel.SINK
        self.values, self.written, self._expect = {}, {}, []
        self._exec(0, 0)  # the host's last token starts warp 0 at block 0; the model goes on

    def feed(self, t: int) -> None:
        """One chip token the model wrote (0 .. CHIP_VOCAB - 1)."""
        p = self.prog
        if p is None or self.done:
            return
        self.tokens += 1
        if EXEC0 <= t < LOAD0:
            x = t - EXEC0
            b, w = x & (1 << kernel.BBITS) - 1, x >> kernel.BBITS & (1 << kernel.WBITS) - 1
            if w >= p.warps:
                self.done, self.block, self.warp = True, None, None
                self.last = "launch done"
                return
            self._exec(b, w)
            return
        if not self._expect:
            return
        item = self._expect.pop(0)
        if item[0] == "reg":
            _, rd, L, h = item
            key = (self.warp, rd, L)
            v = self.values.get(key, 0)
            self.values[key] = (v & 0xFFFF0000) | t if h == 0 else (v & 0xFFFF) | t << 16
            self.regs[rd][L] = 1.0
            if h == 1:
                self.reg_writes += 1
                self.last = f"warp {self.warp} lane {L}: r{rd} = {_signed(self.values[key])}"
        elif item[0] == "addr":
            self._addr = t - MEM0
            if self._addr < kernel.SINK:
                self.mem[self._addr] = 1.0
        elif item[0] == "lo":
            self._lo = t
        elif self._addr < kernel.SINK:  # "hi"; inactive lanes write the sink word: no light
            self.written[self._addr] = self._lo | t << 16
            self.mem_writes += 1
            self.last = f"lane {item[1]} writes memory word {self._addr}"

    def _exec(self, b: int, w: int) -> None:
        """EXEC(b, w): warp w runs block b; its record's values follow."""
        p = self.prog
        self.block, self.warp = b, w
        self.records += 1
        self.warps[w] = self.blocks[b] = 1.0
        for s in range(len(p.stages[b])):
            if p.stages[b][s] != "nop":
                self.stages[s] = 1.0
        self._expect = [("reg", rd, L, h) for _, rd in p.live[b]
                        for L in range(p.lanes) for h in (0, 1)]
        if p.writeback[b]:
            self._expect += [(part, L) for L in range(p.lanes) for part in ("addr", "lo", "hi")]
        ops = " ".join(o for o in p.stages[b] if o != "nop")
        self.last = f"warp {w} runs block {b}: {ops}"

    def tick(self) -> None:
        """One frame: every light fades."""
        for lights in (self.warps, self.blocks, self.stages, self.mem):
            for i, v in enumerate(lights):
                lights[i] = v * FADE
        for row in self.regs:
            for i, v in enumerate(row):
                row[i] = v * FADE


def _signed(v: int) -> int:
    return v - (1 << 32) if v >> 31 else v
