"""The GPU circuit: kernels with global memory, where stores cost no tokens.

The burst machine (burst.py) plus a warp loop, global loads and stores. Instructions:

    add  rd, ra, rb|imm      rd = ra + rb (32-bit)
    tid  rd                  rd = global thread id = lanes * warp + lane  (ctaid.x * ntid.x + tid.x)
    ld   rd, [ra + imm]      rd = in[ra + imm]         (word addresses, mod 128)
    st   [ra + imm], rb      out[ra + imm] = rb        (optionally guarded: @p / @!p, per lane)
    setp pd, ra, rb|imm      pd = ra <cmp> rb          (eq ne lt le gt ge, signed or unsigned)
    mad  rd, ra, rb|imm, rc  rd = ra * rb + rc (low 32 bits; only in the multiply stages)
    nop                      padding, so that a mad lands in a multiply stage
    sreg rd, map             rd bit i = bit map_i of the linear thread index t (map_i = 7: 0);
                             imm = map_0 | map_1 << 3 | ...: tid.x / tid.y / ctaid.* of a
                             power-of-two launch are bit fields of t
    max / min rd, ra, rb|imm signed or unsigned (the compare adder picks a side)
    sub  rd, ra, rb|imm      rd = ra - rb
    selp rd, ra, rb|imm, p   rd = p ? ra : rb, per lane (p: the guard field, an in-block setp)
    mulhi rd, ra, rb|imm     rd = (ra * rb) >> 32, unsigned (multiply stages; shifts right too)
    and / or / xor rd, ra, rb|imm   bitwise (only in the logic stages)
    atom [ra + imm], rb|imm  mem[a] += v, per lane, no result (red.add; only in the atomic
                             stages, and last in its block). Lanes that hit the same word are
                             combined: the first such active lane writes old + all their values
                             back through MEM records the circuit emits after the live-outs
                             (MEM(a) VAL VAL per lane); the other lanes write 0 to SINK.
    stw  [ra + imm], rb      a store whose value later records can load (shared memory): written
                             back like atom, every active lane in lane order; last in its block
                             and after no load in it (so a lane's write cannot change what a
                             later lane, recomputed after it, would store)

Immediates are 32-bit. Predicates p0..p7 live inside a block only: the driver never makes one a
live-out, and a guard reads the latest setp of the same block that wrote its predicate.

Global memory is split as for a kernel whose inputs are const __restrict__: `in` is loaded by
the host and read-only during the launch; `out` is write-only. A store is a write into a memory
head *above* the stage that computes it, at the EXEC token, in the same forward pass: nothing
later in the pass needs it, and the host reads `out` from the model's recurrent state. So a
warp that only stores costs one token. Registers are per warp; warps run one after another.

Records: EXEC(b, w) then 16 VALs per live-out, then EXEC(b + 1, w), or EXEC(0, w + 1) after
the kernel's last block, or the target of the block's branch (see MBlock). Host records:
PROG(b, s) VAL VAL and PROGI(b, s) VAL VAL (the fields and the immediate of one instruction, two
ROM heads), LIVE(b) VAL VAL (live-outs and branch), MEM(a) VAL VAL (in[a]),
LOAD(r) VAL x 16.
"""

from dataclasses import dataclass

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, Chan, Target
from chipformer.q35.prims import Circ, from_bits

from qgpu.circuit import writeonly
from qgpu.circuit.blocks import split
from qgpu.circuit.burst import add_nib, mul_nib_wide, nib_bits
from qgpu.circuit.gadgets import counter, thead, thermo

LANES = 8
NREG = 16
S = 5
LMAX = 4  # live-outs per block
BBITS = 5  # blocks per kernel
WBITS = 5  # warps per launch (and threads <= 2^TBITS)
ABITS = 7  # word addresses
CHUNKS = 2 * LANES
GROUPS = LANES // 2
NIB = 8
OPS = ("add", "tid", "ld", "st", "setp", "mad", "nop", "sreg", "max", "min", "sub", "selp",
       "mulhi", "and", "or", "xor", "atom", "stw")
MUL_OPS, LOGIC_OPS, ATOM_OPS = {"mad", "mulhi"}, {"and", "or", "xor"}, {"atom", "stw"}
WPB_MAX = 8  # warps per thread block when it has barriers (per-warp register rows: w mod 8)
SINK = (1 << ABITS) - 1  # a word the driver never gives a buffer: absorbs atomics' extra writes
TBITS = 7  # bits of the linear thread index t = lanes * warp + lane
CMPS = ("eq", "ne", "lt", "le", "gt", "ge")
NPRED = 8
NCODE = 17  # output code: the predicted token's id bits (every chip token is < 2^17)
CODE_AMP = 128  # the input code in a chip token's embedding
KAPPA = 2.0**-20  # the output code after the exit: tiny, so it never disturbs Qwen's norms

VAL_MAX = 1 << 16
EXEC0 = VAL_MAX
XBITS = BBITS + WBITS + 1  # EXEC(b, w, phase): block bits, warp bits, a phase-start bit
LOAD0 = EXEC0 + (1 << XBITS)
PROG0 = LOAD0 + 256
PROGI0 = PROG0 + (8 << BBITS)
LIVE0 = PROGI0 + (8 << BBITS)
MEM0 = LIVE0 + 256


def EXEC(b: int, w: int, phase: bool = False) -> int:
    """phase: this record starts a phase (the warp's resume block R is set to b): the first
    record of a launch, of a thread block, and after a barrier that every warp has reached."""
    return EXEC0 + (int(phase) << (BBITS + WBITS)) + (w << BBITS) + b


def LOAD(r: int) -> int:
    return LOAD0 + r


def PROG(b: int, s: int) -> int:
    return PROG0 + 8 * b + s


def PROGI(b: int, s: int) -> int:
    return PROGI0 + 8 * b + s


def LIVE(b: int) -> int:
    return LIVE0 + b


def MEM(a: int) -> int:
    return MEM0 + a


# In a merged checkpoint every Qwen text token reaches the circuit as this one token: its
# embedding row (shared by Qwen's whole vocabulary) carries these features
TEXT = MEM0 + (1 << ABITS)
CHIP_VOCAB = TEXT  # chip tokens are 0 .. CHIP_VOCAB - 1
TAU_TEXT = 2.0**-40


def kind(t: int) -> str:
    if t == TEXT:
        return "text"
    for name, lo in (("mem", MEM0), ("live", LIVE0), ("prog", PROG0), ("load", LOAD0),
                     ("exec", EXEC0)):
        if t >= lo:
            return name
    return "val"


@dataclass(frozen=True)
class KInstr:
    op: str
    rd: int = 0
    ra: int = 0
    rb: int = 0
    imm: int = 0
    pd: int = 0  # setp: predicate written
    cmp: str = "eq"
    signed: bool = False
    useimm: bool = False  # setp / add / mad: imm instead of rb
    guard: int | None = None  # st: predicate that enables each lane
    gneg: bool = False  # st: @!p rather than @p
    rc: int = 0  # mad: the addend register

    def __post_init__(self):
        assert self.op in OPS and self.cmp in CMPS
        assert self.guard is None or self.op in ("st", "selp", "atom", "stw"), \
            "only st / selp / atom / stw take a guard"
        assert self.op != "selp" or self.guard is not None, "selp needs its predicate"
        assert self.op != "stw" or not self.useimm, "stw stores rb (imm is the address offset)"

    def rw(self):
        """(register written, registers read); predicate p is register -1 - p. Register numbers
        may be virtual (the driver allocates them); encode() checks they fit."""
        g = () if self.guard is None else (-1 - self.guard,)
        two = (self.ra,) if self.useimm else (self.ra, self.rb)
        return {"add": (self.rd, two), "max": (self.rd, two), "min": (self.rd, two),
                "selp": (self.rd, two + g), "mulhi": (self.rd, two), "and": (self.rd, two),
                "or": (self.rd, two), "xor": (self.rd, two),
                "sub": (self.rd, two), "sreg": (self.rd, ()),
                "tid": (self.rd, ()),
                "ld": (self.rd, (self.ra,)), "st": (None, (self.ra, self.rb) + g),
                "atom": (None, ((self.ra,) if self.useimm else (self.ra, self.rb)) + g),
                "stw": (None, (self.ra, self.rb) + g),
                "setp": (-1 - self.pd, (self.ra,) if self.useimm else (self.ra, self.rb)),
                "mad": (self.rd, (self.ra, self.rc) if self.useimm
                        else (self.ra, self.rb, self.rc)),
                "nop": (None, ())}[self.op]

    def encode(self) -> tuple[int, int, int, int]:
        """Two 16-bit payload pairs: the fields (PROG row), then the 32-bit immediate. A mad
        keeps rc where a setp keeps cmp and signed; a nop is a row with valid = 0."""
        assert all(0 <= r < NREG for r in (self.rd, self.ra, self.rb, self.rc)), self
        assert 0 <= self.pd < NPRED and (self.guard or 0) < NPRED
        mid = (self.rc if self.op == "mad" else CMPS.index(self.cmp) | self.signed << 3)
        f = (OPS.index(self.op) | self.rd << 5 | self.ra << 9 | self.rb << 13
             | (self.op != "nop") << 17 | self.pd << 18 | mid << 21
             | self.useimm << 25 | (self.guard is not None) << 26 | (self.guard or 0) << 27
             | self.gneg << 30)
        i = self.imm & 0xFFFFFFFF
        return f & 0xFFFF, f >> 16, i & 0xFFFF, i >> 16


def placer(stages: int, mul_stages=frozenset(), logic_stages=frozenset(),
           atom_stages=frozenset()):
    """Stage slots for a candidate block: instructions in order, a mad / mulhi only in a
    multiply stage and and / or / xor only in a logic stage (no-ops pad up to the next one), or
    None when the block does not fit."""
    def place(instrs):
        slots = []
        for ins in instrs:
            if slots and slots[-1].op in ATOM_OPS:
                return None  # a write-back ends its block: later loads must see its writes
            if ins.op == "stw" and any(x.op == "ld" for x in slots):
                return None  # its record must not recompute loads of memory it is changing
            need = (mul_stages if ins.op in MUL_OPS else logic_stages if ins.op in LOGIC_OPS
                    else atom_stages if ins.op in ATOM_OPS else None)
            if need is not None:
                while len(slots) < stages and len(slots) not in need:
                    slots.append(KInstr("nop"))
            if len(slots) >= stages:
                return None
            slots.append(ins)
        return slots
    return place


def compare(a: int, b: int, cmp: str, signed_: bool) -> bool:
    if signed_:
        a, b = (x - (1 << 32) if x >> 31 else x for x in (a, b))
    return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b,
            "ge": a >= b}[cmp]


def chunks(values: list[int]) -> list[int]:
    return [(values[k // 2] >> 16 * (k % 2)) & 0xFFFF for k in range(2 * len(values))]


# --- reference -----------------------------------------------------------------------------


@dataclass
class MBlock:
    """A machine block: stage slots, live-outs in emission order, and where control goes next:
    ("next",) to b + 1, ("jump", t), ("cond", stage, neg, t[, any]): to t when the predicate from
    that stage's setp is true (false if neg), else b + 1: lane 0's predicate (a uniform branch),
    or with any=True whether any lane's is (a loop that runs while some lane still needs it);
    ("end",): the warp is done; ("bar",): a barrier: the next warp of the thread block resumes
    at the phase's start R, and after the block's last warp every warp continues at b + 1."""

    instrs: list[KInstr]
    live: list[tuple[int, int]]
    branch: tuple = ("next",)
    wpb: int = 1  # warps per thread block (a launch constant; barriers need it)


BRANCHES = ("next", "jump", "cond", "end", "bar")


@dataclass
class Trace:
    tokens: list[int]
    owned: list[int]
    out: dict[int, int]  # word address -> value stored
    blocks: list[MBlock]
    mem: dict[int, int] = None  # global memory at the end (atomics update it)


def trace(program: list[KInstr], mem: dict[int, int], warps: int, outputs: set[int] = frozenset(),
          stages: int = S, lmax: int = LMAX, lanes: int = LANES,
          mul_stages=frozenset(), logic_stages=frozenset(), atom_stages=frozenset()) -> Trace:
    """A straight-line program: blocks formed by `split`, the last one ends the warp."""
    blocks = [MBlock(b.instrs, b.live) for b in
              split(program, set(outputs), stages, lmax,
                    placer(stages, mul_stages, logic_stages, atom_stages))]
    blocks[-1].branch = ("end",)
    return run_blocks(blocks, mem, warps, lanes)


LV_ANY = 22 + BBITS  # LIVE payload: fields after the branch target


def encode_live(blk: MBlock) -> tuple[int, int, int]:
    kind = BRANCHES.index(blk.branch[0])
    stage, neg, target = (blk.branch[1:4] if kind == 2 else
                          (0, 0, blk.branch[1]) if kind == 1 else (0, 0, 0))
    anyl = kind == 2 and len(blk.branch) > 4 and blk.branch[4]
    atoms = [s for s, ins in enumerate(blk.instrs) if ins.op in ATOM_OPS]
    assert len(atoms) <= 1, "one atomic per block"
    assert blk.wpb in (1, 2, 4, 8)
    v = (len(blk.live) | sum(s << 3 + 3 * i for i, (s, _) in enumerate(blk.live))
         | kind << 15 | stage << 18 | int(neg) << 21 | target << 22
         | int(anyl) << LV_ANY | int(bool(atoms)) << LV_ANY + 1
         | (atoms[0] if atoms else 0) << LV_ANY + 2 | (blk.wpb.bit_length() - 1) << LV_ANY + 5)
    return v & 0xFFFF, v >> 16 & 0xFFFF, v >> 32


def program_tokens(blocks: list[MBlock]) -> list[int]:
    """The program image as host records: PROG / PROGI per instruction, LIVE per block."""
    toks: list[int] = []
    for b, blk in enumerate(blocks):
        for s, ins in enumerate(blk.instrs):
            f0, f1, i0, i1 = ins.encode()
            toks += [PROG(b, s), f0, f1, PROGI(b, s), i0, i1]
        toks += [LIVE(b), *encode_live(blk)]
    return toks


def run_blocks(blocks: list[MBlock], mem: dict[int, int], warps: int, lanes: int = LANES,
               max_records: int = 100000) -> Trace:
    """Reference execution of machine blocks: the token stream the circuit must produce."""
    toks = program_tokens(blocks)
    for a, v in mem.items():
        toks += [MEM(a), v & 0xFFFF, v >> 16 & 0xFFFF]
    out: dict[int, int] = {}
    mem = dict(mem)  # global memory: loads read it, atomics write it
    owned = []
    for part, rec in records(blocks, mem, warps, lanes, max_records, out=out):
        if part != "end":
            owned += list(range(len(toks), len(toks) + len(rec)))
        toks += rec
    return Trace(toks, owned, out, blocks, mem)


def records(blocks: list[MBlock], mem: dict[int, int], warps: int, lanes: int = LANES,
            max_records: int = 100000, out: dict | None = None, apply: bool = True,
            regfile: dict | None = None):
    """A launch, record by record, as the circuit writes it. For each record it yields
    ("exec", [EXEC(b, w)]), which only control flow decides, then ("body", tokens): the
    live-outs' VALs and a write-back's MEM records, computed when the generator is resumed after
    the EXEC token, i.e. from memory as it is then. Finally ("end", [EXEC(0, warps)]).

    `mem` is global memory (a dict, word address -> uint32): loads read it. With `apply`, a
    record's write-back updates it here; otherwise the caller applies the MEM records it feeds
    (a device whose stores to a peer aperture leave over a link). Between records the caller may
    write `mem` too: that is where a peer's stores arrive. `out` collects plain (non write-back)
    stores; `regfile` (per-warp register rows) persists across launches when given."""
    # warps < 2^WBITS: the end-of-launch token EXEC(0, warps) must still be one
    assert len(blocks) <= 1 << BBITS and warps < 1 << WBITS and warps * lanes <= 1 << TBITS
    for blk in blocks:
        assert all(0 <= rd < NREG for _, rd in blk.live), "a predicate cannot be a live-out"
        assert len(blk.live) <= LMAX
    out = {} if out is None else out
    # per-warp register rows (w mod 8), as in the circuit; they persist across warps
    if regfile is None:
        regfile = {}
    for wl in range(WPB_MAX):
        regfile.setdefault(wl, [[0] * lanes for _ in range(NREG)])
    n_records = 0
    w, b, phase_start, R = 0, 0, True, 0
    while w < warps:
        n_records += 1
        assert n_records <= max_records, "no end after max_records blocks"
        regs = regfile[w % WPB_MAX]
        if phase_start:
            R = b
        blk = blocks[b]
        yield "exec", [EXEC(b, w, phase_start)]
        phase_start = False
        # as the circuit: a block reads committed registers, or a result an earlier stage
        # of the same block wrote (forwarding); only its live-outs are committed
        work = [list(r) for r in regs]
        results, preds = [], {}  # predicates of this block
        stage_pred = {}
        atom_writes = []
        for s, ins in enumerate(blk.instrs):
            res = None
            if ins.op == "setp":
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                preds[ins.pd] = [compare(x, y, ins.cmp, ins.signed)
                                 for x, y in zip(work[ins.ra], rhs)]
                stage_pred[s] = preds[ins.pd]
            elif ins.op == "mad":
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                res = [(x * y + z) & 0xFFFFFFFF
                       for x, y, z in zip(work[ins.ra], rhs, work[ins.rc])]
            elif ins.op == "nop":
                pass
            elif ins.op in ("mulhi", "and", "or", "xor"):
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                f = {"mulhi": lambda x, y: x * y >> 32, "and": int.__and__,
                     "or": int.__or__, "xor": int.__xor__}[ins.op]
                res = [f(x, y) & 0xFFFFFFFF for x, y in zip(work[ins.ra], rhs)]
            elif ins.op == "selp":
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                res = [x if p != ins.gneg else y
                       for x, y, p in zip(work[ins.ra], rhs, preds[ins.guard])]
            elif ins.op in ("max", "min", "sub"):
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                if ins.op == "sub":
                    res = [(x - y) & 0xFFFFFFFF for x, y in zip(work[ins.ra], rhs)]
                else:
                    lt = [compare(x, y, "lt", ins.signed) for x, y in zip(work[ins.ra], rhs)]
                    res = [(y if t else x) if ins.op == "max" else (x if t else y)
                           for x, y, t in zip(work[ins.ra], rhs, lt)]
            elif ins.op == "sreg":
                t = [lanes * w + L for L in range(lanes)]
                res = [sum((tt >> (ins.imm >> 3 * i & 7) & 1) << i for i in range(TBITS)
                           if ins.imm >> 3 * i & 7 != 7) for tt in t]
            elif ins.op == "add":
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                res = [(x + y) & 0xFFFFFFFF for x, y in zip(work[ins.ra], rhs)]
            elif ins.op == "tid":
                res = [lanes * w + L for L in range(lanes)]
            elif ins.op == "ld":
                res = [mem.get((x + ins.imm) % (1 << ABITS), 0) & 0xFFFFFFFF
                       for x in work[ins.ra]]
            elif ins.op == "atom":
                rhs = [ins.imm & 0xFFFFFFFF] * lanes if ins.useimm else work[ins.rb]
                on = ([True] * lanes if ins.guard is None
                      else [p != ins.gneg for p in preds[ins.guard]])
                addr = [(x + ins.imm) % (1 << ABITS) for x in work[ins.ra]]
                atom_writes = []
                for L in range(lanes):
                    first = on[L] and not any(on[q] and addr[q] == addr[L] for q in range(L))
                    total = sum(rhs[q] for q in range(lanes) if on[q] and addr[q] == addr[L])
                    final = (mem.get(addr[L], 0) + total) & 0xFFFFFFFF
                    atom_writes.append((addr[L], final) if first else (SINK, 0))
            elif ins.op == "stw":
                on = ([True] * lanes if ins.guard is None
                      else [p != ins.gneg for p in preds[ins.guard]])
                atom_writes = [((x + ins.imm) % (1 << ABITS), y) if e else (SINK, 0)
                               for x, y, e in zip(work[ins.ra], work[ins.rb], on)]
            else:
                on = ([True] * lanes if ins.guard is None
                      else [p != ins.gneg for p in preds[ins.guard]])
                for x, y, e in zip(work[ins.ra], work[ins.rb], on):
                    if e:
                        out[(x + ins.imm) % (1 << ABITS)] = y
            results.append(res)
            if res is not None:
                work[ins.rd] = res
        for s_, rd in blk.live:
            regs[rd] = results[s_]
        toks = []
        for s, _ in blk.live:
            toks += chunks(results[s])
        if any(ins.op in ATOM_OPS for ins in blk.instrs):
            for a, v in atom_writes:  # the write-back: MEM(a) VAL VAL per lane
                toks += [MEM(a), v & 0xFFFF, v >> 16]
                if apply:
                    mem[a] = v
        yield "body", toks
        kind = blk.branch[0]
        last_warp = w % blk.wpb == blk.wpb - 1
        if kind in ("end", "bar"):
            if kind == "bar" and last_warp:  # every warp got here: on to b + 1
                w, b, phase_start = w - (blk.wpb - 1), b + 1, True
            elif kind == "end" and last_warp:  # the next thread block
                w, b, phase_start = w + 1, 0, True
            else:  # the next warp of the block resumes where this phase started
                w, b = w + 1, R
            continue
        if kind == "next":
            b += 1
        elif kind == "jump":
            b = blk.branch[1]
        else:
            _, s, neg, target = blk.branch[:4]
            p = stage_pred[s]
            if len(blk.branch) > 4 and blk.branch[4]:  # any lane (then neg flips it)
                taken = any(p) != neg
            else:
                assert len(set(p)) == 1, f"divergent branch in block {b}: {p}"
                taken = p[0] != neg
            b = target if taken else b + 1
    yield "end", [EXEC(0, warps, phase_start)]


# --- circuit -------------------------------------------------------------------------------


class Kernel(Circ):
    def __init__(self, target: Target | None = None, lanes: int = LANES, stages: int = S,
                 mul_stages=frozenset(), logic_stages=frozenset(),
                 atom_stages=frozenset()) -> None:
        super().__init__(target or Target())
        c, one = self, self.one
        assert lanes in (2, 4, 8) and 1 <= stages <= 8
        assert all(0 <= m < stages for m in set(mul_stages) | set(logic_stages))
        self.lanes, self.stages, self.mul_stages = lanes, stages, frozenset(mul_stages)
        self.logic_stages = frozenset(logic_stages)
        self.atom_stages = frozenset(atom_stages)
        assert all(0 <= a < stages for a in self.atom_stages)
        groups, nchunk = lanes // 2, 2 * lanes

        def feat(name, fn):
            return c.token_feature(name, fn, full=True)

        def lag(e, k):
            return Chan(e, tuple(1.0 if i == k else 0.0 for i in range(4)))

        is_val = feat("is_val", lambda t: int(kind(t) == "val"))
        is_exec = feat("is_exec", lambda t: int(kind(t) == "exec"))
        is_load = feat("is_load", lambda t: int(kind(t) == "load"))
        is_mem = feat("is_mem", lambda t: int(kind(t) == "mem"))
        # a record starts at EXEC / LOAD / PROG / LIVE; MEM tokens also come inside an EXEC
        # record (an atomic's write-back), so they do not start one
        start = one - is_val - is_mem
        pay = [feat(f"pay{b}", lambda t, b=b: int(kind(t) == "val" and t >> b & 1))
               for b in range(16)]
        eb = [feat(f"eb{b}", lambda t, b=b: int(kind(t) == "exec" and (t - EXEC0) >> b & 1))
              for b in range(BBITS + WBITS)]
        ldr = [feat(f"ldr{b}", lambda t, b=b: int(kind(t) == "load" and (t - LOAD0) >> b & 1))
               for b in range(4)]
        nblk, nadr = 1 << BBITS, 1 << ABITS

        # --- record, position ---------------------------------------------------------
        (rec,) = thead(c, "rec", {0: start}, {0: one},
                       [{"data": {2: is_exec, 3: is_load,
                                  **{4 + b: eb[b] for b in range(BBITS + WBITS)},
                                  **{4 + BBITS + WBITS + b: ldr[b] for b in range(4)}},
                         "ref": one}])
        r_exec, r_load = rec[0], rec[1]
        r_blk, r_warp = rec[2:2 + BBITS], rec[2 + BBITS:2 + BBITS + WBITS]
        r_ldr = rec[2 + BBITS + WBITS:6 + BBITS + WBITS]
        # per-warp registers: row 16 * (w mod 8) + r, so warps can take turns at a barrier
        wl_oh = c.onehot_of_bits(r_warp[:3], "wlocal")
        # R, the block where the current phase started: written at EXEC tokens that start one
        phase_tok = feat("phase", lambda t: int(kind(t) == "exec" and (t - EXEC0) >> (BBITS + WBITS) & 1))
        (r_phase,) = thead(c, "phase", {0: phase_tok}, {0: one},
                           [{"data": {2 + b: eb[b] for b in range(BBITS)}, "ref": one}])
        jmax = nchunk * LMAX + 2 * lanes  # live-out chunks, then an atomic's write-back
        jt = thermo(c, counter(c, "j", start, is_val), start, jmax + 1, "j")

        def j_is(k: int) -> Expr:
            return (one if k == 0 else jt[k - 1]) - jt[k]

        at_exec = c.and_([r_exec, is_exec], "at_exec")  # the record's first token

        # --- fetch ---------------------------------------------------------------------
        blk_oh = c.onehot_of_bits(r_blk, "blk")
        q = {b: c.and_([r_exec, blk_oh[b]], f"blkq{b}") for b in range(nblk)}
        stage = []
        for s in range(stages):
            rows = []
            for tok, nbits, nm in ((PROG, 31, "rom"), (PROGI, 32, "romi")):
                keys = {b: lag(feat(f"{nm}{b},{s}", lambda t, b=b, s=s, tok=tok:
                                    int(t == tok(b, s))), 2) for b in range(nblk)}
                data = {2 + i: lag(pay[i], 1) for i in range(16)}
                data |= {18 + i: lag(pay[i], 0) for i in range(nbits - 16)}
                (row,) = thead(c, f"{nm}{s}", keys, q, [{"data": data, "ref": one}])
                rows.append(row)
            row, irow = rows
            st = {"op": c.onehot_of_bits(row[0:5], f"s{s}.op", len(OPS)), "rd": row[5:9],
                  "ra": row[9:13], "rb": row[13:17], "valid": row[17], "pd": row[18:21],
                  "cmp": c.onehot_of_bits(row[21:24], f"s{s}.cmp", len(CMPS)),
                  "signed": row[24], "useimm": row[25], "gvalid": row[26], "g": row[27:30],
                  "gneg": row[30], "imm": [from_bits(irow[4 * k:4 * k + 4]) for k in range(NIB)],
                  "rc": row[21:25], "ibits": irow}
            for f in ("rd", "ra", "rb", "pd", "g") + (("rc",) if s in mul_stages else ()):
                st[f + "_oh"] = c.onehot_of_bits(st[f], f"s{s}.{f}")
            op_st, op_setp = st["op"][3], st["op"][4]
            st["writes"] = c.and_([st["valid"], one - op_st - op_setp - st["op"][OPS.index("atom")]
                                   - st["op"][OPS.index("stw")]], f"s{s}.writes")
            st["pwrites"] = c.and_([st["valid"], op_setp], f"s{s}.pwrites")
            stage.append(st)
        keys = {b: lag(feat(f"live{b}", lambda t, b=b: int(t == LIVE(b))), 3)
                for b in range(nblk)}
        ldata = {2 + i: lag(pay[i], 2) for i in range(16)}
        ldata |= {18 + i: lag(pay[i], 1) for i in range(16)}
        ldata |= {34 + i: lag(pay[i], 0) for i in range(LV_ANY + 7 - 32)}
        (lv,) = thead(c, "livetab", keys, q, [{"data": ldata, "ref": one}])
        nl_oh = c.onehot_of_bits(lv[0:3], "nlive", LMAX + 1)
        src_oh = [c.onehot_of_bits(lv[3 + 3 * i:6 + 3 * i], f"src{i}", stages)
                  for i in range(LMAX)]
        br_next, br_jump, br_cond, br_end, br_bar = c.onehot_of_bits(lv[15:18], "branch",
                                                                     len(BRANCHES))
        br_stage = c.onehot_of_bits(lv[18:21], "bstage", stages)
        br_neg, br_target, br_any = lv[21], lv[22:22 + BBITS], lv[LV_ANY]
        has_atom = lv[LV_ANY + 1]  # the block ends with a write-back: it follows the live-outs
        atom_stage = c.onehot_of_bits(lv[LV_ANY + 2:LV_ANY + 5], "astage", stages)
        wpb_oh = c.onehot_of_bits(lv[LV_ANY + 5:LV_ANY + 7], "wpb")  # log2 warps per block

        # --- register commits (as in burst.py) ------------------------------------------
        # only for live-outs the block has: an atomic's write-back follows them in the record
        live_has = [sum((nl_oh[n] for n in range(i + 1, LMAX + 1)), Expr()) for i in range(LMAX)]
        rdl = [[c.node([t for s in range(stages)
                        for t in c.t_and([src_oh[i][s], stage[s]["rd_oh"][r]])], f"rdl{i}.{r}")
                for r in range(NREG)] for i in range(LMAX)]
        ldr_oh = c.onehot_of_bits(r_ldr, "ldr")
        wkeys = []
        for h in range(groups):
            ks = {}
            for r in range(NREG):
                terms = c.t_and([r_load, is_val, j_is(4 * h + 4), ldr_oh[r]])
                for i in range(LMAX):
                    terms += c.t_and([r_exec, is_val, j_is(nchunk * i + 4 * h + 4), rdl[i][r],
                                      live_has[i]])
                ks[r] = c.node(terms, f"wkey{h}.{r}")
            wkeys.append(ks)
        rdata = {}
        for lg in range(4):
            pos = 3 - lg
            for b in range(16):
                rdata[2 + 32 * (pos // 2) + 16 * (pos % 2) + b] = lag(pay[b], lg)

        # commits and reads address row 16 * (w mod 8) + r: the sum of two one-hots is exactly
        # 1 more than the rest only on that row
        wkeys = [{16 * w + r: ks[r] + wl_oh[w] - 1 for w in range(WPB_MAX) for r in range(NREG)}
                 for ks in wkeys]

        def read_regs(port_oh, name):
            qq = {16 * w + r: port_oh[r] + wl_oh[w] + r_exec - 2
                  for w in range(WPB_MAX) for r in range(NREG)}
            per_lane = [None] * lanes
            for h in range(groups):
                (row,) = thead(c, f"{name}{h}", wkeys[h], qq, [{"data": rdata, "ref": one}])
                per_lane[2 * h], per_lane[2 * h + 1] = row[0:32], row[32:64]
            return [[from_bits(bits[4 * k:4 * k + 4]) for k in range(NIB)] for bits in per_lane]

        mem_keys = {a: lag(feat(f"mem{a}", lambda t, a=a: int(t == MEM(a))), 2)
                    for a in range(nadr)}
        mem_data = {2 + i: lag(pay[i], 1) for i in range(16)}
        mem_data |= {18 + i: lag(pay[i], 0) for i in range(16)}

        # --- stages ----------------------------------------------------------------------
        self.vals, self.preds, self.stores = [], [], []  # stores[s][lane]: that lane's `out`
        self.atoms = {}  # atomic stage -> per lane (write-back address bits, final nibbles)
        for s, st in enumerate(stage):
            (op_add, op_tid, op_ld, op_st, op_setp, op_mad, _, op_sreg, op_max, op_min,
             op_sub, op_selp, op_mulhi, op_and, op_or, op_xor, op_atom, op_stw) = st["op"]
            lane_atom = []  # per lane, for the atomic unit: (address groups, bits, active, old)
            op_stw = st["op"][OPS.index("stw")]
            cmpmode = op_setp + op_max + op_min + op_sub  # the adder computes ra - rhs
            # sreg: output bit i takes bit map_i of t (map_i = 7: none)
            smap = [c.onehot_of_bits(st["ibits"][3 * i:3 * i + 3], f"s{s}.smap{i}")
                    for i in range(TBITS)]
            # the guard: the latest earlier setp of this block that wrote predicate g
            peq = [c.node([t for p in range(NPRED) for t in
                           c.t_and([stage[s2]["pwrites"], stage[s2]["pd_oh"][p], st["g_oh"][p]])],
                          f"s{s}.geq{s2}") for s2 in range(s)]
            gwin = [[peq[s2]] + [one - e for e in peq[s2 + 1:]] for s2 in range(s)]
            ops = []
            for port in ("ra", "rb") + (("rc",) if s in self.mul_stages else ()):
                committed = read_regs(st[port + "_oh"], f"s{s}{port}")
                eq = [c.node([t for r in range(NREG) for t in
                              c.t_and([stage[s2]["writes"], stage[s2]["rd_oh"][r],
                                       st[port + "_oh"][r]])], f"s{s}{port}.eq{s2}")
                      for s2 in range(s)]
                win = [[eq[s2]] + [one - e for e in eq[s2 + 1:]] for s2 in range(s)]
                none = [one - e for e in eq]
                per_port = []
                for L in range(lanes):
                    if not s:
                        per_port.append(committed[L])
                        continue
                    nibs = []
                    for k in range(NIB):
                        terms = c.t_and(none + [r_exec], a=committed[L][k])
                        for s2 in range(s):
                            terms += c.t_and(win[s2], a=self.vals[s2][L][k])
                        nibs.append(c.node(terms, f"s{s}{port}.L{L}.n{k}"))
                    per_port.append(nibs)
                ops.append(per_port)
            A, B = ops[:2]
            vals, preds, stores = [], [], []
            imm, useimm = st["imm"], st["useimm"]
            for L in range(lanes):
                nm = f"s{s}.L{L}"
                # second adder input: rb for add, the immediate for ld / st, and for setp the
                # complement of rb or imm (with carry-in 1: ra - rhs)
                b2 = [c.node(c.t_and([op_add, one - useimm], a=B[L][k])
                             + c.t_and([op_add, useimm], a=imm[k])
                             + c.t_and([op_ld + op_st + op_atom + op_stw], a=imm[k])
                             + c.t_and([cmpmode, useimm], a=15 * one - imm[k])
                             + c.t_and([cmpmode, one - useimm], a=15 * one - B[L][k]),
                             f"{nm}.b2n{k}") for k in range(NIB)]
                total, cout = add_nib(c, A[L], b2, f"{nm}.add", carry_in=cmpmode,
                                      with_cout=True)
                rhs = [c.node(c.t_and([useimm], a=imm[k]) + c.t_and([one - useimm], a=B[L][k]),
                              f"{nm}.rhs{k}") for k in range(NIB)]
                # setp: carry out = [ra >= rhs] unsigned; the sign nibbles fix signed compares
                zero = [c.node(c.t_lt(total[k], 1, 0, 15), f"{nm}.z{k}") for k in range(NIB)]
                eq = c.and_(zero, f"{nm}.eq")
                sa = c.node(c.t_ge(A[L][7], 8, 0, 15), f"{nm}.sa")
                sb = c.node(c.t_ge(rhs[7], 8, 0, 15), f"{nm}.sb")
                sgn = st["signed"]
                lt = c.node(c.t_and([one - sgn, one - cout]) + c.t_and([sgn, sa, one - sb])
                            + c.t_and([sgn, sa, sb, one - cout])
                            + c.t_and([sgn, one - sa, one - sb, one - cout]), f"{nm}.lt")
                ceq, cne, clt, cle, cgt, cge = st["cmp"]
                pv = [st["pwrites"]]
                preds.append(c.node(c.t_and(pv + [ceq, eq]) + c.t_and(pv + [cne, one - eq])
                                    + c.t_and(pv + [clt, lt]) + c.t_and(pv + [cle, lt])
                                    + c.t_and(pv + [cle, eq])
                                    + c.t_and(pv + [cgt, one - lt, one - eq])
                                    + c.t_and(pv + [cge, one - lt]), f"{nm}.pred"))
                # store enable: no guard, or the guard predicate (negated for @!p)
                gv = (c.node([t for s2 in range(s) for t in
                              c.t_and(gwin[s2] + [self.preds[s2][L]])], f"{nm}.gv")
                      if s else Expr())
                gvalid, gneg = st["gvalid"], st["gneg"]
                enable = c.node(c.t_and([one - gvalid]) + c.t_and([gvalid, gneg, one - gv])
                                + c.t_and([gvalid, one - gneg, gv]), f"{nm}.en")
                # selp's condition: the predicate, negated for @!p
                pick = c.node(c.t_and([one - gneg, gv]) + c.t_and([gneg, one - gv]),
                              f"{nm}.pick")
                abits = nib_bits(c, total[0], f"{nm}.a0") + nib_bits(c, total[1], f"{nm}.a1")[:3]
                addr_bits = abits
                # the address as 2-bit one-hot groups (Qwenputer's store does the same): the
                # key / query form of row a is a plain sum, exactly 1 on row a and <= 0 elsewhere
                agroups = [c.onehot_of_bits(abits[i:i + 2], f"{nm}.ag{i // 2}")
                           for i in range(0, ABITS, 2)]

                def at(a, gate, agroups=agroups):  # [address == a] AND the gate literals
                    lits = [g[a >> 2 * i & (len(g) - 1)] for i, g in enumerate(agroups)] + gate
                    return sum(lits, Expr()) - (len(lits) - 1)

                (ldv,) = thead(c, f"{nm}.in", mem_keys,
                               {a: at(a, [r_exec, op_ld + op_atom]) for a in range(nadr)},
                               [{"data": mem_data, "ref": one}])
                if s == 0 and L == 0:
                    self.mem_head = self.nodes[-1]  # global memory, as the host reads it back
                loaded = [from_bits(ldv[4 * k:4 * k + 4]) for k in range(NIB)]
                if s in self.atom_stages:
                    active = c.and_([op_atom + op_stw, st["valid"], enable], f"{nm}.aact")
                    lane_atom.append((agroups, addr_bits, active, loaded, rhs))
                # tid = lanes * w + L: the lane bits, then the warp bits
                tbits = [one * (L >> i & 1) for i in range(lanes.bit_length() - 1)] + r_warp
                tid = [sum((tbits[4 * k + i] * (1 << i) for i in range(4)
                            if 4 * k + i < len(tbits)), Expr()) for k in range(2)]
                lane_bits = lanes.bit_length() - 1
                sbits = []
                for i in range(TBITS):
                    terms = []
                    for src in range(min(len(tbits), 7)):
                        if src < lane_bits:  # a constant bit of this lane's index
                            terms += c.t_and([smap[i][src]]) if L >> src & 1 else []
                        else:
                            terms += c.t_and([smap[i][src], r_warp[src - lane_bits]])
                    sbits.append(c.node(terms, f"{nm}.sb{i}") if terms else Expr())
                snib = [sum((sbits[4 * k + i] * (1 << i) for i in range(4) if 4 * k + i < TBITS),
                            Expr()) for k in range(2)]
                prod, high = [Expr()] * NIB, [Expr()] * NIB
                if s in self.mul_stages:  # this stage has a 64-bit multiplier
                    # the addend only for mad: mulhi's high half is of the bare product
                    addend = [c.node(c.t_and([op_mad], a=ops[2][L][k]), f"{nm}.addend{k}")
                              for k in range(NIB)]
                    prod, high = mul_nib_wide(c, A[L], rhs, addend, f"{nm}.mul")
                band = [Expr()] * NIB
                if s in self.logic_stages:  # bits of ra and rhs; or and xor follow from and
                    abits = [b for k in range(NIB) for b in nib_bits(c, A[L][k], f"{nm}.la{k}")]
                    bbits = [b for k in range(NIB) for b in nib_bits(c, B[L][k], f"{nm}.lb{k}")]
                    ibits = st["ibits"]
                    band = [sum((c.node(c.t_and([abits[4 * k + i], useimm, ibits[4 * k + i]])
                                        + c.t_and([abits[4 * k + i], one - useimm,
                                                   bbits[4 * k + i]]),
                                        f"{nm}.and{4 * k + i}") * (1 << i) for i in range(4)),
                                Expr()) for k in range(NIB)]
                vals.append([c.node(c.t_and([op_add + op_sub], a=total[k])
                                    + (c.t_and([op_tid], a=tid[k]) if k < 2 else [])
                                    + (c.t_and([op_sreg], a=snib[k]) if k < 2 else [])
                                    + c.t_and([op_ld], a=loaded[k])
                                    + c.t_and([op_max, lt], a=rhs[k])
                                    + c.t_and([op_max, one - lt], a=A[L][k])
                                    + c.t_and([op_min, lt], a=A[L][k])
                                    + c.t_and([op_min, one - lt], a=rhs[k])
                                    + c.t_and([op_selp, pick], a=A[L][k])
                                    + c.t_and([op_selp, one - pick], a=rhs[k])
                                    + (c.t_and([op_mad], a=prod[k]) if prod[k].terms else [])
                                    + (c.t_and([op_mulhi], a=high[k]) if high[k].terms else [])
                                    + (c.t_and([op_and], a=band[k])
                                       + c.t_and([op_or], a=A[L][k] + rhs[k] - band[k])
                                       + c.t_and([op_xor], a=A[L][k] + rhs[k] - 2 * band[k])
                                       if band[k].terms else []),
                                    f"{nm}.v{k}")
                             for k in range(NIB)])
                # the store: a write into this lane's `out` head, at the EXEC token only
                # nibble k of the stored word in column 2 + k (a write-only head)
                skeys = {a: at(a, [at_exec, op_st, st["valid"], enable]) for a in range(nadr)}
                thead(c, f"{nm}.out", skeys, {},
                      [{"data": {2 + k: B[L][k] for k in range(NIB)}, "ref": one}])
                stores.append(writeonly.mark(self.nodes[-1]))
            self.vals.append(vals)
            self.preds.append(preds)
            self.stores.append(stores)
            if s in self.atom_stages:
                self.atoms[s] = atomic_unit(c, lane_atom, op_atom, op_stw, f"s{s}.atom")

        # --- next token --------------------------------------------------------------------
        # after the live-outs (j = 16 n), an atomic's write-back: per lane L, MEM(a_L) when
        # j = 16 n + 2 L and the current token is not that MEM, then its two VALs
        pre_mem, lo, hi = [], [], []
        for L in range(lanes if self.atom_stages else 0):
            pre_mem.append(c.node([t for n in range(LMAX + 1) for t in c.t_and(
                [r_exec, has_atom, nl_oh[n], j_is(nchunk * n + 2 * L), one - is_mem])],
                f"atom.mem{L}"))
            lo.append(c.node([t for n in range(LMAX + 1) for t in c.t_and(
                [r_exec, has_atom, nl_oh[n], j_is(nchunk * n + 2 * L), is_mem])], f"atom.lo{L}"))
            hi.append(c.node([t for n in range(LMAX + 1) for t in c.t_and(
                [r_exec, has_atom, nl_oh[n], j_is(nchunk * n + 2 * L + 1)])], f"atom.hi{L}"))
        last = c.node([t for n in range(LMAX + 1)
                       for t in c.t_and([r_exec, nl_oh[n], one - has_atom, j_is(nchunk * n)])
                       + c.t_and([r_exec, nl_oh[n], has_atom, one - is_mem,
                                  j_is(nchunk * n + 2 * lanes)])], "last")
        self.out_exec = last
        self.out_val = c.node([t for n in range(1, LMAX + 1)
                               for t in c.t_and([r_exec, nl_oh[n]], a=one - jt[nchunk * n - 1])]
                              + [t for x in lo + hi for t in c.t_and([x])], "out.val")
        self.out_mem = c.node([t for x in pre_mem for t in c.t_and([x])], "out.mem")
        has = [sum((nl_oh[n] for n in range(i + 1, LMAX + 1)), Expr()) for i in range(LMAX)]
        slots = []
        for n in range(4):
            terms = []
            for i in range(LMAX):
                for k in range(nchunk):
                    for s in range(stages):
                        terms += c.t_and([r_exec, has[i], j_is(nchunk * i + k), src_oh[i][s]],
                                         a=self.vals[s][k // 2][4 * (k % 2) + n])
            for s, per_lane in self.atoms.items():
                for L, (_, final) in enumerate(per_lane):
                    terms += c.t_and([lo[L], atom_stage[s]], a=final[n])
                    terms += c.t_and([hi[L], atom_stage[s]], a=final[4 + n])
            slots.append(c.node(terms, f"slot{n}"))
        # next EXEC: (b + 1, w), (target, w) or, when the warp ends, (0, w + 1). A cond branch
        # follows lane 0's predicate: the driver only emits uniform ones.
        # the branch predicate: lane 0's (uniform branches) or whether any lane's is set
        lane0 = c.node([t for s in range(stages)
                        for t in c.t_and([br_stage[s], one - br_any, self.preds[s][0]])
                        + c.t_ge(sum(self.preds[s], Expr()), 1, 0, lanes,
                                 lits=[br_stage[s], br_any])], "br.p")
        taken_now = c.node(c.t_and([br_cond, one - br_neg, lane0])
                           + c.t_and([br_cond, br_neg, one - lane0]), "br.taken_now")
        # latch it at the EXEC token: later tokens of the record recompute the block after
        # live-outs were committed (a loop counter is both), so only the first token is right
        ((taken,),) = thead(c, "br.latch", {0: at_exec}, {0: r_exec},
                            [{"data": {2: taken_now}, "ref": one}])
        to_target = c.node(c.t_and([br_jump]) + c.t_and([taken]), "br.to_target")
        to_next = c.node(c.t_and([br_next]) + c.t_and([br_cond, one - taken]), "br.to_next")
        cb = [one] + [c.and_(r_blk[:i], f"incb.c{i}") for i in range(1, BBITS)]
        cw = [one] + [c.and_(r_warp[:i], f"incw.c{i}") for i in range(1, WBITS)]
        # the thread block's last warp: its low log2(wpb) warp bits are all 1
        lastw = c.node([t for k in range(4) for t in c.t_and([wpb_oh[k]] + r_warp[:k])],
                       "lastwarp")
        # end / bar: a warp that is not its block's last hands over to the next warp, which
        # resumes at R; after the last one, bar continues at b + 1 (a new phase) and end at 0
        to_inc = c.node(c.t_and([to_next]) + c.t_and([br_bar, lastw]), "nx.inc")
        to_r = c.node(c.t_and([br_bar, one - lastw]) + c.t_and([br_end, one - lastw]), "nx.r")
        nxt = {}
        for i in range(BBITS):  # block bits: b + 1, the target, R, or 0
            nxt[i] = (c.t_and([last, to_inc, r_blk[i], one - cb[i]])
                      + c.t_and([last, to_inc, one - r_blk[i], cb[i]])
                      + c.t_and([last, to_target, br_target[i]])
                      + c.t_and([last, to_r, r_phase[i]]))
        keep = [sum((wpb_oh[k] for k in range(min(i, 3) + 1)), Expr()) for i in range(WBITS)]
        to_wplus = c.node(c.t_and([br_end]) + c.t_and([br_bar, one - lastw]), "nx.wplus")
        stay = br_next + br_jump + br_cond
        for i in range(WBITS):  # warp bits: w, w + 1, or back to the block's first warp
            nxt[BBITS + i] = (c.t_and([last, stay, r_warp[i]])
                              + c.t_and([last, to_wplus, r_warp[i], one - cw[i]])
                              + c.t_and([last, to_wplus, one - r_warp[i], cw[i]])
                              + c.t_and([last, br_bar, lastw, keep[i], r_warp[i]]))
        # the phase-start bit
        nxt[BBITS + WBITS] = c.t_and([last, br_bar, lastw]) + c.t_and([last, br_end, lastw])
        for s, per_lane in self.atoms.items():  # the MEM token's address
            for L, (addr, _) in enumerate(per_lane):
                for i in range(ABITS):
                    nxt.setdefault(i, [])
                    nxt[i] += c.t_and([pre_mem[L], atom_stage[s], addr[i]])
        self.out_bits = []
        for n in range(4):
            bits = nib_bits(c, slots[n], f"out.n{n}")
            for b in range(4):
                bb = 4 * n + b
                self.out_bits.append(c.node(c.t_and([bits[b]]) + nxt.get(bb, []),
                                            f"out.b{bb}"))
        self.output_code()

    def output_code(self) -> None:
        """The next token as an output code the model's own lm_head decodes (Qwenputer's
        scheme): each chip token's embedding carries CODE_AMP * (2 bit - 1) for the NCODE bits of
        its id; the last sublayer clears that and writes KAPPA * (2 bit - 1) of the predicted
        token where the circuit predicts (0 elsewhere). With the final norm's gain on these dims
        and tied embeddings, the predicted token has the largest logit."""
        c, one = self, self.one
        assert MEM0 + (1 << ABITS) <= 1 << NCODE
        f = c.token_feature
        self.code = [f(f"code{i}", lambda t, i=i: 0 if t == TEXT else
                       CODE_AMP * (2 * (t >> i & 1) - 1), full=True) for i in range(NCODE)]
        # mu: 1 on chip tokens (Qwen's layers mute on it); tau: -1 on chip tokens (cancelled at
        # the exit), a tiny value on text, which the final norm amplifies to keep chip tokens
        # out of Qwen's text
        self.mu = f("mu", lambda t: int(t != TEXT), full=True)
        self.is_text = f("is_text", lambda t: int(t == TEXT), full=True)  # the overlay's switch
        self.tau = f("tau", lambda t: TAU_TEXT if t == TEXT else -1, full=True)
        # who predicts, with predicted()'s priority (VAL, then EXEC, then MEM), on chip tokens
        mu = self.mu
        kv = c.and_([self.out_val, mu], "own.val")
        ke = c.and_([self.out_exec, one - self.out_val, mu], "own.exec")
        kinds = [(kv, 0, 16), (ke, EXEC0, XBITS)]
        if self.out_mem.terms:
            kinds.append((c.and_([self.out_mem, one - self.out_val, one - self.out_exec, mu],
                                 "own.mem"), MEM0, ABITS))
        own = sum((k for k, _, _ in kinds), Expr())
        nb = []
        for i in range(NCODE):
            terms = []
            for k, base, width in kinds:
                assert base & ((1 << width) - 1) == 0  # payload bits never carry into the base
                if i < width:
                    terms += c.t_and([k, self.out_bits[i]])
                elif base >> i & 1:
                    terms += c.t_and([k])
            nb.append(c.node(terms, f"next.b{i}"))
        for i, e in enumerate(self.code):  # the input code goes first, so the exit adds onto 0
            c.ffn([(-1, one, e), (1, one, -1 * e)], f"code.clear{i}", into=e)
        top = max(n.level for n in c.nodes)
        for i, e in enumerate(self.code):
            terms = (c.t_and([nb[i]], 2 * KAPPA) if nb[i].terms else []) + c.t_and([own], -KAPPA)
            c.ffn(terms, f"exit.code{i}", into=e, min_level=top)
        c.ffn([(1, mu, one)], "exit.tau", into=self.tau, min_level=top)
        self.next_bits = nb
        self.owner = own
        # what a merged checkpoint keeps after the circuit's layers
        self.keep = [next(iter(e.terms)) for e in self.code + [self.tau, mu]]


def dim(e: Expr) -> int:
    (d,) = e.terms
    return d


def predicted(c: Kernel, m, t: int) -> int | None:
    bits = sum(int(m[t, dim(e)]) << b for b, e in enumerate(c.out_bits) if e.terms)
    if m[t, dim(c.out_val)] == 1:
        return bits
    if m[t, dim(c.out_exec)] == 1:
        return EXEC0 + (bits & ((1 << XBITS) - 1))
    if c.out_mem.terms and m[t, dim(c.out_mem)] == 1:
        return MEM0 + (bits & ((1 << ABITS) - 1))
    return None


def memory(c: Kernel, carry: dict) -> dict[int, int]:
    """Global memory after a run, from one copy of the `in` heads (MEM records, the host's and
    the atomics' write-backs, update every copy alike)."""
    index = {id(n): i for i, n in enumerate(c.nodes)}
    state = carry.get((index[id(c.mem_head)], "S", 0), {})
    return {row: sum(int(vec[2 + i] > 0) << i for i in range(32)) for row, vec in state.items()}


def atomic_unit(c: Circ, lanes_: list, op_atom: Expr, op_stw: Expr, name: str) -> list:
    """The write-back stage. atom: lane L writes to a_L if it is the first active lane with
    that address, the value old(a_L) + the values of all active lanes whose address is a_L;
    any other lane writes 0 to SINK. stw: every active lane writes its value to its address
    (in lane order, so the last one wins), inactive lanes 0 to SINK."""
    one = c.one
    n = len(lanes_)
    eq = {}
    for L in range(n):
        for q in range(L):
            groups = [c.node([t for x in range(len(gl)) for t in c.t_and([gl[x], gq[x]])],
                             f"{name}.eq{L}.{q}.g{g}")
                      for g, (gl, gq) in enumerate(zip(lanes_[L][0], lanes_[q][0]))]
            eq[L, q] = eq[q, L] = c.and_(groups, f"{name}.eq{L}.{q}")
    out = []
    for L, (_, bits, active, old, _) in enumerate(lanes_):
        earlier = [c.and_([lanes_[q][2], eq[L, q]], f"{name}.hit{L}.{q}") for q in range(L)]
        first = (c.node(c.t_lt(sum(earlier, Expr()), 1, 0, L, lits=[active]), f"{name}.first{L}")
                 if earlier else active)
        tot = [c.node(c.t_and([active], a=lanes_[L][4][k])
                      + [t for q in range(n) if q != L
                         for t in c.t_and([lanes_[q][2], eq[L, q]], a=lanes_[q][4][k])],
                      f"{name}.tot{L}.{k}") for k in range(NIB)]
        sums = [old[k] + tot[k] for k in range(NIB)]  # <= 15 + 8 * 15
        q_ = [c.node([t for th in range(1, 9) for t in c.t_ge(sums[k], 16 * th, 0, 135)],
                     f"{name}.q{L}.{k}") for k in range(NIB)]
        final = add_nib(c, [sums[k] - 16 * q_[k] for k in range(NIB)], [Expr()] + q_[:NIB - 1],
                        f"{name}.fin{L}")
        addr = [c.node(c.t_and([op_atom, first, bits[i]]) + c.t_and([op_atom, one - first])
                       + c.t_and([op_stw, active, bits[i]]) + c.t_and([op_stw, one - active]),
                       f"{name}.addr{L}.{i}") for i in range(ABITS)]
        # a sink write carries 0: its value would otherwise depend on memory the first lane
        # has already updated by the time this lane's tokens are recomputed
        value = [c.node(c.t_and([op_atom, first], a=final[k])
                        + c.t_and([op_stw, active], a=lanes_[L][4][k]), f"{name}.val{L}.{k}")
                 for k in range(NIB)]
        out.append((addr, value))
    return out


def stored(c: Kernel, carry: dict) -> dict[int, int]:
    """Word address -> value, read from the `out` heads' recurrent state after a run
    (evaluate(..., carry=...)): what the host reads back from the model's cache."""
    out = {}
    index = {id(n): i for i, n in enumerate(c.nodes)}
    for per_stage in c.stores:
        for node in per_stage:
            state = carry.get((index[id(node)], "S", 0), {})
            for row, vec in state.items():
                out[row] = word(vec[2:2 + NIB] / vec[0])
    return out


def word(nibbles) -> int:
    """A stored word from its 8 nibble columns, each given relative to the row's ballast
    (column 0): nibble = column / ballast * A_REF / A_BIT."""
    out = 0
    for k, x in enumerate(nibbles):
        v = float(x) * A_REF / A_BIT
        assert v == round(v) and 0 <= v < 16, v
        out |= int(v) << 4 * k
    return out
