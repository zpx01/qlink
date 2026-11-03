"""The driver: lower real PTX to the circuit's machine blocks (circuit/kernel.py) at launch.

NVIDIA's driver compiles PTX to the GPU's machine code when a kernel is loaded; this does the
same for our machine, for the PTX it understands, and raises Unsupported for the rest. It runs
at launch, so kernel parameters are known values and the code is specialized on them.

Values. Each PTX register with one definition is tracked symbolically when it can be:
a constant (`ld.param` scalars, immediates, folded arithmetic), a buffer's base pointer, a
special register, 4 * r (`mul.wide r, 4`) or an address r + base (`add.s64`). Registers with
several definitions (loop variables) and everything computed per lane get a machine register.
Pointers are kept as **word indices**: every buffer sits in one 128-word window, so 64-bit
pointer arithmetic only needs byte offsets / 4 (multiples of 4 are checked).

Shared memory. `.shared` arrays sit in the window after the buffers (thread blocks run one
after another, so one copy serves all); `ld.shared` is a load, `st.shared` a write-through
store (`stw`) that later loads see; nvcc's 32-bit shared addresses (`mov.u32 r, sym`, shl,
add.s32) are pointer arithmetic like clang's 64-bit ones. `bar.sync 0` ends a group with a
"bar" terminator: the warps of a thread block take turns, phase by phase.

Control flow. PTX basic blocks are first laid out so each loop is contiguous, header first
(`relayout`); each becomes one or more machine blocks of up to S stages (blocks.split, with
live-outs from liveness over the CFG). Branches:
- `@p bra` to the final `ret` with a per-lane p, before any other branch: an early exit,
  if-converted: later stores are guarded by @!p, and the setp is recomputed next to each store
  (predicates never cross blocks). Loads and register writes stay unguarded: nothing after the
  exit reads them, and our memory never faults.
- a predicate that is the same on every lane of a warp (it depends only on constants, ctaid,
  ntid and other such registers): a machine branch on lane 0's value.
- a predicate known at launch: folded into a jump or a fall-through.
- anything else is a divergent branch: the whole kernel is predicated (every block runs under a
  per-lane mask; `full_blocks` finds the blocks every lane runs, which need no mask).

Instructions: ld.param, cvta.to.global, mov (registers, immediates, %tid / %ntid / %ctaid .x),
mad.lo(ctaid, ntid, tid) -> tid, mad.lo.s32 -> mad, add.s32 / add.s64 (pointers), mul.wide and
shl.b64 by multiples of 4, setp (eq ne lt le gt ge, s32 / u32 / b32), ld / st .global / .shared
32-bit, atom.add, bar.sync 0, bra, ret (and the ALU ops lower_bits handles).
"""

import dataclasses
from dataclasses import dataclass, field

from qgpu import ptx
from qgpu.circuit.blocks import split
from qgpu.circuit.kernel import BBITS, LMAX, NREG, SINK, TBITS, WBITS, KInstr, MBlock, placer

WINDOW = SINK  # words for buffers: the last one absorbs atomics' extra writes
ZERO = NREG - 1  # a register the driver never writes: reads 0
VZERO = 1 << 20  # its virtual name, before register allocation
SREGS_UNIFORM = {"%ctaid.x", "%ntid.x", "%nctaid.x", "%ctaid.y", "%ntid.y", "%nctaid.y"}
FOLD = {"and": int.__and__, "or": int.__or__, "xor": int.__xor__}
MIRROR = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}
EXIT_P, BRANCH_P, ACTIVE_P, EDGE_P, SEL_P = 7, 6, 5, 4, 3  # predicate numbers the driver uses
# Memory-model qualifiers of ld / st (PTX ISA 8): this machine runs one warp at a time, and a
# load reads memory as the latest record left it, so every access is already sequentially
# consistent: volatile, relaxed, acquire and release (at any scope) change nothing.
ORDER = {"volatile", "relaxed", "acquire", "release", "weak", "cta", "gpu", "sys", "cluster",
         "mmio"}


def access(ins) -> list[str]:
    """An ld / st / atom opcode without its memory-ordering qualifiers."""
    return [p for p in ins.parts if p not in ORDER]


class Unsupported(Exception):
    pass


@dataclass(frozen=True)
class Const:
    v: int  # a 32-bit value, or a byte offset


@dataclass(frozen=True)
class Ptr:  # a buffer base, as a word index
    word: int


@dataclass(frozen=True)
class DevPtr:
    """A pointer argument to device memory at a fixed word address (cudaMalloc'd memory that
    persists across launches, or a peer aperture): the host passes it instead of a buffer id."""
    word: int


@dataclass(frozen=True)
class Sreg:
    name: str


@dataclass(frozen=True)
class Reg:  # a machine register; kind "i32" or "ptr" (a word index)
    r: int
    kind: str = "i32"


@dataclass(frozen=True)
class Scaled:  # a byte offset 4 * k * register: k words per unit
    r: int
    k: int = 1


@dataclass(frozen=True)
class Addr:  # word index register + word
    r: int
    word: int


@dataclass(frozen=True)
class Pred:  # a setp, emitted where it is used
    cmp: str
    signed: bool
    ra: int
    rhs: object  # Reg or Const


@dataclass(frozen=True)
class Known:  # a predicate known at launch
    v: bool


@dataclass(frozen=True)
class Wide:  # a 32-bit value widened to 64 bits (cvt), only usable as an address part
    v: object


@dataclass(frozen=True)
class Gather:  # thread-index bits (t-bit indices, low first), or "linear": t itself
    bits: object


@dataclass
class Launch:
    blocks: list[MBlock]
    mem: dict[int, int]  # initial `in` words
    warps: int
    buffers: dict[int, int] = field(default_factory=dict)  # buffer id -> base word
    dead: frozenset = frozenset()  # CFG edges known at launch never to be taken


def basic_blocks(k: ptx.Kernel) -> list[tuple[int, int]]:
    """[start, end) instruction ranges of PTX basic blocks, in order."""
    starts = {0} | set(k.labels.values())
    for i, ins in enumerate(k.instrs):
        if ins.parts[0] in ("bra", "ret"):
            starts.add(i + 1)
    starts = sorted(s for s in starts if s <= len(k.instrs))
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(k.instrs)]) if a < b]


def relayout(k: ptx.Kernel) -> ptx.Kernel:
    """Reorder basic blocks so each loop is contiguous, header first, and the only branches
    backwards are back-edges (compilers rotate loops: clang puts reduce's latch before its
    header). Falls-through that no longer hold become `bra.uni`; a conditional branch whose
    target is now next is inverted. A structured kernel comes back unchanged."""
    bbs = basic_blocks(k)
    n, at = len(bbs), {a: i for i, (a, _) in enumerate(bbs)}

    def edges(i):  # (taken target or None, fall-through or None)
        last = k.instrs[bbs[i][1] - 1]
        fall = i + 1 if i + 1 < n else None
        if last.parts[0] == "ret":
            return None, None
        if last.parts[0] == "bra":
            t = at[k.labels[last.args[0].value]]
            return t, (fall if last.guard is not None else None)
        return None, fall

    succ = [[s for s in edges(i) if s is not None] for i in range(n)]
    pred = [[j for j in range(n) if i in succ[j]] for i in range(n)]
    dom = [set(range(n)) for _ in range(n)]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            d = {i} | (set.intersection(*(dom[j] for j in pred[i])) if pred[i] else set())
            if d != dom[i]:
                dom[i], changed = d, True
    loops: dict[int, set] = {}  # header -> natural loop body
    for u in range(n):
        for h in succ[u]:
            if h in dom[u]:
                body, work = loops.setdefault(h, {h}), [u]
                while work:
                    x = work.pop()
                    if x not in body:
                        body.add(x)
                        work += pred[x]

    def order(region: set, head) -> list[int]:
        inner = [h for h in loops if h in region and h != head and loops[h] <= region]
        top = [h for h in inner if not any(h != g and loops[h] < loops[g] for g in inner)]
        rep_of = {x: x for x in region}
        for h in top:
            rep_of |= {x: h for x in loops[h]}
        nodes = set(rep_of.values())
        indeg = {x: 0 for x in nodes}
        out = {x: set() for x in nodes}
        for x in region:
            for y in succ[x]:
                if (y in region and y != head and rep_of[y] != rep_of[x]
                        and rep_of[y] not in out[rep_of[x]]):
                    out[rep_of[x]].add(rep_of[y])
                    indeg[rep_of[y]] += 1
        seq, ready = [], sorted(x for x in nodes if indeg[x] == 0)
        while ready:
            x = ready.pop(0)
            seq += order(loops[x], x) if x in top else [x]
            for y in out[x]:
                indeg[y] -= 1
                if indeg[y] == 0:
                    ready = sorted(ready + [y])
        assert len(set(seq)) == len(region), "an irreducible CFG"
        return seq

    seq = order(set(range(n)), None)
    if seq == list(range(n)):
        return k
    name = {at[a]: lbl for lbl, a in k.labels.items()}
    for i in range(n):
        name.setdefault(i, f"$L__qgpu{i}")
    instrs, labels = [], {}
    for pos, i in enumerate(seq):
        nxt = seq[pos + 1] if pos + 1 < len(seq) else None
        labels[name[i]] = len(instrs)
        a, e = bbs[i]
        body = list(k.instrs[a:e])
        t, fall = edges(i)
        last = body[-1]
        if last.parts[0] == "bra" and last.guard is not None and fall != nxt and t == nxt:
            # invert: branch to the old fall-through, fall into the target
            body[-1] = dataclasses.replace(last, neg=not last.neg,
                                           args=[ptx.Operand("label", name[fall])])
            fall = None
        if fall is not None and fall != nxt:
            body.append(ptx.Instr("bra.uni", [ptx.Operand("label", name[fall])]))
        instrs += body
    return dataclasses.replace(k, instrs=instrs, labels=labels)


def full_blocks(k: ptx.Kernel, bbs, bb_at, latch_of: dict[int, int], uniform: set[str],
                dead=frozenset()) -> dict[int, bool]:
    """Blocks every lane runs whenever they run, so they need no mask. Block c is full if its
    immediate dominator d is and c post-dominates d in the CFG without back-edges: every lane
    at d reaches c (a block after a loop runs once all lanes have left the loop). A loop header also needs its
    latch full and the latch's branch uniform: a predicate whose registers are only ever
    written in full blocks, from uniform values."""
    n = len(bbs)
    back = {(latch, h) for h, latch in latch_of.items()}
    outs = []
    for bi, (_, end) in enumerate(bbs):
        last = k.instrs[end - 1]
        o = [bb_at[k.labels[last.args[0].value]]] if last.parts[0] == "bra" else []
        if last.parts[0] != "ret" and not (last.parts[0] == "bra" and last.guard is None):
            o.append(bi + 1)
        outs.append([c for c in o if c < n and (bi, c) not in dead])
    sink = n
    # a block after a loop runs once every lane has left it (masks gather them), so only
    # the back-edge itself is dropped
    fwd = [[c for c in o if (bi, c) not in back] or [sink] for bi, o in enumerate(outs)]
    pdom = {sink: {sink}}
    for bi in reversed(range(n)):  # forward edges go forward (relayout)
        pdom[bi] = {bi} | set.intersection(*(pdom[c] for c in fwd[bi]))
    dom = {0: {0}}
    for c in range(1, n):
        preds = [p for p in range(c) if c in fwd[p]]
        dom[c] = {c} | (set.intersection(*(dom[p] for p in preds)) if preds else set())
    defs: dict[str, set] = {}  # register -> (blocks writing it, registers its values read)
    for bi, (start, end) in enumerate(bbs):
        for ins in k.instrs[start:end]:
            if (d := defines(ins)) is not None:
                blocks_, reads = defs.setdefault(d, (set(), set()))
                blocks_.add(bi)
                reads |= {a.value for a in ins.args[1:] if a.kind == "reg"}

    def cone(r: str) -> set[str]:
        seen, todo = set(), [r]
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                todo += list(defs.get(x, (set(), set()))[1])
        return seen

    ok = set(latch_of)  # headers assumed full until shown otherwise
    while True:
        full = {}
        for c in range(n):
            if c == 0:
                f = True
            else:
                d = max(dom[c] - {c}, default=None)
                f = d is not None and full[d] and c in pdom[d]
            full[c] = f and (c not in latch_of or c in ok)
        bad = set()
        for h in ok:
            last = k.instrs[bbs[latch_of[h]][1] - 1]
            g = last.guard
            uni = g is None or (g in uniform and all(
                full[b] for r in cone(g) for b in defs.get(r, (set(), set()))[0]))
            if not (full[latch_of[h]] and uni):
                bad.add(h)
        if not bad:
            return full
        ok -= bad


def compare_const(cmp: str, a: int, b: int, signed: bool) -> bool:
    if signed:
        a, b = (x - (1 << 32) if x >> 31 else x for x in (a & 0xFFFFFFFF, b & 0xFFFFFFFF))
    return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b,
            "ge": a >= b}[cmp]


def defines(ins) -> str | None:
    if ins.args and ins.args[0].kind == "reg" and ins.parts[0] not in ("st", "bra"):
        return ins.args[0].value
    return None


def uniform_regs(k: ptx.Kernel) -> set[str]:
    """PTX registers equal on every lane of a warp (fixpoint: every definition uses only
    immediates, parameters, uniform special registers and uniform registers)."""
    defs: dict[str, list] = {}
    for ins in k.instrs:
        if (d := defines(ins)) is not None:
            defs.setdefault(d, []).append(ins)
    uni = set(defs)

    def ok(ins) -> bool:
        # a load is uniform when its address is: every lane of the warp reads the same word in
        # the same record (a flag a peer writes, a loop bound in memory). Its address operand is
        # checked below with the other operands.
        if ins.parts[0] == "ld" and access(ins)[1] not in ("param", "global", "shared"):
            return False
        for a in ins.args[1:]:
            if a.kind == "reg" and a.value not in uni:
                return False
            if a.kind == "sreg" and a.value not in SREGS_UNIFORM:
                return False
            if a.kind == "mem" and a.value[0].kind == "reg" and a.value[0].value not in uni:
                return False
        return True

    changed = True
    while changed:
        changed = False
        for r in list(uni):
            if not all(ok(i) for i in defs[r]):
                uni.discard(r)
                changed = True
    return uni


WRITES = {"add", "tid", "ld", "mad", "sreg", "max", "min", "sub", "selp", "mulhi", "and", "or",
          "xor"}
NO_WRITE = {"st", "setp", "nop", "atom", "stw"}


def fields_used(ins: KInstr) -> list[str]:
    """The register fields an instruction really uses (the others are don't-care zeros)."""
    used = ["rd"] if ins.op in WRITES else []
    if ins.op not in ("tid", "sreg", "nop"):
        used.append("ra")
    if ins.op in ("st", "atom", "stw") or (ins.op in ("add", "max", "min", "sub", "setp", "mad", "selp",
                                                "mulhi",
                                     "and", "or", "xor") and not ins.useimm):
        used.append("rb")
    if ins.op == "mad":
        used.append("rc")
    return used


def allocate(code: list[list[KInstr]], live_out: list[set]) -> tuple[list, list]:
    """Map virtual registers to the 15 physical ones (ZERO is reserved): a register interferes
    with every register live where it is written; greedy coloring in order of appearance.
    A register preferably avoids those its group reads: a machine block whose live-out
    overwrites an input another live-out still reads can only be emitted in some orders, and
    two such live-outs (a swap) in none (blocks.plan)."""
    edges: dict[int, set] = {}
    order: list[int] = []
    reads_near: dict[int, set] = {}  # register -> registers read in the groups writing it
    for gi, instrs in enumerate(code):
        read = {x for ins in instrs for x in ins.rw()[1] if x >= 0}
        for ins in instrs:
            if (rd := ins.rw()[0]) is not None and rd >= 0:
                reads_near.setdefault(rd, set()).update(read - {rd})
        live = {r for r in live_out[gi] if r >= 0}
        for ins in reversed(instrs):
            rd, srcs = ins.rw()
            if rd is not None and rd >= 0:
                edges.setdefault(rd, set()).update(x for x in live if x != rd)
                for x in live:
                    if x != rd:
                        edges.setdefault(x, set()).add(rd)
                live.discard(rd)
            live |= {x for x in srcs if x >= 0}
        for ins in instrs:
            for f in fields_used(ins):
                r = getattr(ins, f)
                if r not in order:
                    order.append(r)
    color = {VZERO: ZERO}
    for r in order:
        if r in color:
            continue
        taken = {color[x] for x in edges.get(r, ()) if x in color}
        free = [c for c in range(ZERO) if c not in taken]
        if not free:
            raise Unsupported("needs more than 15 registers")
        avoid = {color[x] for x in reads_near.get(r, ()) if x in color}
        color[r] = next((c for c in free if c not in avoid), free[0])
    phys = [[dataclasses.replace(ins, **{f: color[getattr(ins, f)] for f in fields_used(ins)},
                                 **{f: 0 for f in ("rd", "ra", "rb", "rc")
                                    if f not in fields_used(ins)})
             for ins in instrs] for instrs in code]
    return phys, [{color[r] for r in lo if r >= 0 and r in color} for lo in live_out]


def lower(k: ptx.Kernel, grid, block, args, buffers: dict[int, list[int]], lanes: int = 8,
          stages: int = 6, mul_stages=frozenset({4}), logic_stages=frozenset({2}),
          atom_stages=frozenset({5}),
          lmax: int = LMAX, write_through: bool = False,
          scratch_base: int | None = None) -> Launch:
    """`args`: the kernel's arguments, pointer arguments given as buffer ids (keys of
    `buffers`, whose values are the initial uint32 words).

    `write_through`: global stores are `stw` too, so every result lands in global memory (the
    `in` heads, which the host's MEM records reset at each launch) and none in the `out`
    heads, whose rows would outlive the launch: what a context with several launches needs.

    `scratch_base`: where launch buffers (given by id), `.shared` arrays and shuffle scratch
    start, above device memory that persists across launches (pointer arguments given as
    DevPtr)."""
    opts = {"lanes": lanes, "stages": stages, "mul_stages": mul_stages,
            "logic_stages": logic_stages, "atom_stages": atom_stages, "lmax": lmax,
            "write_through": write_through, "scratch_base": scratch_base}
    # twice: branches known at launch (found by the first pass) let more blocks run unmasked
    first = _lower(k, grid, block, args, buffers, **opts)
    launch = _lower(k, grid, block, args, buffers, dead=first.dead, **opts) if first.dead \
        else first
    if len(launch.blocks) > 1 << BBITS:
        raise Unsupported(f"{len(launch.blocks)} machine blocks, the ROM holds {1 << BBITS}")
    return launch


def _lower(k: ptx.Kernel, grid, block, args, buffers: dict[int, list[int]], lanes: int,
           stages: int, mul_stages, logic_stages, atom_stages, lmax: int,
           write_through: bool = False, dead: frozenset = frozenset(),
           scratch_base: int | None = None) -> Launch:
    k = relayout(k)
    grid = tuple(grid) + (1,) * (3 - len(grid))
    block = tuple(block) + (1,) * (3 - len(block))
    if grid[2] != 1 or block[2] != 1:
        raise Unsupported("3D launches")
    (gx, gy), (bx, by) = grid[:2], block[:2]
    if (bx * by) % lanes:
        raise Unsupported(f"threads per block must be a multiple of {lanes}")
    most = min(lanes * ((1 << WBITS) - 1), 1 << TBITS)  # the end token needs warps < 2^WBITS
    if gx * gy * bx * by > most:
        raise Unsupported(f"more than {most // lanes} warps ({most} threads)")
    one_d = gy == 1 and by == 1

    def pow2(v):
        return v & (v - 1) == 0

    # the linear thread index t = ((ctaid.y * gx + ctaid.x) * by + tid.y) * bx + tid.x: with
    # power-of-two sizes every special register is a bit field of t
    lbx, lby, lgx = (v.bit_length() - 1 for v in (bx, by, gx))
    fields = {"%tid.x": (0, lbx, bx), "%tid.y": (lbx, lby, by),
              "%ctaid.x": (lbx + lby, lgx, gx), "%ctaid.y": (lbx + lby + lgx, None, gy)}
    consts = {"%ntid.x": bx, "%ntid.y": by, "%nctaid.x": gx, "%nctaid.y": gy}

    def sreg_bits(name: str) -> list[int]:
        """t-bit indices of a special register, low bit first."""
        lo, width, size = fields[name]
        if size == 1:
            return []
        if not pow2(size) and not (name == "%ctaid.y" or (name == "%ctaid.x" and gy == 1)):
            raise Unsupported(f"{name} of a non-power-of-two launch size")
        if name in ("%tid.y", "%ctaid.x") and not pow2(bx * (by if name == "%ctaid.x" else 1)):
            raise Unsupported(f"{name} with a non-power-of-two block")
        top = TBITS if width is None or (name == "%ctaid.x" and gy == 1) else lo + width
        return list(range(lo, top))
    base, top, mem = {}, scratch_base or 0, {}
    for bid, words in buffers.items():
        base[bid] = top
        for i, w in enumerate(words):
            mem[top + i] = w & 0xFFFFFFFF
        top += len(words)
    shared = {}  # `.shared` arrays follow the buffers; thread blocks run one after another
    for sh in k.shared:
        if sh.size % 4:
            raise Unsupported(f"shared {sh.name} is not whole words")
        shared[sh.name] = top
        top += sh.size // 4
    # warp shuffles: a CUDA warp is 32 threads, ours 8, so a shuffle crosses machine warps; it
    # goes through a scratch array in memory (one word per thread of the block) and a barrier.
    # Two scratch arrays alternate (one barrier per shuffle) unless a shuffle sits in a loop
    # or they do not fit, then one array and a second barrier after the read
    shfls = [i for i, ins in enumerate(k.instrs) if ins.parts[0] == "shfl"]
    nthreads = bx * by
    scratch: list[int] = []
    if shfls:
        if nthreads % 32 or not pow2(nthreads):
            raise Unsupported(f"shuffles in a block of {nthreads} threads (not whole warps)")
        backs = [(k.labels[ins.args[0].value], i) for i, ins in enumerate(k.instrs)
                 if ins.parts[0] == "bra" and k.labels[ins.args[0].value] <= i]
        looped = any(t <= i <= j for i in shfls for t, j in backs)
        for _ in range(1 if looped or top + 2 * nthreads > WINDOW else 2):
            scratch.append(top)
            top += nthreads
    if top > WINDOW:
        raise Unsupported(f"buffers and shared memory need {top} words, the window has "
                          f"{WINDOW}")
    barriers = any(ins.parts[0] in ("bar", "barrier", "shfl") for ins in k.instrs)
    wpb = bx * by // lanes if barriers else 1  # warps per thread block, for barriers
    if wpb not in (1, 2, 4, 8):
        raise Unsupported(f"a barrier with {wpb} warps per thread block")
    params = {p.name: (Ptr(a.word) if isinstance(a, DevPtr) else Ptr(base[a])) if p.size == 8
              else Const(a & 0xFFFFFFFF) for p, a in zip(k.params, args)}

    ndefs: dict[str, int] = {}
    for ins in k.instrs:
        if (d := defines(ins)) is not None:
            ndefs[d] = ndefs.get(d, 0) + 1
    uniform = uniform_regs(k)
    bbs = basic_blocks(k)
    bb_at = {a: i for i, (a, _) in enumerate(bbs)}
    ret_blocks = {i for i, (a, _) in enumerate(bbs) if k.instrs[a].op == "ret"}

    # PTX blocks that only fall into each other (the early exit is if-converted, so it does
    # not end a block) form one group: one code list, split into machine blocks later
    targets = {bb_at[k.labels[ins.args[0].value]] for ins in k.instrs if ins.parts[0] == "bra"}
    group_of: list[int] = []
    vals: dict[str, object] = {}
    known: dict[str, Const] = {}  # loop variables whose value here is a known constant
    mreg: dict[str, int] = {}
    nreg = [0]  # virtual registers; allocated to physical ones at the end
    exit_pred: list[Pred] = []
    exit_at: dict[int, int] = {}  # group -> position of its latest exit setp
    code: list[list[KInstr]] = []
    terms: list[tuple] = []  # per group
    gathered: dict = {}  # thread-index gathers already in a register, in the current group

    def fresh() -> int:
        nreg[0] += 1
        return nreg[0] - 1

    def home(name: str) -> int:
        if name not in mreg:
            mreg[name] = fresh()
        return mreg[name]

    def multi(name: str) -> bool:
        return ndefs.get(name, 0) > 1

    def val(a: ptx.Operand, fold: bool = False, raw: bool = False):
        """An operand's value; special registers become constants or thread-index gathers
        (with raw=True they stay Sreg, for the global-index pattern)."""
        if a.kind == "reg":
            if fold and a.value in known:
                v = known[a.value]
            elif a.value in vals:
                v = vals[a.value]
            else:
                raise Unsupported(f"{a.value} read before it is defined")
        elif a.kind == "imm":
            v = Const(a.value & 0xFFFFFFFF)
        elif a.kind == "sreg":
            v = Sreg(a.value)
        elif a.kind == "sym" and a.value in shared:
            v = Ptr(shared[a.value])
        else:
            raise Unsupported(f"operand {a}")
        return v if raw or not isinstance(v, Sreg) else sreg_value(v.name)

    def global_index(x, y, z):
        """mad.lo(ctaid.a, ntid.a, tid.a): the global index along axis a, or None."""
        for ax in "xy":
            if {x, y} == {Sreg(f"%ctaid.{ax}"), Sreg(f"%ntid.{ax}")} and z == Sreg(f"%tid.{ax}"):
                if ax == "x" and one_d:
                    return Gather("linear")
                size = bx if ax == "x" else by
                if not pow2(size):
                    raise Unsupported(f"global {ax} index with ntid.{ax} = {size}")
                return Gather(tuple(sreg_bits(f"%tid.{ax}") + sreg_bits(f"%ctaid.{ax}")))
        return None

    def reg_of(v, out: list[KInstr]) -> int:
        """A machine register holding v (a 32-bit value, or a pointer's word index)."""
        if isinstance(v, Reg):
            return v.r
        r = fresh()
        if isinstance(v, Const):
            out.append(KInstr("add", rd=r, ra=VZERO, imm=v.v, useimm=True))
        elif isinstance(v, Ptr):
            out.append(KInstr("add", rd=r, ra=VZERO, imm=v.word, useimm=True))
        elif isinstance(v, Addr):
            out.append(KInstr("add", rd=r, ra=v.r, imm=v.word, useimm=True))
        elif isinstance(v, Scaled):  # used as a number after all: 4 k r
            out.append(KInstr("mad", rd=r, ra=v.r, imm=4 * v.k, useimm=True, rc=VZERO))
        elif isinstance(v, Gather):
            if v in gathered:  # computed earlier in this group (straight-line code)
                return gathered[v]
            gathered[v] = r
            if v.bits == "linear":
                out.append(KInstr("tid", rd=r))
            else:
                bits = list(v.bits) + [7] * (TBITS - len(v.bits))
                out.append(KInstr("sreg", rd=r, imm=sum(b << 3 * i for i, b in enumerate(bits))))
        else:
            raise Unsupported(f"cannot materialize {v}")
        return r

    def sreg_value(name: str):
        """A special register as a constant or a gather of thread-index bits."""
        if name in consts:
            return Const(consts[name])
        if name not in fields:
            raise Unsupported(f"special register {name}")
        bits = sreg_bits(name)
        return Gather(tuple(bits)) if bits else Const(0)

    def define(name: str, v, out: list[KInstr], ptr: bool = False):
        """Bind a PTX register. One with several definitions (a loop variable) lives in its own
        machine register, so every definition writes there."""
        if not multi(name):
            vals[name] = v
            return
        r = dest(name)
        known.pop(name, None)
        if isinstance(v, Const) and not pmode:
            known[name] = v
        if isinstance(v, Reg):
            if v.r != r:
                out.append(KInstr("add", rd=r, ra=v.r, imm=0, useimm=True))
        elif isinstance(v, Const):
            out.append(KInstr("add", rd=r, ra=VZERO, imm=v.v, useimm=True))
        elif isinstance(v, (Ptr, Addr)):
            src = VZERO if isinstance(v, Ptr) else v.r
            out.append(KInstr("add", rd=r, ra=src, imm=v.word, useimm=True))
        elif isinstance(v, Wide):  # a 64-bit loop variable: its register holds the 32-bit value
            out.append(KInstr("add", rd=r, ra=reg_of(v.v, out), imm=0, useimm=True))
        else:
            raise Unsupported(f"{name}: a loop variable holding {v}")
        is_ptr = ptr or isinstance(v, (Ptr, Addr)) or (isinstance(v, Reg) and v.kind == "ptr")
        vals[name] = Reg(r, "ptr" if is_ptr else "i32")
        commit(out)

    def dest(name: str) -> int:
        """The register an instruction computing PTX register `name` writes. Under a lane mask
        a loop variable is written through a temporary that commit() selects in."""
        if not multi(name):
            return fresh()
        if pmode and not full[cur[0]]:
            t = fresh()
            pending.append((name, t, mask[cur[0]]))  # the mask of the block it is written in
            return t
        return home(name)

    def commit(out: list[KInstr]):
        """home = active ? temporary : home, for the loop variables written just now."""
        while pending:
            name, t, m = pending.pop()
            out.append(KInstr("setp", ra=m, imm=0, useimm=True, pd=ACTIVE_P, cmp="ne"))
            out.append(KInstr("selp", rd=home(name), ra=t, rb=home(name), guard=ACTIVE_P))
            v = vals.get(name)
            vals[name] = Reg(home(name), v.kind if isinstance(v, Reg) else "i32")

    def emit_setp(p: Pred, pd: int, out: list[KInstr]):
        rhs = p.rhs
        out.append(KInstr("setp", ra=p.ra, rb=rhs.r if isinstance(rhs, Reg) else 0,
                          imm=rhs.v if isinstance(rhs, Const) else 0,
                          useimm=isinstance(rhs, Const), pd=pd, cmp=p.cmp, signed=p.signed))

    def word_addr(a: ptx.Operand) -> tuple[int, int]:
        base_op, off = a.value
        if off % 4:
            raise Unsupported("a byte offset that is not a whole word")
        v = (vals.get(base_op.value) if base_op.kind == "reg"
             else Ptr(shared[base_op.value]) if base_op.value in shared else None)
        if isinstance(v, Addr):
            return v.r, v.word + off // 4
        if isinstance(v, Reg) and v.kind == "ptr":
            return v.r, off // 4
        if isinstance(v, Ptr):
            return VZERO, v.word + off // 4
        raise Unsupported(f"address {v}")

    def scaled(r: int, k: int, out: list[KInstr]) -> Scaled:
        """Scaled(r, k), safe to use later: a loop variable's register may be rewritten before
        then, so its word offset is computed now."""
        if r not in mreg.values():
            return Scaled(r, k)
        w = fresh()
        out.append(KInstr("add", rd=w, ra=r, imm=0, useimm=True) if k == 1 else
                   KInstr("mad", rd=w, ra=r, imm=k, useimm=True, rc=VZERO))
        return Scaled(w, 1)

    def words_reg(sc: Scaled, out: list[KInstr], plus: int | None = None) -> int:
        """A register holding the word offset k * r (+ register `plus`): one add or mad."""
        if sc.k == 1 and plus is None:
            return sc.r
        t = fresh()
        if sc.k == 1:
            out.append(KInstr("add", rd=t, ra=plus, rb=sc.r))
        else:
            out.append(KInstr("mad", rd=t, ra=sc.r, imm=sc.k, useimm=True,
                              rc=VZERO if plus is None else plus))
        return t

    def as_wide(x):
        """A 64-bit operand as Wide (a 64-bit loop variable is a register of its value)."""
        if isinstance(x, Reg) and x.kind == "i32":
            return Wide(x)
        return x

    def whole_words(c: Const) -> int:
        if c.v % 4:
            raise Unsupported("pointer + bytes that are not a whole word")
        return c.v // 4

    def barrier(gi: int) -> tuple[list[KInstr], int]:
        """End group gi with a barrier; the code after it goes in a new group."""
        terms[gi] = ("bar",)
        code.append([])
        terms.append(("next",))
        gathered.clear()
        return code[-1], len(code) - 1

    def shuffle(ins, bi: int, out: list[KInstr], gi: int) -> tuple[list[KInstr], int]:
        """shfl.sync.down.b32 d, a, b, c, mask: d = a of lane + b of the 32-thread warp if
        that lane is within the clamp c's segment, else a (PTX ISA). Every thread writes a to
        scratch[i] (i: its index in the block), a barrier, then reads scratch[i + b]."""
        if ins.op != "shfl.sync.down.b32":
            raise Unsupported(ins.op)
        d, src, bop, cop, _ = ins.args
        if d.kind == "pair":  # nvcc's "%r|%p": the predicate (lane in range) is not modeled
            pname = d.value[1].value
            if any(o.kind == "reg" and o.value == pname or other.guard == pname
                   for other in k.instrs for o in other.args[1:]):
                raise Unsupported("a shuffle's in-range predicate is used")
            d = d.value[0]
        b, cv = val(bop), val(cop)
        if not (isinstance(b, Const) and isinstance(cv, Const)):
            raise Unsupported("a shuffle offset or clamp that is not known at launch")
        if pmode and not full[bi]:
            raise Unsupported("a shuffle in divergent code")
        seg = cv.v >> 8 & 31
        width = ~seg & 31  # the lane bits that move within a segment
        if width & (width + 1):
            raise Unsupported(f"shuffle clamp {cv.v:#x}")
        xr = reg_of(val(src), out)
        buf = scratch[shuffle_count[0] % len(scratch)]
        shuffle_count[0] += 1
        lb = nthreads.bit_length() - 1
        idx = reg_of(Gather(tuple(range(lb))), out)
        commit(out)
        out.append(KInstr("stw", ra=idx, rb=xr, imm=buf))
        out, gi = barrier(gi)
        top_ok = (cv.v & width) - (b.v & 31)  # lane (its moving bits) <= this: in range
        r = dest(d.value)
        if top_ok < 0:
            out.append(KInstr("add", rd=r, ra=xr, imm=0, useimm=True))
        else:
            got = fresh()
            out.append(KInstr("ld", rd=got, ra=reg_of(Gather(tuple(range(lb))), out),
                              imm=buf + (b.v & 31)))
            lane = reg_of(Gather(tuple(range(width.bit_length()))), out)
            out.append(KInstr("setp", ra=lane, imm=top_ok, useimm=True, pd=SEL_P, cmp="le"))
            out.append(KInstr("selp", rd=r, ra=got, rb=xr, guard=SEL_P))
        vals[d.value] = Reg(r)
        if len(scratch) == 1:  # the next shuffle reuses the array: everyone has read it
            out, gi = barrier(gi)
        return out, gi

    shuffle_count = [0]

    # loop variables get their register before any block reads them
    for name in ndefs:
        if multi(name):
            vals[name] = Reg(home(name))

    def emit_edges(bi: int, out: list[KInstr], gi: int):
        """Pass block bi's mask along its outgoing edges: a later block's mask gains the lanes
        that branch there (max of 0 / 1 masks); a back-edge sets the loop header's mask to the
        lanes that go round again and ends the group with "branch if any lane is on"."""
        p, neg, t = edge.get(bi, (None, False, bi + 1))
        if (start_end := bbs[bi]) and k.instrs[start_end[1] - 1].parts[0] != "bra":
            p, neg, t = None, False, bi + 1  # falls through
        m = mask[bi]
        if full[bi]:  # every lane runs it: its mask is not kept, it is all ones
            m = fresh()
            out.append(KInstr("add", rd=m, ra=VZERO, imm=1, useimm=True))
        if p == "never":  # a launch-time-false branch: falls through
            p, t = None, bi + 1
        if p is None:
            taken, fall = m, None
        else:
            emit_setp(p, EDGE_P, out)
            taken, fall = fresh(), fresh()
            out.append(KInstr("selp", rd=taken, ra=m, rb=VZERO, guard=EDGE_P, gneg=neg))
            out.append(KInstr("selp", rd=fall, ra=VZERO, rb=m, guard=EDGE_P, gneg=neg))
        for c, lanes_ in ((t, taken), (bi + 1, fall)):
            if lanes_ is None or c in ret_blocks or (c > bi and full[c]):
                continue
            if c > bi and (c in accumulate or c in assigned):
                out.append(KInstr("max", rd=mask[c], ra=mask[c], rb=lanes_))
            elif c > bi:
                out.append(KInstr("add", rd=mask[c], ra=lanes_, imm=0, useimm=True))
                assigned.add(c)
            elif latch_of.get(c) == bi:  # the back-edge
                out.append(KInstr("add", rd=mask[c], ra=lanes_, imm=0, useimm=True))
                out.append(KInstr("setp", ra=mask[c], imm=0, useimm=True, pd=BRANCH_P,
                                  cmp="ne"))
                terms[gi] = ("cond_any", c)
            else:
                raise Unsupported("a branch backwards that is not a loop's back-edge")

    def mul_hi(x, y, signed: bool, name: str, out: list[KInstr]) -> int:
        """mul.hi of a register and a register or constant. The circuit's mulhi is unsigned;
        signed: hi_s = hi_u - (x < 0 ? y : 0) - (y < 0 ? x : 0)  (mod 2^32)."""
        ra = reg_of(x, out)
        rb = 0 if isinstance(y, Const) else reg_of(y, out)
        imm = y.v if isinstance(y, Const) else 0
        r = dest(name) if not signed else fresh()
        out.append(KInstr("mulhi", rd=r, ra=ra, rb=rb, imm=imm, useimm=isinstance(y, Const)))
        if not signed:
            return r
        for neg_of, other in ((ra, y), (None if isinstance(y, Const) else rb, Reg(ra))):
            if neg_of is None:  # y is a constant: its sign is known
                if y.v >> 31:
                    t = fresh()
                    out.append(KInstr("sub", rd=t, ra=r, rb=ra))
                    r = t
                continue
            emit_setp(Pred("lt", True, neg_of, Const(0)), SEL_P, out)
            corr, t = fresh(), fresh()
            out.append(KInstr("selp", rd=corr, ra=reg_of(other, out), rb=0, imm=0,
                              useimm=True, guard=SEL_P))
            out.append(KInstr("sub", rd=t, ra=r, rb=corr))
            r = t
        if multi(name):
            define(name, Reg(r), out)
            return home(name)
        return r

    def lower_bits(base_op: str, ty: str, name: str, xs: list, out: list[KInstr]):
        """and / or / xor / not on per-thread values (the logic stage); shifts by a constant
        through the multiplier: x << k = x * 2^k, x >> k = mulhi(x, 2^(32 - k)), and the
        arithmetic shift adds -2^(32 - k) where x < 0."""
        if base_op in FOLD or base_op == "not":
            x, y = (xs[0], Const(0xFFFFFFFF)) if base_op == "not" else xs
            if isinstance(x, Const):
                x, y = y, x
            ra = reg_of(x, out)
            rb = 0 if isinstance(y, Const) else reg_of(y, out)
            r = dest(name)
            out.append(KInstr("xor" if base_op == "not" else base_op, rd=r, ra=ra, rb=rb,
                              imm=y.v if isinstance(y, Const) else 0,
                              useimm=isinstance(y, Const)))
            vals[name] = Reg(r)
            return
        x, k = xs
        if not isinstance(k, Const):
            raise Unsupported(f"{base_op} by a per-thread amount")
        k = k.v
        if k == 0:
            define(name, x, out)
            return
        if base_op == "shl" or k >= 32 and ty != "s32":
            if k >= 32:
                define(name, Const(0), out)
                return
            if base_op == "shl" and k >= 2 and not multi(name):
                # 4 * 2^(k-2) * x: a byte offset if it meets a pointer (nvcc's shared
                # addresses), else materialized where it is used as a number
                vals[name] = scaled(reg_of(x, out), 1 << (k - 2), out)
                return
            ra = reg_of(x, out)
            r = dest(name)
            if base_op == "shl" and k == 1:  # x + x: an add places in any stage
                out.append(KInstr("add", rd=r, ra=ra, rb=ra))
            else:
                out.append(KInstr("mad", rd=r, ra=ra, imm=1 << k, useimm=True, rc=VZERO))
            vals[name] = Reg(r)
            return
        k = min(k, 31)
        ra = reg_of(x, out)
        u = fresh() if ty == "s32" else dest(name)
        out.append(KInstr("mulhi", rd=u, ra=ra, imm=1 << (32 - k), useimm=True))
        if ty != "s32":
            vals[name] = Reg(u)
            return
        emit_setp(Pred("lt", True, ra, Const(0)), SEL_P, out)
        fill, corr = fresh(), fresh()
        out.append(KInstr("add", rd=fill, ra=VZERO, imm=-(1 << (32 - k)) & 0xFFFFFFFF,
                          useimm=True))
        out.append(KInstr("selp", rd=corr, ra=fill, imm=0, useimm=True, guard=SEL_P))
        r = dest(name)
        out.append(KInstr("add", rd=r, ra=u, rb=corr))
        vals[name] = Reg(r)

    def udiv_const(ra: int, d: int, out: list[KInstr]) -> int:
        """n / d for a register n and a constant d > 0, unsigned: Granlund and Montgomery's
        multiply-high sequence (what compilers emit; NVIDIA GPUs have no integer divider
        either), exact for every 32-bit n: l = ceil(log2 d), m' = floor(2^32 (2^l - d) / d)
        + 1, t = mulhi(n, m'), q = (t + ((n - t) >> 1)) >> (l - 1)."""
        def shr(r, k):
            if k == 0:
                return r
            q = fresh()
            out.append(KInstr("mulhi", rd=q, ra=r, imm=1 << (32 - k), useimm=True))
            return q

        if d == 1:
            return ra
        l = (d - 1).bit_length()
        if d == 1 << l:
            return shr(ra, l)
        m = ((1 << 32) * ((1 << l) - d)) // d + 1
        t, u, v = fresh(), fresh(), fresh()
        out.append(KInstr("mulhi", rd=t, ra=ra, imm=m, useimm=True))
        out.append(KInstr("sub", rd=u, ra=ra, rb=t))
        out.append(KInstr("add", rd=v, ra=shr(u, 1), rb=t))
        return shr(v, l - 1)

    def div_rem(op: str, name: str, x, y, out: list[KInstr]) -> None:
        """div / rem .u32 / .s32 by a value known at launch (a constant or a kernel
        parameter): C semantics, the quotient truncated toward zero, the remainder n - q d
        with the sign of n."""
        signed = op.endswith("s32")
        if not isinstance(y, Const):
            raise Unsupported(f"{op} by a per-thread value (only by a value known at launch)")
        d = y.v - (1 << 32) if signed and y.v >> 31 else y.v
        if d == 0:
            raise Unsupported(f"{op} by zero")
        if isinstance(x, Const):
            n = x.v - (1 << 32) if signed and x.v >> 31 else x.v
            q = abs(n) // abs(d) * (1 if (n < 0) == (d < 0) else -1)
            define(name, Const((q if op.startswith("div") else n - q * d) & 0xFFFFFFFF), out)
            return
        rn = reg_of(x, out)
        if not signed:
            q = udiv_const(rn, d, out)
        else:  # through |n|, then the sign back: n < 0 xor d < 0
            neg, a, qa, qn, q = fresh(), fresh(), None, fresh(), fresh()
            out.append(KInstr("sub", rd=neg, ra=VZERO, rb=rn))
            emit_setp(Pred("lt", True, rn, Const(0)), SEL_P, out)
            out.append(KInstr("selp", rd=a, ra=neg, rb=rn, guard=SEL_P))
            qa = udiv_const(a, abs(d), out)
            out.append(KInstr("sub", rd=qn, ra=VZERO, rb=qa))
            emit_setp(Pred("lt", True, rn, Const(0)), SEL_P, out)  # again: never across blocks
            hi, lo = (qn, qa) if d > 0 else (qa, qn)
            out.append(KInstr("selp", rd=q, ra=hi, rb=lo, guard=SEL_P))
        r = dest(name)
        if op.startswith("div"):
            out.append(KInstr("add", rd=r, ra=q, imm=0, useimm=True))
        else:
            out.append(KInstr("mad", rd=r, ra=q, imm=-d & 0xFFFFFFFF, useimm=True, rc=rn))
        vals[name] = Reg(r)

    # divergent control flow (a per-lane branch other than the early exit): every block runs
    # for all lanes, in program order, under a per-lane mask register (0 / 1); a loop's
    # back-edge becomes a machine branch taken while any lane's mask says continue
    # An early exit may also come after other branches when every other branch is uniform
    # and the exit is in no loop (`if (threadIdx.x != 0) return;` after a spin-wait loop, to
    # post a flag from one thread): each lane passes it once, so later stores are guarded alike.
    bras = [(i, ins) for i, ins in enumerate(k.instrs) if ins.parts[0] == "bra"]
    loops_at = [(k.labels[ins.args[0].value], i) for i, ins in bras
                if k.labels[ins.args[0].value] <= i]
    late_exit = set()
    for i, ins in bras:
        t = bb_at[k.labels[ins.args[0].value]]
        if (ins.guard is not None and ins.guard not in uniform and t in ret_blocks
                and not ins.neg and not any(h <= i <= j for h, j in loops_at)
                and all(o.guard is None or o.guard in uniform for j, o in bras if j != i)):
            late_exit.add(i)
    if len(late_exit) > 1:
        late_exit = set()
    divergent, seen = False, False
    for i, ins in enumerate(k.instrs):
        if ins.parts[0] != "bra":
            continue
        t = bb_at[k.labels[ins.args[0].value]]
        exit_like = (ins.guard is not None and t in ret_blocks and not seen and not ins.neg
                     ) or i in late_exit
        if ins.guard is not None and ins.guard not in uniform and not exit_like:
            divergent = True
        seen = seen or not exit_like
    pmode = divergent
    headers, latch_of = set(), {}
    for bi, (start, end) in enumerate(bbs):
        last = k.instrs[end - 1]
        if last.parts[0] == "bra":
            t = bb_at[k.labels[last.args[0].value]]
            if t <= bi:
                if t in headers:
                    raise Unsupported("a loop with two back-edges")
                headers.add(t)
                latch_of[t] = bi
    if pmode and (len(ret_blocks) != 1 or max(ret_blocks) != len(bbs) - 1):
        raise Unsupported("divergent code with a ret that is not the last block")
    mask = {bi: fresh() for bi in range(len(bbs))} if pmode else {}
    # a block's mask is assigned by the first edge into it (in program order, which is the
    # order blocks run in), and maxed by the others; only a block that a loop exits to
    # gathers lanes over iterations, so only those start at 0
    accumulate, assigned = set(), set()
    for bi, (start, end) in enumerate(bbs):
        last = k.instrs[end - 1]
        outs = [bb_at[k.labels[last.args[0].value]]] if last.parts[0] == "bra" else []
        if last.parts[0] != "ret" and not (last.parts[0] == "bra" and last.guard is None):
            outs.append(bi + 1)
        accumulate |= {c for c in outs if c > bi and any(
            h <= bi <= latch and not h <= c <= latch for h, latch in latch_of.items())}
    full = full_blocks(k, bbs, bb_at, latch_of, uniform, dead) if pmode else {}
    found_dead = set()  # edges a branch known at launch never takes
    pending: list[tuple[str, int, int]] = []
    cur = [0]
    edge: dict[int, tuple] = {}  # PTX block -> its branch, emitted at the block's end

    branched = False
    for bi, (start, end) in enumerate(bbs):
        new_group = (bi == 0 or bi in headers or (bi - 1) in latch_of.values()) if pmode else (
            bi == 0 or bi in targets or terms[-1][0] != "next")
        if new_group:
            code.append([])
            terms.append(("next",))
            gathered.clear()
            known.clear()  # a block entered from elsewhere: loop variables are not known
        group_of.append(len(code) - 1)
        out, gi = code[-1], len(code) - 1
        cur[0] = bi
        if pmode and bi == 0:  # masks that gather lanes start at 0 (the entry block is full)
            for b in sorted(accumulate):
                out.append(KInstr("add", rd=mask[b], ra=VZERO, imm=0, useimm=True))
        if pmode and bi in headers:  # each iteration gathers inner loops' exits afresh
            for b in range(bi + 1, latch_of[bi] + 1):
                if b in accumulate:
                    out.append(KInstr("add", rd=mask[b], ra=VZERO, imm=0, useimm=True))
        for i in range(start, end):
            commit(out)  # the previous instruction's masked writes, before anything reads them
            ins = k.instrs[i]
            op, a, base_op = ins.op, ins.args, ins.parts[0]
            if ins.guard is not None and base_op != "bra":
                raise Unsupported(f"guarded {op}")
            if base_op == "ret":
                terms[gi] = ("end",)
            elif op.startswith("ld.param"):
                define(a[0].value, params[a[1].value[0].value], out)
            elif op == "cvta.to.global.u64":
                define(a[0].value, val(a[1]), out, ptr=True)
            elif base_op == "mov":
                define(a[0].value, val(a[1], raw=True), out)
            elif op == "mad.lo.s32":
                g = global_index(*(val(o, raw=True) for o in a[1:]))
                x, y, z = (val(o) for o in a[1:]) if g is None else (None, None, None)
                if g is not None:
                    r = reg_of(g, out)
                    if multi(a[0].value):
                        define(a[0].value, Reg(r), out)
                        continue
                elif all(isinstance(v, Const) for v in (x, y, z)):
                    define(a[0].value, Const((x.v * y.v + z.v) & 0xFFFFFFFF), out)
                    continue
                else:
                    if isinstance(x, Const):
                        x, y = y, x
                    ra, rc = reg_of(x, out), reg_of(z, out)
                    rb = 0 if isinstance(y, Const) else reg_of(y, out)
                    r = dest(a[0].value)
                    out.append(KInstr("mad", rd=r, ra=ra, rb=rb,
                                      imm=y.v if isinstance(y, Const) else 0,
                                      useimm=isinstance(y, Const), rc=rc))
                vals[a[0].value] = Reg(r)
            elif op == "mul.lo.s32":
                x, y = val(a[1]), val(a[2])
                if isinstance(x, Const) and isinstance(y, Const):
                    define(a[0].value, Const((x.v * y.v) & 0xFFFFFFFF), out)
                    continue
                if isinstance(x, Const):
                    x, y = y, x
                ra = reg_of(x, out)
                rb = 0 if isinstance(y, Const) else reg_of(y, out)
                r = dest(a[0].value)
                out.append(KInstr("mad", rd=r, ra=ra, rb=rb, imm=y.v if isinstance(y, Const)
                                  else 0, useimm=isinstance(y, Const), rc=VZERO))
                vals[a[0].value] = Reg(r)
            elif op not in ("or.pred", "and.pred") and (
                    base_op in FOLD or base_op in ("shl", "shr", "not")) and ins.parts[1] in (
                    "b32", "u32", "s32"):
                xs = [val(o) for o in a[1:]]
                if not all(isinstance(v, Const) for v in xs):
                    lower_bits(base_op, ins.parts[1], a[0].value, xs, out)
                    continue
                if base_op in FOLD:
                    v = FOLD[base_op](xs[0].v, xs[1].v)
                elif base_op == "not":
                    v = ~xs[0].v
                elif base_op == "shl":
                    v = xs[0].v << xs[1].v if xs[1].v < 32 else 0
                else:
                    sv = xs[0].v - (1 << 32) if ins.parts[1] == "s32" and xs[0].v >> 31 else xs[0].v
                    v = sv >> min(xs[1].v, 32)
                define(a[0].value, Const(v & 0xFFFFFFFF), out)
            elif base_op in ("max", "min", "sub") and ins.parts[1] in ("s32", "u32"):
                x, y = val(a[1]), val(a[2])
                signed = ins.parts[1] == "s32"
                if isinstance(x, Const) and isinstance(y, Const):
                    if base_op == "sub":
                        v = x.v - y.v
                    else:
                        lt = compare_const("lt", x.v, y.v, signed)
                        v = (y.v if lt else x.v) if base_op == "max" else (x.v if lt else y.v)
                    define(a[0].value, Const(v & 0xFFFFFFFF), out)
                    continue
                if base_op == "sub" and isinstance(y, Const):
                    r = dest(a[0].value)
                    out.append(KInstr("add", rd=r, ra=reg_of(x, out), imm=-y.v & 0xFFFFFFFF,
                                      useimm=True))
                    vals[a[0].value] = Reg(r)
                    continue
                if base_op != "sub" and isinstance(x, Const):
                    x, y = y, x
                ra = reg_of(x, out)
                rb = 0 if isinstance(y, Const) else reg_of(y, out)
                r = dest(a[0].value)
                out.append(KInstr(base_op, rd=r, ra=ra, rb=rb, imm=y.v if isinstance(y, Const)
                                  else 0, useimm=isinstance(y, Const), signed=signed))
                vals[a[0].value] = Reg(r)
            elif op in ("div.u32", "div.s32", "rem.u32", "rem.s32"):
                div_rem(op, a[0].value, val(a[1]), val(a[2]), out)
            elif op == "neg.s32":
                x = val(a[1])
                if isinstance(x, Const):
                    define(a[0].value, Const(-x.v & 0xFFFFFFFF), out)
                    continue
                r = dest(a[0].value)
                out.append(KInstr("sub", rd=r, ra=VZERO, rb=reg_of(x, out)))
                vals[a[0].value] = Reg(r)
            elif op in ("mul.hi.u32", "mul.hi.s32"):
                x, y = val(a[1]), val(a[2])
                if isinstance(x, Const) and isinstance(y, Const):
                    sx, sy = ((v.v - (1 << 32) if op.endswith("s32") and v.v >> 31 else v.v)
                              for v in (x, y))
                    define(a[0].value, Const((sx * sy >> 32) & 0xFFFFFFFF), out)
                    continue
                if isinstance(x, Const):
                    x, y = y, x
                vals[a[0].value] = Reg(mul_hi(x, y, op.endswith("s32"), a[0].value, out))
            elif op == "selp.b32" or op == "selp.s32" or op == "selp.u32":
                x, y, p = val(a[1]), val(a[2]), vals[a[3].value]
                if isinstance(p, Known):
                    define(a[0].value, x if p.v else y, out)
                    continue
                emit_setp(p, SEL_P, out)
                ra = reg_of(x, out)
                rb = 0 if isinstance(y, Const) else reg_of(y, out)
                r = dest(a[0].value)
                out.append(KInstr("selp", rd=r, ra=ra, rb=rb, imm=y.v if isinstance(y, Const)
                                  else 0, useimm=isinstance(y, Const), guard=SEL_P))
                vals[a[0].value] = Reg(r)
            elif op in ("or.pred", "and.pred"):
                p, q = val(a[1]), val(a[2])
                if isinstance(p, Known) or isinstance(q, Known):
                    kp, other = (p, q) if isinstance(p, Known) else (q, p)
                    absorb = kp.v if op == "or.pred" else not kp.v
                    vals[a[0].value] = Known(kp.v) if absorb else other
                    continue
                if not (isinstance(p, Pred) and isinstance(q, Pred)):
                    raise Unsupported(f"{op} of {p}, {q}")
                # (a >= n) or (b >= n) is max(a, b) >= n; and is min; the other way round for <
                if not (p.cmp == q.cmp and p.signed == q.signed and p.rhs == q.rhs
                        and p.cmp in MIRROR and p.cmp not in ("eq", "ne")):
                    # any two: each as a 0 / 1 register (setp, selp), and / or, then != 0
                    bits = []
                    for x in (p, q):
                        emit_setp(x, SEL_P, out)
                        b = fresh()
                        out.append(KInstr("selp", rd=b, ra=reg_of(Const(1), out), imm=0, useimm=True,
                                          guard=SEL_P))
                        bits.append(b)
                    r = fresh()
                    out.append(KInstr(op.split(".")[0], rd=r, ra=bits[0], rb=bits[1]))
                    vals[a[0].value] = Pred("ne", False, r, Const(0))
                    continue
                upper = p.cmp in ("ge", "gt")
                kind = "max" if upper == (op == "or.pred") else "min"
                r = fresh()
                out.append(KInstr(kind, rd=r, ra=p.ra, rb=q.ra, signed=p.signed))
                vals[a[0].value] = Pred(p.cmp, p.signed, r, p.rhs)
            elif op == "add.s64" or (op == "add.s32" and any(
                    isinstance(v, (Ptr, Addr)) or (isinstance(v, Reg) and v.kind == "ptr")
                    for v in (val(a[1]), val(a[2])))):
                # 64-bit pointers, or nvcc's 32-bit shared-memory addresses
                x, y = val(a[1]), val(a[2])
                if isinstance(x, (Scaled, Const)) and not isinstance(y, (Scaled, Const)):
                    x, y = y, x
                if isinstance(x, Ptr) and isinstance(y, Scaled):
                    v = Addr(words_reg(y, out), x.word)
                elif isinstance(x, Addr) and isinstance(y, Scaled):
                    v = Addr(words_reg(y, out, plus=x.r), x.word)
                elif isinstance(x, Ptr) and isinstance(y, Const):
                    v = Ptr(x.word + whole_words(y))
                elif isinstance(x, Addr) and isinstance(y, Const):
                    v = Addr(x.r, x.word + whole_words(y))
                elif isinstance(x, Reg) and x.kind == "ptr" and isinstance(y, Const):
                    r = dest(a[0].value)
                    out.append(KInstr("add", rd=r, ra=x.r, imm=whole_words(y), useimm=True))
                    v = Reg(r, "ptr")
                elif isinstance(x, Reg) and x.kind == "ptr" and isinstance(y, Scaled):
                    r = dest(a[0].value)
                    out.append(KInstr("add", rd=r, ra=x.r, rb=words_reg(y, out)))
                    v = Reg(r, "ptr")
                else:
                    raise Unsupported(f"{op} of {x}, {y}")
                define(a[0].value, v, out, ptr=True)
            elif op == "add.s32":
                x, y = val(a[1]), val(a[2])
                if isinstance(x, Const) and isinstance(y, Const):
                    define(a[0].value, Const((x.v + y.v) & 0xFFFFFFFF), out)
                    continue
                if isinstance(x, Const):
                    x, y = y, x
                ra = reg_of(x, out)
                rb = None if isinstance(y, Const) else reg_of(y, out)
                r = dest(a[0].value)
                if rb is None:
                    out.append(KInstr("add", rd=r, ra=ra, imm=y.v, useimm=True))
                else:
                    out.append(KInstr("add", rd=r, ra=ra, rb=rb))
                vals[a[0].value] = Reg(r)
            elif op in ("cvt.s64.s32", "cvt.u64.u32"):
                x = val(a[1])  # widening is invisible to word-index addressing
                if isinstance(x, (Gather, Scaled)):
                    x = Reg(reg_of(x, out))
                if not isinstance(x, (Reg, Const)):
                    raise Unsupported(f"{op} of {x}")
                if isinstance(x, Reg) and x.r in mreg.values() and not multi(a[0].value):
                    t = fresh()  # a loop variable: keep this value, not its register
                    out.append(KInstr("add", rd=t, ra=x.r, imm=0, useimm=True))
                    x = Reg(t)
                define(a[0].value, Wide(x), out)
            elif op == "cvt.u32.u64":
                x = as_wide(val(a[1]))
                if not isinstance(x, Wide):
                    raise Unsupported(f"{op} of {x}")
                define(a[0].value, x.v, out)
            elif op == "and.b64":
                x, y = val(a[1]), val(a[2])
                if isinstance(x, Const):
                    x, y = y, x
                if not (isinstance(x, Wide) and isinstance(y, Const) and y.v % 4 == 0):
                    raise Unsupported(f"and.b64 of {x}, {y}")
                if isinstance(x.v, Const):
                    vals[a[0].value] = Const(x.v.v & y.v)
                    continue
                # a byte offset masked to whole words: the word index is (r & mask) >> 2
                t, wd = fresh(), fresh()
                out.append(KInstr("and", rd=t, ra=x.v.r, imm=y.v & 0xFFFFFFFF, useimm=True))
                out.append(KInstr("mulhi", rd=wd, ra=t, imm=1 << 30, useimm=True))
                vals[a[0].value] = Scaled(wd)
            elif op == "shl.b64":
                x, y = as_wide(val(a[1])), val(a[2])
                if not (isinstance(x, Wide) and isinstance(y, Const) and 2 <= y.v < 32):
                    raise Unsupported(f"shl.b64 of {x}, {y}")
                vals[a[0].value] = (Const(x.v.v << y.v) if isinstance(x.v, Const)
                                    else scaled(x.v.r, 1 << (y.v - 2), out))
            elif op in ("mul.wide.s32", "mul.wide.u32"):
                x, y = val(a[1]), val(a[2])
                if not (isinstance(y, Const) and y.v % 4 == 0 and y.v):
                    raise Unsupported(f"mul.wide by {y}, not a multiple of 4")
                if isinstance(x, Const):
                    define(a[0].value, Const(y.v * x.v), out)
                    continue
                if isinstance(x, Gather):
                    x = Reg(reg_of(x, out))
                if isinstance(x, Scaled):  # x = 4 k r already (a shifted index): 4 k y r bytes
                    vals[a[0].value] = Scaled(x.r, x.k * y.v)
                    continue
                if not isinstance(x, Reg):
                    raise Unsupported(f"mul.wide of {x}")
                vals[a[0].value] = scaled(x.r, y.v // 4, out)
            elif op.startswith("setp."):
                _, cmp, ty = op.split(".")
                if ty not in ("s32", "u32", "b32"):
                    raise Unsupported(op)
                x, y = val(a[1], fold=True), val(a[2], fold=True)
                signed = ty == "s32"
                if isinstance(x, Const) and isinstance(y, Const):
                    vals[a[0].value] = Known(compare_const(cmp, x.v, y.v, signed))
                    continue
                if isinstance(x, Const):
                    x, y, cmp = y, x, MIRROR[cmp]
                rhs = y if isinstance(y, Const) else Reg(reg_of(y, out))
                vals[a[0].value] = Pred(cmp, signed, reg_of(x, out), rhs)
            elif base_op == "bra" and pmode:
                t = bb_at[k.labels[a[0].value]]
                p = None if ins.guard is None else vals[ins.guard]
                if isinstance(p, Known):
                    p = None if p.v != ins.neg else "never"
                edge[bi] = (p, ins.neg, t)
                if p == "never":
                    found_dead.add((bi, t))
                elif p is None and ins.guard is not None:
                    found_dead.add((bi, bi + 1))
            elif base_op == "bra":
                t = bb_at[k.labels[a[0].value]]
                if ins.guard is None:
                    terms[gi] = ("jump", t)
                    branched = True
                    continue
                p = vals[ins.guard]
                if isinstance(p, Known):
                    if p.v != ins.neg:
                        terms[gi] = ("jump", t)
                        branched = True
                    continue
                if t in ret_blocks and (not branched or i in late_exit) and \
                        ins.guard not in uniform:
                    if ins.neg or exit_pred:
                        raise Unsupported("more than one early exit, or @!p bra to ret")
                    exit_pred.append(p)
                    emit_setp(p, EXIT_P, out)  # early: nearby stores of this group reuse it
                    exit_at[gi] = len(out) - 1
                    continue
                if ins.guard not in uniform:
                    raise Unsupported(f"divergent branch on {ins.guard}")
                terms[gi] = ("cond", p, ins.neg, t)
                branched = True
            elif base_op in ("membar", "fence"):
                # __threadfence*(): one warp runs at a time and its stores leave in program
                # order over FIFO links, so every store is visible to every observer, local or
                # peer, before any later one: the ordering a fence asks for already holds
                pass
            elif base_op in ("bar", "barrier") and ins.parts[1:] == ["sync"] and a[0].value == 0:
                # a barrier ends the group: every warp of the thread block runs up to here
                # before any goes on (circuit: warps take turns, phase by phase)
                commit(out)
                out, gi = barrier(gi)
            elif base_op == "shfl":
                out, gi = shuffle(ins, bi, out, gi)
            elif base_op == "ld" and access(ins)[1] in ("global", "shared") and \
                    ins.parts[-1][1:] == "32":
                ra, w = word_addr(a[1])
                r = dest(a[0].value)
                out.append(KInstr("ld", rd=r, ra=ra, imm=w))
                vals[a[0].value] = Reg(r)
            elif base_op in ("st", "atom") and ("global" in ins.parts or "shared" in ins.parts) \
                    and ins.parts[-1][1:] == "32" and (base_op == "atom" or len(access(ins)) == 3):
                if "shared" in ins.parts and base_op == "atom":
                    raise Unsupported("shared atomics")
                if base_op == "atom":
                    if ins.parts[-2:] != ["add", "u32"] and ins.parts[-2:] != ["add", "s32"]:
                        raise Unsupported(op)
                    if any(o.kind == "reg" and o.value == a[0].value
                           for other in k.instrs if other is not ins for o in other.args[1:]):
                        raise Unsupported("an atomic whose old value is used")
                    a = a[1:]  # [address], value
                ra, w = word_addr(a[0])
                src = reg_of(val(a[1]), out)
                g = {}
                if pmode and not full[bi]:
                    out.append(KInstr("setp", ra=mask[bi], imm=0, useimm=True, pd=ACTIVE_P,
                                      cmp="ne"))
                    g = {"guard": ACTIVE_P}
                elif exit_pred:
                    # reuse it only within a block's reach (predicates never cross blocks),
                    # else recompute it next to the store. A write-back store (stw) cannot
                    # share a block with a load, so between the two it is always recomputed
                    wb = base_op == "st" and ("shared" in ins.parts or write_through)
                    if wb or gi not in exit_at or len(out) - exit_at[gi] > stages - 2:
                        emit_setp(exit_pred[0], EXIT_P, out)
                        exit_at[gi] = len(out) - 1
                    g = {"guard": EXIT_P, "gneg": True}
                # a shared store writes through to memory later loads read: stw
                kop = "stw" if base_op == "st" and ("shared" in ins.parts or write_through) \
                    else base_op
                out.append(KInstr(kop, ra=ra, rb=src, imm=w, **g))
            else:
                raise Unsupported(op)
            commit(out)
        commit(out)  # a block's masked writes land before its masks move on
        if pmode and bi not in ret_blocks:
            emit_edges(bi, out, gi)

    if pmode:  # a block no edge reaches (a launch-time-false branch): no lane runs it
        code[0][1:1] = [KInstr("add", rd=mask[b], ra=VZERO, imm=0, useimm=True)
                        for b in range(1, len(bbs))
                        if b not in assigned | accumulate | ret_blocks and not full[b]]

    # the setp of a cond branch goes last in its block, so it is in the final machine block
    for bi, t in enumerate(terms):
        if t[0] == "cond":
            emit_setp(t[1], BRANCH_P, code[bi])
            terms[bi] = ("cond", t[2], t[3])

    # branch targets as groups
    for gi, t in enumerate(terms):
        if t[0] in ("jump", "cond", "cond_any"):
            terms[gi] = t[:-1] + (group_of[t[-1]],)
    ng = len(code)

    # liveness over the group CFG, on machine registers
    def succs(gi):
        t = terms[gi]
        nxt = [gi + 1] if gi + 1 < ng else []
        return {"next": nxt, "bar": nxt, "end": [], "jump": [t[-1]], "cond": nxt + [t[-1]],
                "cond_any": nxt + [t[-1]]}[t[0]]

    use, dfn = [], []
    for instrs in code:
        u, d = set(), set()
        for ins in instrs:
            rd, srcs = ins.rw()
            u |= {s for s in srcs if s not in d}
            if rd is not None:
                d.add(rd)
        use.append(u)
        dfn.append(d)
    live_in = [set() for _ in range(ng)]
    live_out = [set() for _ in range(ng)]
    changed = True
    while changed:
        changed = False
        for bi in reversed(range(ng)):
            lo = set().union(*(live_in[s] for s in succs(bi)))
            li = use[bi] | (lo - dfn[bi])
            if lo != live_out[bi] or li != live_in[bi]:
                live_out[bi], live_in[bi], changed = lo, li, True

    # register allocation: color the interference graph of the virtual registers
    code, live_out = allocate(code, live_out)

    # machine blocks: each group split into stage-sized pieces
    place = placer(stages, mul_stages, logic_stages, atom_stages)
    pieces = []
    for gi, instrs in enumerate(code):
        try:
            pieces.append(split(instrs, {r for r in live_out[gi] if r < NREG}, stages, lmax,
                                place) if instrs else [])
        except AssertionError as e:
            raise Unsupported(f"no block split for group {gi}: {e}") from None
    first, idx = [], 0
    for ps in pieces:
        first.append(idx)
        idx += max(len(ps), 1)
    blocks: list[MBlock] = []
    for bi, ps in enumerate(pieces):
        blocks += [MBlock(p.instrs, p.live, wpb=wpb) for p in ps] or [MBlock([], [], wpb=wpb)]
        t, last = terms[bi], blocks[-1]
        if t[0] == "bar":
            last.branch = ("bar",)
        elif t[0] == "end":
            last.branch = ("end",)
        elif t[0] == "jump":
            last.branch = ("jump", first[t[1]])
        elif t[0] == "cond":
            stage = max(s for s, ins in enumerate(last.instrs)
                        if ins.op == "setp" and ins.pd == BRANCH_P)
            last.branch = ("cond", stage, t[1], first[t[2]])
        elif t[0] == "cond_any":
            stage = max(s for s, ins in enumerate(last.instrs)
                        if ins.op == "setp" and ins.pd == BRANCH_P)
            last.branch = ("cond", stage, False, first[t[1]], True)
    if blocks[-1].branch[0] == "next":
        raise Unsupported("control falls off the end")
    # falling into an empty block that only ends the warp: end right here (one token less)
    for b, blk in enumerate(blocks[:-1]):
        nxt = blocks[b + 1]
        if blk.branch == ("next",) and not nxt.instrs and nxt.branch == ("end",):
            blk.branch = ("end",)
    return Launch(blocks, mem, gx * gy * bx * by // lanes, base, frozenset(found_dead))
