"""Bursts: a block of up to S warp instructions per forward pass; only live-outs use tokens.

A SIMT warp (add.s32 over LANES lanes, NREG registers) where one record runs a whole block
(blocks.py forms them):

    EXEC(b)  VAL x 16 per live-out, in the driver's order        then EXEC(b + 1)

Inside the pass, stage s fetches instruction s of block b from its own ROM head, reads its
operands either from the committed register file or, when an earlier stage of the same block
wrote that register, from that stage's result (forwarding in the MLP), and adds. Values move
between stages as nibble integers 0..15: the adder needs only nibble sums, so bits are made only
for the 16-bit chunk being emitted.

Host records: PROG(b, s) VAL(rd | ra << 4 | rb << 8 | 1 << 12) per instruction,
LIVE(b) VAL(count | src0 << 3 | src1 << 5 ...) per block, LOAD(r) VAL x 16 per register.
"""

from dataclasses import dataclass

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Chan, Target
from chipformer.q35.prims import Circ, from_bits

from qgpu.circuit.blocks import Block, Instr, split
from qgpu.circuit.gadgets import counter, thead, thermo

LANES = 8
NREG = 16
S = 4  # stages per pass
LMAX = 2  # live-outs per block
BBITS = 4  # up to 16 blocks
CHUNKS = 2 * LANES
GROUPS = LANES // 2
NIB = 8  # nibbles per 32-bit lane value

VAL_MAX = 1 << 16
EXEC0 = VAL_MAX
LOAD0 = EXEC0 + 256
PROG0 = LOAD0 + 256
LIVE0 = PROG0 + 256


def EXEC(b: int) -> int:
    return EXEC0 + b


def LOAD(r: int) -> int:
    return LOAD0 + r


def PROG(b: int, s: int) -> int:
    return PROG0 + 8 * b + s


def LIVE(b: int) -> int:
    return LIVE0 + b


def kind(t: int) -> str:
    if t < VAL_MAX:
        return "val"
    if t < LOAD0:
        return "exec"
    if t < PROG0:
        return "load"
    if t < LIVE0:
        return "prog"
    return "live"


def chunks(values: list[int]) -> list[int]:
    return [(values[k // 2] >> 16 * (k % 2)) & 0xFFFF for k in range(CHUNKS)]


# --- reference -----------------------------------------------------------------------------


@dataclass
class Trace:
    tokens: list[int]
    owned: list[int]
    regs: list[list[int]]
    blocks: list[Block]


def trace(program: list[Instr], init: dict[int, list[int]], outputs: set[int],
          stages: int = S, lmax: int = LMAX) -> Trace:
    blocks = split(program, outputs, stages, lmax)
    assert len(blocks) < 1 << BBITS
    regs = [[0] * LANES for _ in range(NREG)]
    toks: list[int] = []
    for b, blk in enumerate(blocks):
        for s, (rd, ra, rb) in enumerate(blk.instrs):
            toks += [PROG(b, s), rd | ra << 4 | rb << 8 | 1 << 12]
        enc = len(blk.live) | sum(s << 3 + 2 * i for i, (s, _) in enumerate(blk.live))
        toks += [LIVE(b), enc]
    for r, vals in init.items():
        regs[r] = [v & 0xFFFFFFFF for v in vals]
        toks += [LOAD(r)] + chunks(regs[r])
    owned = []
    for b, blk in enumerate(blocks):
        results = []
        for rd, ra, rb in blk.instrs:
            results.append([(x + y) & 0xFFFFFFFF for x, y in zip(regs[ra], regs[rb])])
            regs[rd] = results[-1]
        start = len(toks)
        toks.append(EXEC(b))
        for s, _ in blk.live:
            toks += chunks(results[s])
        owned += list(range(start, len(toks)))
    toks.append(EXEC(len(blocks)))
    return Trace(toks, owned, regs, blocks)


# --- circuit -------------------------------------------------------------------------------


class Burst(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or Target())
        c, one = self, self.one

        def feat(name, fn):
            return c.token_feature(name, fn, full=True)

        def lag1(e):
            return Chan(e, (0.0, 1.0, 0.0, 0.0))

        is_val = feat("is_val", lambda t: int(kind(t) == "val"))
        is_exec = feat("is_exec", lambda t: int(kind(t) == "exec"))
        is_load = feat("is_load", lambda t: int(kind(t) == "load"))
        start = (is_exec + is_load + feat("is_prog", lambda t: int(kind(t) == "prog"))
                 + feat("is_live", lambda t: int(kind(t) == "live")))
        pay = [feat(f"pay{b}", lambda t, b=b: int(kind(t) == "val" and t >> b & 1))
               for b in range(16)]
        blk_b = [feat(f"blk{b}", lambda t, b=b: int(kind(t) == "exec" and (t - EXEC0) >> b & 1))
                 for b in range(BBITS)]
        ldr = [feat(f"ldr{b}", lambda t, b=b: int(kind(t) == "load" and (t - LOAD0) >> b & 1))
               for b in range(4)]
        nblk = 1 << BBITS

        # --- record, position ---------------------------------------------------------
        (rec,) = thead(c, "rec", {0: start}, {0: one},
                       [{"data": {2: is_exec, 3: is_load,
                                  **{4 + b: blk_b[b] for b in range(BBITS)},
                                  **{8 + b: ldr[b] for b in range(4)}},
                         "ref": one}])
        r_exec, r_load = rec[0], rec[1]
        r_blk, r_ldr = rec[2:2 + BBITS], rec[2 + BBITS:6 + BBITS]
        jmax = CHUNKS * LMAX
        xj = counter(c, "j", start, is_val)
        jt = thermo(c, xj, start, jmax + 1, "j")  # jt[m] = [j > m]

        def j_is(k: int) -> Expr:
            return (one if k == 0 else jt[k - 1]) - jt[k]

        def j_lt(k: int) -> Expr:
            return one if k > jmax else (one - jt[k - 1] if k > 0 else Expr())

        # --- fetch: one ROM head per stage, and the live-out table ----------------------
        blk_oh = c.onehot_of_bits(r_blk, "blk")
        q = {b: c.and_([r_exec, blk_oh[b]], f"blkq{b}") for b in range(nblk)}
        stage = []
        for s in range(S):
            keys = {b: lag1(feat(f"prog{b},{s}", lambda t, b=b, s=s: int(t == PROG(b, s))))
                    for b in range(nblk)}
            (row,) = thead(c, f"rom{s}", keys, q,
                           [{"data": {2 + i: Chan(pay[i]) for i in range(13)}, "ref": one}])
            stage.append({"rd": row[0:4], "ra": row[4:8], "rb": row[8:12], "valid": row[12]})
        keys = {b: lag1(feat(f"live{b}", lambda t, b=b: int(t == LIVE(b)))) for b in range(nblk)}
        (lv,) = thead(c, "livetab", keys, q,
                      [{"data": {2 + i: Chan(pay[i]) for i in range(3 + 2 * LMAX)}, "ref": one}])
        n_live = lv[0:3]
        src_oh = [c.onehot_of_bits(lv[3 + 2 * i:5 + 2 * i], f"src{i}") for i in range(LMAX)]
        nl_oh = c.onehot_of_bits(n_live[:2], "nlive", LMAX + 1)

        for s, st in enumerate(stage):
            for f in ("rd", "ra", "rb"):
                st[f + "_oh"] = c.onehot_of_bits(st[f], f"s{s}.{f}")

        # --- commits: live-out i, lane group h, at counter j = 16 i + 4 h + 4 -----------
        rdl = [[c.node([t for s in range(S)
                        for t in c.t_and([src_oh[i][s], stage[s]["rd_oh"][r]])], f"rdl{i}.{r}")
                for r in range(NREG)] for i in range(LMAX)]
        ldr_oh = c.onehot_of_bits(r_ldr, "ldr")
        wkeys = []
        for h in range(GROUPS):
            ks = {}
            for r in range(NREG):
                terms = c.t_and([r_load, is_val, j_is(4 * h + 4), ldr_oh[r]])
                for i in range(LMAX):
                    terms += c.t_and([r_exec, is_val, j_is(CHUNKS * i + 4 * h + 4), rdl[i][r]])
                ks[r] = c.node(terms, f"wkey{h}.{r}")
            wkeys.append(ks)
        data = {}
        for lag in range(4):
            pos = 3 - lag
            for b in range(16):
                col = 2 + 32 * (pos // 2) + 16 * (pos % 2) + b
                data[col] = Chan(pay[b], tuple(1.0 if i == lag else 0.0 for i in range(4)))

        # --- committed operand reads, one port pair per stage ---------------------------
        def read(port_oh, name):
            qq = {r: c.and_([r_exec, port_oh[r]], f"{name}q{r}") for r in range(NREG)}
            lanes = [None] * LANES
            for h in range(GROUPS):
                (row,) = thead(c, f"{name}{h}", wkeys[h], qq, [{"data": data, "ref": one}])
                lanes[2 * h], lanes[2 * h + 1] = row[0:32], row[32:64]
            return [[from_bits(bits[4 * k:4 * k + 4]) for k in range(NIB)] for bits in lanes]

        # --- stages ----------------------------------------------------------------------
        self.vals = []  # vals[s][lane][nibble]: stage s result as nibble Exprs
        for s, st in enumerate(stage):
            ops = []
            for port in ("ra", "rb"):
                committed = read(st[port + "_oh"], f"s{s}{port}")
                # eq[s2]: stage s2 < s is valid and wrote this operand's register
                eq = [c.node([t for r in range(NREG) for t in
                              c.t_and([stage[s2]["valid"], stage[s2]["rd_oh"][r],
                                       st[port + "_oh"][r]])], f"s{s}{port}.eq{s2}")
                      for s2 in range(s)]
                # the latest such stage wins; none: the committed value
                win = [[eq[s2]] + [one - e for e in eq[s2 + 1:]] for s2 in range(s)]
                none = [one - e for e in eq]
                lanes = []
                for L in range(LANES):
                    nibs = []
                    for k in range(NIB):
                        terms = c.t_and(none + [r_exec], a=committed[L][k]) if s else []
                        for s2 in range(s):
                            terms += c.t_and(win[s2], a=self.vals[s2][L][k])
                        nibs.append(c.node(terms, f"s{s}{port}.L{L}.n{k}") if s
                                    else committed[L][k])
                    lanes.append(nibs)
                ops.append(lanes)
            self.vals.append([add_nib(c, ops[0][L], ops[1][L], f"s{s}.add{L}")
                              for L in range(LANES)])

        # --- next token --------------------------------------------------------------------
        # j < 16 n: VAL chunk j % 16 of live-out j // 16;  j == 16 n: EXEC(b + 1)
        last = c.node([t for n in range(LMAX + 1)
                       for t in c.t_and([r_exec, nl_oh[n], j_is(CHUNKS * n)])], "last")
        self.out_exec = last
        self.out_val = c.node([t for n in range(1, LMAX + 1)
                               for t in c.t_and([r_exec, nl_oh[n]], a=j_lt(CHUNKS * n))],
                              "out.val")
        has = [sum((nl_oh[n] for n in range(i + 1, LMAX + 1)), Expr())
               for i in range(LMAX)]  # [live-out i exists]
        slots = []  # the 4 nibbles of the chunk to emit
        for n in range(4):
            terms = []
            for i in range(LMAX):
                for k in range(CHUNKS):
                    for s in range(S):
                        terms += c.t_and([r_exec, has[i], j_is(CHUNKS * i + k), src_oh[i][s]],
                                         a=self.vals[s][k // 2][4 * (k % 2) + n])
            slots.append(c.node(terms, f"slot{n}"))
        carry = [one] + [c.and_(r_blk[:i], f"inc.c{i}") for i in range(1, BBITS)]
        self.out_bits = []
        for n in range(4):
            bits = nib_bits(c, slots[n], f"out.n{n}")
            for b in range(4):
                terms = [] if bits[b] is None else c.t_and([bits[b]])
                bb = 4 * n + b
                if bb < BBITS:  # EXEC payload: (b + 1) bit bb
                    terms += c.t_and([last, r_blk[bb], one - carry[bb]])
                    terms += c.t_and([last, one - r_blk[bb], carry[bb]])
                self.out_bits.append(c.node(terms, f"out.b{bb}"))


def add_nib(c: Circ, a: list, b: list, name: str, carry_in: Expr | None = None,
            with_cout: bool = False, n: int = NIB):
    """(a + b + carry_in) mod 2^32 on nibble integers: per-nibble generate / propagate, carry
    lookahead, then each result nibble is linear: s + carry_in - 16 carry_out. With `with_cout`
    also returns the carry out of the top nibble (for compares)."""
    sums = [a[k] + b[k] for k in range(n)]
    hi = 30
    if carry_in is not None:  # a 0/1 form, added to the lowest nibble
        sums[0] = sums[0] + carry_in
        hi = 31
    g = [c.node(c.t_ge(sums[k], 16, 0, hi), f"{name}.g{k}") for k in range(n)]
    p = [c.node(c.t_eq(sums[k], 15, 0, hi), f"{name}.p{k}") for k in range(n)]

    def sources(k):  # mutually exclusive: a carry enters nibble k
        return [[g[j]] + p[j + 1:k] for j in range(k)]

    out = []
    for k in range(n):
        cin = c.node([t for src in sources(k) for t in c.t_and(src)], f"{name}.c{k}") \
            if k else Expr()
        cout = g[k] + (c.node([t for src in sources(k) for t in c.t_and(src + [p[k]])],
                              f"{name}.pc{k}") if k else Expr())
        out.append(sums[k] + cin - 16 * cout)
    if with_cout:
        top = c.node([t for src in sources(n) for t in c.t_and(src)], f"{name}.cout")
        return out, top
    return out


def mul_nib(c: Circ, a: list, b: list, addend: list, name: str) -> list:
    """(a * b + addend) mod 2^32 on nibble integers (mad.lo.s32: the low 32 bits are the same
    for signed and unsigned operands).

    A partial product a_i * b_j (<= 225) is one MLP unit: a unit computes coef * a * relu(b)
    exactly when b is a nibble. Each is split into high and low nibbles (14 thresholds), so a
    column sums at most 8 low + 7 high nibbles + the addend (<= 233, a bf16 integer). Columns
    are split again (column = 16 q + r) and the result is the add of r and q shifted one nibble,
    whose nibble sums stay <= 29."""
    lo = {k: [] for k in range(NIB)}
    hi = {k: [] for k in range(NIB)}
    for i in range(NIB):
        for j in range(NIB - i):
            p = c.node([(1, a[i], b[j])], f"{name}.p{i}{j}")
            h = c.node([t for th in range(1, 15) for t in c.t_ge(p, 16 * th, 0, 225)],
                       f"{name}.h{i}{j}")
            lo[i + j].append(p - 16 * h)
            if i + j + 1 < NIB:
                hi[i + j + 1].append(h)
    cols = [sum(lo[k] + hi[k], Expr()) + addend[k] for k in range(NIB)]
    q = [c.node([t for th in range(1, 15) for t in c.t_ge(cols[k], 16 * th, 0, 233)],
                f"{name}.q{k}") for k in range(NIB)]
    r = [cols[k] - 16 * q[k] for k in range(NIB)]
    return add_nib(c, r, [Expr()] + q[:NIB - 1], f"{name}.sum")


def mul_nib_wide(c: Circ, a: list, b: list, addend: list, name: str) -> tuple[list, list]:
    """The full 64-bit a * b + addend on nibble integers: (low 8 nibbles, high 8 nibbles), the
    low half as in mul_nib (mad.lo) and the high half for mul.hi (with a zero addend). All 64
    partial products are split; a column sums at most 8 low + 8 high nibbles + the addend's
    (<= 247, still a bf16 integer)."""
    lo = {k: [] for k in range(2 * NIB)}
    hi = {k: [] for k in range(2 * NIB)}
    for i in range(NIB):
        for j in range(NIB):
            p = c.node([(1, a[i], b[j])], f"{name}.p{i}{j}")
            h = c.node([t for th in range(1, 15) for t in c.t_ge(p, 16 * th, 0, 225)],
                       f"{name}.h{i}{j}")
            lo[i + j].append(p - 16 * h)
            if i + j + 1 < 2 * NIB:
                hi[i + j + 1].append(h)
    cols = [sum(lo[k] + hi[k], Expr()) + (addend[k] if k < NIB else Expr())
            for k in range(2 * NIB)]
    q = [c.node([t for th in range(1, 16) for t in c.t_ge(cols[k], 16 * th, 0, 247)],
                f"{name}.q{k}") for k in range(2 * NIB)]
    r = [cols[k] - 16 * q[k] for k in range(2 * NIB)]
    out = add_nib(c, r, [Expr()] + q[:2 * NIB - 1], f"{name}.sum", n=2 * NIB)
    return out[:NIB], out[NIB:]


def nib_bits(c: Circ, v: Expr, name: str) -> list:
    """Bits of an integer nibble 0..15 (one node per bit, summed thresholds at its flips)."""
    out = []
    for b in range(4):
        terms = []
        for theta in range(1, 16):
            d = (theta >> b & 1) - ((theta - 1) >> b & 1)
            if d:
                terms += c.t_ge(v, theta, 0, 15, d)
        out.append(c.node(terms, f"{name}.b{b}"))
    return out


def dim(e: Expr) -> int:
    (d,) = e.terms
    return d


def predicted(c: Burst, m, t: int) -> int | None:
    bits = sum(int(m[t, dim(e)]) << b for b, e in enumerate(c.out_bits) if e.terms)
    if m[t, dim(c.out_val)] == 1:
        return bits
    if m[t, dim(c.out_exec)] == 1:
        return EXEC(bits & ((1 << BBITS) - 1))
    return None
