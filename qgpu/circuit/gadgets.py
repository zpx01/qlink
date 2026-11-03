"""Circuit gadgets on top of Qwenputer's q35 compiler (chipformer.q35, MIT, Divyansh Agrawal).

`sel`, `chan`, `thead` come from Qwenputer's RISC-V fork (experimental/riscv/chipformer/q35/
prims.py); `counter`, `thermo`, `stage1_digits`, `adder_stage2`, `add_nibbles` from its
rv/core.py. They are copied unchanged in behaviour so we depend only on the main tree's IR.
"""

from chipformer.q35.ir import A_BIT, A_REF, REF_COLS, Chan, K
from chipformer.q35.prims import Circ, from_bits


def sel(c: Circ, e):
    """Key / query pre-activation K * (2e - 1) for a 0/1 (or <= 0 when false) form e."""
    if isinstance(e, Chan):
        return Chan(e.expr * (2 * K) - c.one * K, e.taps)
    return Chan(e * (2 * K) - c.one * K)


def chan(e, scale):
    if isinstance(e, Chan):
        return Chan(e.expr * scale, e.taps)
    return Chan(e * scale)


def thead(c: Circ, name, keys, queries, vheads, tiny=0.0):
    """A head of 0/1 tables: one dict per value head with data {column: 0/1 form or Chan},
    ref (1 where a written row carries the ballast), optional gate (0/1: its outputs), beta
    (write gate) and into (dims to add the outputs onto). Returns each value head's outputs."""
    specs = []
    for x in vheads:
        v = {col: chan(e, A_BIT) for col, e in x["data"].items()}
        for col in REF_COLS:
            v[col] = chan(x["ref"], A_REF)
        gate = c.one if x.get("gate") is None else x["gate"]
        cols = sorted(x["data"])
        specs.append({"v": v, "a": x.get("a", 0), "z": {col: K * gate for col in cols},
                      "outs": [(col, 2.0**7) for col in cols], "beta": x.get("beta"),
                      "into": x.get("into")})
    first, rest = specs[0], specs[1:]
    outs = c.head(name, {s: sel(c, e) for s, e in queries.items()},
                  {s: sel(c, e) for s, e in keys.items()}, first["v"], first["a"], first["z"],
                  first["outs"], tiny=tiny, beta=first["beta"], into=first["into"], vx=rest)
    res, i = [], 0
    for x in specs:
        n = len(x["outs"])
        res.append(outs[i:i + n])
        i += n
    return res


def counter(c: Circ, name, reset, step):
    """A 2^-k counter head: k = the `step` tokens since the latest `reset` token (0 there).

    Row 0 holds the data bit written at the reset; row 1 is a ballast-only row rewritten at
    every other token. Steps halve the state (a = 110), resets clear it (a = 238), and the
    2-hot read sums the two rows, so the ballast stays fixed and the data reads 2^-k exactly."""
    one = c.one
    ((x,),) = thead(c, name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                    [{"data": {2: reset}, "ref": one - reset,
                      "a": 110 * step + 238 * reset}], tiny=2.0**-90)
    return x


def thermo(c: Circ, x, reset, n, name):
    """[k > m] for m < n, from x = 2^-k (0 at reset tokens)."""
    one = c.one
    return [c.node([(1, one - reset, 7 * one - 8 * 2**m * x),
                    (-1, one - reset, 6 * one - 8 * 2**m * x)], f"{name}>{m}")
            for m in range(n)]


def adder_stage2(c: Circ, g, p_, d, n_nib, name):
    """Result bits of a carry-lookahead add from per-nibble generate / propagate / digit bits;
    also the carry out of the top nibble."""
    one = c.one

    def carry_sources(k):
        """Mutually exclusive literal sets, one of them true iff a carry enters nibble k."""
        return [[g[j]] + p_[j + 1:k] for j in range(k)]

    r = []
    for k in range(n_nib):
        for b in range(4):
            terms = c.t_and([d[k][b]])
            for src in carry_sources(k):
                low = d[k][:b]
                terms += c.t_and(src + low + [one - d[k][b]])
                terms += c.t_and(src + low + [d[k][b]], -1)
            r.append(c.node(terms, f"{name}.r{4 * k + b}"))
    cout = c.node([t for src in carry_sources(n_nib) for t in c.t_and(src)], f"{name}.cout")
    return r, cout


def stage1_digits(c: Circ, sums, name, hi):
    """Generate, propagate and digit bits of nibble sums; sums[k] = [(int Expr, literals)]."""
    g, p_, d = [], [], []
    for k, forms in enumerate(sums):
        gt, pt, dt = [], [], [[] for _ in range(4)]
        for s, lits in forms:
            gt += c.t_ge(s, 16, 0, hi, lits=lits)
            pt += c.t_eq(s, 15, 0, hi, lits=lits)
            for b in range(4):
                for theta in range(1, hi + 1):
                    dd = (theta % 16 >> b & 1) - ((theta - 1) % 16 >> b & 1)
                    if dd:
                        dt[b] += c.t_ge(s, theta, 0, hi, dd, lits=lits)
        g.append(c.node(gt, f"{name}.g{k}"))
        p_.append(c.node(pt, f"{name}.p{k}"))
        d.append([c.node(dt[b], f"{name}.d{k}.{b}") for b in range(4)])
    return g, p_, d


def add_nibbles(c: Circ, a_bits, b_bits, n_nib, lits, name):
    """A plain carry-lookahead add of two bit vectors (n_nib nibbles), gated by literals."""
    sums = [[(from_bits(a_bits[4 * k:4 * k + 4]) + from_bits(b_bits[4 * k:4 * k + 4]), lits)]
            for k in range(n_nib)]
    g, p_, d = stage1_digits(c, sums, name, 30)
    r, _ = adder_stage2(c, g, p_, d, n_nib, name)
    return r
