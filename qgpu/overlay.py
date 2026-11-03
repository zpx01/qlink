"""Overlay: circuit dims that do not matter on text tokens share Qwen's dims in a merged checkpoint.

Qwenputer's `chipformer.q35.overlay` (read its docstring) measures, over a corpus of mixed
sessions, what every dim, FFN unit, head key and output gate does at text positions, plans which
dims can live in Qwen's dims (0 or unneeded on text, every reader and writer switched off on
text by -KILL * is_text on its gate) and applies the switches to the graph. Its analysis is
generic; what ties it to the CHIP-8 bridge is the corpus and which tokens are text. Here every
text token is kernel.TEXT, and the corpus is text around real launches: every kernel of the
suite, both compilers, lowered for the circuit's own lanes and stages.
"""

from contextlib import contextmanager
from unittest import mock

import numpy as np
from chipformer.q35 import overlay as ov

from qgpu import ptx
from qgpu.circuit.kernel import TEXT, Kernel, run_blocks
from qgpu.lower import lower


def text_mask(tokens: list[int]) -> np.ndarray:
    return np.array([t == TEXT for t in tokens])


@contextmanager
def our_text():
    """Qwenputer's measurement and checks, with kernel.TEXT as the text token."""
    with mock.patch.object(ov, "_text_mask", text_mask):
        yield


def launches(rng: np.random.Generator) -> list[tuple]:
    """(kernel, grid, block, args, buffers): small launches of every kernel in the suite."""
    x = [int(v) for v in rng.integers(0, 2**32, 32)]
    sm = [int(v) for v in rng.integers(0, 16, 16)]
    n8, z8 = 8, [0] * 8
    return [
        ("vadd", (1,), (8,), [0, 1, 2, 6], {0: x[:6], 1: x[6:12], 2: [0] * 6}),
        ("saxpy", (2,), (8,), [-3, 0, 1, 12], {0: x[:12], 1: x[12:24]}),
        ("histogram", (1,), (16,), [0, 1, 13], {0: x[:13], 1: [0] * 16}),
        ("divloop", (1,), (8,), [0, 1, 7], {0: sm[:7], 1: [0] * 7}),
        ("branchy", (1,), (8,), [0, 1, 2, n8, 1], {0: [v % 5 - 2 & 0xFFFFFFFF for v in x[:8]],
                                                    1: z8, 2: z8}),
        ("bitops", (1,), (8,), [0, 1, n8], {0: x[:8], 1: z8}),
        ("diverge", (1,), (8,), [0, 1, n8], {0: x[:8], 1: z8}),
        ("collatz", (1,), (8,), [0, 1, 5], {0: [v % 9 for v in sm[:5]], 1: [0] * 5}),
        ("colsum", (1,), (8,), [0, 1, 8, 2], {0: x[:16], 1: z8}),
        ("matmul", (1, 2), (4, 2), [0, 1, 2, 3], {0: sm[:9], 1: sm[7:16], 2: [0] * 9}),
        ("reduce", (2,), (8,), [0, 1, 13], {0: x[:13], 1: [5]}),
        ("scan", (1,), (16,), [0, 1, 11], {0: x[:11], 1: [0] * 11}),
        ("warpsum", (1,), (32,), [0, 1, 20], {0: x[:20], 1: [0]}),
        ("shfl", (1,), (32,), [0, 1, 2, 9], {0: x[:9], 1: [0] * 9, 2: [0] * 9}),
    ]


def corpus(c: Kernel, cfg: dict, seed: int = 7, docs: int = 6) -> list[list[int]]:
    """Sessions of text and launches: each document runs every kernel once (both compilers
    across documents), in a random order, with text of random length before, between and
    after."""
    rng = np.random.default_rng(seed)
    traces = []
    for src in ("kernels", "kernels/nvcc"):
        for name, grid, block, args, bufs in launches(rng):
            k = ptx.load(f"{src}/{name}.ptx")[name]
            lw = lower(k, grid, block, args, bufs, lanes=c.lanes, **cfg)
            traces.append(run_blocks(lw.blocks, lw.mem, lw.warps, lanes=c.lanes).tokens)
    out = []
    for d in range(docs):
        order = [i for i in rng.permutation(len(traces)) if i % 2 == d % 2]
        doc = [TEXT] * int(rng.integers(1, 6))
        for i in order:
            doc += traces[i] + [TEXT] * int(rng.integers(1, 6))
        out.append(doc)
    return out


def zero_on_text(f):
    return lambda t: 0 if t == TEXT else f(t)


def apply(c: Kernel, p: ov.Plan) -> None:
    """Qwenputer's `overlay.apply` with kernel.TEXT as the text token."""
    off = ov.KILL * c.is_text
    for i, j in p.terms:
        n = c.nodes[i]
        coef, a, b = n.terms[j]
        n.terms[j] = (coef, a, b - off)
    for i, ch_i in p.keys:
        ch = c.nodes[i].k[ch_i]
        c.nodes[i].k[ch_i] = ov.Chan(ch.expr - off, ch.taps)
    for i, x_, col in p.gates:
        z = ov._vz(c.nodes[i], x_)
        z[col] = z[col] - off
    for d in p.dims & set(c.token_fns):  # overlaid features read 0 on text (Qwen's dims)
        c.token_fns[d] = zero_on_text(c.token_fns[d])


QMAX = 1.0  # |Qwen embedding value| the switch-offs must absorb (Qwen3.5 0.8B / 4B: <= 0.23)


def tighten(c: Kernel, act: ov.Activity, p: ov.Plan, qmax: float = QMAX) -> ov.Plan:
    """Drop overlay dims until every switched-off unit, key and output gate stays dead with
    Qwen values up to qmax in the overlay dims (`plan` only reports the bound, q_bound). A dim
    read with a large coefficient, like a counter's thermometer, cannot share Qwen's dims. The
    switch-offs stay: each was allowed (its output does not matter on text, or is 0 there)."""
    dims = set(p.dims)

    def noise(e):
        return sum(abs(float(v)) for d, v in e.terms.items() if d in dims)

    def weak(e, top, margin) -> set[int]:
        """The overlay dims e reads, if Qwen values up to qmax there could wake its switch."""
        s = noise(e)
        if s and (ov.KILL - max(top, 0) - margin) / s < qmax:
            return {d for d in e.terms if d in dims}
        return set()

    while True:
        drop = set()
        for i, j in p.terms:
            drop |= weak(c.nodes[i].terms[j][2], act.gate_max[i, j], 104 / ov.K)
        for i, ch_i in p.keys:
            ch = c.nodes[i].k[ch_i]
            drop |= weak(ch.expr, act.key_max[i, ch_i], 104 / ch.taps[0])
        for i, x_, col in p.gates:
            drop |= weak(ov._vz(c.nodes[i], x_)[col], act.z_max.get((i, x_, col), 0), 104)
        if not drop:
            break
        dims -= drop
    out = ov.Plan(dims, p.terms, p.keys, p.gates)
    bounds = [np.inf]
    for i, j in p.terms:
        s = noise(c.nodes[i].terms[j][2])
        if s:
            bounds.append((ov.KILL - max(act.gate_max[i, j], 0) - 104 / ov.K) / s)
    out.q_bound = float(min(bounds))
    return out


def build(c: Kernel, keep: set[int], cfg: dict, traces: list[list[int]] | None = None):
    """Measure, plan, tighten and apply the overlay of circuit c (in place)."""
    with our_text():
        act = ov.activity(c, corpus(c, cfg) if traces is None else traces)
    p = tighten(c, act, ov.plan(c, act, keep))
    apply(c, p)
    return p


def check_noisy(c: Kernel, p: ov.Plan, tokens: list[int], rng: np.random.Generator,
                scale: float = QMAX / 4) -> None:
    """With random Qwen-like values (normal, std scale) in the overlay dims at text positions,
    every other dim equals the clean evaluation (Qwenputer's check)."""
    with our_text():
        ov.check_noisy(c, p, tokens, rng, scale)
