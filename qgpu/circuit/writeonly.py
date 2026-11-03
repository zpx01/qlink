"""Write-only heads: rows the circuit writes but never reads (the kernel's `out` memory).

Qwenputer's IR evaluator requires every written data column to be 0 or A_BIT, because a read
divides the row by its ballast inside the head's output RMSNorm and must stay exact. A head that
is never queried has no read, so that rule protects nothing there, and its rows can hold small
integers instead: a stored 32-bit word is 8 nibble columns (value * A_BIT) instead of 32 bits,
which saves the MLP units that turn nibbles into bits. The host reads the state directly.

`mark(node)` flags such a head. Importing this module wraps the IR's head evaluator: a flagged
head gets the same checks (one-hot keys, write gate, decay, ballast) except that data columns may
hold any of 0, A_BIT, ..., 15 * A_BIT, and it must have no queries. Everything else is unchanged.
"""

import numpy as np
from chipformer.q35 import ir

NIBBLE_LEVELS = [k * ir.A_BIT for k in range(16)]

_orig_eval_head = ir.eval_head


def mark(node: ir.Head) -> ir.Head:
    assert not node.q, "a write-only head has no queries"
    node.write_only = True
    return node


def eval_write_only(node: ir.Head, m: np.ndarray, chip: np.ndarray, check: bool,
                    carry: dict | None = None, ni: int = 0) -> None:
    n = m.shape[0]
    k = (np.stack([ir.conv(c, m, check, carry, (ni, "k", s)) for s, c in node.k.items()], 1)
         if node.k else np.zeros((n, 0)))
    ki = list(node.k)
    if check:
        nz = (k > 0).sum(1)
        ir.require(nz <= 1, "key must be one-hot", nz)
        ir.require(~((nz > 0) & ~chip), "key on a text token", nz)
    has_key = k.any(1) if k.shape[1] else np.zeros(n, dtype=bool)
    key_row = [ki[int(np.argmax(k[t]))] if has_key[t] else None for t in range(n)]
    for x_, vh in enumerate(node.vheads()):
        v = np.zeros((n, node.n_cols))
        for col, c in vh.v.items():
            v[:, col] = ir.conv(c, m, check, carry, (ni, "v", x_, col))
        a = ir.ev(vh.a, m)
        beta = np.ones(n) if vh.beta is None else ir.ev(vh.beta, m)
        writes = has_key & (beta > 0)
        if check:
            ir.require(ir.is_bf16(a) & np.isin(a, list(ir.DECAY)), "decay selector a", a)
            ir.require(np.isin(beta, [0.0, 1.0]), "write gate beta", beta)
            written = v[writes]
            if len(written):
                data = np.delete(written, ir.REF_COLS, 1)
                ir.require(np.isin(data, NIBBLE_LEVELS), "write-only data column", data)
                ref = written[:, list(ir.REF_COLS)]
                ir.require((ref == 0) | (ref == ir.A_REF), "state row ballast column", ref)
        state: dict = {} if carry is None else carry.get((ni, "S", x_), {})
        decay = np.vectorize(ir.DECAY.get)(a)
        for t in np.nonzero((decay != 1.0) | writes)[0]:
            if decay[t] == 0.0:
                state.clear()
            elif decay[t] == 0.5:
                state = {s: row * 0.5 for s, row in state.items()}
            if writes[t]:
                state[key_row[t]] = v[t].copy()
        if carry is not None:
            carry[ni, "S", x_] = state
        for _, d, _ in vh.out:  # never read: the outputs are 0
            if not vh.into:
                m[:, d] = 0.0


def eval_head(node, m, chip, check, carry=None, ni=0):
    if getattr(node, "write_only", False):
        return eval_write_only(node, m, chip, check, carry, ni)
    return _orig_eval_head(node, m, chip, check, carry, ni)


ir.eval_head = eval_head
