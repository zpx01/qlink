"""Load-time analysis, the part NVIDIA's driver does when it turns PTX into SASS.

For every branch we compute its reconvergence point: the immediate post-dominator of the
branch in the instruction-level control-flow graph. A warp that diverges at a branch runs each
side with its own lane mask and joins again there (the classic IPDOM stack).
"""

from qgpu.ptx import Kernel

EXIT = -1  # virtual exit node: every `ret` leads here


def successors(k: Kernel) -> list[list[int]]:
    n = len(k.instrs)
    succ = []
    for i, ins in enumerate(k.instrs):
        base = ins.parts[0]
        if base == "ret" or base == "exit":
            s = [EXIT]
        elif base == "bra":
            t = k.labels[ins.args[0].value]
            s = [t] if ins.guard is None else sorted({t, i + 1})
        else:
            s = [i + 1]
        assert all(x == EXIT or 0 <= x < n for x in s), f"{k.name}: falls off the end at {i}"
        succ.append(s)
    return succ


def postdominators(k: Kernel) -> list[set[int]]:
    """pdom[i]: the nodes that every path from i to the exit passes through (i included)."""
    succ = successors(k)
    n = len(k.instrs)
    every = set(range(n)) | {EXIT}
    pdom = [set(every) for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            sets = [{EXIT} if s == EXIT else pdom[s] for s in succ[i]]
            new = set.intersection(*sets) | {i}
            if new != pdom[i]:
                pdom[i] = new
                changed = True
    return pdom


def reconvergence(k: Kernel) -> dict[int, int]:
    """Branch index -> its immediate post-dominator (EXIT when the sides only meet at exit).

    Strict post-dominators of a node form a chain; the immediate one is the chain element that
    all the others post-dominate, i.e. the one with the largest post-dominator set."""
    pdom = postdominators(k)
    out = {}
    for i, ins in enumerate(k.instrs):
        if ins.parts[0] == "bra" and ins.guard is not None:
            strict = pdom[i] - {i}
            out[i] = max(strict, key=lambda d: len(pdom[d]) if d != EXIT else 0)
    return out
