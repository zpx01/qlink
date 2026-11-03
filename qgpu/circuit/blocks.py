"""Load-time block formation for burst execution, the driver's side.

A burst runs up to S consecutive instructions inside one forward pass; only *live-out* values go
back through tokens. The driver splits a straight-line program into blocks and, per block,
decides which stage results to emit and in which order:

- live-out: a register the block writes whose last value is read after the block (or is a
  program output);
- the circuit recomputes the block at every token of its record while live-outs are committed
  lane group by lane group, so a live-out must be emitted before any live-out that overwrites
  one of the block inputs it depends on. Lanes are independent, so a live-out may depend on its
  own register (r1 = r1 + r2). A cyclic constraint, or more than LMAX live-outs, ends the block
  one instruction earlier.
"""

from dataclasses import dataclass

Instr = tuple[int, int, int]  # (rd, ra, rb): add.s32 rd, ra, rb


def rw(ins) -> tuple[int | None, tuple[int, ...]]:
    """(register written or None, registers read). Plain tuples are adds; other instructions
    say it themselves (kernel.KInstr)."""
    if isinstance(ins, tuple):
        rd, ra, rb = ins
        return rd, (ra, rb)
    return ins.rw()


@dataclass
class Block:
    instrs: list[Instr]
    live: list[tuple[int, int]]  # emission order: (stage, rd)


def liveness(program: list[Instr], outputs: set[int]) -> list[set[int]]:
    """live[i]: registers live after instruction i."""
    live, out = set(outputs), [set() for _ in program]
    for i in range(len(program) - 1, -1, -1):
        out[i] = set(live)
        rd, srcs = rw(program[i])
        live = (live - {rd}) | set(srcs)
    return out


def plan(instrs: list[Instr], live_after: set[int], lmax: int) -> list[tuple[int, int]] | None:
    """Emission order of the block's live-outs, or None if the block is not allowed."""
    deps, last = [], {}  # deps[s]: block inputs stage s depends on
    for s, ins in enumerate(instrs):
        rd, srcs = rw(ins)
        d = set()
        for x in srcs:
            d |= deps[last[x]] if x in last else {x}
        deps.append(d)
        if rd is not None:
            last[rd] = s
    outs = [(s, rd) for rd, s in last.items() if rd in live_after]
    if len(outs) > lmax or any(rd < 0 for _, rd in outs):  # < 0: predicates (KInstr.rw)
        return None  # too many live-outs, or a predicate would cross into another block
    # an atomic / stw writes back after every live-out is committed, recomputing the block:
    # its address, value and guard must not read a register a live-out overwrites
    if any(getattr(ins, "op", None) in ("atom", "stw") and deps[s] & {rd for _, rd in outs}
           for s, ins in enumerate(instrs)):
        return None
    order, todo = [], sorted(outs)
    while todo:
        # x may go first unless another pending live-out reads the register x overwrites
        ready = [x for x in todo
                 if not any(y != x and x[1] in deps[y[0]] for y in todo)]
        if not ready:
            return None
        order.append(ready[0])
        todo.remove(ready[0])
    return order


def split(program: list[Instr], outputs: set[int], stages: int, lmax: int,
          place=None) -> list[Block]:
    """Greedy blocks of up to `stages` instructions. `place(instrs)` maps a candidate block to
    its stage slots (padding with no-ops where an instruction needs a particular stage) or
    None if it does not fit; by default instruction s goes to stage s."""
    place = place or (lambda instrs: instrs if len(instrs) <= stages else None)
    live = liveness(program, outputs)
    blocks, i = [], 0
    while i < len(program):
        best = None
        for j in range(i + 1, len(program) + 1):
            slots = place(program[i:j])
            if slots is None:
                break
            order = plan(slots, live[j - 1], lmax)
            if order is not None:
                best = (j, slots, order)
        assert best is not None, f"instruction {i} fits no block"
        j, slots, order = best
        blocks.append(Block(slots, order))
        i = j
    return blocks
