"""Reference SIMT interpreter for the PTX subset: the ground truth the weights must match.

Execution model (as on an NVIDIA GPU):
- a launch is a grid of thread blocks; threads of a block are numbered x fastest, then y, z,
  and grouped into warps of 32 consecutive threads;
- a warp runs one instruction at a time for all its active lanes;
- a branch on which lanes disagree pushes both sides onto the warp's reconvergence stack with
  their own lane masks; they join at the branch's immediate post-dominator (driver.py);
- `bar.sync` holds a warp until every warp of its block with live threads has arrived;
- blocks run one after another (CUDA guarantees no order between blocks).

Registers hold unsigned bit patterns of their declared width; each instruction's type suffix
says how to read them. Global memory sits at a 64-bit base so address arithmetic is really
64-bit, as with real device pointers.
"""

from dataclasses import dataclass, field

from qgpu import driver
from qgpu.ptx import Instr, Kernel, Operand

WARP = 32
GLOBAL_BASE = 0x7F00_0000_0000


def mask(bits: int) -> int:
    return (1 << bits) - 1


def signed(v: int, bits: int) -> int:
    v &= mask(bits)
    return v - (1 << bits) if v >> (bits - 1) else v


class Memory:
    """Flat little-endian byte space starting at `base`, with a bump allocator."""

    def __init__(self, size: int, base: int = 0) -> None:
        self.base = base
        self.data = bytearray(size)
        self.top = 0

    def alloc(self, nbytes: int, align: int = 256) -> int:
        self.top = (self.top + align - 1) // align * align
        addr = self.base + self.top
        self.top += nbytes
        assert self.top <= len(self.data), "out of memory"
        return addr

    def _off(self, addr: int, n: int) -> int:
        off = addr - self.base
        if not (0 <= off and off + n <= len(self.data)):
            raise IndexError(f"address {addr:#x} out of range")
        if addr % n:
            raise ValueError(f"misaligned {n}-byte access at {addr:#x}")
        return off

    def read(self, addr: int, n: int) -> int:
        off = self._off(addr, n)
        return int.from_bytes(self.data[off:off + n], "little")

    def write(self, addr: int, n: int, v: int) -> None:
        off = self._off(addr, n)
        self.data[off:off + n] = (v & mask(8 * n)).to_bytes(n, "little")

    def write_i32(self, addr: int, values) -> None:
        for j, v in enumerate(values):
            self.write(addr + 4 * j, 4, int(v))

    def read_i32(self, addr: int, count: int, signed_: bool = True) -> list[int]:
        out = [self.read(addr + 4 * j, 4) for j in range(count)]
        return [signed(v, 32) for v in out] if signed_ else out


@dataclass
class Entry:
    """One level of a warp's reconvergence stack."""

    pc: int
    rpc: int  # reconvergence pc: pop this entry when pc reaches it
    mask: int  # lanes this entry runs


@dataclass
class Warp:
    lanes: list[tuple[int, int, int]]  # (tid.x, tid.y, tid.z) of each lane
    regs: list[dict[str, int]]
    stack: list[Entry]
    exited: int = 0
    at_barrier: bool = False

    @property
    def done(self) -> bool:
        return not self.stack


@dataclass
class Stats:
    warp_instrs: int = 0  # instructions issued (one per warp step)
    lane_instrs: int = 0  # instructions times active lanes
    divergent_branches: int = 0
    by_op: dict[str, int] = field(default_factory=dict)


class Launch:
    def __init__(self, k: Kernel, mem: Memory, grid, block, args) -> None:
        self.k, self.mem = k, mem
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.block = tuple(block) + (1,) * (3 - len(block))
        assert len(args) == len(k.params), f"{k.name} takes {len(k.params)} arguments"
        self.params = {p.name: int(a) & mask(8 * p.size) for p, a in zip(k.params, args)}
        self.param_size = {p.name: p.size for p in k.params}
        self.rpc = driver.reconvergence(k)
        self.shared_addr, top = {}, 0
        for s in k.shared:
            top = (top + s.align - 1) // s.align * s.align
            self.shared_addr[s.name] = top
            top += s.size
        self.shared_size = top
        self.stats = Stats()

    # --- operands -------------------------------------------------------------

    def width(self, reg: str) -> int:
        prefix = reg[1:].rstrip("0123456789")
        return self.k.reg_width[prefix]

    def get(self, a: Operand, w: Warp, lane: int, ctaid) -> int:
        if a.kind == "reg":
            return w.regs[lane].get(a.value, 0)
        if a.kind == "imm":
            return a.value
        if a.kind == "sreg":
            name, _, dim = a.value[1:].partition(".")
            d = "xyz".index(dim) if dim else 0
            if name == "tid":
                return w.lanes[lane][d]
            if name == "ntid":
                return self.block[d]
            if name == "ctaid":
                return ctaid[d]
            if name == "nctaid":
                return self.grid[d]
            if name == "laneid":
                return lane
        if a.kind == "sym" and a.value in self.shared_addr:
            return self.shared_addr[a.value]
        raise ValueError(f"cannot read operand {a}")

    def put(self, a: Operand, w: Warp, lane: int, v: int) -> None:
        assert a.kind == "reg", a
        w.regs[lane][a.value] = v & mask(self.width(a.value))

    def address(self, a: Operand, w: Warp, lane: int, ctaid) -> tuple[str | None, int]:
        """(param name or None, address) of a memory operand."""
        base, off = a.value
        if base.kind == "sym" and base.value in self.params:
            return base.value, off
        return None, self.get(base, w, lane, ctaid) + off

    # --- execution --------------------------------------------------------------

    def run(self) -> Stats:
        gx, gy, gz = self.grid
        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    self.run_block((x, y, z))
        return self.stats

    def run_block(self, ctaid) -> None:
        bx, by, bz = self.block
        tids = [(x, y, z) for z in range(bz) for y in range(by) for x in range(bx)]
        warps = []
        for s in range(0, len(tids), WARP):
            lanes = tids[s:s + WARP]
            warps.append(Warp(lanes, [{} for _ in lanes],
                              [Entry(0, driver.EXIT, mask(len(lanes)))]))
        shared = Memory(max(self.shared_size, 4))
        while True:
            live = [w for w in warps if not w.done]
            if not live:
                return
            if all(w.at_barrier for w in live):
                for w in live:  # every live warp arrived: release them
                    w.at_barrier = False
                    w.stack[-1].pc += 1
                continue
            for w in live:
                while not w.done and not w.at_barrier:
                    self.step(w, ctaid, shared)

    def step(self, w: Warp, ctaid, shared: Memory) -> None:
        top = w.stack[-1]
        active = top.mask & ~w.exited
        if not active or top.pc == top.rpc:
            w.stack.pop()
            return
        pc = top.pc
        ins = self.k.instrs[pc]
        lanes = [i for i in range(len(w.lanes)) if active >> i & 1]
        go = [i for i in lanes if self.guard(ins, w, i)]
        self.stats.warp_instrs += 1
        self.stats.lane_instrs += len(go)
        self.stats.by_op[ins.op] = self.stats.by_op.get(ins.op, 0) + 1
        base = ins.parts[0]

        if base == "bra":
            target = self.k.labels[ins.args[0].value]
            taken = sum(1 << i for i in go)
            rest = active & ~taken
            if not rest:
                top.pc = target
            elif not taken:
                top.pc = pc + 1
            else:
                self.stats.divergent_branches += 1
                r = self.rpc[pc]
                top.pc = r
                w.stack.append(Entry(pc + 1, r, rest))
                w.stack.append(Entry(target, r, taken))
            return
        if base in ("ret", "exit"):
            w.exited |= sum(1 << i for i in go)
            if active & ~w.exited:
                top.pc = pc + 1
            return
        if base == "bar":
            assert ins.op == "bar.sync" and ins.args[0].value == 0, ins.op
            w.at_barrier = True
            return
        if base == "shfl":
            self.shfl(ins, w, go, ctaid)
        else:
            for i in go:
                self.execute(ins, w, i, ctaid, shared)
        top.pc = pc + 1

    def guard(self, ins: Instr, w: Warp, lane: int) -> bool:
        if ins.guard is None:
            return True
        return bool(w.regs[lane].get(ins.guard, 0)) != ins.neg

    def shfl(self, ins: Instr, w: Warp, go: list[int], ctaid) -> None:
        """shfl.sync.down.b32 d, a, b, c, membermask (PTX ISA semantics)."""
        assert ins.op == "shfl.sync.down.b32", ins.op
        d, a, b, c, _ = ins.args
        src = [w.regs[i].get(a.value, 0) if a.kind == "reg" else a.value
               for i in range(len(w.lanes))]
        out = {}
        for i in go:
            bval = self.get(b, w, i, ctaid) & 31
            cval = self.get(c, w, i, ctaid)
            segmask = cval >> 8 & 31
            max_lane = (i & segmask) | (cval & 31 & ~segmask)
            j = i + bval
            ok = j <= max_lane and j < len(w.lanes)
            out[i] = (src[j] if ok else src[i], int(ok))
        for i, (v, ok) in out.items():
            if d.kind == "pair":  # "%r|%p": the value and whether the source lane was in range
                self.put(d.value[0], w, i, v)
                self.put(d.value[1], w, i, ok)
            else:
                self.put(d, w, i, v)

    def execute(self, ins: Instr, w: Warp, i: int, ctaid, shared: Memory) -> None:
        p = ins.parts
        base = p[0]
        args = ins.args

        def val(n: int) -> int:
            return self.get(args[n], w, i, ctaid)

        if base in ("ld", "st"):
            space = p[1]
            nbytes = int(p[-1][1:]) // 8
            name, addr = self.address(args[1] if base == "ld" else args[0], w, i, ctaid)
            if space == "param":
                assert base == "ld" and name is not None and addr == 0, ins.op
                assert self.param_size[name] == nbytes
                self.put(args[0], w, i, self.params[name])
                return
            m = self.mem if space == "global" else shared if space == "shared" else None
            assert m is not None, f"unsupported state space {space}"
            if base == "ld":
                self.put(args[0], w, i, m.read(addr, nbytes))
            else:
                m.write(addr, nbytes, val(1))
            return
        if base == "atom":
            # atom[.sem][.scope].global.add.u32 (clang adds .relaxed.sys, nvcc writes none)
            assert p[-3:] == ["global", "add", "u32"], ins.op
            _, addr = self.address(args[1], w, i, ctaid)
            old = self.mem.read(addr, 4)
            self.mem.write(addr, 4, old + val(2))
            self.put(args[0], w, i, old)
            return

        ty = p[-1]
        bits = 1 if ty == "pred" else int(ty[1:])  # and/or/xor/not .pred: 1-bit logic
        sgn = ty[0] == "s"

        def num(n: int, b: int = bits, s: bool | None = None) -> int:
            v = val(n)
            return signed(v, b) if (sgn if s is None else s) else v & mask(b)

        if base == "mov" or base == "cvta":
            self.put(args[0], w, i, val(1))
        elif base == "add":
            self.put(args[0], w, i, val(1) + val(2))
        elif base == "sub":
            self.put(args[0], w, i, val(1) - val(2))
        elif base == "neg":
            self.put(args[0], w, i, -val(1))
        elif base in ("and", "or", "xor"):
            f = {"and": int.__and__, "or": int.__or__, "xor": int.__xor__}[base]
            self.put(args[0], w, i, f(val(1) & mask(bits), val(2) & mask(bits)))
        elif base == "not":
            self.put(args[0], w, i, ~val(1))
        elif base == "shl":
            n = val(2) & mask(32)
            self.put(args[0], w, i, 0 if n >= bits else val(1) << n)
        elif base == "shr":
            n = min(val(2) & mask(32), bits)
            self.put(args[0], w, i, num(1) >> n)
        elif base in ("min", "max"):
            f = min if base == "min" else max
            self.put(args[0], w, i, f(num(1), num(2)))
        elif base == "mul":
            mode = p[1]
            prod = num(1) * num(2)
            if mode == "lo":
                self.put(args[0], w, i, prod)
            elif mode == "hi":
                self.put(args[0], w, i, prod >> bits)
            elif mode == "wide":
                self.put(args[0], w, i, prod)
            else:
                raise NotImplementedError(ins.op)
        elif base == "mad":
            assert p[1] == "lo", ins.op
            self.put(args[0], w, i, num(1) * num(2) + num(3))
        elif base == "setp":
            cmp_ = p[1]
            a, b = num(1), num(2)
            r = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b,
                 "ge": a >= b}[cmp_]
            self.put(args[0], w, i, int(r))
        elif base == "selp":
            self.put(args[0], w, i, val(1) if val(3) else val(2))
        elif base == "cvt":
            src = p[2]  # cvt.<dst>.<src>: the destination width is the register's
            sb = int(src[1:])
            v = signed(val(1), sb) if src[0] == "s" else val(1) & mask(sb)
            self.put(args[0], w, i, v)
        else:
            raise NotImplementedError(f"{self.k.name}: {ins.op}")


def launch(k: Kernel, mem: Memory, grid, block, args) -> Stats:
    return Launch(k, mem, grid, block, args).run()
