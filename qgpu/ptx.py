"""Parser for the PTX subset that clang's NVPTX backend emits for our kernels.

A module holds `.entry` kernels. For each one we keep its parameters, register declarations,
`.shared` variables, and a flat list of instructions with labels resolved to indices.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Param:
    name: str
    type: str  # ".u64", ".u32", ".b32", ...
    size: int


@dataclass
class Shared:
    name: str
    align: int
    size: int


@dataclass
class Operand:
    kind: str  # "reg" | "sreg" | "imm" | "sym" | "label" | "mem" | "pair"
    # reg/sreg/sym/label name, int immediate, (base Operand, offset) for mem, or two Operands
    # for a "%r|%p" destination pair (shfl's value and valid predicate)
    value: object


@dataclass
class Instr:
    op: str  # full opcode with modifiers, e.g. "setp.ge.s32"
    args: list[Operand]
    guard: str | None = None  # predicate register guarding the instruction
    neg: bool = False  # guard is negated (@!%p)
    line: int = 0

    @property
    def parts(self) -> list[str]:
        return self.op.split(".")


@dataclass
class Kernel:
    name: str
    params: list[Param]
    reg_width: dict[str, int]  # register prefix -> bits ("r" -> 32, "rd" -> 64, "p" -> 1)
    shared: list[Shared]
    instrs: list[Instr]
    labels: dict[str, int] = field(default_factory=dict)


SIZES = {"8": 1, "16": 2, "32": 4, "64": 8}
SREG = re.compile(r"^%(tid|ntid|ctaid|nctaid)\.[xyz]$|^%laneid$|^%warpid$")


def parse_int(s: str) -> int:
    s = s.strip().lstrip("+")  # "[%rd1+-4]" gives "+-4"
    neg = s.startswith("-")
    s = s.lstrip("-")
    v = int(s, 16) if s.lower().startswith("0x") else int(s.rstrip("Uu"))
    return -v if neg else v


def parse_operand(s: str) -> Operand:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].replace(" ", "")
        m = re.fullmatch(r"([^+\-]+)([+\-]\S+)?", inner)
        base, off = m.group(1), m.group(2)
        return Operand("mem", (parse_operand(base), parse_int(off) if off else 0))
    if "|" in s:
        a, b = s.split("|")
        return Operand("pair", (parse_operand(a), parse_operand(b)))
    if SREG.match(s):
        return Operand("sreg", s)
    if s.startswith("%"):
        return Operand("reg", s)
    if s.startswith("$"):
        return Operand("label", s)
    if re.fullmatch(r"-?(0[xX][0-9a-fA-F]+|\d+)[Uu]?", s):
        return Operand("imm", parse_int(s))
    return Operand("sym", s)


def split_args(s: str) -> list[str]:
    """Split on commas outside brackets."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def parse(text: str) -> dict[str, Kernel]:
    kernels: dict[str, Kernel] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s*\.visible\s+\.entry\s+(\w+)\(", lines[i])
        if not m:
            i += 1
            continue
        k = Kernel(m.group(1), [], {}, [], [])
        i += 1
        while not lines[i].strip().startswith(")"):
            pm = re.match(r"\s*\.param\s+(\.\w+)(?:\s+\.ptr.*?)?\s+(\w+),?\s*$", lines[i])
            if pm:
                ty = pm.group(1)
                k.params.append(Param(pm.group(2), ty, SIZES[ty[2:]]))
            i += 1
        i += 1
        while lines[i].strip() != "{":
            i += 1
        i += 1
        while lines[i].strip() != "}":
            raw = lines[i]
            s = raw.split("//")[0].strip()
            i += 1
            if not s:
                continue
            if s.endswith(":"):
                k.labels[s[:-1]] = len(k.instrs)
                continue
            rm = re.match(r"\.reg\s+\.(\w+)\s+%(\w+)<\d+>;", s)
            if rm:
                ty = rm.group(1)
                k.reg_width[rm.group(2)] = 1 if ty == "pred" else int(ty[1:])
                continue
            sm = re.match(r"\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\];", s)
            if sm:
                k.shared.append(Shared(sm.group(2), int(sm.group(1)), int(sm.group(3))))
                continue
            if s.startswith(".pragma"):  # optimizer hints (nvcc's "nounroll"): no semantics
                continue
            if s.startswith("."):
                raise ValueError(f"line {i}: unsupported directive {s!r}")
            assert s.endswith(";"), s
            s = s[:-1].strip()
            guard, neg = None, False
            gm = re.match(r"@(!?)(%p\d+)\s+(.*)", s)
            if gm:
                neg, guard, s = gm.group(1) == "!", gm.group(2), gm.group(3)
            op, _, rest = s.partition(" ")
            args = [parse_operand(a) for a in split_args(rest)] if rest.strip() else []
            k.instrs.append(Instr(op, args, guard, neg, i))
        for ins in k.instrs:
            for a in ins.args:
                if a.kind == "label" and a.value not in k.labels:
                    raise ValueError(f"{k.name}: unknown label {a.value}")
        kernels[k.name] = k
        i += 1
    return kernels


def load(path) -> dict[str, Kernel]:
    with open(path) as f:
        return parse(f.read())
