"""CUDA source to a launch, as NVRTC and the CUDA runtime do on a real GPU.

Qwen writes one `__global__` kernel; `compile_cuda` compiles it to PTX with clang (Homebrew
LLVM's NVPTX backend, kernels/qcuda.h for CUDA's spellings), and `plan` lays out its
arguments (arrays become buffers in global memory), picks the launch shape and lowers the PTX
for this launch (qgpu.lower, the driver). The model then runs it (qgpu/bridge.py).

KERNEL_RULES is advanced mode's system prompt: a short preface and docs/qgpu-cuda.md, the
GPU's CUDA reference.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from qgpu import ptx
from qgpu.circuit import kernel
from qgpu.lower import Unsupported, lower
from qgpu.merge import STAGES
from qgpu.runtime import GPUError, Prepared

ROOT = Path(__file__).resolve().parent.parent
CLANG = os.environ.get("CLANG", "/opt/homebrew/opt/llvm/bin/clang")
MAX_THREADS = 128

# advanced mode's system prompt: what to do, then the GPU's CUDA reference (docs/qgpu-cuda.md;
# tests/test_cuda.py runs its examples and checks its numbers against the machine)
DOC = (ROOT / "docs" / "qgpu-cuda.md").read_text()
KERNEL_RULES = """You have a cuda tool that runs a CUDA kernel on the GPU inside your weights.
When a message needs a calculation, write a kernel for this GPU, following the reference
below, and call cuda. If a result does not answer the question, the kernel computed something
else: fix it and call cuda again. Answer only with numbers the GPU returned.

""" + DOC


class CUDAError(GPUError):
    """The source does not compile, or the GPU cannot run it; the message goes to Qwen."""


@dataclass
class Param:
    name: str
    array: bool
    const: bool
    signed: bool


@dataclass
class Compiled:
    name: str
    params: list[Param]
    kernel: ptx.Kernel


SIGNATURE = re.compile(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", re.DOTALL)
FLOATS = re.compile(r"\b(float|double|half|__half)\b")


def signature(source: str) -> tuple[str, list[Param]]:
    """The kernel's name and parameters, from its source."""
    sigs = SIGNATURE.findall(source)
    if len(sigs) != 1:
        if re.search(r"__global__\s+void\s*<", source):
            raise CUDAError("the kernel's name must be a C identifier, like compute (no < >)")
        raise CUDAError("write exactly one __global__ kernel" if not sigs else
                        f"write one __global__ kernel, not {len(sigs)}")
    name, args = sigs[0]
    params = []
    for a in filter(None, (s.strip() for s in args.split(","))):
        m = re.match(r"(.*?)(\w+)\s*(\[\s*\])?$", a)
        if not m:
            raise CUDAError(f"cannot read the parameter {a!r}")
        ty, pname, brackets = m.group(1), m.group(2), m.group(3)
        if FLOATS.search(ty):
            raise CUDAError(f"parameter {pname} is {ty.strip()}: the GPU computes integers only")
        params.append(Param(pname, "*" in ty or bool(brackets), "const" in ty.split(),
                            "unsigned" not in ty and "uint" not in ty))
    return name, params


def compile_cuda(source: str) -> Compiled:
    """CUDA C to PTX (the kernel's only entry), or the compiler's errors."""
    if FLOATS.search(source):
        raise CUDAError("the GPU computes 32-bit integers only: no float or double")
    name, params = signature(source)
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#include"))
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "kernel.cu", Path(tmp) / "kernel.ptx"
        src.write_text(f'#include "{ROOT / "kernels" / "qcuda.h"}"\n{body}\n')
        try:
            r = subprocess.run([CLANG, "-x", "cuda", "--cuda-device-only", "-nocudainc",
                                "-nocudalib", "--cuda-gpu-arch=sm_70", "-Xclang",
                                "-target-feature", "-Xclang", "+ptx70", "-O2", "-S", str(src),
                                "-o", str(out)], check=False, capture_output=True, text=True,
                               timeout=60)
        except FileNotFoundError as e:
            raise CUDAError(f"no CUDA compiler on the host ({CLANG})") from e
        if r.returncode:
            errs = [ln.replace(str(src), "kernel.cu") for ln in r.stderr.splitlines()
                    if "error:" in ln]
            msg = "the CUDA does not compile: " + "; ".join(errs[:4])
            names = sorted(set(re.findall(r"undeclared identifier '(\w+)'", r.stderr)))
            if names:  # usually an array the kernel uses but did not take as a parameter
                msg += (f". Every array the kernel uses is a parameter: declare "
                        f"{', '.join(names)} in the kernel's parameters, int* for an output, "
                        "const int* for an input")
            raise CUDAError(msg)
        kernels = ptx.parse(out.read_text())
    if len(kernels) != 1:
        raise CUDAError(f"write one __global__ kernel, not {len(kernels)}")
    k = next(iter(kernels.values()))
    if len(k.params) != len(params):
        raise CUDAError("cannot match the kernel's parameters")
    return Compiled(name, params, k)


def number(v):
    """An integer, also when written as a string ("4200"); None otherwise."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    return None


def args_shape(c: Compiled) -> str:
    """The args a kernel takes, from its signature (in the refusals Qwen reads): its input
    arrays, as lists; the outputs and the count are the host's."""
    ints = [p for p in c.params if not p.array]
    given = [f'"{p.name}": [numbers]' for p in c.params if p.array and p.const]
    given += [f'"{p.name}": number' for p in ints if len(ints) > 1]
    return f"args for {c.name} is {{{', '.join(given)}}}"


def plan(c: Compiled, args: dict, threads: int | None, lanes: int, stages: int) -> Prepared:
    """Buffers for the array arguments (by parameter name), the launch shape, and the lowered
    kernel, as cudaMalloc, cudaMemcpy and the launch would. An array args gives numbers for is
    an input; any other array (left out, or given only zeros) is an output the host makes:
    zeros, as long as the inputs, or {"zeros": k}. A kernel's one int is the count: the number
    of items, whatever args says. What comes back is the outputs (or, with none, the arrays the
    kernel may change in place). threads defaults to the number of items."""
    if not isinstance(args, dict):
        raise CUDAError(f"args is one JSON object by parameter name: {args_shape(c)}")
    unknown = [k for k in args if k not in {p.name for p in c.params}]
    if unknown:
        raise CUDAError(f"{c.name} has no parameter {', '.join(unknown)}: {args_shape(c)}")
    if all(p.const for p in c.params if p.array):
        raise CUDAError(f"{c.name} has no output array: add an int* parameter for the results "
                        "and write them into it; only output arrays come back")
    inputs, sized = {}, {}
    for p in c.params:
        a = args.get(p.name)
        if not p.array:
            continue
        if isinstance(a, list) and a and all(number(v) is not None for v in a) \
                and any(number(v) for v in a):
            inputs[p.name] = [number(v) for v in a]
        elif isinstance(a, dict) and set(a) == {"zeros"} and (number(a["zeros"]) or 0) > 0:
            sized[p.name] = number(a["zeros"])
        elif p.const or (a is not None and not isinstance(a, list)):
            raise CUDAError(f"give the input array {p.name} as a list of its integers: "
                            f"{args_shape(c)}")
    longest = max((len(v) for v in inputs.values()), default=1)
    ints = [p for p in c.params if not p.array]
    kargs, bufs, outs, changed, shown = [], {}, {}, {}, {}
    for p in c.params:
        if p.array:
            data = inputs.get(p.name) or [0] * sized.get(p.name, longest)
            bid = len(bufs)
            bufs[bid] = [v & 0xFFFFFFFF for v in data]
            if p.name not in inputs:
                outs[p.name] = (bid, len(data))
            elif not p.const:
                changed[p.name] = (bid, len(data))
            kargs.append(bid)
            shown[p.name] = data if p.name in inputs else {"zeros": len(data)}
        else:
            v = longest if len(ints) == 1 else number(args.get(p.name))
            if v is None:
                raise CUDAError(f"give {p.name} as a number: {args_shape(c)}")
            kargs.append(v & 0xFFFFFFFF)
            shown[p.name] = v
    outs = outs or changed
    threads = number(threads) if threads is not None else longest
    if threads is None or not 1 <= threads <= MAX_THREADS:
        raise CUDAError(f"threads must be a number from 1 to {MAX_THREADS}")
    if sum(len(b) for b in bufs.values()) >= kernel.SINK:
        raise CUDAError(f"the arrays hold more than the GPU's {kernel.SINK - 1} numbers")
    total = max(1 << (threads - 1).bit_length(), lanes)
    block = min(total, 32, lanes * kernel.WPB_MAX)
    mul, logic, atom = STAGES[stages]
    try:
        lw = lower(c.kernel, (total // block,), (block,), kargs, bufs, lanes=lanes,
                   stages=stages, mul_stages=mul, logic_stages=logic, atom_stages=atom,
                   write_through=True)
    except (Unsupported, AssertionError, KeyError, ValueError) as e:
        hint = ("; loops cost registers: shorten the kernel, with no loop over the items "
                "(each thread handles its own item; a total is one atomicAdd per thread)") \
            if "registers" in str(e) else ""
        raise CUDAError(f"the GPU cannot run this kernel: {e}{hint}") from e
    if max(lw.mem, default=0) >= kernel.SINK:
        raise CUDAError("the kernel needs more memory than the GPU has")
    signed = all(p.signed for p in c.params if p.array and not p.const)
    return Prepared(c.name, lw, outs, signed, shown)
