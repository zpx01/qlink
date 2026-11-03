"""calc: an expression in Python's arithmetic, run with the GPU's library kernels.

Qwen's calc tool, as in Qwenputer: Qwen writes the arithmetic and a program written ahead of
time computes it. Here that program is a library of CUDA kernels (qgpu.runtime.KERNELS, each
verified exact through the circuit), and `evaluate` runs every operation as one launch on the
GPU in the model's weights, as CuPy or PyTorch do on a real GPU. The host only parses the
expression and moves data; it computes nothing (a number alone runs as a list of one).

Expressions (a Python subset, parsed with `ast`):
- integers, lists [1, 2, 3], square matrices [[1, 2], [3, 4]] (n <= 4); 32-bit integers
- + - * // % element-wise, with Python's rules (// and % floor); a number (or a list of one,
  as NumPy broadcasts) on either side applies to every element; x or × between two operands
  is *; unary -
- / divides by one number, to two decimals
- a @ b: the matrix product of two matrices (* between two matrices is refused: it means @
  in maths, element by element in NumPy), the dot product of two lists
- sum(x), mean(x) (two decimals), max(x), min(x), cumsum(x) (running totals), dot(a, b),
  count(x) (how many times each value 0..15 occurs)
- several expressions separated by commas: a list of results; computed list elements
"""

import ast
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass

from qgpu.runtime import GPUError, prepare, reference


@dataclass
class Arr:
    """A value: a number (scalar), a list, or an n x n matrix (shape n, row-major data)."""
    data: list[int]
    shape: int | None = None
    scalar: bool = False

    def show(self):
        if self.scalar:
            return self.data[0]
        if self.shape:
            n = self.shape
            return [self.data[r * n:(r + 1) * n] for r in range(n)]
        return self.data


@dataclass
class Final:
    """A result no operation may take further: a decimal (/, mean), count's table."""
    value: object

    def show(self):
        return self.value


# element-wise operators: (list op list, list op number, number op list)
ELEMENTWISE = {ast.Add: ("vadd", "addk", "addk"), ast.Sub: ("vsub", "subk", "rsubk"),
               ast.Mult: ("vmul", "mulk", "mulk"), ast.Div: (None, "divdec", None),
               ast.FloorDiv: (None, "fdivk", None), ast.Mod: (None, "fdivk", None)}
FUNCTIONS = ("sum", "mean", "max", "min", "cumsum", "dot", "count")
TIMES = re.compile(r"(?<=[\]\d)])\s*[x×]\s*(?=[\[\d(-])")  # "[1, 2] x 3": multiplication

# launch(kernel, data, n) runs one launch and returns its outputs, yielding progress events
Launch = Callable[[str, dict, int], Generator]


def parse(text: str) -> ast.expr:
    """The expression's syntax tree; only the forms the runtime runs are accepted."""
    text = TIMES.sub(" * ", text)
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise GPUError(f"not an expression: {e.msg} in {text.strip()!r}") from None
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp,
                               ast.Call, ast.Name, ast.Load, ast.operator, ast.unaryop))
        if not ok or isinstance(node, ast.Constant) and (
                not isinstance(node.value, int) or isinstance(node.value, bool)):
            raise GPUError(f"cannot run {ast.unparse(node)!r}: use integers, lists, + - * / % "
                           f"@ and {', '.join(FUNCTIONS)}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in FUNCTIONS):
            raise GPUError(f"unknown function {ast.unparse(node.func)!r}; the functions are "
                           f"{', '.join(FUNCTIONS)}")
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS:
            raise GPUError(f"unknown name {node.id!r}: write the numbers themselves")
    return tree


def literal(node: ast.expr) -> Arr | None:
    """A number, list or matrix written out (with signs); None for anything computed."""
    if isinstance(node, ast.Constant):
        return Arr([node.value], scalar=True)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
            and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        return Arr([-v if isinstance(node.op, ast.USub) else v], scalar=True)
    if isinstance(node, ast.List):
        items = [literal(e) for e in node.elts]
        if not items:
            raise GPUError("an empty list")
        if any(i is None for i in items):  # computed elements: _eval runs them
            return None
        if all(i is not None and i.scalar for i in items):
            return Arr([i.data[0] for i in items])
        if all(i is not None and not i.scalar and i.shape is None for i in items):
            n = len(items)
            if any(len(i.data) != n for i in items):
                raise GPUError("a matrix must be square: n rows of n numbers")
            return Arr([v for i in items for v in i.data], shape=n)
        raise GPUError("lists hold numbers written out (or rows of a square matrix)")
    return None


def evaluate(text: str, launch: Launch) -> Generator:
    """Run an expression; every operation is a launch. Returns the result for the user."""
    tree = parse(text)
    parts = tree.elts if isinstance(tree, ast.Tuple) else [tree]
    results = []
    for part in parts:
        v = yield from _eval(part, launch)
        results.append(v.show())
    return results if isinstance(tree, ast.Tuple) else results[0]


def _arr(v, what: str) -> Arr:
    if isinstance(v, Final):
        raise GPUError(f"{what} takes a number or list; a decimal (from / or mean) and "
                       "count(...) can only be "
                       "the last operation")
    return v


def _eval(node: ast.expr, launch: Launch) -> Generator:
    lit = literal(node)
    if lit is not None:
        return lit
    if isinstance(node, ast.List):  # computed elements: each a number, computed on the GPU
        items = []
        for e in node.elts:
            v = _arr((yield from _eval(e, launch)), "a list")
            if not v.scalar:
                raise GPUError("a list's elements are numbers (a matrix: rows written out)")
            items.append(v.data[0])
        return Arr(items)
    if isinstance(node, ast.UnaryOp):
        x = _arr((yield from _eval(node.operand, launch)), "-")
        if isinstance(node.op, ast.UAdd):
            return x
        out = yield from launch("mulk", {"x": x.data, "k": -1}, len(x.data))
        return Arr(out["y"], x.shape, x.scalar)
    if isinstance(node, ast.BinOp):
        a = _arr((yield from _eval(node.left, launch)), "an operator")
        b = _arr((yield from _eval(node.right, launch)), "an operator")
        return (yield from _binop(node.op, a, b, launch))
    if isinstance(node, ast.Call):
        name = node.func.id
        args = []
        for x in node.args:
            args.append(_arr((yield from _eval(x, launch)), f"{name}(...)"))
        return (yield from _call(name, args, launch))
    raise GPUError(f"cannot run {ast.unparse(node)!r}")


def _one(v: Arr) -> Arr:
    """A list of one broadcasts like a number (NumPy's rule)."""
    return Arr(v.data, scalar=True) if not v.scalar and not v.shape and len(v.data) == 1 \
        else v


def _binop(op: ast.operator, a: Arr, b: Arr, launch: Launch) -> Generator:
    sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
           ast.Mod: "%", ast.MatMult: "@", ast.Pow: "**", ast.BitXor: "^", ast.BitAnd: "&",
           ast.BitOr: "|", ast.LShift: "<<", ast.RShift: ">>"}[type(op)]
    if isinstance(op, ast.MatMult):
        if a.shape and b.shape:
            if a.shape != b.shape:
                raise GPUError("@ needs two matrices of the same size")
            out = yield from launch("matmul", {"A": a.data, "B": b.data}, a.shape)
            return Arr(out["C"], shape=a.shape)
        if not (a.scalar or b.scalar or a.shape or b.shape):
            return (yield from _call("dot", [a, b], launch))
        raise GPUError("@ multiplies two matrices or two lists")
    if type(op) not in ELEMENTWISE:
        power = " (for a power, multiply: x*x)" if sym in ("**", "^") else ""
        raise GPUError(f"the operator {sym} is not supported: use + - * / % @{power}")
    if isinstance(op, ast.Mult) and a.shape and b.shape:  # A × B of matrices means @
        raise GPUError("* between two matrices: for the matrix product write A @ B")
    vv, vk, kv = ELEMENTWISE[type(op)]
    b = _one(b)  # a list of one on the right acts as a number; on the left, against a longer
    if not b.scalar:  # list (the result keeps the left side's shape otherwise)
        a = _one(a)
    divide = isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod))
    if b.scalar:  # list op number, or number op number (a list of one)
        if divide and b.data[0] == 0:
            raise GPUError("division by zero")
        out = yield from launch(vk, {"x": a.data, "k": b.data[0]}, len(a.data))
        if isinstance(op, ast.Div):  # the GPU's whole parts and hundredths, as decimals
            dec = [decimal(w, h) for w, h in zip(out["w"], out["h"])]
            return Final(dec[0] if a.scalar else
                         [dec[r * a.shape:(r + 1) * a.shape] for r in range(a.shape)]
                         if a.shape else dec)
        y = out["r"] if isinstance(op, ast.Mod) else out["q"] if divide else out["y"]
        return Arr(y, a.shape, a.scalar)
    if a.scalar:  # number op list
        if kv is None:
            raise GPUError(f"a number {sym} a list is not supported: {sym} by one number only")
        out = yield from launch(kv, {"x": b.data, "k": a.data[0]}, len(b.data))
        return Arr(out["y"], b.shape)
    if len(a.data) != len(b.data) or a.shape != b.shape:
        raise GPUError(f"{sym} needs two lists of the same length (or a list and a number); got "
                       f"{len(a.data)} and {len(b.data)} numbers")
    if vv is None:
        raise GPUError(f"{sym} by a list is not supported: {sym} by one number only")
    out = yield from launch(vv, {"a": a.data, "b": b.data}, len(a.data))
    return Arr(out["c"], a.shape)


def _call(name: str, args: list[Arr], launch: Launch) -> Generator:
    want = 2 if name == "dot" else 1
    if len(args) != want or any(x.scalar or x.shape for x in args):
        raise GPUError(f"{name} takes {'two lists' if want == 2 else 'one list'}")
    x = args[0]
    n = len(x.data)
    if name == "dot":
        if len(args[1].data) != n:
            raise GPUError("dot needs two lists of the same length")
        out = yield from launch("dot", {"a": x.data, "b": args[1].data}, n)
        return Arr(out["out"], scalar=True)
    if name == "sum":
        out = yield from launch("reduce", {"x": x.data}, n)
        return Arr(out["out"], scalar=True)
    if name == "max":
        out = yield from launch("vmax", {"x": x.data}, n)
        return Arr(out["out"], scalar=True)
    if name == "min":  # -max(-x)
        neg = yield from launch("mulk", {"x": x.data, "k": -1}, n)
        top = yield from launch("vmax", {"x": neg["y"]}, n)
        out = yield from launch("mulk", {"x": top["out"], "k": -1}, 1)
        return Arr(out["y"], scalar=True)
    if name == "cumsum":
        out = yield from launch("scan", {"x": x.data}, n)
        return Arr(out["y"])
    if name == "mean":  # the GPU's whole part and hundredths, written as one number
        whole, hund = (yield from launch("mean", {"x": x.data}, n))["out"]
        return Final(decimal(whole, hund))
    out = yield from launch("histogram", {"x": x.data}, n)  # count
    return Final({str(k): c for k, c in enumerate(out["bins"]) if c})


def decimal(whole: int, hundredths: int) -> str:
    """A decimal from the GPU's whole part and hundredths (both carry the sign)."""
    sign = "-" if whole < 0 or hundredths < 0 else ""
    return f"{sign}{abs(whole)}.{abs(hundredths):02d}"


def on_reference(lanes: int = 4, stages: int = 4) -> Launch:
    """A launcher on the reference interpreter (no model): for tests and checks."""
    def launch(name, data, n):
        return reference(prepare(name, data, n, lanes, stages), lanes)
        yield
    return launch
