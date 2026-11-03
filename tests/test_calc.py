"""calc: expressions in Python's arithmetic against Python, every operation a launch of a
library kernel (on the reference interpreter)."""

import pytest

from qgpu.calc import evaluate, on_reference
from qgpu.runtime import GPUError


def run(expr: str, launch=None):
    g = evaluate(expr, launch or on_reference())
    try:
        while True:
            next(g)
    except StopIteration as s:
        return s.value


@pytest.mark.parametrize("expr,want", [
    ("4*30 + 3*55", 285),
    ("[1, 2, 3, 4, 5] * [3, 2, 1, 4, 5] - 3", [0, 1, 0, 13, 22]),
    ("[2, 3, 4, 5, 6] * [5, 4, 3, 4, 2] * 3", [30, 36, 36, 60, 36]),
    ("sum([4, 3] * [30, 55])", 285),
    ("mean([12, 7, 30, 5])", "13.50"),
    ("mean([-12, 7, -30, 5])", "-7.50"),
    ("max([10, 20, 30] - 4)", 26),
    ("min([7, -3, 12])", -3),
    ("cumsum([4200, 8100, 6500])", [4200, 12300, 18800]),
    ("[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]", [[19, 22], [43, 50]]),
    ("[[1, 2], [3, 4]] * 2 + 1", [[3, 5], [7, 9]]),
    ("[1, 2, 3] @ [4, 5, 6]", 32),
    ("dot([1, 2, 3], [4, 5, 6])", 32),
    ("count([3, 6, 1, 3, 3])", {"1": 1, "3": 3, "6": 1}),
    ("50 // 6, 50 % 6, 50 / 6", [8, 2, "8.33"]),  # Python: // and % floor, / a decimal
    ("[-7, 7] // 2, [-7, 7] % 2, [-7, 7] / 2", [[-4, 3], [1, 1], ["-3.50", "3.50"]]),
    ("84 / 18, 7 // -2, 7 % -2", ["4.66", -4, -1]),
    ("12 * 7 + 1", 85),
    ("-[1, 2] + 10", [9, 8]),
    ("100 - [1, 2]", [99, 98]),
    ("2147483647 + 1", -2147483648),  # 32-bit integers wrap
    ("max([10 - 4, 20 - 4, 30 - 4])", 26),  # computed elements
    ("[1, 2, 3] x [3, 2, 1] - 3", [0, 1, 0]),  # x for *
    ("[2, 3] x 4, 5 × 6", [[8, 12], 30]),
    ("[4, 3] * [30], [50] // [6], [50] % [6]", [[120, 90], [8], [2]]),  # a list of one broadcasts
])
def test_expression(expr, want):
    assert run(expr) == want


@pytest.mark.parametrize("expr,says", [
    ("[1, 2] / [1, 2]", "by one number"),
    ("mean([1, 2]) + 1", "last operation"),
    ("10 / 4 * 2", "last operation"),
    ("foo([1])", "unknown function"),
    ("x + 1", "unknown name"),
    ("[1, 2] + [1, 2, 3]", "same length"),
    ("5 / 0", "division by zero"),
    ("1.5 * 2", "integers"),
    ("import os", "not an expression"),
    ("sum(3)", "one list"),
    ("[[1, 2], [3]]", "square"),
    ("[[1, 2], [3, 4]] * [[5, 6], [7, 8]]", "A @ B"),
    ("[[1, 2], 3 + 4]", "elements are numbers"),
])
def test_refused(expr, says):
    with pytest.raises(GPUError, match=says):
        run(expr)


def test_every_operation_is_a_launch():
    """Nothing is computed by the host: a number alone runs as a list of one."""
    names = []

    def launch(name, data, n):
        names.append(name)
        out = yield from on_reference()(name, data, n)
        return out

    assert run("min([4, 9]) * 2 + sum([1, 2])", launch) == 11
    assert names == ["mulk", "vmax", "mulk", "mulk", "reduce", "addk"]  # number + number: addk


def test_power_is_refused_with_a_hint():
    def launch(*_):
        raise AssertionError("no launch")
        yield

    for e in ("3^2", "3**2"):
        with pytest.raises(GPUError, match=r"not supported.*x\*x"):
            list(evaluate(e, launch))
