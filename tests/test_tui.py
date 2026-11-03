"""The chat app, headless, on a scripted session (no model): every event renders, and the GPU
panel follows real launch token streams (from the reference)."""

import asyncio
import time

import pytest

pytest.importorskip("textual")

from qgpu import tui
from qgpu.bridge import Event
from qgpu.circuit.kernel import run_blocks
from qgpu.monitor import Program
from qgpu.runtime import host_tokens, prepare, read


class FakeSession:
    name, dev = "fake", "cpu"
    delay = 0.0  # seconds per chip token (screenshots use it to catch a launch running)

    def __init__(self, merged, device, think, check, mode="calc"):
        import sys

        print("a library warning", file=sys.stderr)  # the real load writes to stderr too
        self.meta = {"lanes": 4, "stages": 4, "chip_layers": 46}
        self.think, self.check, self.cancel, self.mode = think, check, False, mode
        self.reset()

    def reset(self):
        self.stats = {"launches": 0, "chip_tokens": 0, "launch_seconds": 0.0, "verified": 0}

    def launch(self, name, data, n):
        p = prepare(name, data, n, 4, 4)
        lw = p.launch
        tr = run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
        yield Event("launch", {"kernel": name, "inputs": data, "blocks": len(lw.blocks),
                               "warps": lw.warps, "host_tokens": len(host_tokens(lw)),
                               "program": Program.of(lw, 4)})
        stream = tr.tokens[len(host_tokens(lw)):]
        for t in stream:
            time.sleep(self.delay)
            yield Event("token", {"t": t})
        self.stats["launches"] += 1
        self.stats["chip_tokens"] += len(stream)
        self.stats["verified"] += 1
        out = read(p, tr.mem)
        yield Event("result", {"kernel": name, "outputs": out, "tokens": len(stream),
                               "seconds": 2.0, "verified": True})
        return out

    def send(self, msg):
        if "matrices" in msg:  # calc mode: an expression; its value is a list, not named arrays
            call = ("<tool_call>\n<function=calc>\n<parameter=expression>\n"
                    "[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]\n</parameter>\n</function>")
            for k in range(0, len(call), 8):  # the call streams in, as Qwen writes it
                time.sleep(self.delay * 20)
                yield Event("text", {"text": call[k:k + 8]})
            self.drafted = True
            yield Event("call", {"expression": "[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]"})
            yield from self.launch("matmul", {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, 2)
            yield Event("value", {"value": [[19, 22], [43, 50]]})
            yield Event("text", {"text": "The product is:\n\n$$ \\begin{bmatrix} 19 & 22 \\\\ "
                                         "43 & 50 \\end{bmatrix} $$"})
            yield Event("done", dict(self.stats))
            return
        if "capital" in msg:  # the router: no calculation, straight to the answer
            yield Event("text", {"text": "Paris."})
            yield Event("done", dict(self.stats))
            return
        yield Event("call", {"source": "__global__ void f(float *x) {}", "args": [[1]],
                             "threads": 1})
        yield Event("refused", {"error": "the GPU computes 32-bit integers only"})
        yield Event("call", {"source": "__global__ void dot(int *a, int *b, int *out, int n)",
                             "args": [[4, 3], [30, 55], {"zeros": 1}, 2], "threads": 2})
        c = yield from self.launch("vmul", {"a": [4, 3], "b": [30, 55]}, 2)
        yield from self.launch("reduce", {"x": c["c"]}, 2)
        yield Event("value", {"value": {"out": [285]}})
        yield Event("text", {"text": "You spent **285**."})
        yield Event("done", dict(self.stats))


async def wait(pilot, cond, steps=100):
    for _ in range(steps):
        await pilot.pause(0.05)
        if cond():
            return


def test_chat_flow(monkeypatch):
    monkeypatch.setattr(tui, "Session", FakeSession)

    async def run():
        app = tui.QGPU("fake", None, False, True)
        async with app.run_test(size=(150, 50)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            assert not app.query_one("#prompt").disabled
            await pilot.click("#prompt")
            await pilot.press(*"How much did I spend?")
            await pilot.press("enter")
            await wait(pilot, lambda: not app.busy)
            assert not app.busy
            tools = list(app.query(tui.Tool))
            assert [t.label for t in tools] == ["cuda", "cuda"]
            assert [t.state for t in tools] == ["err", "ok"]
            lines = [str(w.render()) for w in tools[1].query(".line")]
            assert lines[0].startswith("  ⎿  ✓ vmul") and "[120, 165]" in lines[0]
            assert "exact" in lines[0] and "reduce" in lines[1] and "[285]" in lines[1]
            assert lines[2].strip() == "out = [285]"
            assert "Error: the GPU computes" in str(tools[0].query(".line").first().render())
            mon = app.query_one(tui.GPUView).mon
            assert mon.done and mon.records > 0 and mon.mem_writes > 0  # it followed the stream
            await pilot.pause(0.3)
            assert app.query(tui.Answer).last().text == "You spent **285**."
            assert "2 launches" in str(app.query_one("#hint").render())
            assert "2 launches" in str(app.query_one("#g-log").render())  # the panel's log
            assert app.send("/new")
            await pilot.pause(0.1)
            assert not app.query(tui.Tool) and app.query(".welcome")
            await pilot.click(tui.Example)  # an example prompt sends itself
            await wait(pilot, lambda: not app.busy and app.query(tui.Tool))
            assert app.query(tui.UserMsg)

    asyncio.run(run())


def test_slash_commands(monkeypatch):
    """Settings are slash commands typed in the input; the hint line shows them."""
    monkeypatch.setattr(tui, "Session", FakeSession)

    async def run():
        app = tui.QGPU("fake", None, False, False)
        async with app.run_test(size=(150, 50)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            hint = lambda: str(app.query_one("#hint").render())
            assert "normal mode" in hint()
            await pilot.click("#prompt")
            await pilot.press(*"/mo")
            assert "/mode" in str(app.query_one("#menu").render())  # the matching commands
            await pilot.press("tab")
            assert app.query_one("#prompt").value == "/mode "
            await pilot.press(*"advanced", "enter")
            await pilot.pause(0.1)
            assert app.session.mode == "cuda" and "advanced mode" in hint()
            app.send("/mode")  # no argument: the other mode
            app.send("/verify on")
            app.send("/think")
            await pilot.pause(0.1)
            s = app.session
            assert s.mode == "calc" and s.check and s.think and "verify" in hint()
            app.send("/panel")
            assert app.query_one("#side").has_class("hidden")
            app.send("/bogus")
            await pilot.pause(0.1)
            assert "Unknown command" in str(app.query(tui.UserMsg).last().reply)

    asyncio.run(run())


def test_plain_answer_has_no_call(monkeypatch):
    monkeypatch.setattr(tui, "Session", FakeSession)

    async def run():
        app = tui.QGPU("fake", None, False, False)
        async with app.run_test(size=(150, 50)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            await pilot.click("#prompt")
            await pilot.press(*"capital of France?")
            await pilot.press("enter")
            await wait(pilot, lambda: not app.busy)
            await pilot.pause(0.3)
            assert not app.query(tui.Tool) and app.query(tui.Answer).last().text == "Paris."

    asyncio.run(run())


def test_calc_value_renders(monkeypatch):
    """A calc call's value (a number, a list, a matrix) shows as "= value"."""
    monkeypatch.setattr(tui, "Session", FakeSession)

    async def run():
        app = tui.QGPU("fake", None, False, True)
        async with app.run_test(size=(150, 50)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            await pilot.click("#prompt")
            await pilot.press(*"Multiply the matrices")
            await pilot.press("enter")
            await wait(pilot, lambda: not app.busy)
            call = app.query_one(tui.Tool)
            assert call.label == "calc" and call.arg.endswith("@ [[5, 6], [7, 8]]")
            assert app.session.drafted and not app.query(tui.Draft)  # the draft became the call
            lines = [str(w.render()) for w in call.query(".line")]
            assert lines[-1].strip() == "= [[19, 22], [43, 50]]" and call.state == "ok"
            answer = app.query(tui.Answer).last().shown
            assert "[[19, 22], [43, 50]]" in answer and "begin" not in answer  # plain math
            assert not any("Error" in str(n.render()) for n in app.query(tui.Notice))

    asyncio.run(run())


def test_plain_math():
    """Qwen's LaTeX as plain text (a terminal cannot typeset it); money keeps its $."""
    t = (r"$$ \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix} $$ and $4 \times 30 + "
         r"\frac{6}{2}$, about $\text{total} \approx 285$, for $5.")
    assert tui.plain_math(t) == (" [[19, 22], [43, 50]]  and 4 × 30 + (6)/(2), about total ≈ "
                                 "285, for $5.")


def test_derivation_lines_stay_text():
    """A math line of only "=" must not turn the line above into a Markdown heading."""
    from io import StringIO

    from rich.console import Console
    from rich.markdown import Markdown

    t = "$$\nA B\n=\nC\n$$"
    c = Console(file=StringIO(), width=60, color_system=None)
    c.print(Markdown(tui.plain_math(t)))
    assert c.file.getvalue().split() == ["A", "B", "=", "C"]  # one plain line, no heading


def test_answer_keeps_line_breaks():
    """Single lines (steps of a sum) stay lines; code blocks are left alone."""
    from io import StringIO

    from rich.console import Console
    from rich.markdown import Markdown

    t = "We add:\n\n2 + 5 = 7\n3 + 6 = 9\n\n```\nx = 1\n```"
    c = Console(file=StringIO(), width=60, color_system=None)
    c.print(Markdown(tui.hard_breaks(t)))
    lines = [ln.strip() for ln in c.file.getvalue().split("\n") if ln.strip()]
    assert "2 + 5 = 7" in lines and "3 + 6 = 9" in lines and "x = 1" in lines


def test_register_values_are_signed():
    assert tui.num(0xFFFFFFFC) == "-4" and tui.num(12345, 4) == "12k"


def test_panel_follows_the_window(monkeypatch):
    """Resizing: a narrow window hides the panel; at every size the shown cards fit."""
    monkeypatch.setattr(tui, "Session", FakeSession)

    async def run():
        app = tui.QGPU("fake", None, False, True)
        async with app.run_test(size=(148, 47)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            app.send("Multiply the matrices")
            await wait(pilot, lambda: not app.busy)
            for w, h in ((148, 47), (120, 30), (110, 24), (105, 20), (80, 24), (220, 60)):
                await pilot.resize_terminal(w, h)
                await pilot.pause(0.3)
                side, gpu = app.query_one("#side"), app.query_one(tui.GPUView)
                assert side.display == (w >= tui.NARROW), (w, h)
                if side.display:
                    used = sum(c.outer_size.height for c in gpu.children if c.display)
                    assert used <= side.size.height and gpu.sm.display, (w, h, used)
            app.send("/panel")  # /panel overrides the automatic choice
            await pilot.pause(0.1)
            assert not app.query_one("#side").display

    asyncio.run(run())


def test_command_menu_keys(monkeypatch):
    """The menu takes the arrow keys: up/down choose a command, left/right its option, enter
    runs it; escape closes it."""
    monkeypatch.setattr(tui, "Session", FakeSession)

    async def run():
        app = tui.QGPU("fake", None, False, False)
        async with app.run_test(size=(150, 50)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            await pilot.click("#prompt")
            await pilot.press("/")
            assert app.menu[0] == "/mode" and app.query_one("#menu").display
            await pilot.press("right", "enter")  # /mode, from normal to advanced
            await pilot.pause(0.1)
            assert app.session.mode == "cuda"
            await pilot.press("/", "down", "left", "enter")  # /verify, from off to on
            await pilot.pause(0.1)
            assert app.session.check and not app.menu
            await pilot.press("/", "down", "down", "enter")  # /think as it stands: off
            await pilot.pause(0.1)
            assert not app.session.think
            await pilot.press("/", "up", "escape")  # up wraps to /quit; escape closes
            assert not app.menu and app.query_one("#prompt").value == "" and app.is_running

    asyncio.run(run())


def test_call_shows_while_written(monkeypatch):
    """While Qwen writes a call, a draft shows the tool and what it has written so far."""
    monkeypatch.setattr(tui, "Session", FakeSession)
    monkeypatch.setattr(FakeSession, "delay", 0.01)

    async def run():
        app = tui.QGPU("fake", None, False, False)
        async with app.run_test(size=(150, 50)) as pilot:
            await wait(pilot, lambda: app.session is not None)
            app.send("Multiply the matrices")
            await wait(pilot, lambda: app.query(tui.Draft)  # drawn on the next frame
                       and "@" in str(app.query_one(tui.Draft).body.render()))
            draft = app.query_one(tui.Draft)
            assert "calc" in str(draft.head.render()) and "writing" in str(draft.head.render())
            assert "[[1, 2], [3, 4]] @" in str(draft.body.render())
            await wait(pilot, lambda: not app.busy, 200)
            assert not app.query(tui.Draft) and app.query_one(tui.Tool).label == "calc"

    asyncio.run(run())
