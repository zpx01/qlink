"""qgpu chat: talk to Qwen, which computes on the GPU inside its own weights.

Your message sits in a box on the right, Qwen's text follows a dot, and each GPU call is
one "● calc(...)" line with its kernel launches and value indented under "⎿". Settings
are slash commands typed in the input (/help lists them). In normal mode Qwen writes the
arithmetic as an expression and the GPU's library of CUDA kernels runs it; in advanced mode
Qwen writes the CUDA kernel itself (qgpu/bridge.py, qgpu/calc.py, qgpu/cuda.py). The panel on
the right is the GPU, live: its warp scheduler, program ROM, pipeline, register file and memory
light up as the model's chip tokens drive them (qgpu/monitor.py).

Run: uv run qgpu [MERGED] [--device mps] [--mode normal|advanced] [--think] [--check]
"""

import json
import re
import time
from collections import deque
from pathlib import Path
from typing import ClassVar

from rich.cells import cell_len
from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static

from qgpu.bridge import Cancelled, Event, Session
from qgpu.circuit import kernel
from qgpu.circuit.kernel import NREG
from qgpu.monitor import Monitor
from qgpu.runtime import KERNELS

MODES = {"normal": "calc", "advanced": "cuda"}  # what the user calls a mode: the session's
MODE_NAME = {v: k for k, v in MODES.items()}
EXAMPLES = [
    "Add the vectors [2, 3, 4] and [5, 6, 7].",
    "What is the dot product of [1, 2, 3] and [4, 5, 6]?",
    "Scale the vector [3, -1, 4] by 5, then add [1, 1, 1].",
    "What is the squared length of the vector [3, 4, 12]?",
    "Multiply the matrices [[1, 2], [3, 4]] and [[5, 6], [7, 8]].",
]
COMMANDS = [  # (command, arguments, what it does)
    ("/mode", "normal|advanced", "switch mode (starts a new chat)"),
    ("/verify", "on|off", "check each launch against the CPU reference"),
    ("/think", "on|off", "let Qwen think before it answers"),
    ("/new", "", "start a new chat"),
    ("/panel", "", "show or hide the GPU panel"),
    ("/help", "", "commands and keys"),
    ("/quit", "", "exit"),
]
MODE_NOTES = [
    ("normal", "Qwen writes the math; ready-made GPU kernels compute it. Dependable."),
    ("advanced", ("Qwen writes its own CUDA kernels. Experimental: a 2B model gets them "
                  "right only some of the time.")),
]
OPTIONS = {"/mode": ("normal", "advanced"), "/verify": ("on", "off"),
           "/think": ("on", "off")}  # a command's choices, picked with left/right in the menu
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STAR = "·✢✳✶✻✽✻✶✳✢"  # the status line's pulse
ACCENT, OK, ERR = "#d7875f", "#87af87", "#d77575"  # muted: one warm accent, a calm green/red
SPARK = " ▁▂▃▄▅▆▇█"
FPS = 12
NARROW = 100  # columns: below this the GPU panel hides itself
SAMPLE = 6  # frames per throughput sample (half a second)


LATEX = [(r"\\(?:times)\b", "×"), (r"\\cdot\b", "·"), (r"\\div\b", "÷"), (r"\\le(?:q)?\b", "≤"),
         (r"\\ge(?:q)?\b", "≥"), (r"\\neq?\b", "≠"), (r"\\approx\b", "≈"), (r"\\pm\b", "±"),
         (r"\\(?:quad|qquad|,|;|!)", " "), (r"\\left|\\right", ""), (r"\\(?:mathbf|text|mathrm)\{([^{}]*)\}", r"\1"),
         (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"), (r"\\sqrt\{([^{}]*)\}", r"√(\1)")]


def plain_math(text: str) -> str:
    """LaTeX in Qwen's answer as plain text: a terminal cannot typeset it. Matrices become
    [[a, b], [c, d]], operators their symbols, and $ / $$ / \\( \\) / \\[ \\] delimiters go."""
    def matrix(m):
        rows = [r for r in re.split(r"\\\\", m.group(1)) if r.strip()]
        return "[" + ", ".join("[" + ", ".join(c.strip() for c in r.split("&")) + "]"
                               for r in rows) + "]"

    text = re.sub(r"\\begin\{[bpvBV]?matrix\}(.*?)\\end\{[bpvBV]?matrix\}", matrix, text,
                  flags=re.DOTALL)
    for pat, rep_ in LATEX:
        text = re.sub(pat, rep_, text)
    text = re.sub(r"\$\$|\\\[|\\\]|\\\(|\\\)", "", text)
    text = re.sub(r"(?<![\w\\])\$([^$\n]+)\$", r"\1", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    # a line of only = or - (a step of a derivation) would make the line above a Markdown
    # heading (setext) or a rule: escape it so it stays text
    return "\n".join("\\" + ln.strip() if ln.strip() and set(ln.strip()) <= set("=-") else ln
                     for ln in text.split("\n"))


def hard_breaks(text: str) -> str:
    """Qwen's line breaks kept: Markdown joins single lines into one paragraph ("2 + 5 = 7"
    and "3 + 6 = 9" on one line), so each line outside a code block ends in a hard break."""
    out, code = [], False
    for ln in text.split("\n"):
        code ^= ln.lstrip().startswith("```")
        out.append(ln if code or ln.lstrip().startswith("```") or not ln.strip() else ln + "  ")
    return "\n".join(out)


def about(merged: str) -> str:
    """What the checkpoint is, from its qgpu.json (written by qgpu.merge)."""
    try:
        m = json.loads((Path(merged) / "qgpu.json").read_text())
        base, gpu, qwen = m["base"], m["chip_layers"], m["qwen_layers"]
    except (OSError, KeyError, ValueError):
        base, gpu, qwen = "Qwen3.5", "its own", None
    own = f"Qwen's {qwen}" if qwen else "Qwen's own"
    return (f"{base} with a GPU compiled into its weights: {gpu} layers in front of {own}, "
            "every weight computed, none learned.")


def short(v, n: int = 40) -> str:
    return short_text(json.dumps(v, separators=(", ", ": ")), n)


def short_text(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def light(h: float, on: tuple, base: tuple) -> str:
    """A light's color: `base` when dark, `on` at full brightness."""
    h = max(0.0, min(1.0, h))
    r, g, b = (int(x + (y - x) * h) for x, y in zip(base, on))
    return f"rgb({r},{g},{b})"


def spark(vals, width: int) -> str:
    vals = list(vals)[-width:]
    top = max(max(vals, default=0), 1e-9)
    s = "".join(SPARK[min(8, round(v / top * 8))] if v > 0 else SPARK[1] for v in vals)
    return s.rjust(width, SPARK[1])


# the GPU's lights: each unit has a dark base and a lit colour
WARP, BLOCK, STAGE, REG, MEM, WROTE = (240, 190, 90), (110, 180, 230), (230, 140, 90), \
    (120, 210, 150), (110, 160, 240), (70, 110, 90)
OFF, LOADED, DIM = (44, 48, 54), (52, 82, 96), (110, 116, 126)


# --- chat widgets -----------------------------------------------------------------------------


class UserMsg(Horizontal):
    """Your message: a box with round corners on the right, as in a chat app."""

    def __init__(self, msg: str, reply: Text | None = None) -> None:
        super().__init__()
        self.msg, self.reply = msg, reply  # reply: a slash command's outcome, under it

    def compose(self) -> ComposeResult:
        t = Text(self.msg)
        if self.reply is not None:
            t.append("\n")
            t.append_text(self.reply)
        yield Static(t, classes="bubble")


class Notice(Static):
    pass


class Example(Static):
    """A clickable example prompt."""

    def __init__(self, prompt: str) -> None:
        super().__init__(Text.assemble(("  › ", "dim"), prompt))
        self.prompt = prompt

    def on_click(self, _: events.Click) -> None:
        self.app.send(self.prompt)


class Holder(Vertical):
    """A container that takes widgets before it is mounted: they wait for it (events can
    arrive faster than Textual mounts)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.waiting: list = []
        self.ready = False  # on_mount has run: mount directly (is_mounted is set later)

    def add(self, *ws) -> None:
        if self.ready:
            self.mount(*ws)
        else:
            self.waiting += ws

    def on_mount(self) -> None:
        self.ready = True
        if self.waiting:
            self.mount(*self.waiting)
            self.waiting = []


class Answer(Horizontal):
    """Qwen's text: a dot, then Markdown (Rich's: plain terminal text, LaTeX made plain)."""

    def __init__(self) -> None:
        super().__init__(classes="answer")
        self.text, self.shown, self.body = "", "", Static("", classes="body")

    def compose(self) -> ComposeResult:
        yield Static("●", classes="dot")
        yield self.body

    def write(self, piece: str) -> None:
        self.text += piece
        self.shown = plain_math(self.text.split("<tool_call>")[0]).strip()
        self.body.update(RichMarkdown(hard_breaks(self.shown)) if self.shown else "")
        self.display = bool(self.shown)


class Draft(Vertical):
    """A tool call while Qwen is still writing it: the tool's name, then each parameter as it
    streams in (CUDA source highlighted). The call event replaces it with a Tool."""

    def __init__(self) -> None:
        super().__init__(classes="tool")
        self.head, self.body = Static("", classes="head"), Static("", classes="source")
        self.call = ""  # the call's text so far, after <tool_call>

    def compose(self) -> ComposeResult:
        yield self.head
        yield self.body

    def draw(self, spin: str) -> None:
        call = re.sub(r"\n?<[^>]*$", "", self.call)  # a tag still being written: not text
        name = re.search(r"<function=([^>\s]+)>", call)
        params = re.findall(r"<parameter=([^>\s]+)>\n?(.*?)(?:\n?</parameter>|$)", call,
                            re.DOTALL)
        self.head.update(Text.assemble((f"{spin} ", ACCENT), (name.group(1) if name else "tool",
                                                                "bold"),
                                       ("  writing the call…", "dim")))
        if params and params[0][0] == "source":
            body = Syntax(params[0][1].rstrip(), "cuda", theme="nord", word_wrap=True,
                          background_color="default")
        else:
            body = Text("\n".join(f"{k} = {' '.join(v.split())}" if len(params) > 1 else v.strip()
                                  for k, v in params), style="dim")
        self.body.update(body)


class Tool(Holder):
    """One step on the GPU, as a "● name(argument)" line and its lines under "⎿"."""

    def __init__(self, name: str, arg: str = "", source: str | None = None) -> None:
        super().__init__(classes="tool")
        self.label, self.arg, self.source = name, arg, source
        self.state = "run"  # run, ok, err
        self.head, self.n = Static(self._head(), classes="head"), 0

    def _head(self) -> Text:
        dot = {"run": "dim", "ok": OK, "err": ERR}[self.state]
        t = Text.assemble(("● ", dot), (self.label, "bold"))
        if self.arg:
            t.append(f"({self.arg})", style="dim" if self.state == "run" else "")
        return t

    def compose(self) -> ComposeResult:
        yield self.head
        if self.source:
            yield Static(Syntax(self.source.strip(), "cuda", theme="nord", word_wrap=True,
                                background_color="default"), classes="source")

    def line(self, text) -> Static:
        """A line under the call: "⎿" before the first, aligned spaces after."""
        t = indented(text if isinstance(text, Text) else Text(str(text)), self.n == 0)
        self.n += 1
        w = Static(t, classes="line")
        self.add(w)
        return w

    def set(self, state: str) -> None:
        self.state = state
        self.head.update(self._head())


def indented(t: Text, first: bool = True) -> Text:
    """Text under a call: "  ⎿  " (or spaces) before its first line, spaces before the rest."""
    out = Text()
    for i, line in enumerate(t.split("\n", allow_blank=True)):
        out.append("\n" if i else "")
        out.append("  ⎿  " if first and i == 0 else "     ", style="dim")
        out.append_text(line)
    return out


def launch_line(d: dict, mark: str, extra: Text | None = None) -> Text:
    """A kernel launch under its call: a mark (spinner, ✓, ✗), the kernel, what it computes,
    then progress or result."""
    doc = KERNELS[d["kernel"]].doc if d["kernel"] in KERNELS else ""
    t = Text.assemble((f"{mark} ", ACCENT if mark in SPIN else OK if mark == "✓" else ERR),
                      (f"{d['kernel']:<8}", ""), (doc, "dim"))
    if extra:
        t.append("  ")
        t.append_text(extra)
    return t


# --- the GPU panel ----------------------------------------------------------------------------

def section(title: str, sub: str = "") -> Text:
    t = Text(title, style="bold")
    if sub:
        t.append(f"  {sub}", style="dim")
    return t


class GPUView(Vertical):
    """The GPU, drawn from the monitor's lights: a status card, the streaming multiprocessor
    (warp scheduler, program ROM, pipeline, register file), global memory, and the launch log."""

    def __init__(self) -> None:
        super().__init__(id="gpu")
        self.mon = Monitor()
        self.kern, self.t0 = "", 0.0
        self.meta = {"lanes": 4, "stages": 4}
        self.model = ""
        self.fed = self.fed_seen = 0  # chip tokens over all launches, at the last sample
        self.rates: deque = deque([0.0] * 34, maxlen=34)  # tok/s, one per SAMPLE frames
        self.history: list[dict] = []
        self.reg_warp = 0  # the warp whose registers the file shows
        self.n = 0
        self.top, self.sm, self.memory, self.launches = (
            Static(classes="card", id=f"g-{k}") for k in ("top", "sm", "mem", "log"))

    def compose(self) -> ComposeResult:
        yield self.top
        yield self.sm
        yield self.memory
        yield self.launches

    def on_mount(self) -> None:
        self.sm.border_title, self.memory.border_title = "SM 0", "global memory"
        self.launches.border_title, self.top.border_title = "launches", "GPU"

    def start(self, d: dict) -> None:
        self.mon.load(d["program"])
        self.kern, self.t0 = d["kernel"], time.time()

    def feed(self, t: int) -> None:
        self.mon.feed(t)
        self.fed += 1

    def finish(self, d: dict) -> None:
        self.history.append(d)

    def frame(self) -> None:
        self.n += 1
        if self.n % SAMPLE == 0:
            self.rates.append((self.fed - self.fed_seen) * FPS / SAMPLE)
            self.fed_seen = self.fed
        if self.mon.warp is not None:
            self.reg_warp = self.mon.warp
        self.top.border_subtitle = self.model
        if "chip_layers" in self.meta:
            self.top.border_title = f"GPU · in {self.meta['chip_layers']} layers of Qwen"
        room = self.parent.size.height if self.parent else 0
        sized = room > 0  # before the first layout every card shows

        w = self.width_of(self.sm)

        def lines(x: Text) -> int:  # a card's height: its lines as wrapped, and its border
            return sum(max(1, -(-cell_len(ln) // w)) for ln in x.plain.split("\n")) + 2

        # The SM first (compact if the full one and a two-line status do not fit), then the
        # status (two lines if the full one does not fit), the launch log, memory: a card
        # that does not fit in what is left is not shown.
        short_top = self.draw_top(short=True)
        sm = self.draw_sm()
        if sized and lines(sm) + lines(short_top) > room:
            sm = self.draw_sm(compact=True)
        top = self.draw_top()
        if sized and lines(sm) + lines(top) > room:
            top = short_top
        for card, text in ((self.sm, sm), (self.top, top), (self.launches, self.draw_log()),
                           (self.memory, self.draw_mem())):
            card.display = not sized or lines(text) <= room
            room -= lines(text) if card.display else 0
            card.update(text)
        self.mon.tick()

    @staticmethod
    def width_of(card: Static) -> int:
        """A card's text width: the drawings grow with the panel (46 before layout)."""
        return card.content_size.width or 46

    @property
    def live(self) -> bool:
        return self.mon.prog is not None and not self.mon.done

    def draw_top(self, short: bool = False) -> Text:
        """The status card; short: only what runs and the throughput."""
        m, t, w = self.mon, Text(), self.width_of(self.top)
        if self.live:
            doc = KERNELS[self.kern].doc if self.kern in KERNELS else ""
            t.append("● running ", style=ACCENT)
            t.append(self.kern, style="bold")
            t.append(f"  {short_text(doc, max(8, w - 13 - len(self.kern)))}", style="dim")
        else:
            t.append("○ idle", style="dim")
            t.append(f" · last: {self.kern}" if self.kern else "", style="dim")
        t.append("\ntok/s ", style="dim")
        t.append(spark(self.rates, w - 13), style=ACCENT if self.live else "dim")
        t.append(f" {self.rates[-1]:5.1f}", style="bold" if self.live else "dim")
        if short:
            return t
        lanes, stages = self.meta.get("lanes", 4), self.meta.get("stages", 4)
        t.append(f"\n{lanes} lanes · {stages} stages · {NREG} regs · {kernel.SINK} words",
                 style="dim")
        if m.prog is not None:
            t.append(f"\n{m.tokens:,} tok · {m.records} records · {m.reg_writes} reg · "
                     f"{m.mem_writes} mem writes", style="" if self.live else "dim")
        return t

    def draw_sm(self, compact: bool = False) -> Text:
        """The SM; compact (a short window): no spacing lines and a one-line pipeline."""
        m, p, live = self.mon, self.mon.prog, self.live
        lanes = p.lanes if p else self.meta.get("lanes", 4)
        w = self.width_of(self.sm)
        t = Text()
        # one line each: the 32 warp slots and the 32 ROM blocks, spaced if the width allows
        nw, nb, slots = (len(m.warps), len(m.blocks), 1 << kernel.WBITS) if p else \
            (0, 0, 1 << kernel.WBITS)
        gap = " " if w >= 9 + 2 * slots else ""
        t.append("warps    ", style="bold")
        for k in range(slots):
            t.append(("●" if live and k == m.warp else "○" if k < nw else "·") + gap,
                     style=light(m.warps[k] if k < nw else 0, WARP, DIM if k < nw else OFF))
        t.append(f"\n{'rom':<9}", style="bold")
        for b in range(1 << kernel.BBITS):
            t.append(("█" if gap else "▮") + gap,
                     style=light(m.blocks[b] if b < nb else 0, BLOCK, LOADED if b < nb else OFF))
        t.append("\n" + " " * 9, style="dim")
        t.append(f"{nw} warp{'s' * (nw != 1)} × {lanes} lanes · " if p else "", style="dim")
        t.append((f"block {m.block} of {nb}" if live and m.block is not None else
                  f"{nb} blocks") if p else f"{1 << kernel.BBITS} blocks free",
                 style="dim")

        stages = len(m.stages) if p else self.meta.get("stages", 4)
        ops = p.stages[m.block] if live and m.block is not None else []
        gap = "\n" if compact else "\n\n"
        t.append(gap)
        if compact:
            t.append(f"{'pipeline':<9}", style="bold")
            for s in range(stages):
                op = ops[s] if s < len(ops) and ops[s] != "nop" else "·"
                c = light(m.stages[s] if live and s < len(m.stages) else 0, STAGE, DIM)
                t.append(f"[{short_text(op, 5):^5}]", style=c)
                t.append("›" if s < stages - 1 else "", style="dim")
            return self._registers(t, gap, lanes, w)
        t.append_text(section("pipeline", "fetch › exec › write back"))
        inner = max(5, min(8, (w - 3 * (stages - 1)) // stages - 2))  # a box's inside
        rows: list[list[tuple[str, str]]] = [[], [], []]
        for s in range(stages):
            op = ops[s] if s < len(ops) and ops[s] != "nop" else "·"
            c = light(m.stages[s] if live and s < len(m.stages) else 0, STAGE, DIM)
            box = (f"┌{'─' * inner}┐", f"│{short_text(op, inner):^{inner}}│",
                   f"└{'─' * inner}┘")
            for i, (row, piece) in enumerate(zip(rows, box)):
                row.append((piece, c))
                if s < stages - 1:
                    row.append(("─›─" if i == 1 else "   ", "dim"))
        for row in rows:
            t.append("\n")
            for piece, st in row:
                t.append(piece, style=st)

        return self._registers(t, gap, lanes, w)

    def _registers(self, t: Text, gap: str, lanes: int, w: int) -> Text:
        m, p = self.mon, self.mon.prog
        t.append(gap)
        t.append_text(section("registers", f"warp {self.reg_warp}" if p else
                              f"{NREG} × {lanes} lanes"))
        cw = max(4, min(6, (w - 9) // (2 * lanes)))  # a register cell
        head = "".join(f"{f'L{L}':>{cw}}" for L in range(lanes))
        t.append(f"\n   {head}     {head}", style="dim")
        half = NREG // 2
        for r in range(half):
            t.append("\n")
            for rr in (r, r + half):
                t.append(f"r{rr:<2}", style="dim")
                for L in range(lanes):
                    v = m.values.get((self.reg_warp, rr, L)) if p else None
                    h = m.regs[rr][L] if p else 0
                    if v is None:
                        t.append(f"{'·':>{cw}}", style=light(h, REG, OFF))
                    else:
                        t.append(f"{num(v, cw - 1):>{cw}}", style=light(h, REG, DIM))
                t.append("  " if rr < half else "")
        return t

    def draw_mem(self) -> Text:
        m, p = self.mon, self.mon.prog
        used = set(p.memory) if p else set()
        w = self.width_of(self.memory)
        per = 32 if w >= 4 + 32 else 16  # words per row
        cell = 2 if w >= 4 + 2 * per else 1
        t = Text()
        for row in range(0, kernel.SINK, per):
            t.append(f"{row:>3} " if row == 0 else f"\n{row:>3} ", style="dim")
            for a in range(row, min(row + per, kernel.SINK)):
                base = WROTE if a in m.written else LOADED if a in used else OFF
                t.append(("■" if a in used or a in m.written else "·").ljust(cell),
                         style=light(m.mem[a] if p else 0, MEM, base))
        t.append("\n    ")
        for label, c in (("free", OFF), ("buffer", LOADED), ("written", WROTE), ("writing", MEM)):
            t.append("■ ", style=light(0, c, c))
            t.append(f"{label}  ", style="dim")
        if p is not None:
            t.append(f"\n    {len(used)} words in buffers · {len(m.written)} written back",
                     style="dim")
        return t

    def draw_log(self) -> Text:
        t = Text()
        if not self.history:
            t.append("no launches yet", style="dim")
            return t
        n, exact = len(self.history), sum(d["verified"] is True for d in self.history)
        ow = max(12, self.width_of(self.launches) - 27)
        for d in self.history[-4:]:
            v = d["verified"]
            t.append("✓ " if v is not False else "✗ ", style=OK if v else ERR if v is False
                     else "dim")
            out = next(iter(d["outputs"].values())) if len(d["outputs"]) == 1 else d["outputs"]
            t.append(f"{d['kernel']:<8}")
            t.append(f"{short(out, ow):<{ow + 1}}")
            t.append(f"{d['tokens']:>5,} tok {d['seconds']:>4.1f}s\n", style="dim")
        toks = sum(d["tokens"] for d in self.history)
        t.append(f"{n} launch{'es' * (n != 1)} · {toks:,} chip tokens", style="dim")
        if any(d["verified"] is not None for d in self.history):
            t.append(f" · {exact}/{n} exact", style="dim")
        return t


def num(v: int, n: int = 4) -> str:
    """A register's 32 bits as a signed value, in at most n characters."""
    v = v - (1 << 32) if v >> 31 else v
    if len(str(v)) <= n:
        return str(v)
    for div, suf in ((1000, "k"), (1_000_000, "M"), (1_000_000_000, "G")):
        if abs(v) // div < (100 if v < 0 else 1000):
            return f"{v // div}{suf}"
    return "big"


# --- the input --------------------------------------------------------------------------------

class Prompt(Input):
    """The message box. While the command menu is open: up/down choose a command, left/right
    its option, enter runs it, tab completes it, escape closes the menu."""

    BINDINGS: ClassVar = [
        Binding("tab", "complete", show=False),
        Binding("up", "menu(-1)", show=False), Binding("down", "menu(1)", show=False),
        Binding("left", "option(-1)", show=False), Binding("right", "option(1)", show=False),
        Binding("escape", "close", show=False),
    ]

    def action_complete(self) -> None:
        if self.app.menu:
            self.value = self.app.menu[self.app.menu_i] + " "
            self.cursor_position = len(self.value)

    def action_menu(self, step: int) -> None:
        if self.app.menu:
            self.app.menu_i = (self.app.menu_i + step) % len(self.app.menu)
            self.app.draw_menu()

    def action_option(self, step: int) -> None:
        cmd = self.app.menu[self.app.menu_i] if self.app.menu else None
        if cmd in OPTIONS:
            self.app.menu_opt[cmd] = (self.app.menu_opt[cmd] + step) % len(OPTIONS[cmd])
            self.app.draw_menu()
        elif step < 0:  # no menu: the arrows move the cursor as usual
            self.action_cursor_left()
        else:
            self.action_cursor_right()

    def action_close(self) -> None:
        if self.app.menu:
            self.value = ""
        else:
            self.app.action_stop()


# --- the app ----------------------------------------------------------------------------------

class QGPU(App):
    TITLE = "qgpu"
    CSS = """
    Screen { background: $background; }
    #main { height: 1fr; }
    #chat, #side { scrollbar-size-vertical: 1; scrollbar-background: $background;
                   scrollbar-color: $panel-lighten-1; scrollbar-color-hover: $panel-lighten-2;
                   scrollbar-color-active: $panel-lighten-3; }
    #chat { width: 1fr; padding: 0 1; }
    #side { width: 34%; min-width: 46; max-width: 84; padding: 0 0 0 1; overflow-y: hidden; }
    #side.hidden { display: none; }
    .card { border: round $panel-lighten-2; border-title-color: $text-muted;
            border-subtitle-color: $text-muted; padding: 0 1; height: auto; }
    .welcome { padding: 0 1; margin: 0 0 1 0; border: round $panel-lighten-2; height: auto; }
    .welcome Static { height: auto; }
    .mode { height: auto; }
    .mode .mode-name { width: 12; padding: 0 0 0 2; text-style: bold; }
    .mode Static { width: 1fr; }
    Example:hover { background: $boost; }
    UserMsg { margin: 1 0 0 0; height: auto; align-horizontal: right; }
    UserMsg .bubble { width: auto; max-width: 75%; height: auto; padding: 0 1;
                      border: round $panel-lighten-3; background: transparent; }
    .answer { height: auto; margin: 1 0 0 0; }
    .answer .dot { width: 2; }
    .answer .body { width: 1fr; height: auto; }
    .tool { height: auto; margin: 1 0 0 0; }
    .tool Static { height: auto; }
    .tool .source { margin: 0 0 0 5; }
    .thought { margin: 1 0 0 0; color: $text-muted; text-style: italic; height: auto; }
    Notice { color: $text-muted; margin: 1 0 0 0; }
    #status { height: 1; padding: 0 1; display: none; }
    #status.on { display: block; }
    #prompt { margin: 0; border: round $panel-lighten-2; }
    #prompt:focus { border: round $panel-lighten-3; }
    #menu { height: auto; padding: 0 2; display: none; }
    #menu.on { display: block; }
    #hint { height: 1; padding: 0 2; color: $text-muted; }
    """
    BINDINGS: ClassVar = [
        Binding("escape", "stop", "Stop", show=False),
        Binding("ctrl+b", "toggle_side", "GPU panel", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
    ]

    def __init__(self, merged: str, device: str | None, think: bool, check: bool,
                 mode: str = "normal"):
        super().__init__()
        self.merged, self.device, self.think, self.check = merged, device, think, check
        self.mode = MODES.get(mode, mode)  # the session's name for it
        self.session: Session | None = None
        self.busy, self.t0 = True, time.time()
        self.status_text = "Loading the model"
        self.frame_no = 0
        self.tool = self.answer = self.line = self.thought = self.draft = None
        self.launch, self.launch_t0 = None, 0.0
        self.side_shown: bool | None = None  # None: by the window's width
        self.menu: list[str] = []  # the slash commands the menu shows
        self.menu_i, self.menu_opt = 0, {}  # the highlighted one; each one's chosen option

    # layout ---------------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with VerticalScroll(id="chat"):
                yield self._welcome()
            with VerticalScroll(id="side"):
                yield GPUView()
        yield Static("", id="status")
        yield Prompt(placeholder="Loading the model…", id="prompt", disabled=True)
        yield Static("", id="menu")
        yield Static("", id="hint")

    def _welcome(self) -> Vertical:
        mode = MODE_NAME[self.session.mode if self.session else self.mode]
        intro = Text()
        intro.append("✻ ", style=ACCENT)
        intro.append("qgpu", style="bold")
        intro.append(f"  {about(self.merged)}\n\n", style="dim")
        intro.append("Try", style="bold")
        modes = Text()
        modes.append("\nModes", style="bold")
        modes.append(f"  now {mode} · /mode to switch", style="dim")
        rows = [Horizontal(Static(name, classes="mode-name"), Static(Text(what, style="dim")),
                           classes="mode")
                for name, what in MODE_NOTES]
        foot = Static(Text("\n/ for commands · esc stops · ctrl+b GPU panel", style="dim"))
        return Vertical(Static(intro), *[Example(e) for e in EXAMPLES], Static(modes), *rows,
                        foot, classes="welcome")

    def _hint(self, width: int | None = None) -> None:
        s, t = self.session, Text()
        if s is None:
            self.query_one("#hint", Static).update("loading…")
            return
        st = s.stats
        # (priority, text): the lowest go first when the line is wider than the window
        parts = [(0, f"{MODE_NAME[s.mode]} mode"), (2, f"{s.name} on {s.dev}"),
                 (1, "thinking" if s.think else ""),
                 (1, (f"verify {st['verified']}/{st['launches']} exact" if st["launches"]
                      else "verify on") if s.check else ""),
                 (3, f"{st['launches']} launch{'es' * (st['launches'] != 1)}"
                  if st["launches"] else ""),
                 (0, "/ for commands")]
        parts = [(k, x) for k, x in parts if x]
        room = (width or self.size.width) - 4
        while len(" · ".join(x for _, x in parts)) > room and len(parts) > 2:
            parts.remove(max(parts, key=lambda kx: kx[0]))
        for i, (_, x) in enumerate(parts):
            t.append(" · " if i else "", style="dim")
            t.append(x, style=ACCENT if i == 0 else "")
        self.query_one("#hint", Static).update(t)

    def _say(self, text: str) -> None:
        self.status_text = text

    # loading --------------------------------------------------------------------------------
    def on_mount(self) -> None:
        # Rich's Markdown draws `code` cyan on a block: plain bold fits the muted palette
        self.console.push_theme(Theme({"markdown.code": "bold", "markdown.code_block": "",
                                       "markdown.link": "underline", "markdown.h1": "bold",
                                       "markdown.h2": "bold", "markdown.item.bullet": "dim"}))
        self._hint()
        self.query_one("#status").add_class("on")
        self.set_interval(1 / FPS, self._frame)
        self.load()

    def _frame(self) -> None:
        """FPS times a second: the GPU panel, the status line, the running launch."""
        if not self.query("#status"):  # the app is closing: its widgets are gone
            return
        self.frame_no += 1
        spin = SPIN[self.frame_no % len(SPIN)]
        gpu = self.query_one(GPUView)
        gpu.frame()
        status = Text()
        if self.busy:
            status.append(f"{STAR[self.frame_no // 2 % len(STAR)]} ", style=ACCENT)
            status.append(self.status_text + "…", style=ACCENT)
            status.append(f"  ({time.time() - self.t0:.0f}s · esc to interrupt)", style="dim")
        self.query_one("#status", Static).update(status)
        if self.draft is not None:
            self.draft.draw(spin)
        if self.line is not None and self.launch is not None:
            rate = gpu.mon.tokens / max(time.time() - self.launch_t0, 1e-6)
            self.line.update(self._under(launch_line(self.launch, spin, Text(
                f"{gpu.mon.tokens:,} tok · {rate:.1f} tok/s", style="dim"))))

    def _under(self, t: Text) -> Text:
        """A running launch's line, keeping its place under the call."""
        head = Text("  ⎿  " if self.line_first else "     ", style="dim")
        head.append_text(t)
        return head

    @work(thread=True, exclusive=True)
    def load(self) -> None:
        import transformers

        transformers.logging.set_verbosity_error()  # its warnings would draw over the screen
        try:
            s = Session(self.merged, self.device, think=self.think, check=self.check,
                        mode=self.mode)
        except Exception as e:  # noqa: BLE001 - shown in the app
            self.call_from_thread(self._loaded, None, str(e))
            return
        self.call_from_thread(self._loaded, s, None)

    def _loaded(self, s: Session | None, error: str | None) -> None:
        self._busy(False)
        if error:
            self._add(Notice(Text(f"Could not load the model: {error}", style=ERR)))
            return
        self.session = s
        gpu = self.query_one(GPUView)
        gpu.meta, gpu.model = s.meta, f"{s.name} · {s.dev}"
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.placeholder = "Ask anything, or / for commands"
        prompt.focus()
        self._hint()

    def _busy(self, on: bool) -> None:
        self.busy = on
        self.query_one("#status").set_class(on, "on")

    # chat -----------------------------------------------------------------------------------
    def _add(self, *ws) -> None:
        self.query_one("#chat").mount(*ws)
        self.query_one("#chat").scroll_end(animate=False)

    @on(Input.Changed, "#prompt")
    def typing(self, ev: Input.Changed) -> None:
        """While a slash command is being typed, the commands that match it, under the box;
        each opens on its current setting."""
        word = ev.value.split(" ")[0]
        menu = [c for c, _, _ in COMMANDS if ev.value.startswith("/") and c.startswith(word)]
        if menu != self.menu:
            self.menu, self.menu_i = menu, 0
            self.menu_opt = {c: OPTIONS[c].index(self._setting(c)) for c in OPTIONS}
        self.draw_menu()

    def _setting(self, cmd: str) -> str:
        s = self.session
        if s is None:
            return OPTIONS[cmd][0]
        if cmd == "/mode":
            return MODE_NAME[s.mode]
        return "on" if (s.check if cmd == "/verify" else s.think) else "off"

    def draw_menu(self) -> None:
        box = self.query_one("#menu", Static)
        box.set_class(bool(self.menu), "on")
        t = Text()
        for i, (c, _, what) in enumerate(x for x in COMMANDS if x[0] in self.menu):
            on = i == self.menu_i
            t.append("\n" if i else "")
            t.append("› " if on else "  ", style=ACCENT)
            t.append(f"{c:<9}", style="bold" if on else "")
            opts = Text()
            for k, o in enumerate(OPTIONS.get(c, ())):
                chosen = k == self.menu_opt.get(c)
                opts.append(f" {o} ", style=f"bold reverse {ACCENT}" if chosen and on else
                            "bold" if chosen else "dim")
            t.append_text(opts)
            t.append(" " * max(1, 22 - opts.cell_len))
            t.append(what, style="" if on else "dim")
        if self.menu:
            t.append("\n  ↑↓ choose · ←→ option · enter run · tab complete · esc close",
                     style="dim")
        box.update(t)

    @on(Input.Submitted, "#prompt")
    def submit(self, ev: Input.Submitted) -> None:
        msg = ev.value.strip()
        if self.menu:  # the highlighted command, with the typed argument or the chosen option
            cmd = self.menu[self.menu_i]
            arg = msg.partition(" ")[2].strip() or (OPTIONS[cmd][self.menu_opt[cmd]]
                                                     if cmd in OPTIONS else "")
            msg = f"{cmd} {arg}".strip()
        if self.send(msg):
            ev.input.value = ""

    def send(self, msg: str) -> bool:
        msg = msg.strip()
        if not msg or self.session is None:
            return False
        if msg.startswith("/"):
            return self.command(msg)
        if self.busy:
            self.notify("Still working: press escape to stop.", severity="warning")
            return False
        for w in self.query(".welcome"):
            w.remove()
        self._add(UserMsg(msg))
        self.tool = self.answer = self.line = self.thought = self.draft = None
        self._busy(True)
        self.t0 = time.time()
        self._say("Reading")
        self.run_chat(msg)
        return True

    def command(self, msg: str) -> bool:
        """A slash command: shown as your line, its outcome under "⎿"."""
        name, _, arg = msg.partition(" ")
        arg, s = arg.strip().lower(), self.session

        def reply(text: str, err: bool = False) -> None:
            self._add(UserMsg(msg, Text(text, style=ERR if err else "dim")))

        def switch(on: bool) -> bool | None:
            return {"on": True, "off": False, "": not on}.get(arg)

        hits = [c for c, _, _ in COMMANDS if c.startswith(name)]
        name = hits[0] if len(hits) == 1 else name
        if self.busy and name in ("/mode", "/new"):
            reply("Still working: press escape first.", True)
        elif name == "/mode":
            want = arg or MODE_NAME["cuda" if s.mode == "calc" else "calc"]
            want = next((m for m in MODES if m.startswith(want)), None)
            if want is None:
                reply("Usage: /mode normal|advanced", True)
            elif MODES[want] == s.mode:
                reply(f"Already in {want} mode.")
            else:
                s.mode = MODES[want]
                self.action_new_chat()
                reply(f"{want.capitalize()} mode. A new chat.")
        elif name in ("/verify", "/think"):
            attr = "check" if name == "/verify" else "think"
            v = switch(getattr(s, attr))
            if v is None:
                reply(f"Usage: {name} on|off", True)
            else:
                setattr(s, attr, v)
                reply(f"{'Verify launches' if attr == 'check' else 'Thinking'}: "
                      f"{'on' if v else 'off'}")
        elif name == "/new":
            self.action_new_chat()
        elif name == "/panel":
            self.action_toggle_side()
        elif name == "/help":
            lines = [f"{c:<9}{a:<17}{w}" for c, a, w in COMMANDS]
            lines += ["", "esc stop · tab complete a command · ctrl+b GPU panel · ctrl+q quit"]
            reply("\n".join(lines))
        elif name in ("/quit", "/exit"):
            self.exit()
        else:
            reply(f"Unknown command {name}. /help lists them.", True)
        self._hint()
        return True

    @work(thread=True, exclusive=True, group="chat")
    def run_chat(self, msg: str) -> None:
        try:
            for e in self.session.send(msg):
                self.call_from_thread(self.show, e)
        except Cancelled:
            self.call_from_thread(self._stopped)
        except Exception as e:  # noqa: BLE001 - shown in the app
            self.call_from_thread(self._failed, repr(e))

    def show(self, e: Event) -> None:
        d = e.data
        gpu = self.query_one(GPUView)
        if e.kind == "token":
            gpu.feed(d["t"])
            return
        if e.kind == "think":
            if self.thought is None:
                self.thought = Static("", classes="thought")
                self.thought.raw = ""
                self._add(self.thought)
            self.thought.raw += d["text"]
            self.thought.update(Text("∴ " + self.thought.raw.strip()))
            self._say("Thinking")
        elif e.kind == "text":
            if self.answer is None:
                self.answer = Answer()
                self._add(self.answer)
            self.answer.write(d["text"])
            if "<tool_call>" in self.answer.text:  # Qwen is writing a call: show it as it goes
                if self.draft is None:
                    self.draft = Draft()
                    self._add(self.draft)
                self.draft.call = self.answer.text.split("<tool_call>", 1)[1]
                self._say("Writing the calculation for the GPU")
            else:
                self._say("Writing")
        elif e.kind == "call":
            if self.answer is not None and not self.answer.display:
                self.answer.remove()  # the turn was only the call
            if self.draft is not None:
                self.draft.remove()  # the call, written: the Tool below shows it
            self.answer = self.draft = None
            if "expression" in d:
                self.tool = Tool("calc", d["expression"])
            else:
                src = d["source"]
                name = re.search(r"void\s+(\w+)", src)
                args = ", ".join(f"{k}={short(v, 24)}" for k, v in (d["args"] or {}).items()
                                 ) if isinstance(d["args"], dict) else short(d["args"], 40)
                self.tool = Tool("cuda", f"{name.group(1) if name else 'kernel'}, {args}", src)
            self._add(self.tool)
            self._say("Computing on the GPU")
        elif e.kind == "launch":
            gpu.start(d)
            self.launch, self.launch_t0 = d, time.time()
            self.line_first = self.tool.n == 0
            self.line = self.tool.line(launch_line(d, SPIN[0]))
            self._say(f"Running {d['kernel']} on the GPU: {d['blocks']} machine blocks, "
                      f"{d['warps']} warp{'s' * (d['warps'] != 1)}")
        elif e.kind == "result":
            ok = d["verified"] is not False
            gpu.finish(d)
            out = next(iter(d["outputs"].values())) if len(d["outputs"]) == 1 else d["outputs"]
            extra = Text.assemble(("→ ", "dim"), (short(out, 36), ""),
                                  (f"  {d['tokens']:,} tok · {d['seconds']:.1f}s", "dim"))
            if d["verified"] is not None:
                extra.append(" · exact" if ok else " · differs from the reference",
                             style="dim" if ok else ERR)
            self.line.update(self._under(launch_line(self.launch, "✓" if ok else "✗", extra)))
            self.line = self.launch = None
            self._hint()
        elif e.kind == "value":
            value = d["value"]  # cuda: the result arrays by name; calc: the expression's value
            named = isinstance(value, dict) and self.tool.label == "cuda"
            for k, v in (value.items() if named else [("", value)]):
                self.tool.line(Text.assemble((f"{k} = " if k else "= ", "bold"),
                                             (json.dumps(v), "bold")))
            self.tool.set("ok")
            self._say("Reading the result")
        elif e.kind == "refused":
            if self.draft is not None:  # a call that never ran (cut off, or not parsed)
                self.draft.remove()
                self.draft = None
            if self.tool is not None and not d.get("repeat") and self.tool.state == "run":
                self.tool.line(Text(f"Error: {d['error']}", style=ERR))
                self.tool.set("err")
            else:
                self._add(Notice(Text(d["error"], style=ERR)))
        elif e.kind == "done":
            self._busy(False)
            self._hint()
            self.query_one("#prompt", Input).focus()
        self.query_one("#chat").scroll_end(animate=False)

    def _stopped(self) -> None:
        self._busy(False)
        self._add(Notice("Interrupted. The context was cleared: the next message starts a new "
                         "chat."))
        self.session.reset()
        self._hint()

    def _failed(self, err: str) -> None:
        self._busy(False)
        self._add(Notice(Text(f"Error: {err}. Started a new chat.", style=ERR)))
        self.session.reset()
        self._hint()

    # actions --------------------------------------------------------------------------------
    def action_new_chat(self) -> None:
        if self.session is None or self.busy:
            return
        self.session.reset()
        self.tool = self.answer = self.line = self.thought = self.draft = None
        chat = self.query_one("#chat")
        chat.remove_children()
        chat.mount(self._welcome())
        self._hint()

    def action_stop(self) -> None:
        if self.session and self.busy:
            self.session.cancel = True

    def action_toggle_side(self) -> None:
        """/panel: show or hide the GPU panel, overriding the automatic choice."""
        self.side_shown = self.query_one("#side").has_class("hidden")
        self._place_side()

    def on_resize(self, ev: events.Resize) -> None:
        self._place_side(ev.size.width)
        self._hint(ev.size.width)

    def _place_side(self, width: int | None = None) -> None:
        """The panel shows when the window is wide enough for it and the chat (or when
        /panel asked for it); a narrow window gives the chat the whole width."""
        width = width or self.size.width
        shown = width >= NARROW if self.side_shown is None else self.side_shown
        self.query_one("#side").set_class(not shown, "hidden")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Chat with Qwen; it computes on the GPU inside "
                                             "its weights")
    ap.add_argument("merged", nargs="?", default="build/merged-2b")
    ap.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
    ap.add_argument("--think", action="store_true", help="think before answering")
    ap.add_argument("--check", action="store_true", help="verify launches against the reference")
    ap.add_argument("--mode", choices=tuple(MODES), default="normal",
                    help="normal: Qwen writes expressions for the GPU's kernel library "
                         "(default); advanced: Qwen writes the CUDA kernels")
    a = ap.parse_args()
    # Loading the model creates a multiprocessing semaphore, which starts Python's resource
    # tracker; started under Textual (stdio replaced) it fails with "bad value(s) in
    # fds_to_keep". Start it while stdio is still the terminal.
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()
    QGPU(a.merged, a.device, a.think, a.check, a.mode).run()


if __name__ == "__main__":
    main()
