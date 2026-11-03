"""The chat session: Qwen computes on the GPU inside its own weights, in one of two modes.

Both modes are one chat with one tool. Qwen calls it as often as it needs, every call runs on
the GPU, its result comes back as the tool response, and Qwen answers from it:
- calc (normal): calc(expression). Qwen writes the arithmetic as a Python-style expression;
  qgpu.calc runs every operation as a launch of a library kernel (written once, verified
  exact through the circuit).
- cuda (advanced): cuda(source, args). Qwen writes the CUDA kernel itself, with KERNEL_RULES
  as its system prompt. The host does what NVRTC and the CUDA runtime do on a real GPU: it
  compiles the source to PTX, lays out the arguments in global memory and lowers the PTX
  (qgpu/cuda.py, qgpu/lower.py); compiler and driver errors go back to Qwen as the response.
A router reads Qwen's yes or no (from its first token's logits) on whether the message asks
for a calculation; if yes, Qwen's turn starts inside the tool call (a required tool choice), so
it writes the expression or the kernel but cannot skip the GPU.

Each launch: the host feeds the program image, the initial memory (all of memory zeroed first
from a context's second launch on) and the first EXEC token, the only chip tokens it writes;
the model writes the launch itself (greedy chip tokens until the end-of-launch token
EXEC(0, warps)); and the host reads the results back from the `in` head's rows in the model's
recurrent state (cudaMemcpy device -> host). The host never runs the reference; `check=True`
only compares each launch's tokens and outputs with it.

`Session.send` yields Events, which the TUI (qgpu/tui.py) renders and `main` prints.

Run: uv run python -m qgpu.bridge MERGED "What is the sum of 12, 7, 30 and 5?"
"""

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from qgpu.calc import evaluate
from qgpu.circuit import kernel
from qgpu.circuit.kernel import EXEC
from qgpu.cuda import KERNEL_RULES, compile_cuda, plan
from qgpu.monitor import Program
from qgpu.runtime import Device, GPUError, Prepared, host_tokens, prepare, read

CALC_TOOL = {"type": "function", "function": {
    "name": "calc",
    "description": "Evaluate an arithmetic expression exactly, on the GPU inside your weights, "
                   "with 32-bit integers and Python's rules: + - * // % and parentheses, / to "
                   "two decimals, lists [1, 2, 3] element by element (a number applies to every "
                   "element), sum, mean, max, min, cumsum (running totals), dot, A @ B for the "
                   "matrix product of square matrices like [[1, 2], [3, 4]], count (how often "
                   "each value 0..15 occurs). "
                   "Separate several expressions with commas. Returns the exact value.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string",
                       "description": "e.g. 4*30 + 3*55 or [2, 3, 4] * [5, 6, 7] - 1 or "
                                      "cumsum([10, 20, 30])"}},
        "required": ["expression"]}}}

CALC = ("You have a calc tool for exact arithmetic. Use it for every arithmetic step instead "
        "of calculating by hand, and answer from its results. If a result does not answer the "
        "question, call calc again with a corrected expression; never redo the arithmetic "
        "yourself.")

TOOL = {"type": "function", "function": {
    "name": "cuda",
    "description": "Compile and run one CUDA kernel on the GPU inside your weights, exactly. "
                   "Returns the result arrays the kernel writes.",
    "parameters": {"type": "object", "properties": {
        "source": {"type": "string", "description": "the CUDA C source of one __global__ kernel"},
        "args": {"type": "object", "description": "one object by parameter name: each input "
                 "array (const int *) as a list of its integers, n as a number; output arrays "
                 "(int *) are left out"}},
        "required": ["source", "args"]}}}

@dataclass(frozen=True)
class Mode:
    """A mode: Qwen's system prompt, its one tool, and the Session method that runs a call."""
    system: str
    tool: dict
    run: str
    lead: str = ""  # how the first parameter's value starts, written for Qwen to continue

    @property
    def call_start(self) -> str:
        """A call's opening into its first parameter's value (a required tool choice)."""
        f = self.tool["function"]
        return (f"<tool_call>\n<function={f['name']}>\n"
                f"<parameter={f['parameters']['required'][0]}>\n{self.lead}")


MODES = {"calc": Mode(CALC, CALC_TOOL, "_calc"), "cuda": Mode(KERNEL_RULES, TOOL, "_cuda")}

# Qwen's text and code tokens are sampled with the model card's settings for precise coding
# (Qwen3.5: temperature 0.6, top_p 0.95, top_k 20, no penalties; greedy decoding makes it
# loop). Chip tokens are always the argmax: the circuit computes exactly one next token.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def parse_call(text: str) -> tuple[str, dict]:
    """The last tool call in Qwen's text: (function name, {parameter: value})."""
    calls = re.findall(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
                       text, re.DOTALL)
    if not calls:
        raise GPUError("the tool call was not complete: write <tool_call><function=cuda>"
                       "<parameter=source>...</parameter><parameter=args>...</parameter>"
                       "</function></tool_call>")
    name, body = calls[-1]
    params = {}
    for p, v in re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", body, re.DOTALL):
        params[p] = v.strip()
    return name, params


def _json(text, what: str):
    try:
        return json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError:
        raise GPUError(f"{what} must be JSON: {text!r}") from None


ROUTER = "Answer with one word: yes or no."
ROUTE = "Does this message ask to calculate something with numbers?\n\nMessage: {message}"


@dataclass
class Event:
    """What Session.send reports, in order:
    think / text (text: a decoded piece of Qwen's output),
    call (source, args, threads), launch (kernel, inputs, blocks, warps, host_tokens, program:
    its shape for qgpu.monitor), token (t: each chip token the model writes), result (kernel,
    outputs, tokens, seconds, verified: True / False / None), value (value: the arrays
    returned), refused (error, repeat), done (stats)."""
    kind: str
    data: dict


@dataclass
class Ctx:
    """One model context: its tokens, the tail not yet run, its cache, its launches."""
    ids: list[int] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)
    cache: object = None
    launches: int = 0


class Cancelled(Exception):
    pass


class Session:
    def __init__(self, merged: str | Path, device: str | None = None, think: bool = False,
                 check: bool = False, max_calls: int = 4, max_turn: int = 1024,
                 seed: int = 0, mode: str = "calc"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        merged = Path(merged)
        self.meta = json.loads((merged / "qgpu.json").read_text())
        self.name = self.meta.get("base", merged.name)  # the Qwen it was built from
        self.v0 = self.meta["chip_token_offset"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        if self.dev == "cuda":  # exact bf16: no reduced-precision reductions, no TF32
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(merged)
        self.model = AutoModelForImageTextToText.from_pretrained(
            merged, dtype=torch.bfloat16).to(self.dev).eval()
        self.gpu = Device(self.meta, self.model.config.text_config.to_dict())
        self.think, self.check = think, check
        self.max_calls, self.max_turn, self.mode = max_calls, max_turn, mode
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        tid = self.tok.convert_tokens_to_ids
        self.im_end, self.call_end = tid("<|im_end|>"), tid("</tool_call>")
        self.think_end = tid("</think>")
        self.cancel = False
        self.reset()

    def reset(self) -> None:
        """A new conversation."""
        self.chat = Ctx()  # the conversation with the user
        self.ctx = self.chat  # the context the model is working in now
        self.stats = {"launches": 0, "chip_tokens": 0, "launch_seconds": 0.0, "verified": 0}

    # --- the model ------------------------------------------------------------------------

    def _enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _push(self, toks: list[int]) -> None:
        self.ctx.ids += toks
        self.ctx.pending += toks

    def _run(self):
        """Run the pending tokens through the model; the logits after the last one."""
        c = self.ctx
        x = self.torch.tensor([c.pending], device=self.dev)
        with self.torch.no_grad():
            out = self.model(input_ids=x, past_key_values=c.cache, use_cache=True,
                             logits_to_keep=1)
        c.cache, c.pending = out.past_key_values, []
        return out.logits[0, -1]

    def _turn(self) -> str:
        """An assistant turn's start, as Qwen3.5's chat template writes it."""
        return "<|im_start|>assistant\n" + ("<think>\n" if self.think else
                                            "<think>\n\n</think>\n\n")

    def _prompt(self, system: str, user: str, tools=None) -> Ctx:
        """A fresh context: a system prompt and one user message, at Qwen's turn."""
        self.ctx = Ctx()
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=tools, add_generation_prompt=False, tokenize=False)
        self._push(self._enc(text + self._turn()))
        return self.ctx

    # --- the three passes -----------------------------------------------------------------

    def send(self, user: str) -> Iterator[Event]:
        """One user message, through the GPU, to Qwen's answer (in this session's mode)."""
        self.cancel = False
        yield from self._chat(user, MODES[self.mode])
        yield Event("done", dict(self.stats))

    def _chat(self, user: str, mode: Mode):
        """One context: Qwen calls the mode's tool as often as it needs, then answers. When
        the router says the message asks for a calculation, Qwen's first turn starts inside
        the call."""
        # with thinking on, the turn starts inside <think>: the call cannot be forced there
        forced = mode.call_start if not self.think and self._needs_gpu(user) else ""
        self.ctx = self.chat
        if not self.chat.ids:
            self._push(self._enc(self.tok.apply_chat_template(
                [{"role": "system", "content": mode.system}, {"role": "user", "content": user}],
                tools=[mode.tool], add_generation_prompt=False, tokenize=False)
                + self._turn()))
        else:
            close = "" if self.chat.ids[-1] == self.im_end else "<|im_end|>"
            self._push(self._enc(f"{close}\n<|im_start|>user\n{user}<|im_end|>\n"
                                 + self._turn()))
        self._push(self._enc(forced))
        if forced:  # shown like a call Qwen wrote itself (displays cut the text at <tool_call>)
            yield Event("text", {"text": forced})
        refused = {}  # call -> the error it got
        for calls in range(self.max_calls + 1):
            said, ended = yield from self._assistant()
            said, forced = forced + said.split("</think>")[-1], ""
            if not ended or "<tool_call>" not in said:  # an answer
                break
            call = "<tool_call>" + said.split("<tool_call>", 1)[1]
            try:
                sig = json.dumps(parse_call(call))
            except GPUError:
                sig = call[-200:]
            if calls == self.max_calls:
                result = {"error": f"at most {self.max_calls} calls per message; answer now"}
                yield Event("refused", {"error": result["error"]})
            elif sig in refused:  # not run again: Qwen is told, and it counts toward the cap
                result = {"error": f"this is the same call that was just refused "
                                   f"({refused[sig]}): change it"}
                yield Event("refused", {"error": "Qwen repeated a refused call: told to change "
                                                 "it", "repeat": True})
            else:
                result = yield from getattr(self, mode.run)(call)
                if "error" in result:
                    refused[sig] = result["error"]
            close = "" if self.ctx.ids[-1] == self.im_end else "<|im_end|>"
            self._push(self._enc(f"{close}\n<|im_start|>user\n<tool_response>\n"
                                 f"{json.dumps(result)}\n</tool_response><|im_end|>\n"
                                 + self._turn()))

    def _calc(self, call: str):
        """One calc call: every operation a library kernel launched on the GPU."""
        try:
            name, params = parse_call(call)
            if name != "calc" or "expression" not in params:
                raise GPUError("call calc with one parameter, expression")
            yield Event("call", {"expression": params["expression"]})
            m = self.meta

            def launch(kern, data, n):
                out = yield from self._launch(prepare(kern, data, n, m["lanes"], m["stages"]))
                return out

            value = yield from evaluate(params["expression"], launch)
        except GPUError as e:
            yield Event("refused", {"error": str(e)})
            return {"error": str(e)}
        yield Event("value", {"value": value})
        return {"result": value}

    def _needs_gpu(self, user: str) -> bool:
        """The router: does the message ask for a calculation? Qwen's first token, yes or no
        (read from the logits: no sampling)."""
        self._prompt(ROUTER, ROUTE.format(message=user))
        logits = self._run()
        tid = self.tok.convert_tokens_to_ids
        score = {w: max(float(logits[t]) for t in (tid(w), tid(w.capitalize())))
                 for w in ("yes", "no")}
        return score["yes"] > score["no"]

    def _assistant(self):
        """Stream one assistant turn in the current context as think / text events. Returns
        (its text, whether it ended on a stop token)."""
        thinking = self.think
        seg, shown, events = [], "", []

        def each(t):
            nonlocal thinking, seg, shown
            if t == self.think_end and thinking:
                thinking, seg, shown = False, [], ""
                return
            if t in (self.im_end, self.call_end):
                return
            seg.append(t)
            text = self.tok.decode(seg)
            if text.endswith("�"):  # half a character: wait for the rest
                return
            events.append(Event("think" if thinking else "text", {"text": text[len(shown):]}))
            shown = text

        stop = {self.im_end, self.call_end}
        new, start = [], len(self.ctx.ids)
        logits = self._run()
        while True:  # the stop token stays pending (the next push follows it)
            if self.cancel:
                raise Cancelled
            t = self._sample(logits)
            new.append(t)
            self._push([t])
            each(t)
            yield from events
            events.clear()
            if t in stop or len(new) >= self.max_turn:
                break
            logits = self._run()
        said = self.tok.decode(self.ctx.ids[start:]).removesuffix("<|im_end|>")
        return said, new[-1] in stop

    def _sample(self, logits) -> int:
        """A text token, with SAMPLING (top_k, then top_p, at the temperature)."""
        torch = self.torch
        top, idx = torch.topk(logits.float().cpu(), SAMPLING["top_k"])
        p = torch.softmax(top / SAMPLING["temperature"], -1)
        keep = torch.cumsum(p, -1) - p < SAMPLING["top_p"]  # the smallest set over top_p
        p = p * keep
        return int(idx[torch.multinomial(p / p.sum(), 1, generator=self.rng)])

    # --- the GPU --------------------------------------------------------------------------

    def _cuda(self, call: str):
        """One cuda call: compile, lay out and run Qwen's kernel; the tool response object."""
        try:
            name, params = parse_call(call)
            if name != "cuda" or "source" not in params:
                raise GPUError("call cuda with source and args")
            source = params["source"]
            args, threads = (_json(params.get(k), k) for k in ("args", "threads"))
            yield Event("call", {"source": source, "args": args, "threads": threads})
            m = self.meta
            p = plan(compile_cuda(source), args, threads, m["lanes"], m["stages"])
            out = yield from self._launch(p)
        except GPUError as e:
            yield Event("refused", {"error": str(e)})
            return {"error": str(e)}
        yield Event("value", {"value": out})
        return out

    def _launch(self, p: Prepared):
        """One kernel launch in the current context, written by the model; its outputs."""
        name, data = p.name, p.inputs
        lw = p.launch
        host = host_tokens(lw, clear=self.ctx.launches > 0)
        yield Event("launch", {"kernel": name, "inputs": data, "blocks": len(lw.blocks),
                               "warps": lw.warps, "host_tokens": len(host),
                               "program": Program.of(lw, self.meta["lanes"])})
        end = {self.v0 + EXEC(0, lw.warps, ph) for ph in (False, True)}
        self._push([self.v0 + t for t in host])
        t0, new = time.time(), []
        logits = self._run()
        while True:  # the model writes the launch
            if self.cancel:
                raise Cancelled
            t = int(logits.argmax())
            self._push([t])
            new.append(t)
            yield Event("token", {"t": t - self.v0})
            if t in end or not self.v0 <= t < self.v0 + kernel.CHIP_VOCAB or len(new) > 200000:
                break
            logits = self._run()
        self.ctx.launches += 1
        dt = time.time() - t0
        if new[-1] not in end:
            raise GPUError(f"the launch of {name} did not end cleanly after {len(new)} tokens")
        out = read(p, self.gpu.read(self.ctx.cache))  # the end token is pending, rest is in
        verified = None
        if self.check:
            tr = kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=self.meta["lanes"])
            want = [t + self.v0 for t in tr.tokens[len(host_tokens(lw)):]]
            verified = new == want and out == read(p, tr.mem)
        self.stats["launches"] += 1
        self.stats["chip_tokens"] += len(new)
        self.stats["launch_seconds"] += dt
        self.stats["verified"] += bool(verified)
        yield Event("result", {"kernel": name, "outputs": out, "tokens": len(new),
                               "seconds": dt, "verified": verified})
        return out


def main() -> None:
    """A plain-text chat: every message given, then (without messages) an input loop."""
    import argparse

    ap = argparse.ArgumentParser(description="Chat with Qwen; it writes CUDA and runs it on the "
                                             "GPU in its weights")
    ap.add_argument("merged")
    ap.add_argument("message", nargs="*")
    ap.add_argument("--check", action="store_true", help="compare launches with the reference")
    ap.add_argument("--think", action="store_true", help="think before answering")
    ap.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
    ap.add_argument("--new-chat", action="store_true", help="a fresh context per message")
    ap.add_argument("--mode", choices=("calc", "cuda"), default="calc",
                    help="calc: Qwen writes expressions for the GPU's kernel library (default); "
                         "cuda: Qwen writes the CUDA kernels")
    a = ap.parse_args()
    s = Session(a.merged, a.device, think=a.think, check=a.check, mode=a.mode)

    def chat(msg):
        print(f"user: {msg}", flush=True)
        for e in s.send(msg):
            d = e.data
            if e.kind in ("think", "text"):
                print(d["text"], end="", flush=True)
            elif e.kind == "call" and "expression" in d:
                print(f"\n[calc {d['expression']}]", flush=True)
            elif e.kind == "call":
                print(f"\n[cuda, args {json.dumps(d['args'])}]\n{d['source']}", flush=True)
            elif e.kind == "launch":
                print(f"  [{d['kernel']} {json.dumps(d['inputs'])}: {d['blocks']} blocks, "
                      f"{d['warps']} warps]", end="", flush=True)
            elif e.kind == "result":
                v = {True: ", verified", False: ", DIFFERS", None: ""}[d["verified"]]
                print(f" -> {json.dumps(d['outputs'])} ({d['tokens']} tokens, "
                      f"{d['seconds']:.0f}s{v})", flush=True)
            elif e.kind == "value":
                print(f"[= {json.dumps(d['value'])}]", flush=True)
            elif e.kind == "refused":
                print(f"\n[refused: {d['error']}]", flush=True)
            elif e.kind == "done":
                print(f"\n[{d['launches']} launches]\n", flush=True)

    for m in a.message:
        if a.new_chat:
            s.reset()
        chat(m)
    if not a.message:
        while True:
            try:
                chat(input("> "))
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
