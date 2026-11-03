"""What computes a QwenGPU's tokens. Three backends, one token stream:

- RefBackend: the reference interpreter (circuit/kernel.records) — the stream the circuit
  must write, fast enough to run a whole cluster in a thread each;
- CircuitBackend: the compiled GPU circuit, evaluated exactly in float64 (chipformer's IR
  evaluator) — the same weights the merged model carries, without Qwen around them;
- ModelBackend: a merged Qwen3.5 checkpoint (qgpu.merge) in bf16 on a CUDA device.

Every backend keeps a context: `feed(tokens)` appends to it (the host's records, packets that
arrived from a peer, or the backend's own predictions), `predict(k)` returns its next-token
predictions (argmax) at the last k positions of the context, running whatever is pending.
`memory()` reads global memory back from the backend's state (cudaMemcpy device -> host).
"""

import json
from pathlib import Path

from qgpu.circuit import kernel
from qgpu.circuit.kernel import EXEC0, LOAD0, MEM0, kind


def is_exec(t: int) -> bool:
    return EXEC0 <= t < LOAD0


class RefBackend:
    """The reference interpreter as a backend. It reads the program from `begin` (the circuit
    reads it from the PROG / LIVE records; the reference keeps the machine blocks) and global
    memory from every MEM record it is fed, as the circuit's memory heads do: the host's, a
    peer's, and its own write-backs (whose address the device may have redirected)."""

    name = "ref"

    def __init__(self, lanes: int = 4):
        self.lanes = lanes
        self.mem: dict[int, int] = {}
        self.regfile: dict = {}  # per-warp register rows; they persist, as the model's do
        self.gen = None  # the running launch (kernel.records)
        self.expect: list[int] = []  # what the circuit will write next, in order
        self._mem: tuple | None = None  # a MEM record being fed: (address, [payload])
        self.n_fed = 0

    def reset(self) -> None:
        """A fresh context (memory is rewritten by the caller)."""
        self.__init__(self.lanes)

    def begin(self, launch) -> None:
        self.gen = kernel.records(launch.blocks, self.mem, launch.warps, self.lanes,
                                  apply=False, regfile=self.regfile)
        part, toks = next(self.gen)
        assert part == "exec"
        self.expect = toks  # the launch's first EXEC token, which the host feeds

    def feed(self, toks: list[int]) -> None:
        for t in toks:
            self.n_fed += 1
            k = kind(t)
            if k == "mem":
                self._mem = (t - MEM0, [])
            elif k == "val" and self._mem is not None:
                self._mem[1].append(t)
                if len(self._mem[1]) == 2:
                    a, (lo, hi) = self._mem
                    self.mem[a] = lo | hi << 16
                    self._mem = None
            else:
                self._mem = None
            if self.expect and (t == self.expect[0] or (k == "mem" and kind(self.expect[0]) ==
                                                        "mem")):
                self.expect.pop(0)
                if is_exec(t) and self.gen is not None:
                    part, body = next(self.gen)  # computed now, from memory as it is now
                    assert part == "body"
                    self.expect = body

    def predict(self, k: int = 1) -> list[int]:
        assert k == 1, "the reference predicts one token at a time"
        if not self.expect and self.gen is not None:
            part, toks = next(self.gen)  # control flow only: the next record's EXEC
            assert part in ("exec", "end")
            self.expect = toks
            if part == "end":  # the launch is over once its end token is fed
                self.gen = None
        assert self.expect, "nothing to predict: no launch is running"
        return [self.expect[0]]

    def memory(self) -> dict[int, int]:
        return dict(self.mem)


class CircuitBackend:
    """The GPU circuit's compiled weights, evaluated exactly (chipformer.q35.ir.evaluate),
    streaming: head states and conv windows carry from one chunk to the next."""

    name = "circuit"

    def __init__(self, circuit=None, lanes: int = 4, stages: int = 4):
        from qgpu.merge import STAGES

        if circuit is None:
            mul, logic, atom = STAGES[stages]
            circuit = kernel.Kernel(lanes=lanes, stages=stages, mul_stages=mul,
                                    logic_stages=logic, atom_stages=atom)
        self.c = circuit
        self.carry: dict = {}
        self.pending: list[int] = []
        self.n_fed = 0

    def begin(self, launch) -> None:
        pass  # the program arrives as PROG / LIVE records

    def reset(self) -> None:
        self.carry, self.pending, self.n_fed = {}, [], 0

    def feed(self, toks: list[int]) -> None:
        self.pending += toks
        self.n_fed += len(toks)

    def predict(self, k: int = 1) -> list[int]:
        from chipformer.q35.ir import evaluate

        assert len(self.pending) >= k
        m = evaluate(self.c, self.pending, carry=self.carry)
        n = len(self.pending)
        self.pending = []
        out = [kernel.predicted(self.c, m, t) for t in range(n - k, n)]
        assert None not in out, "the circuit predicts nothing at a host position"
        return out

    def flush(self) -> None:
        from chipformer.q35.ir import evaluate

        if self.pending:
            evaluate(self.c, self.pending, carry=self.carry)
            self.pending = []

    def check(self, expect: list[tuple[int, int]]) -> tuple[int, tuple | None]:
        """As ModelBackend.check: evaluate everything pending, compare predictions."""
        from chipformer.q35.ir import evaluate

        if not self.pending:
            assert not expect
            return 0, None
        start = self.n_fed - len(self.pending)
        m = evaluate(self.c, self.pending, carry=self.carry)
        self.pending = []
        for p, want in expect:
            got = kernel.predicted(self.c, m, p - start)
            if got != want:
                return len(expect), (p, got, want)
        return len(expect), None

    def memory(self) -> dict[int, int]:
        self.flush()
        return kernel.memory(self.c, self.carry)


class ModelBackend:
    """A merged Qwen3.5 checkpoint on one CUDA device. Every prediction is the model's own
    argmax over its whole vocabulary (text tokens included): the circuit's layers put the GPU's
    next token on top, and Qwen's layers, muted on chip tokens, leave it there."""

    name = "model"

    def __init__(self, merged: str | Path, device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText

        from qgpu.runtime import Device

        merged = Path(merged)
        self.meta = json.loads((merged / "qgpu.json").read_text())
        self.v0 = self.meta["chip_token_offset"]
        if device.startswith("cuda"):  # exact bf16: no reduced-precision reductions, no TF32
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.torch, self.dev = torch, device
        self.model = AutoModelForImageTextToText.from_pretrained(
            merged, dtype=torch.bfloat16).to(device).eval()
        self.gpu = Device(self.meta, self.model.config.text_config.to_dict())
        # one throwaway pass (no cache) loads torch's lazily initialised linalg libraries now:
        # their lazy wrappers are not thread-safe, and ranks may run as threads of one process
        with torch.no_grad():
            self.model(input_ids=torch.tensor([[self.v0 + kernel.EXEC0] * 3], device=device),
                       use_cache=False, logits_to_keep=1)
        self.cache = None
        self.pending: list[int] = []
        self.n_fed = 0
        self.forwards = 0  # forward passes run
        self.positions = 0  # token positions they processed

    def begin(self, launch) -> None:
        pass

    def reset(self) -> None:
        """A fresh context: no cache (memory is rewritten by the caller)."""
        self.cache, self.pending, self.n_fed, self.positions = None, [], 0, 0
        self.torch.cuda.empty_cache()

    def feed(self, toks: list[int]) -> None:
        self.pending += toks
        self.n_fed += len(toks)

    def _forward(self, k: int):
        torch = self.torch
        x = torch.tensor([[self.v0 + t for t in self.pending]], device=self.dev)
        with torch.no_grad():
            out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True,
                             logits_to_keep=k)
        self.cache = out.past_key_values
        self.forwards += 1
        self.positions += len(self.pending)
        self.pending = []
        return out.logits[0]

    def predict(self, k: int = 1) -> list[int]:
        assert len(self.pending) >= k
        ids = self._forward(k).argmax(-1).tolist()
        out = [i - self.v0 for i in ids]
        bad = [i for i in out if not 0 <= i < kernel.CHIP_VOCAB]
        if bad:
            raise RuntimeError(f"the model wrote a text token inside a launch: {bad[0]}")
        return out

    def flush(self) -> None:
        if self.pending:
            self._forward(1)

    def check(self, expect: list[tuple[int, int]]) -> tuple[int, tuple | None]:
        """Run everything pending in one forward pass (teacher forcing) and compare the model's
        argmax with `expect`: (absolute position, token the model must predict there). Returns
        (positions checked, first mismatch as (position, got, want) or None)."""
        torch = self.torch
        if not expect:  # only host or peer tokens since the last check
            self.flush()
            return 0, None
        start = self.positions
        idx = sorted({p - start for p, _ in expect})
        assert all(0 <= i < len(self.pending) for i in idx), "a check outside the pending span"
        x = torch.tensor([[self.v0 + t for t in self.pending]], device=self.dev)
        keep = torch.tensor(idx, dtype=torch.long, device=self.dev)
        with torch.no_grad():
            out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True,
                             logits_to_keep=keep)
        self.cache = out.past_key_values
        self.forwards += 1
        self.positions += len(self.pending)
        self.pending = []
        got = dict(zip(idx, (out.logits[0].argmax(-1) - self.v0).tolist()))
        for p, want in expect:
            if got[p - start] != want:
                return len(expect), (p, got[p - start], want)
        return len(expect), None

    def memory(self) -> dict[int, int]:
        self.flush()
        return self.gpu.read(self.cache)
