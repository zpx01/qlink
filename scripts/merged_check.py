"""Check a merged checkpoint (qgpu.merge): Qwen's text is unchanged and the model runs a kernel.

1. Text: the merged model's next-token distributions on plain prompts against the base model's
   (KL, and how often the top token agrees).
2. A launch inside text: text, then a vadd launch (host records + the launch, as chip tokens),
   then text again, in one teacher-forced pass: at every position the circuit owns, the merged
   model's own argmax must be the next chip token.
3. Generation: from the host's part of that sequence, greedy `generate` must write the launch.

Run: uv run python scripts/merged_check.py BASE MERGED [vadd histogram divloop reduce warpsum]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from qgpu import ptx
from qgpu.circuit import kernel
from qgpu.lower import lower

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "CUDA kernels run on thousands of threads grouped into warps of",
    "Photosynthesis converts light energy into",
]


def logits(model, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        dev = next(model.parameters()).device
        return model(input_ids=torch.tensor([ids], device=dev)).logits[0].float().cpu()


def launch(model, tok, v0, lanes, cfg, name, path, grid, block, args, bufs) -> bool:
    """One launch between two text spans: teacher-forced argmax at every owned position, then
    greedy generation from the host's part."""
    k = ptx.load(path)[name]
    lw = lower(k, grid, block, args, bufs, lanes=lanes, **cfg)
    tr = kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=lanes)
    before = tok.encode(f"Running {name} on the GPU inside the model:")
    after = tok.encode(" done.")
    seq = before + [v0 + t for t in tr.tokens] + after
    owned = [len(before) + t for t in tr.owned if t + 1 < len(tr.tokens)]
    t0 = time.time()
    top = logits(model, seq).argmax(-1).tolist()
    bad = [t for t in owned if top[t] != seq[t + 1]]
    first = len(before) + tr.owned[0]
    want = seq[first + 1:len(before) + len(tr.tokens)]
    dev = next(model.parameters()).device
    ids = torch.tensor([seq[:first + 1]], device=dev)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=len(want),
                             do_sample=False, eos_token_id=want[-1], pad_token_id=want[-1])
    got = out[0, ids.shape[1]:].tolist()
    ok = not bad and got == want
    print(f"{name} ({path}): {len(tr.tokens)} chip tokens between text; teacher-forced argmax "
          f"right at {len(owned) - len(bad)}/{len(owned)} owned positions ({t1 - t0:.0f}s); "
          f"generate wrote {len(got)}/{len(want)} tokens in {time.time() - t1:.0f}s: "
          f"{'identical to the reference' if got == want else 'DIFFERENT'}", flush=True)
    return ok


def main() -> None:
    base_dir, merged_dir = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or ["vadd"]
    dev = "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"  # MPS is exact too
    if dev == "cuda":  # exact bf16: no reduced-precision reductions, no TF32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    meta = json.loads((merged_dir / "qgpu.json").read_text())
    v0, lanes, stages = meta["chip_token_offset"], meta["lanes"], meta["stages"]
    tok = AutoTokenizer.from_pretrained(base_dir)
    prompts = [tok.encode(p) for p in PROMPTS]

    base = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16)
    base = base.to(dev).eval()
    want = [logits(base, ids)[:, :v0] for ids in prompts]
    del base

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(merged_dir, dtype=torch.bfloat16)
    model = model.to(dev).eval()
    print(f"merged: {model.config.text_config.num_hidden_layers} layers, hidden "
          f"{model.config.text_config.hidden_size}, loaded on {dev} in {time.time() - t0:.0f}s",
          flush=True)
    kls, agree, n = [], 0, 0
    for ids, w in zip(prompts, want):
        g = logits(model, ids)
        assert (g[:, v0:].max(1).values < g[:, :v0].max(1).values).all(), "a chip token in text"
        p, q = torch.log_softmax(w, -1), torch.log_softmax(g[:, :v0], -1)
        kls.append((p.exp() * (p - q)).sum(-1))
        agree += int((w.argmax(-1) == g[:, :v0].argmax(-1)).sum())
        n += len(ids)
    kl = torch.cat(kls)
    print(f"text: KL(base || merged) mean {kl.mean():.2e}, max {kl.max():.2e}; top token agrees "
          f"at {agree}/{n} positions", flush=True)

    cfg = {"stages": stages, "mul_stages": {stages - 2}, "logic_stages": {stages - 3},
           "atom_stages": {stages - 1}}
    rng = np.random.default_rng(1)
    x = [int(v) for v in rng.integers(0, 2**32, 12)]
    small = [int(v) for v in rng.integers(0, 16, 8)]
    launches = {
        "vadd": ("kernels/vadd.ptx", (1,), (8,), [0, 1, 2, 6], {0: x[:6], 1: x[6:], 2: [0] * 6}),
        "histogram": ("kernels/nvcc/histogram.ptx", (1,), (8,), [0, 1, 8],
                      {0: x[:8], 1: [0] * 16}),
        "divloop": ("kernels/divloop.ptx", (1,), (8,), [0, 1, 6], {0: small[:6], 1: [0] * 6}),
        "reduce": ("kernels/nvcc/reduce.ptx", (1,), (8,), [0, 1, 8], {0: x[:8], 1: [7]}),
        "warpsum": ("kernels/warpsum.ptx", (1,), (32,), [0, 1, 12], {0: x, 1: [0]}),
    }
    results = [launch(model, tok, v0, lanes, cfg, name, *launches[name]) for name in names]
    assert all(results)


if __name__ == "__main__":
    main()
