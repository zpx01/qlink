"""Merge the GPU circuit into a stock Qwen3.5 checkpoint ("Qwen runs CUDA").

Follows Qwenputer's `chipformer.q35.merge.build_merged` (v2 layout), with our circuit and our chip
tokens: the circuit's layers go in front of Qwen's, the residual is widened (Qwen's matrices are
zero-padded and its wide norm gains rescaled, so its text computation is unchanged), chip tokens
are appended to the vocabulary, and Qwen's layers are muted on chip tokens (mute.py). Every Qwen
text token carries the circuit features of kernel.TEXT; the circuit's exit clears every chip dim
but the output code, tau and mu, so Qwen's layers see their usual input on text.

The host plays NVIDIA's driver: on a tool call it lowers the PTX (qgpu.lower) and feeds the
program image, memory and the launch's first EXEC token; the model writes the launch.

With the overlay (qgpu/overlay.py, on by default), circuit dims that do not matter on text tokens
live in Qwen's dims, which makes the checkpoint much narrower.

Run: uv run python -m qgpu.merge SRC DST [--lanes 4] [--stages 4] [--no-mute] [--no-overlay]
"""

import json
import re
import shutil
from pathlib import Path

import torch

from qgpu.circuit import kernel
from qgpu.circuit.kernel import (
    ABITS,
    CHIP_VOCAB,
    EXEC0,
    LIVE0,
    LOAD0,
    MEM0,
    PROG0,
    PROGI0,
    TEXT,
    Kernel,
)

STAGES = {6: ({4}, {2}, {5}), 4: ({2}, {1}, {3}), 3: ({1}, {0}, {2})}
TEXT_PREFIX = "model.language_model."


def token_name(t: int) -> str:
    """The tokenizer string of chip token t."""
    if t < EXEC0:
        return f"<|gpu={t:04x}|>"
    if t < LOAD0:
        x = t - EXEC0
        b, w = x & (1 << kernel.BBITS) - 1, x >> kernel.BBITS & (1 << kernel.WBITS) - 1
        phase = x >> (kernel.BBITS + kernel.WBITS) & 1
        return f"<|gpu_exec {b} {w}{' phase' if phase else ''}|>"
    if t < PROG0:
        return f"<|gpu_load {t - LOAD0}|>"
    if t < PROGI0:
        return f"<|gpu_prog {(t - PROG0) // 8} {(t - PROG0) % 8}|>"
    if t < LIVE0:
        return f"<|gpu_progi {(t - PROGI0) // 8} {(t - PROGI0) % 8}|>"
    if t < MEM0:
        return f"<|gpu_live {t - LIVE0}|>"
    assert t < MEM0 + (1 << ABITS)
    return f"<|gpu_mem {t - MEM0}|>"


def extend_tokenizer(src: Path, dst: Path, first_id: int) -> None:
    """Copy the tokenizer, padding with reserved tokens up to first_id, then the chip tokens."""
    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    names = [token_name(t) for t in range(CHIP_VOCAB)]
    assert len(set(names)) == len(names)
    pad = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first_id - n)]
    chip = [AddedToken(s, special=True) for s in names]
    tok.add_special_tokens(pad + chip)
    assert tok.token_to_id(names[0]) == first_id
    dst.mkdir(parents=True, exist_ok=True)
    tok.save(str(dst / "tokenizer.json"))
    for f in ("chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "generation_config.json", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    dec = tc.setdefault("added_tokens_decoder", {})
    for i, t in enumerate(pad + chip):
        dec[str(n + i)] = {"content": t.content, "lstrip": False, "normalized": False,
                           "rstrip": False, "single_word": False, "special": True}
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False),
                                               encoding="utf-8")


def build_merged(src, dst, lanes: int = 4, stages: int = 4, mute_qwen: bool = True,
                 overlay: bool = True):
    """Write Qwen3.5 + the GPU circuit as one checkpoint; returns (circuit, layout)."""
    from chipformer.q35 import merge as qm
    from chipformer.q35 import tiny
    from chipformer.q35.emit import Layout, allocate, check_bf16, embedding_rows, emit_layers
    from chipformer.q35.ir import Target
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    cfg, sd = qm.load(src)
    tc = cfg["text_config"]
    d0, v0, inter = tc["hidden_size"], tc["vocab_size"], tc["intermediate_size"]
    mul, logic, atom = STAGES[stages]
    c = Kernel(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"], inter),
               lanes=lanes, stages=stages, mul_stages=mul, logic_stages=logic, atom_stages=atom)
    keep = set(c.keep)
    plan = None
    if overlay:
        from qgpu import overlay as qov

        plan = qov.build(c, keep, {"stages": stages, "mul_stages": mul, "logic_stages": logic,
                                   "atom_stages": atom})
        print(plan.summary(), flush=True)
    alloc = allocate(c, None, keep, inter, overlay=plan.dims if plan else None, overlay_cap=d0)
    hidden, m = qm.pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots, over=alloc.over)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    print(f"circuit: {n_chip} layers, {alloc.n_slots} slots; hidden {d0} -> {hidden}", flush=True)

    cfg, sd = qm.widen(cfg, sd, hidden)
    tc = cfg["text_config"]
    n0 = tc["num_hidden_layers"]
    shift = re.compile(r"^(model\.language_model\.layers\.)(\d+)\.")
    sd = {shift.sub(lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k): v for k, v in sd.items()}
    tc["layer_types"] = ["linear_attention"] * n_chip + tc["layer_types"]
    tc["num_hidden_layers"] = n_chip + n0
    vocab = -(-(v0 + CHIP_VOCAB) // 64) * 64
    tc["vocab_size"] = vocab
    mu, tau = c.keep[-1], c.keep[-2]
    if mute_qwen:
        from chipformer.q35.mute import fit_anti_sink, mute

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mute(sd, tc, n0, lay.dim(mu), hidden**0.5, sinks=fit_anti_sink(src, dev), first=n_chip)
    layers = emit_layers(c, lay, tc, 0, TEXT_PREFIX, alloc, keep, chip=c.mu)
    sd.update({k: v.to(torch.bfloat16) for k, v in layers.items()})
    del layers

    # embeddings: Qwen's rows keep their Qwen part and get the TEXT token's chip features
    full = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    full[:v0, :d0] = sd[TEXT_PREFIX + "embed_tokens.weight"][:, :d0]
    text_row = embedding_rows(c, lay, [TEXT], hidden)[0].to(torch.bfloat16)
    full[:v0, d0:] = text_row[d0:]
    full[v0:v0 + CHIP_VOCAB] = embedding_rows(c, lay, list(range(CHIP_VOCAB)),
                                              hidden).to(torch.bfloat16)
    sd[TEXT_PREFIX + "embed_tokens.weight"] = full
    bias = "model.visual.merger.linear_fc2.bias"  # image tokens are text to the circuit too
    if bias in sd:
        sd[bias] = sd[bias].clone()
        sd[bias][d0:] = text_row[d0:].to(sd[bias].dtype)

    norm = sd[TEXT_PREFIX + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims are invisible to the lm_head ...
    for e in c.code:  # ... except the output code and tau
        norm[lay.dim(next(iter(e.terms)))] = tiny.code_gain(lay)
    norm[lay.dim(tau)] = qm.TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[TEXT_PREFIX + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    shards, shard, size = [], {}, 0
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * 2
        if size > 2**31:
            shards.append(shard)
            shard, size = {}, 0
    shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())},
             "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    extend_tokenizer(src, dst, v0)
    meta = {"base": src.name, "chip_token_offset": v0, "chip_vocab": CHIP_VOCAB,
            "chip_layers": n_chip, "qwen_layers_from": n_chip,
            "qwen_layers": n0,
            "hidden": hidden, "chip_dims_offset": d0,
            "ballast_dims": m, "slots": alloc.n_slots, "lanes": lanes, "stages": stages,
            "muted": mute_qwen, "overlay": plan.summary() if plan else None,
            "overlay_slots": len(alloc.final_over)}
    (dst / "qgpu.json").write_text(json.dumps(meta, indent=2))
    return c, lay


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge the GPU circuit into a Qwen3.5 checkpoint")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--no-mute", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()
    build_merged(a.src, a.dst, a.lanes, a.stages, mute_qwen=not a.no_mute,
                 overlay=not a.no_overlay)


if __name__ == "__main__":
    main()
