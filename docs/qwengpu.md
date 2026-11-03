> This is qwenGPU's README, kept from [vixhal-baraiya/qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) at commit `e2695f6`, which QLink starts from. Paths and commands refer to the qwenGPU parts of this repository.

# qgpu: Qwen runs CUDA

**qgpu is an experimental model: Qwen3.5-2B with a SIMT GPU compiled and inserted directly into
its weights.** It is not a plugin, a tool or an emulator. The GPU is 45 transformer layers in
front of Qwen's own 24, and none of their weights are learned: a compiler computes every one.
It runs real CUDA, compiled to NVIDIA PTX, and the model itself writes every step of each
launch.

A real launch, token by token: see the demo in the [qwenGPU repository](https://github.com/vixhal-baraiya/qwenGPU#readme).

## How it works

**The GPU.** A 4-lane SIMT machine with 15 registers per lane, a 4-stage pipeline, a 32-block
program ROM and 126 words of global memory, built from Qwen3.5's own layer types (Gated
DeltaNet linear attention and SwiGLU MLPs). Registers and memory live in the layers'
recurrent state; each record the model writes runs one machine block (up to 4 instructions)
for one warp.
It supports integer arithmetic, comparisons and predicates, divergent branches, loops, global
and shared memory, `atomicAdd`, `__syncthreads()` and warp shuffles.

**A launch.** The host does what NVRTC and the CUDA driver do on a real GPU. It compiles the
CUDA source to PTX with clang, lowers the PTX to the GPU's machine blocks
(`qgpu/lower.py`), and feeds the program image and the input arrays as tokens. From there the
model writes the launch itself: each generated token is a record of which block a warp ran
and the values it produced, until an end-of-launch token. The host then reads the result
arrays from the model's recurrent state. It never computes anything itself.

**The merge.** `qgpu.merge` compiles the circuit with [Qwenputer](https://github.com/divagr18/qwenputer)'s
transformer compiler, widens Qwen's residual stream, puts the circuit layers in front of
Qwen's and adds the GPU's tokens to the vocabulary. Circuit dimensions that do not matter on
text share Qwen's dimensions (the overlay), and the circuit is muted on text tokens, so
Qwen's text is unchanged: the merged model's next-token distribution stays within a KL of
about 7e-4 of the original.

**The chat.** `qgpu` is a terminal chat with the merged model. When a message asks for a
calculation, Qwen computes it on the GPU in its weights, in one of two modes:

- **normal**: Qwen writes the arithmetic as an expression (`[1, 2, 3] @ [4, 5, 6]`), and each
  operation runs as a ready-made CUDA kernel from `kernels/`.
- **advanced**: Qwen writes the CUDA kernel itself, guided by the
  [qgpu CUDA reference](docs/qgpu-cuda.md) in its system prompt.

A panel next to the chat shows the GPU live: the warp scheduler, program ROM, pipeline,
register file and memory light up as the model's tokens drive them.

## Gated DeltaNet as GPU state

Like [Qwenputer](https://github.com/divagr18/qwenputer), qgpu keeps its machine state in the
recurrent state of Qwen3.5's Gated DeltaNet layers. In the circuit's heads the gate is held
open and the write strength is 1, and every key is one-hot, so each update overwrites exactly
one row of a head's state and a one-hot query reads it back: each row is a memory cell.

- **Program ROM**: each machine block's instructions, one instruction per stage, in heads of
  their own.
- **Registers**: 16 per lane for each warp (one always reads zero), in rows chosen by warp and
  register.
- **Global memory**: the kernel's arrays, one 32-bit word per row. Inputs are written by the
  host before the launch; results are written by the circuit during it.

A launch is a stream of tokens. The host writes the setup records: `PROG` and `LIVE` records
load the program, `MEM` records load the inputs, and an `EXEC` token starts warp 0 at block 0.
Then the model writes the rest. Each `EXEC(b, w)` token runs block `b` for warp `w` in one
forward pass, and the `VAL` tokens after it carry the block's results, two 16-bit tokens per
lane for each register a later block reads. Memory writes come back as `MEM` records, and
`EXEC(0, warps)` ends the launch. The token stream is the GPU's execution log, and every value
in it is exact in bf16.

## What runs where

Qwen asks for a calculation with a tool call, in the format Qwen3.5 was trained on. Unlike
Qwenputer, the host does act on that call: it does what a real GPU's host side does.

- **The host**: compiles the CUDA to PTX, lowers the PTX to machine blocks, writes the
  program and the input arrays as tokens, and reads the result arrays from the model's state
  when the launch ends. It decides nothing about the values and computes none of them.
- **The weights**: every step of the launch. The model writes each `EXEC` and `VAL` token by
  greedy decoding through its own layers: which block runs next, every register value, every
  branch and every memory write.

With `--check`, each launch is replayed on a reference interpreter of the circuit and the
model's tokens must match it exactly.

## Results

On Qwen3.5-2B, on Apple silicon (MPS, bf16):

- Every launch in our runs was exact. With `--check`, each launch is compared token for token
  with a reference interpreter of the circuit, and that interpreter matched a real NVIDIA
  Tesla T4 on 49 of 49 test kernels.
- Text: the merged model's next-token distribution is within a KL of about 7e-4 of the
  original.
- Normal mode answered 20 of 24 sampled chat questions with numbers computed on the GPU.
- Advanced mode wrote a correct kernel for 4 of 5 vector and matrix questions on the first
  try (dot product, matrix product, squared length, vector addition). A 2B model's CUDA is still
  unreliable; use normal mode for answers you rely on.

A launch takes tens to a few hundred generated tokens, about 10-15 tokens per second on MPS.

## Quickstart

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/),
[just](https://github.com/casey/just), about 16 GB of free memory for
the 2B model (Apple silicon, an NVIDIA GPU or a CPU), and for advanced mode a clang with the
NVPTX backend (`brew install llvm`, or set `CLANG` to its path).

```bash
git clone https://github.com/vixhal-baraiya/qwenGPU && cd qwenGPU
just setup
just build
just chat
```

`just build` downloads Qwen3.5-2B and compiles the GPU into it. In the chat, type `/` for
commands. `just` lists every recipe, including `just check` and `just test`.

Other entry points:

```bash
uv run python -m qgpu.bridge build/merged-2b "What is the sum of 12, 7, 30 and 5?"
uv run python scripts/merged_check.py models/Qwen3.5-2B build/merged-2b
uv run python scripts/eval_chat.py build/merged-2b --samples 3
```

## Repository

| Path | What it is |
|---|---|
| `qgpu/circuit/` | The GPU as transformer layers: the kernel circuit and its parts |
| `qgpu/ptx.py`, `qgpu/lower.py` | The PTX parser and the driver that lowers PTX to machine blocks |
| `qgpu/sim.py`, `qgpu/driver.py` | A reference SIMT interpreter for the PTX subset |
| `qgpu/merge.py`, `qgpu/overlay.py` | Building a merged Qwen3.5 checkpoint |
| `qgpu/runtime.py` | Launching a kernel on the merged model and reading results back |
| `qgpu/cuda.py`, `qgpu/calc.py` | Advanced mode (Qwen's CUDA) and normal mode (expressions) |
| `qgpu/bridge.py`, `qgpu/tui.py`, `qgpu/monitor.py` | The chat session, the terminal app and the GPU panel |
| `kernels/` | The CUDA kernel library and the PTX tests run against |
| `docs/qgpu-cuda.md` | The CUDA reference for this GPU |
| `tests/` | Circuit, driver, runtime, kernel and app tests |

Run the tests with `just test`. The circuit tests evaluate the compiled weights exactly;
the slowest take a few minutes.

## Credits

The transformer compiler is [Qwenputer](https://github.com/divagr18/qwenputer) (MIT), which
compiles a RISC-V CPU into Qwen3.5-4B; qgpu uses it as a library and builds its own GPU,
driver and token format on top. If you use this work, please cite Qwenputer:

```bibtex
@software{agrawal2026qwenputer,
  author  = {Agrawal, Divyansh},
  title   = {{Qwenputer}: a computer compiled into the weights of a pretrained language model},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/divagr18/qwenputer},
  license = {MIT}
}
```

## License

The code is MIT, see [LICENSE](LICENSE). A checkpoint built with `qgpu.merge` is a modified
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) model (its config is widened and GPU layers
are added; Qwen's own weights are unchanged), so its weights fall under Qwen's Apache-2.0
license.
