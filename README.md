# QLink

By Zeeshan Patel & Claude Opus 5.5 · [Blog post](https://www.zeeshanp.me/blog/qlink/)

**QwenGPUs that talk NCCL.** [qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) compiles a
small SIMT GPU into the weights of Qwen3.5-2B. QLink connects several of those models the way
NVLink connects GPUs, runs NCCL's collectives across them, and trains neural networks on them,
data-parallel. The models compute every value. The host only launches kernels and routes
packets.

| | |
|---|---|
| Collectives | AllReduce, ReduceScatter, AllGather, Broadcast, Reduce and SendRecv on 4 merged Qwen3.5-2B models (one per GB200), pure greedy decoding: every result is bit-identical to real NCCL on the same GPUs |
| Training | a 2-8-1 MLP learns XOR data-parallel on 4 QwenGPUs: 53% → 98.8% test accuracy in 40 steps, replicas bit-identical at every step and equal to an exact numpy replay |
| Tensor parallel | MNIST (86.9% test accuracy) and two diffusion models, one drawing a 7 as a point cloud and one drawing handwritten digits in a latent space, each split across 4 QwenGPUs the way Megatron splits a layer; every checkpoint and every generated sample equals an exact numpy replay |
| Portability | the same CUDA kernels and the same schedule run on the real GPUs over NVLink P2P and give the same parameters, bit for bit |
| Cost | no new tokens, no new weights: the link reuses the GPU's own store record |

## How it works

**A store record is a packet.** In qwenGPU, global memory is the recurrent state of a Gated
DeltaNet head, one 32-bit word per row. The host writes a word with three tokens,
`<|gpu_mem a|> <|gpu=lo|> <|gpu=hi|>`, and a kernel's `st.global` comes out of the model in
exactly that form. So a store written by one model can be fed into another model's context,
and that model's layers write it into its memory, as they would the host's `cudaMemcpy`.

**Apertures.** A device has address ranges whose stores leave it, like an NVLink peer mapping.
When the model writes `MEM(a)` for an aperture address, the device forwards the record to the
peers and feeds `MEM(127)` (the sink) locally. A multicast aperture keeps the local write too
(NVLink SHARP's `multimem.st`). Loads are never routed.

**Record boundaries.** A packet is fed between two records: after the model has predicted the
next `EXEC` token, before that token goes in. There it can only write a memory row, and the next
record reads memory at its `EXEC`, so a kernel spinning on a flag sees a peer's store on its next
iteration. The compiled circuit, evaluated exactly, and the bf16 model both confirm this token
for token.

**qccl.** NCCL's ring algorithms and its Simple protocol, on one CUDA kernel
([`kernels/qccl/prim.cu`](kernels/qccl/prim.cu)). A FIFO of slots at the receiver, a `tail` flag
the sender posts after its data, a `head` flag the receiver posts when a slot is free. Every
rank's kernels synchronize by polling flags their peers write, never through the host.
Multicast windows give NVLS-style AllGather and a two-shot AllReduce, and a ZeRO-1 training
step where each rank's shard update lands in every replica.

```
H2D        each rank's minibatch (its data loader)
forward    mlp_forward    o[s] += W2[k] relu(W1[k] x[s] + b1[k])      (atomicAdd)
backward   mlp_backward   per-sample gradients summed by atomicAdd
comm       qccl ring AllReduce(g)   or   ReduceScatter(g) + multicast shard update (ZeRO-1)
update     sgd            p -= round(g / 2^6)
```

## Results

| Run | What | Result |
|---|---|---|
| E3 | 2-rank AllReduce, 2 models, greedy | 1,020/1,020 tokens per rank identical to the reference; = numpy |
| E4 | 6 collectives, 4 models, greedy, torchrun | 6/6 bit-identical to real NCCL; every rank's ~7,600-token stream identical to the reference system |
| E5 | the same collectives on the real GPUs (CUDA, NVLink P2P) | 6/6 correct, synchronized only by device flags |
| E7 | 6 collectives on 8 QwenGPUs, two models per GPU | 6/6 correct on every rank; every stream identical to the reference system |
| T1 | MLP, ring AllReduce, 40 steps, 4 models | 40/40 steps exact vs numpy on every rank; 98.8% test accuracy |
| T2 | MLP, ZeRO-1 over multicast, 40 steps, 4 models | identical parameters to T1 at every step; half the communication tokens |
| T1-CUDA | T1's program on 4 real GPUs | final parameters identical to T1 |
| G1 | T1's first 2 steps, pure greedy decoding, every token checked as written | identical to T1 |
| M1 | MNIST: a linear classifier on 5×5 crops, tensor-parallel over 4 QwenGPUs, 3,000 images | 80.6% test accuracy (the model's least-squares ceiling is 80.9%); weights equal the numpy replay at all 30 checkpoints |
| D1 | a DDPM drawing a "7": 3→32→2 denoiser, tensor-parallel over 4 QwenGPUs, 6,000 steps, sampling on the QwenGPUs | parameters equal the numpy replay at all 12 checkpoints; all 256 generated points equal the replay at every DDIM step; ~40M tokens per model, every one checked |
| L1 | latent image diffusion on MNIST: a fixed PCA encoder (8 numbers per image), a 9→16→8 denoiser tensor-parallel over 4 QwenGPUs, 6,000 steps, 64 digits sampled on the QwenGPUs | parameters equal the numpy replay at all 6 checkpoints; all 64 generated digits equal the replay at every DDIM step; the [blog post](https://www.zeeshanp.me/blog/qlink/)'s WebGPU demo reproduces them bit for bit |
| M2 | MNIST again on the same 4 QwenGPUs and 80 words each: 31 whitened principal components and a one-vs-rest squared hinge, 6,000 images | 86.9% test accuracy (M1: 80.6%); weights equal the numpy replay at all 60 checkpoints. 100% is out of reach for any model; 63 components would give ~90% but need 8 ranks, which do not fit in 127 words |

The machine-readable summaries are in [`results/`](results), the lab notebook with charts is
[`docs/notebook.html`](docs/notebook.html), and the
[blog post](https://www.zeeshanp.me/blog/qlink/) walks through the design, with animations
rendered from these runs and the digit diffusion model running live in the page (WebGPU, the
QwenGPUs' exact integer arithmetic).

## Quickstart

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just),
a clang with the NVPTX backend, and one CUDA GPU per QwenGPU (each holds a 15 GB merged model).

```bash
just setup
just build                  # download Qwen3.5-2B and compile the GPU into it (qwenGPU)
just test                   # circuit, driver and QLink tests
just collectives 4          # 6 collectives on 4 QwenGPUs vs real NCCL (--mode greedy --check)
just train 4 --mode verify  # data-parallel MLP training on 4 QwenGPUs (--zero for ZeRO-1)
just train-cuda 4           # the same training on the real GPUs
```

MNIST and the diffusion model (`uv sync --group fast` installs flash-linear-attention's Gated
DeltaNet kernels, which keep every check exact and make verification about 4x faster):

```bash
uv run --with pyarrow python scripts/get_mnist.py
uv run python -m torch.distributed.run --nproc-per-node 4 scripts/train_mnist.py build/merged-2b --burst 8 --out runs/mnist
uv run python -m torch.distributed.run --nproc-per-node 4 scripts/train_diffusion.py build/merged-2b --burst 8 --out runs/diffusion
uv run python -m torch.distributed.run --nproc-per-node 4 scripts/train_latent.py build/merged-2b --burst 8 --out runs/latent
```

**Three ways to run the models.** `greedy` decodes autoregressively, one forward pass per token
(about 7 tokens/s per model on a GB200). `draft` has the reference interpreter draft each record
and the model check it in one pass. `verify` lets the reference drive the system with real
packet exchange, then has each model run the exact context it would have seen at the end of
every launch: its argmax at every position it owns must equal the token there. If that holds for
every model, the greedy system writes the same streams, by induction over ticks. It is the same
teacher-forced check qwenGPU's `merged_check` uses, and about 20 times faster than greedy.

## Repository

| Path | What it is |
|---|---|
| `qgpu/dist/device.py` | A QwenGPU as a device: heap, stream, apertures, record boundaries, decoding modes |
| `qgpu/dist/backends.py` | Reference interpreter, exact circuit, merged model on CUDA |
| `qgpu/dist/transport.py` | The fabric: threads, or torch.distributed between processes |
| `qgpu/dist/qccl.py` | Communicators and collectives (ring, multicast) |
| `qgpu/dist/train.py` | Data-parallel training and its exact numpy replay |
| `qgpu/dist/realgpu.py` | The same kernels and schedule on real GPUs over NVLink P2P |
| `qgpu/dist/mnist.py`, `qgpu/dist/diffusion.py`, `qgpu/dist/latent.py` | Tensor-parallel MNIST, a diffusion model on points and a latent diffusion model on images, with exact numpy replays |
| `kernels/qccl/`, `kernels/train/`, `kernels/mnist/`, `kernels/diffusion/` | The CUDA kernels |
| `scripts/` | `qlink_run.py` (collectives vs NCCL), `train_dp.py`, `train_cuda.py`, `train_mnist.py`, `train_diffusion.py`, `train_latent.py` |
| `qgpu/`, `kernels/` (the rest) | qwenGPU: the GPU circuit, the PTX driver, the merge; see [docs/qwengpu.md](docs/qwengpu.md) |

QLink changes to qwenGPU's driver (`qgpu/lower.py`): pointer arguments at fixed device
addresses, so memory persists across launches; `ld.volatile` and fences; warp-uniform loads, so
spin loops stay unpredicated; and an early exit after uniform loops. The reference interpreter
(`qgpu/circuit/kernel.py`) now runs record by record.

## Limits

- QLink carries stores only, so collectives push data. Remote loads and remote atomics would
  need the peer to compute a reply.
- Each QwenGPU has 127 words of memory and a 32-block program ROM.
- Ranks advance in lockstep, one record per tick, which makes every run deterministic and
  replayable. The protocol itself does not depend on it: it also runs on real, asynchronous GPUs.

## Credits

QLink is built on two projects, and would not exist without them:

- **[qwenGPU](https://github.com/vixhal-baraiya/qwenGPU)** by vixhal-baraiya (MIT): the SIMT
  GPU compiled into Qwen3.5's weights, its PTX driver and the merged checkpoint. This repository
  starts from qwenGPU at commit `e2695f6`; its README is kept as [docs/qwengpu.md](docs/qwengpu.md).
- **[Qwenputer](https://github.com/divagr18/qwenputer)** by Divyansh Agrawal (MIT): the
  transformer compiler (`chipformer`) that qwenGPU, and so QLink, uses as a library. As its
  authors ask, if you use this work, please cite Qwenputer:

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

MIT, see [LICENSE](LICENSE), which keeps qwenGPU's copyright notice. A checkpoint built with
`qgpu.merge` is a modified [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) model and falls
under Qwen's Apache-2.0 license.
