model := "Qwen3.5-2B"
merged := "build/merged-2b"

# list the recipes
default:
    @just --list

# install the dependencies
setup:
    uv sync

# download Qwen3.5-2B and compile the GPU into it
build:
    uv run hf download Qwen/{{model}} --local-dir models/{{model}}
    uv run python -m qgpu.merge models/{{model}} {{merged}}

# chat with Qwen and its GPU (e.g. just chat --mode advanced)
chat *args:
    uv run qgpu {{merged}} {{args}}

# check the build: Qwen's text unchanged, kernels exact
check:
    uv run python scripts/merged_check.py models/{{model}} {{merged}}

# run the tests
test *args:
    uv run pytest {{args}}

# every qccl collective on n QwenGPUs (one process per GPU), checked against real NCCL
collectives n="4" *args:
    uv run torchrun --nproc-per-node {{n}} scripts/qlink_run.py {{merged}} {{args}}

# data-parallel MLP training on n QwenGPUs (e.g. just train 4 --mode verify --zero)
train n="4" *args:
    uv run torchrun --nproc-per-node {{n}} scripts/train_dp.py {{merged}} --out runs/train {{args}}

# the same training on n real GPUs over NVLink P2P
train-cuda n="4":
    uv run python scripts/train_cuda.py --world {{n}}
