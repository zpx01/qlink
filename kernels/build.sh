#!/bin/sh
# Compile every kernel here to PTX with Homebrew LLVM's NVPTX backend (no CUDA SDK needed).
set -e
cd "$(dirname "$0")"
CLANG=${CLANG:-/opt/homebrew/opt/llvm/bin/clang}
for f in *.cu; do
  "$CLANG" -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -Xclang -target-feature -Xclang +ptx70 -O2 -S \
    "$f" -o "${f%.cu}.ptx"
done
