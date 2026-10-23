#!/usr/bin/env bash
# flashblock-20261023-prep: build the vendored custom FlashAttention fork.
# RTX3090 is sm86; upstream emits compute_86 PTX for this option.
# Warm up driver JIT before timed measurements.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${FLASHBLOCK_RUNTIME:-/data/research/flashblock-reproduction-runtime}"
PY="${RUNTIME}/venv/bin/python"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export FLASH_ATTN_CUDA_ARCHS=86
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES=GPU-REDACTED

cd "${ROOT}/flash-attention"
"${PY}" -m pip install -e . -v --no-build-isolation --no-deps
