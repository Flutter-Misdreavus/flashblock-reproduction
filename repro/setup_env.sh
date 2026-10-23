#!/usr/bin/env bash
# flashblock-20261023-prep: documented, reproducible creation of the isolated runtime.
# The torch stack is reused read-only from the own MemLoop runtime (same interpreter
# 3.10.12, torch 2.2.2+cu121, triton 2.2.0, numpy 1.26.3); nothing is installed into
# or modified inside that environment.
set -euo pipefail

RUNTIME="${FLASHBLOCK_RUNTIME:-/data/research/flashblock-reproduction-runtime}"
MEMLOOP_SITE="/data/research/MemLoop/runtime/venv/lib/python3.10/site-packages"
VENV="${RUNTIME}/venv"

python3 -m venv --without-pip "${VENV}"
curl -sSL -o "${RUNTIME}/get-pip.py" https://bootstrap.pypa.io/get-pip.py
"${VENV}/bin/python" "${RUNTIME}/get-pip.py" --no-cache-dir

# Link the already-verified torch/triton/numpy stack (read-only source).
for e in "${MEMLOOP_SITE}"/*; do
  b="$(basename "$e")"
  [ -e "${VENV}/lib/python3.10/site-packages/${b}" ] && continue
  ln -s "$e" "${VENV}/lib/python3.10/site-packages/${b}"
done

"${VENV}/bin/pip" install --no-cache-dir -r "$(dirname "${BASH_SOURCE[0]}")/requirements-prep.txt"
