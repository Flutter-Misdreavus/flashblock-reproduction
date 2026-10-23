#!/usr/bin/env bash
set -euo pipefail
RUNTIME="${FLASHBLOCK_RUNTIME:-/data/research/flashblock-reproduction-runtime}"
VENV="${RUNTIME}/venv"
# Official PyPI stalled after ~4 MiB on 2026-10-23. A reachable pinned mirror is
# permitted for this bounded network repair; override with PIP_INDEX_URL.
INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --without-pip "$VENV"
  curl -sSL --fail --max-time 120 -o "$RUNTIME/get-pip.py" https://bootstrap.pypa.io/get-pip.py
  "$VENV/bin/python" "$RUNTIME/get-pip.py" --no-cache-dir --index-url "$INDEX_URL"
fi
if [[ -n "$(find "$VENV/lib/python3.10/site-packages" -maxdepth 1 -type l -print -quit)" ]]; then
  echo 'Refuse installation into symlinked site-packages' >&2
  exit 1
fi
"$VENV/bin/pip" install --no-cache-dir --no-build-isolation --index-url "$INDEX_URL" -r "$(dirname "${BASH_SOURCE[0]}")/requirements-prep.txt"
"$VENV/bin/pip" install --force-reinstall --no-deps --no-cache-dir --no-build-isolation --index-url "$INDEX_URL" flashinfer-python==0.2.9
