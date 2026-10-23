# Preparation-gate reproduction CLI (flashblock-20261023-prep)

Frozen scope: preparation gate for the FlashBlock mechanism reproduction.
No training, no quantization, no held-out test evaluation.

Pinned upstream: `Caesarhhh/Flash_Block` @ `a322fb80f29df0d4cc842fbc011929bc85ba8f36`
(vendored under `dllm/` and `flash-attention/`).

## Files

| file | purpose |
|---|---|
| `requirements-prep.txt` | exact Python package pins installed into the isolated runtime venv |
| `setup_env.sh` | documents venv creation + reuse of the read-only own torch 2.2.2+cu121 stack |
| `build_flash_attn.sh` | builds the vendored custom FlashAttention fork for sm86 only |
| `fetch_model.py` | downloads the pinned TraDo-4B BF16 checkpoint and writes `locks.json` |
| `freeze_ids.py` | freezes GSM8K train100..115 and 128 seeded test ids (ids only) |
| `validate_attention.py` | independent FP32 split/recombine reference + custom-kernel BF16 checks |
| `run_smoke.py` | real-model smoke, baseline (no reuse) vs s2 tau2, train100..101, 64 new tokens |

## Environment

Isolated venv: `/data/research/flashblock-reproduction-runtime/venv`.
`torch==2.2.2+cu121`, `triton==2.2.0`, `numpy==1.26.3`, `flashinfer-python==0.2.9`,
vendored `flash_attn 2.7.3` built from `flash-attention/` for `FLASH_ATTN_CUDA_ARCHS=86`.

GPU is selected by UUID `GPU-REDACTED` via
`CUDA_VISIBLE_DEVICES`. CPU threads capped at 8, `MAX_JOBS=2`.
