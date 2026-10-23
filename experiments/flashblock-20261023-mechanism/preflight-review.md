# Mechanism preflight — PASSED (attempt 1)

Run: `flashblock-20261023-mechanism`, execution commit
`4c00e4be50b8899ffc5407a80b59b4372135d3d7` (clean worktree at start).
Attempt 1, 2026-10-23T02:39:06Z–02:39:42Z. Raw evidence and subprocess logs are in
`runs/flashblock-20261023-mechanism/preflight/` (outside Git).

## Synthetic numerical checks (`unit`)

- Prefill block-causal GQA parity, 32/8 heads, head_dim 128, lengths 75 and 128:
  BF16 relL2 0.002414, max abs 0.009951; frozen limits 0.01 / 0.05. PASS.
- Repaired paged KV scatter with slots `[5,-1,260,3,0,-1,511]`: bitwise equal to
  direct indexing for K and V; unwritten sentinel slot 1023 unchanged. PASS.
- FP32 split/recombine: nonempty relL2 1.495e-7, max abs 1.788e-7; empty-history
  exact; frozen limits 1e-5. PASS.
- True-history page mapping with non-contiguous block table `[3,0,2]`, 600 tokens:
  exact. PASS.

## Real-model neutrality (`train-100`, s2/tau2, max 16 tokens, seed 42)

- No-hook and hook-enabled runs produced the **same 16 token IDs**
  (SHA256 `e19ddbd3...12696`); no mask tokens; output nonempty.
- Hook calls per selected layer: 0:13, 2:13, 17:13, 35:13. All required hooks
  nonzero. 512 aligned observations recorded; `capture_errors` empty.
- Peak allocated 11068.6 MiB (limit 16384), minimum sampled physical free
  8563.1 MiB (limit >=3072). No OOM/NaN. Finite logits.

## Repair history

The four live-supervisor corrections (sentinel slot, reuse-only stale merge,
post-schedule dirty count, explicit denominators) are in commit `4c00e4be` and
`supervisor-feedback-resolution.md`. Two pre-commit integration failures
(`KeyError('kv_heads')`, missing `ninja` on PATH) are documented there; raw
development runs remain outside Git. No tolerance, model, data split or sampling
control was changed.
