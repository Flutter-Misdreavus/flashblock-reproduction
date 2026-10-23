# Mechanism preflight — PASSED on attempt 2 (hard stops active)

Run: `flashblock-20261023-mechanism`, accepted execution commit
`ecb4974a6ad1245c875b0142e43551c2ee99da07` (clean worktree at start).
Attempt 2, 2026-10-23T02:42:48Z–02:43:25Z, wall 37.166 s. Raw subprocess logs and
JSON are in `runs/flashblock-20261023-mechanism/preflight/` (outside Git).

Attempt 1 passed the same numerical checks at commit `4c00e4be`, but predated the
supervisor hard-stop addendum; it is preserved in the git history and is not the
accepted preflight.

## Synthetic numerical checks (`unit`)

- Prefill block-causal GQA parity, 32/8 heads, head_dim 128, lengths 75 and 128:
  BF16 relL2 0.002414, max abs 0.009951; frozen limits 0.01 / 0.05. PASS.
- Repaired paged KV scatter with slots `[5,-1,260,3,0,-1,511]`: bitwise equal to
  direct indexing for K and V; genuinely unwritten sentinel slot 1023 unchanged.
  PASS.
- FP32 split/recombine: nonempty relL2 1.495e-7, max abs 1.788e-7; empty-history
  exact; frozen limits 1e-5. PASS.
- True-history page mapping with non-contiguous block table `[3,0,2]`, 600 tokens:
  exact. PASS.

## Real-model neutrality (`train-100`, s2/tau2, max 16 tokens, seed 42)

- No-hook and hook-enabled runs produced the **same 16 token IDs**
  (SHA256 `e19ddbd3...12696`); no mask tokens; output nonempty.
- Hook calls per selected layer: 0:13, 2:13, 17:13, 35:13. All required hooks
  nonzero. 512 aligned observations recorded; `capture_errors` empty.
- Peak allocated 11068.6 MiB (hard limit 16384), minimum sampled physical free
  8563.1 MiB (hard limit >=3072). Sampler `breach` is null under the enforced
  resource guard. No OOM/NaN. Finite logits.

## Hard stops exercised by this attempt

Non-finite logits now raise; cache-label identity mismatch (>0.01 relL2 or
>0.01 |dLSE|) raises; allocated >16384 MiB or sampled free <3072 MiB raises
through the scheduler wrapper; a successful exit requires zero non-finite logits
and zero capture errors. Approximate stale-vs-full error thresholds were not
changed. This attempt ran the guarded code end-to-end without a breach.

## Repair history

The four live-supervisor corrections (sentinel slot, reuse-only stale merge,
post-schedule dirty count, explicit denominators) and the hard-stop addendum are
in commits `4c00e4be` and `ecb4974a`, documented in
`supervisor-feedback-resolution.md`. Development probes and their conservative
130 s charge are in `development-probes.json`; they are explicitly unaccepted.
No tolerance, model, data split or sampling control was changed.
