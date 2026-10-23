# FlashBlock preparation r3 — bounded repair result

Run: `flashblock-20261023-prep-r3` (parent `flashblock-20261023-prep-r2`), seed 42.
Status: **SUCCEEDED** for the preparation gate. No mechanism, performance or test-set
stage was run. This is not an answer-quality claim.

## Outcome

- Custom sm86 BF16 head_dim128 FlashAttention `.so` loaded and independently validated:
  9/9 correctness groups pass. Dense parity and no-reuse decomposition remain at
  BF16 relative L2 ≈ 0.002 (max 0.00218), within the frozen 1e-2 / 5e-2 tolerances.
- Genuine cached reuse was exercised with persistent A/L buffers in the actual model
  geometry (GQA 32/8, head_dim 128, block 4): unchanged-Q reuse rel L2 0.00107 vs the
  store call; changed-Q stale-cache composition rel L2 0.00171 vs independent
  log-space composition. Reset and batch-slot swap are exact (slot swap rel L2 0.0).
- Both real prompts 100 and 101 generated exactly 64 valid, nonempty, finite tokens
  under dense and official s2/tau2. No NaN/Inf parameter (4,411,424,256 elements) or
  logits (56–57 audited calls) were observed.
- Peak process allocation 11.08 GiB, physical free ≥ 8.35 GiB, wall 935 s,
  charged 0.10 GPU h (stage limit 0.40, campaign 0.20/8.0). Engine `exit()` returned ok.

## Seven supervisor findings addressed

1. **Seeding.** `random`, NumPy, torch CPU and all CUDA devices are seeded (42) before
   engine creation and again immediately before generation; actual state values are
   recorded. Dense cold and warm runs produced identical token IDs and text, so the
   paired sampling seed reset is effective.
2. **Genuine cached reuse.** New tests keep the same persistent A/L buffers across
   calls and cover unchanged q, changed q, LSE checks, dirty mask, cache reset and
   batch-slot reorder. Unchanged-Q reuse is exact to BF16 rounding; changed-Q output
   matches the independent stale per-query composition.
3. **Static shape support.** Cached tests use the real model GQA 32/8, hd128, block 4.
   No hd64 or arbitrary-length case is claimed. No tolerance was changed after results.
4. **Real smoke validity.** Status is derived from actual outputs, token-ID range,
   nonempty text, full parameter/logit finiteness, KV capacity, memory and, for s2,
   observed reuse counters. The hard-coded `all_tokens_finite=True` claim is gone.
5. **LSE and empty history.** No-reuse tests compare independent LSE, require zero
   history output and `-inf` (not NaN) LSE for empty history; NaN fails the gate.
6. **Provenance and streaming.** Actual execution SHAs are recorded separately below.
   The streaming reuse-buffer swap now uses a buffer sized to the active batch;
   without that, s2 failed at the second scheduler step. No test-set evaluation.
7. **Prefill store stub and cleanup.** The unconditional `assert 1==0` in
   `store_kvcache_kernel` was replaced with the standard vectorized Triton paged
   scatter (`slot_mapping` preserved); prefill then reached the attention kernel
   successfully. Peak memory is read in `finally`, and the real engine `exit()` is
   called in `finally`.

## Additional compatibility fixes found by real execution

- `liger-kernel==0.5.10` was the only missing runtime requirement; installed in the
  isolated venv and pinned. No torch/triton/flashinfer reinstall.
- Triton 3.0 autotune key lookup failed for the upstream prefill kernel because the
  keyed constexprs followed config-supplied `BLOCK_M/BLOCK_N`. The keyed constexprs
  were moved before those config kwargs and passed positionally. Attention math was
  not changed.
- `LLM(batch_size=max(4,n))` with `max_active=n` made the streaming reuse-buffer swap
  copy 2 rows into a 4-row buffer. The buffer is now sized to the active batch.

## Reuse audit (s2)

106 scheduler decisions: 84 refresh, 22 reuse. Reuse occurred on 12 distinct block
indices (19–45) for both sequences, so it is genuine multi-block reuse, not a single
step. `dirty_mask` was passed but has no effect in this path; this is documented, not
hidden: the cached A/L are history-only (prefix excluding the current block), so the
current block is always recomputed and changed query rows still compose correctly.
The changed-Q test intentionally uses a fully unrelated q2, so the full-recompute
error (~1.10) measures staleness stress, not realistic per-step drift.

## Timing (no speed claim)

Dense cold 84.65 s (includes one-time Triton/CUDA compilation), dense warm 5.20 s,
s2 5.78 s for 128 generated tokens. s2 is not faster than warm dense in this block-4
smoke, so no mechanism/efficiency conclusion is drawn. The cold/warm difference is a
compilation confound, not a reuse effect.

## Preserved failures (bound to their commits)

- Correctness attempt 1 (`4f319012`): `_kernel_call` passed a mapping as positional
  `dict()` arguments; block-size paths raised TypeError. Fixed in `1d3bc964`.
- Correctness attempt 2 (`1d3bc964`): changed-Q target used fresh history for q2
  instead of the actual stale q1 A/L; 2 cases failed. Fixed in `79b94b86` (test target
  only, kernel math untouched).
- Smoke attempt 1 (`79b94b86`): missing `liger_kernel`; pinned in `adfb9af7`.
- Smoke attempts 2–3 (`adfb9af7`, `c7723532`): Triton 3.0 autotune key IndexError;
  compatibility repair completed in `737c4f90`.
- s2 attempt 1 (`737c4f90`): reuse-buffer batch mismatch; fixed in `904d7689`.

All raw logs, raw generated text and scheduler traces remain outside Git in the run
directory. Only sanitized small evidence is committed under `experiments/`.
