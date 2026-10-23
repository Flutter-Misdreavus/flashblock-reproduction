# Resolution of live supervisor feedback (pre-formal16)

Supervisor checkpoint commit: `08ba307f737151c7cfad70d4d7b26744f97b383f`
(`experiments/flashblock-20261023-mechanism/supervisor-live-review.md`).

All four points were addressed before any official preflight or formal run. No
scientific scope, tolerance, model, data or sampling control changed.

1. **Paged-scatter negative-slot assertion.** The test now checks an unwritten
   sentinel slot, `1023` (block 3, offset 255), instead of slot 5, which positive
   input index 0 writes. The full bitwise `k_cache == direct_k` and
   `v_cache == direct_v` comparisons are retained. The scatter implementation and
   tolerances were not changed. Unit test returns `pass=true`.
2. **Stale-merge accumulators.** `PromptCollector.observe_step` now computes and
   appends `stale_merge_rel` / `stale_vs_actual` only when `meta["reuse"]` is true.
   Hypothetical merges on fresh-recompute/store steps are no longer mixed into the
   reuse residual summary. Summary keys name the reuse denominator explicitly.
3. **Dirty-count label.** Pre-schedule snapshots are still used for the true
   `block_table`, `num_tokens`, and current-block token IDs. The dirty count used
   for metric labels is now read from `seq.sparsity_params` **after** `schedule()`
   returns, so the step-0 reset is reflected; cache age still refers to actual A/L
   stores.
4. **Explicit denominators.** `mechanism-summary.json` now carries a `denominators`
   block: bootstrap unit is the prompt (16 units), aligned-row counts are summed
   over selected layer/head means, and reuse-step/reuse-row denominators are listed
   per layer/head. Raw row counts are never treated as independent bootstrap units.

Development-only integration checks performed before that commit (not formal
attempts; raw evidence remains under `runs/flashblock-20261023-mechanism/dev-hook/`):
a `KeyError('kv_heads')` from an initial hook-registration bug, and a missing
`ninja` executable because the isolated venv was not on `PATH` for the flashinfer
JIT. Both were fixed in the driver; a same-seed s2 train-100 max-16 run then
produced identical 16-token IDs with hooks enabled and disabled, cache-label
relL2 0.00242, and finite logits. No numerical method was altered to make these
checks pass.
