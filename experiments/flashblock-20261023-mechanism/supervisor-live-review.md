# Live GPT supervisor review — address before formal16

This note clarifies existing task correctness/metric requirements, no scientific scope change. Keep prior failures, commit corrections before GPU rerun. Review based on initial mechanism_cli.py1744lines.

1. run_unit_tests paged_kv_scatter negative-slot check currently compares k_cache[0,5] to sentinel7, but slot5 is explicitly written by input0. Use a genuinely unwritten sentinel slot (e.g last slot1023), and retain full bitwise comparison to direct_k/direct_v. Do not alter scatter implementation or relax tolerance to fix a faulty assertion.
2. PromptCollector.observe_step computes sm_rel/sm_act for ALL s2 steps, appends to stale_merge_rel/stale_vs_actual unconditionally, but summarize names these mean_stale_merge_rel_l2_reuse. Filter these accumulations by meta['reuse'] (or separate explicit all-step and reuse-only metrics). Fresh full recompute/store steps use old pre-call A/L for an irrelevant hypothetical merge, so mixing them invalidates the reuse residual summary. This must be fixed before formal data, otherwise preserve raw per-step condition and recompute by genuine reuse mask.
3. Dirty count snapshot before schedule can be stale at new block because schedule resets dirty count to0 at step0. Keep pre-schedule true block_table/num_tokens/tokenIDs for reference, but read actual dirty count used by the policy from sp AFTER orig() for metric labels. Cache-age should refer to actual A/L stores as already implemented.
4. Reuse-error row/layer/head denominators must be explicit. 16prompt primary equal-weight means correct; do not count all query rows as independent bootstrap units. The existing independent reference/history snapshot design otherwise looks consistent on source review.

Only mechanism stage authorized. This note is genuine supervisor feedback, not a new worker task or approval to exceed budgets/retries. Include resolution in stage evidence.
