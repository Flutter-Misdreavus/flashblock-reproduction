# Resolution of GPT pre-execution review (2026-10-23T03:14Z)

The review arrived as an untracked file after the vocab-bound fix commit and is
preserved here with this resolution. No frozen scientific parameter changed.

1. **Multiple engines / OOM hazard.** `cmd_matrix` now constructs exactly one
   engine at a time. The B1 engine is exited, all references dropped, and
   `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` run
   before B4 construction. There is no concurrent engine and no `engines` dict.
2. **A/L identity.** The preflight recorder now clones and compares both
   `attn_output_past` (A) and `logsumexp` (L) per `(layer, seq_id)` at every real
   reuse and after every completion reorder. `identity_enabled` is set per
   `_run_once`, so the tiny timed runs do not enter the identity path with an
   empty expected map.
3. **Resource guards.** Every step checks the NVML sampler breach and
   `torch.cuda.max_memory_allocated()` against the 16 GiB limit; each run also
   checks physical free/allocated/reserved at its boundary. `_compute_apps` is
   filtered with `nvidia-smi -i <selected UUID>`, never all GPUs.
4. **Finite logits.** The untimed partial-completion preflight wraps
   `model_runner.run_model` with `torch.isfinite(logits).all()` and aborts on the
   first non-finite call, reporting the call count. The hook is removed before the
   tiny timing sanity runs. Matrix/profile timing has no per-step `.item()` sync;
   a device-side finite flag is not claimed for primary rows.
5. **Profile / ledger.** `cmd_profile` writes only `separate-profile.json` (no
   preflight copy), records its measured command wall and per-run walls, and
   enforces the cumulative budget. The ledger sums every retained
   `preflight*.json` attempt, the matrix and the profile, so failed/aborted GPU
   commands are charged.
6. **New-request identity.** Per-sequence store/reuse/reorder counters are
   recorded. Preflight asserts nonzero A and L reuse checks, nonzero reorder
   checks, and at least one newly inserted request that both stored and reused.
7. **Clean guards.** Frozen config, driver, manifest and environment snapshot
   were committed sequentially before the first GPU preflight. No `--allow-dirty`
   path exists in this stage. Raw run files stay outside Git.

8. **Matrix attempt 1 OOM (2026-10-23T03:27Z).** The first matrix attempt
   completed all 30 B1 rows, then OOM'd while constructing the B4 engine
   (19.18 GiB still allocated by PyTorch). The `schedule` recorder closure held a
   direct reference to `model_runner`, so `LLM.exit()` did not actually free the
   B1 model. Fix: store the original bound `schedule`, restore it (dropping the
   recorder closure) before `exit`, then drop every `h` reference, `gc.collect()`
   and `torch.cuda.empty_cache()`. A committed `smoke` command now constructs B1,
   releases it, asserts process allocation <=1024 MiB, then constructs B4, before
   any full rerun. Attempt 1 raw rows/log are retained as `*-attempt1.*` and are
   not used in the 60-row summary.
