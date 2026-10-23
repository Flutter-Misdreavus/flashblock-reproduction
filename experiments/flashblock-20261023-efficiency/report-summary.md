# Efficiency stage summary — flashblock-20261023-efficiency

Status: **SUCCEEDED as an execution** (frozen 60-row matrix completed, all raw rows
retained). **The frozen `>=1.10` primary engineering target was NOT met.**

## Frozen question and workload

Original TraDo-4B-Instruct BF16 (`d350259d`), one shared RTX3090
(`GPU-REDACTED`), block4 / denoise8 / dynamic0.9 / temp1 / top_p1 / top_k0 /
consistent sampling, seed42 reset per run, exactly 256 new tokens per request,
`ignore_eos=True` only for this performance workload. Six settings: B1/B4 ×
per-sequence initial token length 1024/4096/8192. Dense (ratio 0) vs official
s2/tau2 cached A/L reuse. Five paired interleaved rounds per setting, frozen
order AB,BA,AB,BA,AB, after a dense and an s2 warmup at the exact shape.
Sequential prefill (add one request, run only its PREFILL, add next, defer
decode) is used identically both conditions because the scheduler does not
enforce `max_num_batched_tokens`. Fastest-rate claim is decoding only for
pretokenized requests; model init, tokenization and detokenization are excluded.

## Primary result (B4L8192)

Mean of five paired `dense_decode_seconds / s2_decode_seconds` ratios:

**0.9661**, descriptive paired-bootstrap 95% CI **[0.9377, 0.9888]**
(10,000 resamples, seed 20261023). Median 0.9826. Target `>=1.10`: **not met**.

| round | dense decode s | s2 decode s | ratio |
|---:|---:|---:|---:|
| 1 | 14.472 | 14.531 | 0.996 |
| 2 | 13.764 | 14.496 | 0.949 |
| 3 | 13.924 | 14.167 | 0.983 |
| 4 | 13.637 | 14.847 | 0.918 |
| 5 | 13.681 | 13.894 | 0.985 |

Dense aggregate 73.70 decode tokens/s; s2 71.16. B4L8192 reuse fraction
**160/3020 = 5.30%** of decode rows. Mean denoise steps 334 (dense) vs 348 (s2);
the small step excess is retained, not normalised away.

## All six settings (5 paired rounds each)

| setting | dense decode s | s2 decode s | mean ratio | 95% CI | dense steps | s2 steps | s2 reuse frac |
|---|---:|---:|---:|---|---:|---:|---:|
| B1L1024 | 15.739 | 16.946 | 0.9290 | 0.9172–0.9404 | 123 | 133 | 0.1777 |
| B1L4096 | 18.693 | 17.880 | 1.0458 | 1.0264–1.0652 | 155 | 145 | 0.2153 |
| B1L8192 | 14.403 | 21.135 | 0.6815 | 0.6769–0.6879 | 110 | 178 | 0.2645 |
| B4L1024 | 17.456 | 15.908 | 1.0975 | 1.0684–1.1376 | 434 | 405 | 0.1104 |
| B4L4096 | 15.478 | 16.726 | 0.9252 | 0.8960–0.9553 | 374 | 418 | 0.1246 |
| B4L8192 | 13.895 | 14.389 | 0.9661 | 0.9377–0.9888 | 334 | 348 | 0.0530 |

There is no monotonic speedup with context. B4L1024 is the only setting near the
target (1.0975); B1L8192 is strongly slower because the s2 trajectory takes 68
more denoise steps. Prefill means at B4L8192 are 8.743 s (dense) / 8.779 s (s2);
prefill+decode E2E means 22.638 s / 23.169 s. Step-count differences are part of
the measured trajectory, not a speedup adjustment.

## Separate profile (B4L8192, one per condition, after untimed warmups)

Profiled decode wall 18.555 s dense / 19.678 s s2, i.e. **1.335× / 1.368×** the
unprofiled matrix means (profiler perturbation disclosed). Total CUDA kernel time
was 7.669 s / 8.438 s; kernel-name-classified attention 4.867 s (63.5%) /
5.581 s (66.1%). The s2 profile contains both
`flash_fwd_splitkv_hist_only` and ordinary `flash_fwd_splitkv` kernels.
`flashinfer::sampling::TopPRenormProb` is ~1.66/1.74 s and is kept in
other/ambiguous, not called attention. The complete 43/51-kernel tables are in
`separate-profile.json`; classification is approximate.

## Memory, load and budget

- Peak torch allocated 14768.6 MiB (limit 16384); peak reserved 14946.0 MiB.
- Minimum sampled physical free 4723.1 MiB (limit >=3072); total GPU 24576 MiB.
  Other users' processes were present throughout (about 4.1–5.1 GiB at boundaries).
- Overall GPU utilization sampled mean 81.95% (matrix), max 100%; shared-load
  noise is retained and the paired CI is descriptive, not a clean-room claim.
- Conservative GPU-command wall including every retained failure/retry:
  **2004.418 s = 0.5568 GPU h** this stage; campaign upper charge
  **1.5568 GPU h** of 8 h. Wall limit 7200 s; stage 1.5 h limit not exceeded.

## Preflight / identity

Passed at commit `6131651c`: 6 train requests (100–105, max tokens
16/32/48/64/16/32), batch/max_active 4, real partial completion and new-request
insertion. 12 A and 12 L reuse checks, 24 reorder checks, all maximum absolute
difference **0.0**; newly inserted seq 6 both stored and reused. 44 finite-logit
calls, 0 nonfinite. Tiny timing sanity dense 1.575 s / s2 1.686 s (16 tokens).
KV 146 blocks, 142 data blocks available, 132 needed for 4×33 pages.

## Provenance, deviations and failures (all retained)

- Retained **B1** rows executed at `0d33c9340c1971b6f44e9d9eb1b8f192997edbcb`;
  the attempt then OOM'd constructing B4 because a recorder closure kept the B1
  model alive (own 19.18 GiB, 7.12 MiB physical free). The 30 B1 rows were kept
  unchanged, with pre/post SHA-256
  `d8669124677b9a6776ee800e6f632abf8c859214f0201b5d157f27b9c1dd066b`.
- **B4** rows executed by strict B4-only resume at
  `5d6fb45146dfb1fa8290645b68babc7c9dba7235`; B1 was not rerun. The final
  60-row file therefore intentionally has mixed provenance (30 B1 + 30 B4).
- An unintended full restart at `6d5ac288` was cancelled by root after ~30 s;
  no rows from it are used. It is charged and logged.
- First preflight failed at `fd94c13f` on a false out-of-vocabulary check
  (`tokenizer.vocab_size` vs `len(tokenizer)`); preserved, then fixed.
- Profile attempt 1 at `5d6fb451` produced an empty kernel table because torch
  2.4 exposes `self_device_time_total` while `self_cuda_time_total` is `None`;
  preserved as `profile-attempt1-empty.json`, then fixed and rerun at `a13e5b09`.
- Primary timed rows do **not** audit every logit; they validate output counts,
  IDs, masks and boundary resources. The real finite-logit audit covers the
  untimed preflight (44 calls) and the two profile warmups (173/175 calls), not
  every timed repetition.

## Decision

The frozen engineering continuation target is not met by this measurement. This
is a valid negative efficiency finding and should go to GPT review together with
the quality question; no heldout QA was run. GPT should independently recompute
the paired statistics from all 60 rows and inspect the mixed-commit provenance.

## Finalization notes demanded by the last review

- Both profile attempts are charged: empty attempt `5d6fb451` 259.130 s and valid
  retry `a13e5b09` 252.625 s. The retry is a profile-analysis-only repair beyond
  the two generative repair cycles; it did not touch any primary timing row.
- The profile command's live cumulative check used the default
  `_prior_gpu_wall()` that excludes `matrix.json`, so it did not see the B4 matrix
  wall. The final ledger includes every command. No claim is made that the live
  profile check enforced the full cumulative budget.
- Evidence hashes for the retained attempt files, current matrix/profile and
  preflight attempts are in `evidence-manifest.json`.
