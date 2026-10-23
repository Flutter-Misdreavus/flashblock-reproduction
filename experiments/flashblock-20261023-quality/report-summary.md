# Quality stage summary — flashblock-20261023-quality

Status: **SUCCEEDED as an execution** (all 256 fixed outputs generated once and
scored). This is an exploratory single-seed, fixed-128-item comparison, not a
paper-table replication.

## Frozen question and workload

Original TraDo-4B-Instruct BF16 (`d350259d64abf9219362ccb1435d68910744eea7`),
GSM8K main test revision `740312add88f781978c0658806c59bc2815b9866`, exactly
128 heldout test row IDs frozen by `sorted(random.Random(42).sample(range(1319),
128))` (`d599ce5b…`). One sample per item per condition, original zero-shot chat
template, ordinary EOS, `max_new_tokens=4096`, block4 / denoise8 /
dynamic0.9 / temp1 / top_p1 / top_k0 / seed42 / consistent sampling 1.
32 stable groups of four sorted IDs; even group index dense→s2, odd s2→dense;
seed and all request/prefix-cache state reset before every group-condition.
One engine for the whole run; batch/max_active 4, TP1, eager; sequential
prefill identical in both conditions; global GPU memory target 0.72
(resource-only adjustment, see deviations). No training, quantization, prompt
tuning, tau/threshold tuning or test-driven changes.

## Primary result

| condition | correct | EM |
|---|---:|---:|
| dense | 118/128 | 92.1875% |
| s2/tau2 | 119/128 | 92.96875% |

Paired numeric exact-match difference (s2 − dense): **+0.78125 percentage
points**. Paired item bootstrap (10,000 resamples, seed 20261023) 95% CI
**[-3.90625, +5.46875] pp**; the exact 128-fold convolution CI is identical.
Discordance: both correct 114, dense-only 4, s2-only 5, neither 5. Exact
McNemar p=1.0 (descriptive only). The frozen `>5pp loss` continuation warning
is **not** triggered.

This does **not** establish equivalence or a genuine quality improvement:
the CI is wide, the analysis is one seed and 128 items, and the small positive
point estimate is exploratory. Negative and positive results are retained as
observed.

## Output validity and extraction

All 256 rows have valid raw IDs (no mask token, no out-of-vocabulary ID), no
truncation, and all 128 outputs per condition ended by ordinary EOS. Generated
lengths: dense mean 223.95, median 203.5, p95 410.65, max 501; s2 mean 227.49,
median 208.5, p95 417.6, max 577. No empty outputs. One strict-box extraction
failure occurred in each condition (the same item, index 206; the last complete
box was not numeric, so it was counted incorrect rather than falling back to an
earlier box). One s2 output (index 788) used the explicitly flagged
last-number fallback and was correct. No gold-aware candidate selection and no
post-hoc answer-aware extraction were used.

## Resources

One engine; generation wall **1395.492 s**. All charged GPU-command wall
(environment probe, three smoke attempts, generation) is **1491.636 s =
0.4143 GPU h**; campaign charge is 1.90 + 0.4143 = **2.3143 h** of the 8 h cap,
leaving the 1.10 h reserve. Peak torch allocated **11,963.6 MiB** (<16,384);
minimum sampled physical free **7,679.1 MiB** (>3,072); KV capacity 93 blocks
(4 calc, 89 free data) covers the worst frozen group need of 68 data pages. No
resource-guard breach. 15,086 audited `run_model` calls with 0 nonfinite
logits; all 291 parameter tensors (4,411,424,256 elements) finite. Engine
released to 169.1 MiB allocated after the run.

## Provenance, deviations and failures

Actual execution commit for all 64 group-condition calls:
`0b39fb9253818ac30e651230564e24bc89187bc1`; config hash
`85bc177814968b49…`; manifest hash `c4bb9f1e…`; scorer hash `7f9225c8…`.
No mixed execution SHA.

- **Smoke attempt 1 failed** at `e9ca4f37411f5078a3c8d5948418c0589b6e5e84`
  after 10.307 s: physical free reached 2,469 MiB (<3,072) during validation,
  before any generated row was counted. The own reserved excess (17,202 MiB)
  was driven primarily by whole-parameter finite-scan temporaries retained by
  the caching allocator, not by an evidenced co-tenant increase. The attempt
  and its secondary `reset with live requests` consequence are preserved; no
  rows from it are used.
- **Smoke attempt 2 passed** at `0373d211…` (41.44 s) with the pre-chunking
  scan; preserved as evidence but superseded.
- **Smoke attempt 3 passed** at `fcab64fb…` (42.28 s) with the final chunked
  scan: s2 reuse 40/251 decode rows, dense reuse 0, no resource breach,
  minimum free 7,739 MiB. This is the gate before heldout generation.
- A CPU freeze attempt before any GPU work failed on a prompt placeholder
  rendering bug; it was fixed and committed, then the manifest was re-frozen.
- Resource-only deviations: global GPU memory target 0.80 → 0.72; full-tensor
  finite scan replaced by the proven 1 Mi-element chunked scan; sequential
  prefill retained identically both conditions. No scientific control, ID,
  template, scorer, seed or max-token parameter changed.
- No heldout output existed when the scorer/tests/config/driver were committed;
  no test-driven tuning was performed.

## Independent check

A separate re-score directly from the raw token IDs/text reproduced all 256
per-item labels, every output SHA-256, group/condition order, the 118/119
counts, the +0.78125 pp delta, and both bootstrap intervals. Raw generated text
and token IDs remain outside Git under `runs/flashblock-20261023-quality/raw/`
(64 bounded group files, 1,180,575 bytes total) with hashes in
`raw-output-manifest.json`; the compact `raw-generated-outputs.json`
(586,715 bytes, 256 rows, no question/source solution) also remains in the
ignored run directory for a GPT hash-verified read-only scoring audit.

## Decision and limitations

Fixed 128 items and one seed on a shared RTX3090, original undistilled model.
No truncation and no >5pp loss were observed, but the paired CI is consistent
with both a small loss and a small gain. Do not claim equivalence, significant
improvement or reproduction of the paper's distilled-model table. GPT should
independently recompute the paired statistics from the committed per-item rows
and decide whether a new, separately frozen multi-seed confirmation is
warranted. No further evaluation or tuning is performed in this run.
