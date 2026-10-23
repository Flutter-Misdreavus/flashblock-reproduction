# Mechanism stage summary — flashblock-20261023-mechanism

Status: **SUCCEEDED** for the frozen mechanism stage. Formal execution commit
`4f4490e9b8a9dd3c32413cc35f034eb21f81626d`; accepted preflight execution commit
`ecb4974a6ad1245c875b0142e43551c2ee99da07`. No heldout/test evaluation, no
training, no quantization, no threshold tuning, no speed or quality claim.

## Primary dense-trajectory result (16 fixed GSM8K train prompts, seed 42)

- Mean paired adjacent-step **A_out cosine minus A_in cosine gap = 0.2201**,
  prompt-bootstrap 95% CI **[0.2051, 0.2364]** (10,000 prompt resamples).
- Mean A_out cosine **0.8998** (95% CI 0.8942–0.9049); mean A_in cosine
  **0.6797** (95% CI 0.6586–0.6991).
- **16/16** selected layer-head means are positive; the layer-head range is
  **0.0574–0.6010**, so heterogeneity is large and retained, not averaged away.
- Changed-input-row stratum gap **0.2701**; unchanged-row stratum gap **0.1822**.
- Mean aligned external-LSE absolute drift **0.6554 nats**. These LSE and
  composed-output diagnostics accompany cosine; cosine alone is not used to infer
  exactness or answer quality.
- Fresh-reference control: actual dense attention vs independent FP32 full
  reference mean relL2 **0.00167–0.00176** per prompt (BF16 rounding scale).

## Official s2/tau2 reuse error on its own trajectory

- 265 genuine reuse steps and 955 actual history-A/L store steps across the 16
  prompts; cache age at reuse is 1–2 denoising steps.
- Actual s2 output vs fresh full reference on reuse steps: mean relL2 **0.2430**
  (95% CI 0.2323–0.2550).
- Independent stale-A/L merge vs fresh full reference: mean relL2 **0.2426**;
  the same stale merge vs actual model output: mean relL2 **0.00192**, showing
  the diagnostic composition reproduces the model's actual merged output.
- On refresh steps, actual output vs fresh full reference: **0.00168**.
- Hard cache-label identity control (stored history A/L vs independent external
  reference) max relL2 **0.00333**, max |dLSE| **9.5e-6**, both below the frozen
  hard limits (0.01 / 0.01). No hard-stop breach, no non-finite logits, no mask
  or out-of-vocabulary output.

## Preflight

Accepted preflight attempt 2 (commit `ecb4974a`, wall 37.2 s): prefill block-causal
GQA parity relL2 0.002414/max abs 0.009951 (limits 0.01/0.05); paged KV scatter
bitwise exact with negative-slot skip; FP32 split/recombine 1.5e-7 relL2; history
page mapping exact; no-hook vs hook same-seed s2 train-100 max-16 token IDs
**identical**; hook counts 13/13/13/13 for layers 0/2/17/35; 512 aligned
observations; no capture errors.

## Resources (conservative whole-stage ledger)

- Formal dense 240.4 s + formal s2 224.8 s = 465.2 s.
- Accepted preflight 37.2 s.
- Unaccepted development probes charged 130 s (see `development-probes.json`).
- Total stage wall 632.4 s = **0.1757 GPU h**; campaign charge after this stage
  **0.5757 GPU h** (limit 8 h; stage limit 1 h).
- Peak allocated 11068.6 MiB (limit 16384), minimum sampled physical free
  8559.1 MiB (limit >=3072). No other-process modification.

## Decision and limitations

The frozen decision rule is satisfied descriptively on this small subset: the mean
paired gap is positive and the bootstrap CI is entirely positive, with all 16
selected layer-head means positive but strongly heterogeneous. Official s2 reuse
has a large measurable approximation error on its own trajectory. This is **not**
evidence of exactness, answer-quality preservation, or efficiency; only 16 train
prompts, one seed and one frozen sampler were used, with no heldout test. GPT should
decide whether to retain the mechanism claim and design the next gate; no further
stage is authorized by this worker.

## Evidence layout

- Git-committed small evidence: `experiments/flashblock-20261023-mechanism/`
  (`preflight.json`, `preflight-review.md`, `mechanism-summary.json`,
  `mechanism-per-prompt.json`, `bounded-small-trace.json`, `resource-ledger.json`,
  `commands-and-provenance.json`, `mechanism-figure-data.csv`, `report-body.json`,
  `run-commits.json`, `development-probes.json`, `supervisor-*.md`).
- Raw generations, per-step traces and subprocess logs: `runs/flashblock-20261023-mechanism/`
  (outside Git).
- `matplotlib` is absent; `mechanism-figure-data.csv` is the exact plotting data
  for the common-scale A_out/A_in cosine figure and LSE/error panels.
- Version control: branch `exp/flashblock-20261023`, repository
  `Flutter-Misdreavus/flashblock-reproduction` (private), relay push by GPT;
  executor push status pending.
