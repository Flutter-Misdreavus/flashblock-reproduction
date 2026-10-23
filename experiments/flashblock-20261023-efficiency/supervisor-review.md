# GPT efficiency review — accepted with explicit corrections and failures

Decision: accept the fixed60-row timing evidence and cached-slot correctness gate; the primary engineering target was NOT met. Accept raw device-profile kernel durations, reject the executor's broad-name attention attribution and replace it with the CPU-audited classification below. Proceed to the frozen128-item quality comparison without scientific-parameter changes. This is a valid negative efficiency result, not replication of the paper's8B/distilled/A100 table.

## Evidence and identities

Completion event176165f536d3d9dc19db4d1bdfabc014e2db561486a0df7a0b617d9e95b03823; taskSHA ff83981bc2d40d59edf07547ba946939dee138b5b64d32f39e4fb9a26634caae. All26 received files independently rehashed against completion manifest. ReportSHA154611e8f4e5db372ebee451d8cf059b7bd9463edb37da22a1bf4724c651c799. Verified committed paired-record file exactly matches manifest-received file, and final first30B1 rows equal preserved original30 records byte-equivalently at the parsed-record level. Raw evidence and logs remain under ignored runs.

Actual accepted preflight SHA6131651c22abf5b0b149a139e851dd483bbe698b; B1 timed SHA0d33c9340c1971b6f44e9d9eb1b8f192997edbcb; B4 timed SHA5d6fb45146dfb1fa8290645b68babc7c9dba7235; valid profile SHAa13e5b0953623e6a95f307d4a5160c8964579dec. Executor final HEAD e52a7ba9 is later than these runs. Independent analysis code was committed as94fd748 before execution; see figures/independent-review.json for fullSHA/script/input hashes. Later supervisor review/task commit is not a GPU execution version. Relay push is performed after this checkpoint; consult branch/remote state for final HEAD.

## Correctness and fixed-work audit

Preflight forced6requests into4slots with partial completions/replacements, limits16/32/48/64/16/32. Both A and L independently matched last-stored state by(layer,seq_id):12reuse checks each,24survivor-reorder checks,zeroidentitymismatch. New request6 actually stored then reused. Untimed short preflight44logit tensors finite. Separate exactB4L8192 warmups173dense/175s2calls finite. Primary timed rows do NOT audit every logit; they validate rawIDs/counts/masks and resources. Do not infer finiteness merely from validIDs.

All60rows in frozen six-setting, five-pair AB/BA order; each outputs exactly B*256 tokens, all problem lists empty. Same condition/setting has one outputhash across5repeats. Distinct first256token prefixes avoid accidental4-way sharedprefix allocation. Scientific controls/config sections are unchanged across B1/B4 execution commits; later config change affects profile attribution only. Sequential prefill applied identically to bothconditions; modelinit/tokenization/detokenization are excluded.

## Independent results

| Batch | Initial tokens/sequence | Mean paired dense/s2 decode time |
|---:|---:|---:|
|1|1024|0.928984|
|1|4096|1.045774|
|1|8192|0.681454|
|4|1024|1.097492|
|4|4096|0.925229|
|4|8192|0.966106|

Primary mean0.966106382862414, median0.982553057790933. Independent exhaustive paired-bootstrap percentile95%CI[0.937687803042057,0.988809876837506], enumerating all5^5=3125resamples; differs slightly from executor's frozen10k MonteCarlo CI method, both retained. All six point estimates agree with executor to1e-12. CI is descriptive for these five pairs on a sharedGPU, not a guarantee about exclusiveGPU performance/general workloads.

Primary mean decode13.894929s dense vs14.389241s s2; aggregate73.695951 vs71.164280tokens/s. Prefill8.742815 vs8.779370s; prefill+decode22.637744 vs23.168612s. This end-to-end definition is pretokenized requests only. Actuals2reuse32/348denoisingrows=9.1954% perrun, or32/604=5.2980% including256savingrows. Dense334denoisingrows/169batcheddecodecalls; s2348/171. s2actual A/Lstores279perrun. Trajectories and number of steps differ despite fixed token work; this is part of the algorithm's observed end-to-end cost. Different context lengths use different frozen training-text workloads, so the curve is descriptive, not an isolated causal effect of length.

## Profile correction

Executor pattern `flash` incorrectly includes `flashinfer::sampling::TopPRenormProbKernel` and other sampling kernels. Complete43/51kernel lists permit exact namespace separation: inspected `void flash::` entries are custom attention (including current-block FP32 and A/L merge); `flashinfer::sampling::` is sampling; remaining device work stays other. Raw artifacts remain unchanged; corrected tables/figures are in figures/independent-review.json and efficiency-profile.*.

Corrected dense attention3.200940s, FlashInfer sampling1.666355s, other2.801989s, totaldevice7.669283s. s2 attention3.839126s, sampling1.741680s, other2.857493s,total8.438299s. Corrected attention fractions41.737%/45.496%, NOT executor63.465%/66.137%. These are single separate instrumented profiles and device-duration sums, not walltimes or evidence of an intrinsic microkernel speedup. Low actualreuse and altered trajectory/caching work are plausible explanations for limited benefit; do not claim a causal decomposition from oneprofilepair.

## Failures, limitations and resource accounting

Firstpreflight falsely rejected addedEOS token151645 because tokenizer.vocab_size151643 excludes addedtokens; fixed bound uses len(tokenizer)=151670. Dirty-worktree guard refused a subsequent launch; no GPU work there. Both retained.

After30validB1rows, second-engine construction OOMed because recorder closure retained old ModelRunner: about19.51GiB processmemory/19.18GiB torchallocated, free7.12MiB. This violated the16GiB/3GiB resource constraint; no claim of whole-stage resource compliance. Fixed closure teardown, committed before release smoke; both B1/B4 released to168.1MiB allocated/free19451MiB. Supervisor stopped an unintended full-matrix restart (only exact ownPID2243396), preserved it, and required B4-only continuation. The original30rows were not replaced. B4 continuation and accepted timedrows fit budgets: maxallocated14768.6MiB, minimumfree4723.1MiB. Existing otherGPU jobs/processes were not targeted.

Firstprofile exported emptytables under legacyPyTorch property but saidSUCCEEDED: attribution rejected, original preserved. One committed profile-only extraction repair used self_device_time_total; extra diagnostic retry beyond two generative repaircycles disclosed. No primaryrow rerun. Full initialpreflight JSON was later replaced with compactcopy during artifactfinalization; identity/logit/timing/resource summaries remain, omitted per-sample fields are not claimed recovered. Manifest records this limitation.

Worker final commandledger2004.418s includes bothprofiles/allfailedcommands, but abortedrestart30s is approximate, and profile livebudgetcheck omitted B4time. GPT uses entire wrapper2968.607s rounded upward to0.90GPUh as a conservative charge, including CPU/idle/failures; this is not measuredGPUbusytime. Campaign prior1.00 +0.90 =1.90GPUh charged. Quality cap5.00h leaves1.10h reserve under8h. Detailed supervisor ledger is authoritative for next task.

## Next task

Original model, tau2 and sampler unchanged. Fixed128GSM8K test IDs, ordinaryEOS/max4096, identical original chat template, scorer/authored tests frozen before heldout generation. Onefreshengine, measured preconstruction headroom, explicit raw-ID/finite checks and complete per-item persistence. No test tuning or inference-speed claim from diagnostic quality timings. Task is independently frozen in ../flashblock-20261023-quality/.
