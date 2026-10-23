# GPT pre-GPU driver static review

Resolve before smoke/test, along with scorer review. No heldout outputs have been inspected.

1. `_install_finite_hook` currently increments nonfinite_calls but does not raise. Make nonfinite logits an immediate hard exception and ensure generate cannot finishSUCCEEDED with nonfinite audit. Persist partial output/failure but never score it as valid. `_run_batch` catches exceptions; caller must stop (it already checksFAILED).
2. Budget check currently only before each group and campaign_cap doesn't subtract prior1.90h. Enforce actual remaining min(stage5h,8h-prior1.90h-reserve1.10h) and live monotonic guard inside prefill/decode steps. Cache priorcommandwall at commandstart; don't reread fullledger everytoken. A4096-token group can exceed the120s margin. Check resource boundary peaks/free too, in addition to inheritedperstepguard.
3. Actual executing source is loaded at commandstart. Freeze executionSHA once after clean guard and use that in EVERYgroup, not `_git_head()` pergroup while CPU/report commits may happen during a long inference process. LaterHEAD is separate metadata, not executingversion.
4. Smoke check asks `counters['need_update_rows']` but accepted efficiency helper emits `update_rows`. Correct to actual key, otherwise falsefailure. Keep s2reuse/nonzero and densezero tests.
5. EOS-only text decoded with skip_special_tokens=False is nonempty string `<|im_end|>` but semantically empty answer. Compute empty_output from decode(skip_special_tokens=True).strip(), preserve rawtext/tokenIDs for scoring/audit; do not retry an empty/wrong answer. Rawinvalid/maskIDs remain stopconditions.
6. Beforegroup assert ignore_eos=False, exactmax4096forheldout, batch_size=max_active4, samefrozenprompt/config. In raw-output resume validation verify each outputhash fromrawtokenIDs, flags/counts/conditionorder/seed/maxlimit/scorer-config hashes, not onlyIDs/prompthash. Do not regenerate any completedgroup duequality.

Use one fresh engine and accepted compaction. Do not alter model/sampler/parenttimingcode. Explicitly commitallfixes/config/tests before GPUexecution. These are staticreview fixes, not GPUattempts or test-informed tuning.
