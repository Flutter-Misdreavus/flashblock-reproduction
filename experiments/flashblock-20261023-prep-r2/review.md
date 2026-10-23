# Preparation r2 review — changes required

Supervisor stopped the worker at 2026-10-23T02:01:13Z to consolidate source-review findings into an explicit bounded repair task. Original completion is CANCELLED; its fallback report is FAILED and is preserved unchanged under runs/flashblock-20261023-prep-r2-received. This is not a passed preparation gate.

Accepted: isolated torch 2.4.1+cu121, custom sm86 BF16 head_dim128 kernel built, provenance hash f82cca5e9ac5ebea3e8d15933190b736b7e9fe68bf94ffc90d1fa03b8c4b6a12; five independent no-reuse correctness groups passed, BF16 relative L2 about 0.002. Both complete model weight shards match the pinned official LFS SHA256; small files checked by Git blob identity. Frozen test IDs exist; no heldout evaluation occurred.

Missing: genuine stale-cache reuse and LSE recomposition tests, real model generation, audited numerical/output validation and seed. Both real smoke attempts failed before model loading due to missing xxhash, preserved in smoke-baseline.json and smoke-s2.json. Dependency pin 0d6891a was committed but its installation/result was not verified before cancellation. Source review lists additional specific defects and risks in supervisor-review.txt. No quality or speed claim can be made.

Actual run commits: correctness was executed from 40d9a882 (reported by worker log), results committed a37257d9; fetch completed using 01b348bf; first smoke used 01b348bf. Subsequent dependency pin 0d6891a is not a successful run version. All original logs retained outside Git. Runtime occupies about16GiB, disk limit respected. GPU usage is not continuously metered; charge a conservative0.08GPUh for r2 probes/correctness/failed import attempts, plus prep1 0.02GPUh. Next repair gets at most0.40GPUh; whole campaign remains8GPUh. Wall budget still ends at02:59:38Z, not reset.

Decision: authorize one final bounded preparation repair stage within remaining original wall/resource limits; at most two repair iterations in that stage, then stop and report any blocker. No new scientific scope, no test evaluation, no environment reinstall. Gate review precedes mechanism stage.
