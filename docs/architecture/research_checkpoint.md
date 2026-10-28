# Research checkpoint — matched cloud development verified

## Current verified status on 2026-10-25

All three matched cloud arms completed 5,000 CUDA FP16 updates, including separately approved C recovery. Selected checkpoints are A4700, B4400 and C4900. Independent Colab CPU development replay is complete; the downloaded export passed outer/inner checksums, checkpoint/source/config matching, per-image reaggregation and paired image-bootstrap recomputation locally. [Verified cloud results](../../reports/cloud_development_results.md). Every interval includes zero; added consistency has no demonstrated reliable improvement and novelty remains unestablished.

The selections and 16 acquired held-out references are frozen without scoring. Final CPU phase evaluation at Q20/Q10/Q40 is prepared, separately gated and unexecuted. Current Mac Pillow/JPEG versions differ from the study, so use the matching Colab CPU codec. No new training, model download or paid resource is required. Older sections below record historical baseline/cloud planning and do not supersede this status.


Approved v1.0/D01–D10 is unchanged. The user separately approved the 5,000-update baseline diagnostic, which completed and passed its declared continuation criterion. [Measured report](../../reports/baseline_diagnostic.md), [final audit](../../reports/final_audit.md), [approved implementation handoff](implementation_handoff.md).

## What changed in the evidence

At the matched 1,000-update budget, B barely beat input and C trailed B. The longer unchanged B recipe now gains **0.954915 dB** worst-phase and **0.857950 dB** full-image PSNR on the eight selection-development images. Supplementary 72 development sources unused for selection gain **0.881471 dB** worst-phase. Runtime 89.12 minutes, 5,000 successful updates, zero skips; selected step 4,800. No protected-test data, downloads, spending or core-pipeline changes.

The first 100 warmup updates exactly replayed earlier B in this fixed environment. Both update budget and cosine horizon differ afterward; this diagnostic does not isolate a pure duration effect. It shows useful baseline learning without proving a unique optimization cause or convergence. The earlier near-identity result is not a basis for automatically replacing the architecture.

## Current problems and decisions

| Problem | Evidence / classification | Severity / impact | Next action / tradeoff |
|---|---|---|---|
| Candidate benefit at useful budget unknown | Only B has a 5,000-update run; earlier C−B is −0.014584 dB at 1,000 updates. Factual gap. | High: comparing unequal budgets or devices would be misleading. | On the newly requested CUDA/FP16 runtime, rerun A/B/C together before longer-budget method claims. |
| Publishable novelty unestablished | JPEG grid/misalignment and consistency prior art overlap; baseline learning is not a new method. | High: academic novelty requirement remains unmet. | Reassess a meaningful distinction and report matched effects/negative results honestly; a pivot needs reviewed approval. |
| Robustness evidence incomplete | One training seed; primary dev selection bias; supplementary dev72 was planned after interim improvement. Protected tests unscored. | High for confirmatory claims. | Assess seed variation and freeze before separately authorized final testing; do not relabel development as test. |
| Cloud/comparator coverage incomplete | Colab operator reports successful T4 setup, two CUDA FP32 updates and a 10+50-update FP16 profile. Long matched CUDA training, transformer/pretrained comparisons, official SR/SIDD replay and LPIPS remain absent. | Medium for breadth/reproduction. | Use measured CUDA budget before long runs; keep staged comparisons explicit. |

## Ordered handoff

**R01 — complete:** preserve the approved baseline artifacts. Selected weights are `experiments/studies/baseline_learning_v1/B_seed42/best_model.pth`, SHA256 `db32a6373fad66a5d2e86bc6b2cc00ffc8e9935bad3aeb8b8e42ff553c0bdcb9`. Source/config/data/codec identities, RNG/state, 50 validation panels, independent replay/aggregation, exact CLI export and full-image references are verified. Do not resume/retrain it or substitute latest-step weights.

**R02 — current: cloud preparation, then bounded-run approval.** The user requested no further long local training on 2026-10-21. Follow the [Colab runbook](../cloud_runbook.md), [notebook](../../notebooks/free_gpu.ipynb) and thin helper under notebooks. Transfer only the acquired 32 training/eight selection-development PNGs and their immutable manifests. No protected-test payload or old weights. Validate actual CUDA/FP16, then profile 10+50 updates capped at 300 seconds plus setup/save. A target T4 is not guaranteed.

Prepare a **new matched A/B/C CUDA/FP16 study**, seed 42, 5,000 updates per arm, patch128, clean microbatch2/accumulation4. A=random/pixel; B=stratified/pixel; C=stratified/pixel+eta0.05. The operator's actual T4 profile completed 10 warmup + 50 measured updates at **1.267546 seconds/update**, 0.322014 GiB allocated/0.375 GiB reserved, zero skips. Update-only projection is **105.63 minutes per arm / 5.28 hours combined**, excluding validation/log/checkpoint/backup/setup. The old 3,600-second cap is infeasible at this rate and is superseded by the [concrete proposal](../../reports/proposals/colab_matched5000_v1/proposal.json): **9,000 seconds per invocation / 27,000 combined (150 minutes per arm / 7.5 hours total)** plus setup/final I/O. These are proposed soft caps, not completion guarantees. Do not combine old MPS/FP32 B with new CUDA/FP16 A/C. Preserve common GPU/software/codec/settings across the new arms. No architecture change or training-source edit is required. $0, no downloads, no protected scoring. The user explicitly approved this cloud budget on 2026-10-22: [bounded-run approval](../../reports/proposals/colab_matched5000_v1/approval.json). Execution is pending the operator running the prepared Colab cells; approval does not imply a started or completed experiment.

Deliverables: verified private transfer, reviewed profile/plan, A/B/C resolved configs and source/data/environment identities, complete/incomplete summaries, best/latest states/logs, resumable private snapshots and selected-dev reports. Acceptance: matching budgets/settings except approved method fields; actual counts/finite results, saved-checkpoint replay and honest comparisons. Fail/incomplete stops before the next arm. Recovery preserves original horizon/environment and requires reviewing remaining invocation budget; no unlimited retries. Operator-reported cloud setup/QA/profile and profile-report Drive copies have passed; full long-run checkpoint persistence and matched CUDA quality results remain pending.

The earlier [local matched A/C proposal](../../reports/proposals/matched5000_candidate_v1/proposal.json) remains a preserved, **unapproved historical alternative** for the exact old MPS/FP32 environment. It is superseded operationally by the user's cloud preference. Its reuse-B strategy is valid only in that matched local environment and must not be copied into the CUDA plan.

**R03 — conditional, unapproved:** review R02 effect sizes, prior-art distinction and feasible seed variability before a larger study. Seeds/test scope need a concrete new budget. Freeze checkpoint/source/codec/test identities before any protected scoring. A positive development effect is not enough for a publication claim.

Architecture/perceptual/pretrained substitutions remain separately gated. [AGENTS.md](../../AGENTS.md) requires specific approval for additional long/expensive runs and large downloads. The user's subsequent shutdown request authorizes finishing and saving the current work, then normal macOS shutdown; it does not authorize additional training or imply scientific project completion.
