# ADR 0001 — Fidelity-first staged restoration system

Status: **Accepted**, 2026-10-20. Authority: [approval record](../approval_record.md), [approved design v1.0](../approved_design.md). The prior plan mixed tasks/models before data/evaluation and exceeded a free-compute budget; the [review](../critical_review.md) records evidence and alternatives.

| Decision | Accepted approach / alternative | Cost and fidelity implications | Validation / failures |
|---|---|---|---|
| D01 | JPEG-first study; other tasks staged / train all families | Less compute, narrower honest conclusions | Executable core; novelty may remain unmet |
| D02 | DnCNN/EDSR baselines; NAFNet optional / transformers only | Lower-cost controls | Task/data parity; label L1 adaptations |
| D03 | Defer LDM / required generative pipeline | Avoid sampling/autoencoder cost and provenance conflict | New provenance/fidelity approval before addition |
| D04 | L1 default, optional perceptual / mandatory composite | Lower baseline memory; visible perception tradeoffs | Ablation and frozen-weight gradient tests |
| D05 | Explicit tensor domains / universal [0,1] | Correct unclipped noise and training gradients | Shape/range/metric-boundary tests |
| D06 | Current CUDA AMP, CPU/MPS FP32 / legacy AMP everywhere | Supported paths; possible MPS throughput loss | Device/numerical checks; no hidden fallback |
| D07 | Named RGB JPEG codec/phase / undefined artifacts | Reproducible task | Codec fixtures; source-JPEG confound |
| D08 | Optional phase consistency / pixel-only sampling | Two-view training, one-pass inference | A/B/C, both-branch gradients; oversmoothing/prior-art risk |
| D09 | Project metrics + separate replay / library defaults | Small CPU cost, meaningful comparisons | Reference parity, correct infinity/aggregation |
| D10 | Minimal single-device ports/profiling/resume / full frameworks | Maintainability, no pooled memory | Measured profile and resume integration |

An existing supported Python/PyTorch pair may be reused after verification. Offline tests use stdlib unittest (pytest-compatible) to avoid unnecessary downloads. These are provisioning choices, not model changes. No adversarial-provenance exception, dataset/weight acquisition, expensive run or novelty claim is approved here.
