# Research completion boundary — 2026-10-26

The user requires a new method or publishable novelty. Engineering completion, successful pretrained inference and favorable point estimates do not establish that requirement. This document does not redefine the objective.

## What the evidence supports

The frozen JPEG study is complete. It finds a small Q20 phase-sampling improvement conditional on one seed and sixteen test sources; the extra consistency term is not supported relative to stratification across the three tested qualities. The full-image check also gives no clear extra benefit from consistency. Earlier shift-invariant JPEG and restoration-consistency prior art is recorded in `reports/novelty_assessment.md`. Additional matched seeds could assess reproducibility but cannot by themselves make a known objective original.

The broader extension adds real approved architectures, official pretrained weights and operator-specific validation. These results are engineering/model comparisons, not a newly trained architecture or an original method. A completed SIDD public-validation replay would also remain an evaluation, not a hidden benchmark submission or novelty proof.

## Remaining concrete gaps

1. Genuine SIDD validation data is inaccessible from the specifically approved authors' Drive mirror due provider quota. Original-host metadata access timed out. Import and model code are implemented/tested; synthetic paired fixtures are not a substitute for real data.
2. A dedicated project-trained Poisson denoiser has not been trained on natural images. A mixed-level L1 run has a concrete configuration, correct photon-level sampling, source-disjoint data and fixed clean dev crops. Actual CUDA profiling and specific long-run approval precede execution.
3. Research novelty remains unestablished. A major new backbone/loss/pairing design needs evidence, a new research decision and a fresh held-out protocol. The already inspected JPEG test16 must not become a tuning set.

## Recommendation

Finish actual task coverage and independent reproducibility first; retain the negative consistency result. Do not automatically train more transformers from scratch on a small free-GPU budget, add generative components, or claim novelty from a new name. A publishable-new-method branch must begin with a focused primary-source review and a falsifiable development-only hypothesis, with any architectural change separately reviewed before implementation. No scientifically justified new architecture is ready to be labeled a successful original contribution today.

There is no honest numerical "100%" acceptance condition for an uncertain research outcome. The project must report the unresolved academic requirement until evidence or an explicit change in requirements resolves it. This is a limitation of the current results, not permission to fabricate, relabel comparisons or recycle test results into method selection.
