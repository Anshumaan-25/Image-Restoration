# Candidate novelty assessment — evidence and limits

Status: **publishable novelty is not established**. The implemented candidate is an experimental training recipe. Completed screening does not support its extra consistency term: C trails B by 0.014584 dB on selected development worst-phase PSNR, and all arms barely exceed JPEG input. [Measured results](results.md). Do not market it as a new restoration architecture or a state-of-the-art method.

## Primary-source comparison

| Prior work | Established overlap | Candidate distinction to investigate | What the source does not establish |
|---|---|---|---|
| [Fu et al., JPEG Artifacts Reduction via Deep Convolutional Sparse Coding, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_JPEG_Artifacts_Reduction_via_Deep_Convolutional_Sparse_Coding_ICCV_2019_paper.pdf) | JPEG restoration and shift-invariant convolutional sparse coding; a compact model-based/learned network. | Our fixed DnCNN training recipe changes JPEG grid origins for identical target content and controls phase exposure with a 64-phase cycle. | Shift-invariant sparse coding is not evidence that the exact two-view loss has already been evaluated, or that our variant is original. |
| [Bai et al., A Regularization-Guided Equivariant Approach for Image Restoration, CVPR 2025 / arXiv:2505.19799](https://arxiv.org/html/2505.19799v1) | Restoration regularization to encourage rotation equivariance, using spatial rotations and cyclic channel shifts of feature maps. | The candidate imposes output agreement across different JPEG compression-grid origins, with clean-target supervision, rather than rotation/channel transformations. | A different transformation alone does not establish a substantial contribution. Equivariance/consistency regularization itself is prior art. |
| [Gu et al., CoDe, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gu_CoDe_An_Explicit_Content_Decoupling_Framework_for_Image_Restoration_CVPR_2024_paper.html) | Image restoration with content decoupling and transform-domain content-consistency loss. | Our unchanged inference model has a pixel-space two-view agreement term and matched phase-sampling controls; no content subnet decomposition. | Sharing the word consistency does not imply the same objective; neither does our combination prove novelty. |
| [Lehtinen et al., Noise2Noise, ICML 2018](https://arxiv.org/abs/1803.04189) | Learning restoration from corrupted observations, supported by statistical assumptions about corruption/targets. | The candidate uses clean supervised targets and agreement across JPEG views. It is not Noise2Noise training. | JPEG quantization errors are not guaranteed independent or zero-mean. Noise2Noise does not prove this agreement term recovers missing details. |

Research queries also covered JPEG grid/phase consistency, stratified sampling and restoration equivariance. These searches are a scoped nearest-prior-art check, not an exhaustive systematic review and not evidence of absence. CVF direct fetches sometimes returned 403; indexed paper text and accessible primary abstracts/full text were used. No benchmark numbers are imported into project result tables.

## Additional JPEG-specific checks during screening

[Jiang et al., Towards Flexible Blind JPEG Artifacts Removal (FBCNN), ICCV 2021](https://arxiv.org/abs/2109.14573) explicitly discusses failure on non-aligned double JPEG images after a one-pixel shift and introduces a double-compression degradation model. Grid misalignment as a restoration problem and training-data augmentation are therefore established prior art. Our current experiment uses single compression from the same PNG reference under two block origins, an unchanged DnCNN, and output agreement with stratified controls. That is a distinction to test, not evidence of publishable novelty; it does not solve the general double-JPEG setting.

[Chen et al., JPEG-Phase-Aware CNN for Steganalysis of JPEG Images](https://ws.binghamton.edu/fridrich/Research/jpeg-phase-aware-Final.pdf) uses JPEG phase for steganalysis. The phase concept itself is established. This is a different task and does not demonstrate that our restoration objective is already known or effective.

## Testable contribution hypothesis

At identical inference topology and training-view/update budgets, phase stratification may reduce sampling variance, and a small agreement term may improve worst-phase fidelity under limited training. Test A=random/pixel, B=stratified/pixel, C=stratified/pixel+agreement. Both views are evaluated together for every arm, with the same clean crops, initialization, seeds, BN microbatch and schedule. Select on the fixed development panel; freeze before protected test.

Primary endpoint: paired image-level difference in worst-phase PSNR. Secondary checks: mean PSNR/SSIM, phase spread, boundary/interior edge error, full-image quality and runtime. Image-grouped uncertainty must not treat correlated phases as independent samples. Three seeds are a minimum variability description, not proof of statistical power.

## Risks and go/no-go

- Agreement can suppress phase-specific detail or favor smoothing. Edge diagnostics and ordinary quality checks are necessary; a higher consistency score alone is not success.
- The approved phase-pair offset of 32 pairs phases separated vertically by four pixels. All phases occur in each cycle, but pair relationships do not cover all possible phase pairs. State this limitation; changing the pairing is a new experimental decision.
- Short learning curves and 16 held-out images cannot establish convergence or general superiority.
- A small JPEG-specific application of known consistency ideas may be too incremental for publication even if it improves scores.

Proceed with a bounded exploratory study only as a candidate-method/ablation investigation. Expand only after empirical feasibility and development results justify it. If benefit is absent or the distinction is insufficient, report a negative or engineering study honestly and request a new research direction. The user's academic novelty requirement remains unresolved until evidence supports it.

## Screening decision

The approved exploratory study has completed; retain its negative result at 1,000 updates. One seed/eight selected development images cannot prove the method always fails. The separately approved [5,000-update baseline diagnostic](baseline_diagnostic.md) now shows useful learning: +0.955 dB primary dev worst-phase / +0.881 dB supplementary dev72. This is an unchanged baseline, not novelty, and cannot be compared as a matched C experiment to the shorter runs. Matched longer-budget controls, seed variability and a meaningful prior-art distinction remain necessary before any successful-method claim. [Next checkpoint](../docs/architecture/research_checkpoint.md).
