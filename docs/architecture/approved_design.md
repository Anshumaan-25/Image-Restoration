# Approved architecture v1.0

Status: **APPROVED v1.0**. Approved 2026-10-20, Asia/Kolkata.

Authority: the user explicitly approved D01–D10. See [approval record](approval_record.md), [accepted ADR](adr/0001-approved-restoration-design.md), and [implementation handoff](implementation_handoff.md). The original proposal is preserved as review history. Design approval permits implementation; data/weight downloads and expensive runs retain their separate gates.

Evidence and alternatives: [critical review](critical_review.md). Execution order and acceptance gates: [proposed roadmap](proposed_roadmap.md).

## Scope and feasibility

Confirmed constraints: submission this week (exact timestamp unconfirmed), $0 spending, local M5/16 GB for development and light inference, one free CUDA GPU when allocated, 128x128 training targets for initial experiments, small microbatches/accumulation, and separate reporting of approved pretrained evaluation. The user requires a new method or publishable novelty.

Recommendation: build a small shared restoration system and **one focused JPEG research study** this week. Add SR and real-denoising adapters and evaluate approved pretrained weights as resources permit. Keep AWGN/Poisson and SR x2/x4 as explicit supported-task milestones; full training across all tasks is later work, not a claimed week-one result. D01 explicitly accepts this staged scope.

A reproducible pilot, a candidate method and an ablation report are feasible targets. Publishable novelty and successful results are not guaranteed. If the academic requirement demands established novelty by submission, the present evidence does not establish that requirement can be met. Do not rename a known augmentation or consistency loss as an original contribution.

## System design

```text
manifest + task config -> dataset / named degradation -> RGB training batch
                                                     |
                                                task-specific model
                                                     |
                              pixel loss (+ approved experimental terms)
                                                     |
                                             trainer / checkpoint

manifest + checkpoint -> inference -> fixed metric/export protocol -> report
```

Each run has one task, one model, one scale and one degradation profile. There is no Restormer/SwinIR/diffusion cascade, no automatic task detector, and no universal jointly trained restoration network. A factory with explicit supported combinations is enough; no plugin framework, distributed trainer or full upstream training framework is needed.

Essential: data manifests; deterministic validation; PSNR/SSIM; identity/bicubic baselines; one small learned baseline; fail-fast tests; resumable single-device trainer; inference and honest reports. Transformers are comparators added after correctness. LPIPS, perceptual-loss experiments, BM3D and NAFNet are optional additions. Diffusion and custom SR Restormer are deferred.

## Model-task compatibility

All public forward methods take one floating RGB BCHW tensor and return BCHW. Scale is fixed at construction, never inferred from tensor dimensions. Different tasks/scales use different checkpoints.

| Model/config | Inputs -> outputs | Supported role | Conditioning, size and limitations |
|---|---|---|---|
| identity | Bx3xHxW -> same | Denoising/JPEG input-quality reference | No learned parameters. |
| bicubic | Bx3xHxW -> Bx3xsHxsW | SR x2/x4 reference | Named interpolation implementation; input is already LR. |
| dncnn_rgb_l1 | Bx3xHxW -> same | Essential learned AWGN baseline; separate SIDD, Poisson and JPEG adaptations | 20 convolutions, width 64: first Conv3x3/ReLU, 18 Conv3x3/BatchNorm/ReLU blocks, final Conv3x3 to RGB; stride 1, padding 1. Output x minus predicted residual; no clamp. L1 training is a project variant, not an exact DnCNN reproduction. |
| edsr_baseline | Bx3xHxW -> Bx3xsHxsW | SR x2/x4 CNN comparison, after the first research pipeline | 16 residual blocks, width 64, two 3x3 convolutions per block, ReLU, no BN, residual scale 1; LR features and pixel shuffle. Internal mean shift matches the pinned source; public range stays normalized. |
| restormer_denoising | Bx3xHxW -> same | Gaussian and SIDD transformer comparison | dim 48; blocks [4,6,6,8]; refinement 4; heads [1,2,4,8]; expansion 2.66; bias false; BiasFree layer norm; dual-pixel false. Pad to multiple 8. Canonical architecture with project-budget training or verified weights. No core SR head. |
| swinir_light_sr | Bx3xHxW -> Bx3xsHxsW | SR x2/x4 transformer comparison | embed_dim 60; depths [6,6,6,6]; heads [6,6,6,6]; window 8; MLP ratio 2; direct pixel shuffle; 1conv residual connection; img_range 1. Separate scale-specific weights. |
| swinir_color_jpeg | Bx3xHxW -> same | Optional established JPEG comparator | Official RGB JPEG variant, window 7 and internal img_range 255; external [0,1] contract. Do not substitute the SR checkpoint. |
| nafnet_width32_l1 | Bx3xHxW -> same | Optional SIDD alternative if Restormer exceeds the profiled budget | Width 32, encoder [2,2,4,8], middle 12, decoder [2,2,2,2]; pad to 16. L1 changes the official recipe. Requires an explicit substitution decision. |
| LDM / custom Restormer-SR | Not specified for core | Deferred | Separate design, provenance and fidelity approval required. |

EDSR architecture details come from its [official implementation](https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/edsr.py). SwinIR task-specific heads/settings come from its [official model](https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py) and [evaluation definitions](https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py). Port only needed licensed architecture code, retain attribution, pin its commit, and test against its reference. No implicit weight downloads. SwinIR real-world adversarially trained variants are excluded; whitelist verified classical/lightweight supervised checkpoints by exact identity.

Dynamic padding: retain original H/W; bottom/right reflection padding to the required multiple, with replication for dimensions too small for legal reflection; unpad outputs to sH/sW. Test H or W equal to 1 and below the window size. Recompute shifted-window masks for actual padded dimensions; do not cache without a bounded key including shape/device/dtype.

## Data and degradation contracts

### Public and internal tensors

- Decoded clean images and targets: RGB float32 [0,1]. Dataset samples are CPU CHW; collation creates BCHW. Model features and perceptual normalization are not subject to C=3 or [0,1].
- Noisy inputs may exceed [0,1] only under an explicitly unclipped noise profile. Model predictions are finite real values without training-time sigmoid/clamp. Clamp at named metric/export boundaries.
- Default file workflow is 8-bit sRGB RGB PNG/JPEG. Apply EXIF orientation consistently; reject grayscale, alpha, CMYK, non-sRGB profiles and higher bit depth with an actionable error unless the user explicitly chooses a documented conversion. Avoid silent bit-depth loss. Raw/Bayer is out of scope.
- PIL is the default I/O layer. If an OpenCV benchmark adapter is enabled, conversions between BGR and RGB must be explicit.

### Manifests and sample contract

JSONL manifest records: id, dataset/version, split, source_id, scene_id when known, relative input_path (nullable for clean-only synthesis), target_path, decoded dimensions, bit_depth, color_space, file SHA256, scale, and degradation/protocol identifiers. Paths resolve against a configured data root. Official block data additionally retains original array index and layout. Validate uniqueness, file existence, matching pairs, scale-aligned dimensions and split disjointness before a run.

Dataset result: `{input: Tensor[3,h,w], target: Tensor[3,s*h,s*w], id: str, metadata: dict}`. Training batches must be shape-compatible. Evaluate variable-sized images one at a time; never pad metric targets into a batch and score the padded pixels. SIDD paths are paired by parsed identifiers, not sorted zip order.

The named two-view JPEG experiment additionally supplies `input_alt: Tensor[3,h,w]` for the same target. Collation preserves both B-sized view tensors; the trainer concatenates them to 2B, runs the unchanged model once, then splits predictions. The ordinary criterion scores concatenated predictions against the target repeated twice; a separate `phase_consistency(pred_a, pred_b) -> scalar_tensor` supplies the optional term. Generic single-view datasets and the public model forward contract remain unchanged.

Randomness is explicit: derive sample/crop/noise keys from run_seed, sample ordinal and view index using a stable documented hash, never Python's randomized hash. Generate noise on CPU for deterministic validation and transfer it with the batch. Training workers do not own scientific randomness. Save the next unconsumed sample ordinal for resume.

### Dataset roles

| Dataset | Proposed use and split policy |
|---|---|
| DIV2K PNG, project research split | Primary JPEG study, avoiding a second JPEG encoding of already-JPEG reference files. Original training IDs 0001–0720 train, 0721–0800 dev; original validation IDs 0801–0900 are the project's held-out JPEG test. Clearly label this repurposed split; it is not the official SR test. Begin with first 32/8/16 IDs per split for screening; expand to the full split for stronger evidence. PNG storage avoids this additional source-codec confound but does not prove photographs are sensor-pristine. |
| DIV2K, literature SR replay | A separate named experiment may train on all 800 official training pairs and use the 100 validation pairs for development, then test on Set5/Set14 or another approved held-out SR benchmark. Never mix results from this split with the JPEG research split. Hidden test HR is not assumed available. |
| BSDS500 RGB | Optional smaller pipeline fixtures with official train/val/test folders. The source photographs include JPEG files; recompression is a double-compression setting and must be labeled. Do not substitute this for the primary JPEG study unnoticed. Use image content, not segmentation labels or metrics. [Official dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html). |
| SIDD-Medium sRGB | Full real-denoising track: measured noisy/clean pairs. Official validation blocks for development, hidden benchmark only if a verified scoring route is usable. Preserve official splits; document scene/content overlap rather than inventing an unseen-scene guarantee. Custom holdouts group parent scene instances and are labeled custom. |
| CBSD68 / Kodak24 | Optional final RGB synthetic-denoising tests after dataset/provenance approval; check overlap against training data. A pretrained model's training-data overlap must also be checked. |

Approval of this design does not authorize specific data/weight downloads. Download manifests, size estimates, licensing/provenance and explicit approval precede acquisition. A no-download dry run uses procedural fixtures and is labeled a pipeline test, not a research experiment.

At review time the [official SIDD page](https://abdokamel.github.io/sidd/) reports its primary evaluation server down and points to Kaggle scoring alternatives. Verify the actual route before scheduling a submission; local public-validation scoring and a hidden-benchmark submission are different deliverables.

### Named operators

1. `awgn_unclipped_v1`: y=x+epsilon*sigma/255, independent standard-normal RGB samples; training sigma uniform [10,50] per image. Fixed validation sigma {15,25,50}, fixed image-derived seeds, no clipping or quantization of noisy inputs. `awgn_clipped_v1` is a separate optional mismatch test, never pooled with it.
2. `poisson_linear_rgb_v1`: apply the documented inverse sRGB transfer to obtain x_lin; sample n~Poisson(p*x_lin), divide by p, clip at the declared sensor-saturation boundary, and convert back with the sRGB transfer. p in {30,60,120}, one level per training sample; evaluate each level separately. This is an approximate RGB shot-noise experiment, not raw sensor simulation or a replacement for SIDD. No Gaussian term unless a new profile defines read-noise units/order.
3. `div2k_bicubic_pairs_v1`: consume official paired LR images. Crop LR at (i,j) and target at (s*i,s*j), with LR crop P/s for target P=128; P divisible by scale. Apply paired geometric transforms. Do not regenerate benchmark LR with a different library. Framework resize for fixtures is `torch_bicubic_fixture_v1`, antialias true, align_corners false, explicitly non-benchmark.
4. `sidd_srgb_pairs_v1`: no added synthetic noise, no rescaling for training crop convenience, synchronized geometric crops. Record MAT key/layout/range for validation blocks; no guessed transpose or division by 255 based only on dtype.
5. `jpeg_rgb_444_v1`: actual Pillow encode/decode, quality in {10,20,40}, subsampling 0 (4:4:4), optimize false, progressive false; record Pillow and linked JPEG library versions. Compress before crop when testing grid-phase effects. No differentiable JPEG approximation in the baseline. This is a project protocol, not automatically equivalent to OpenCV/MATLAB or grayscale Classic5/LIVE1 results.

Random flips/90-degree rotations are applied to clean context before JPEG; arbitrary rotations/interpolation are excluded. Paired measured SIDD/SR images use identical transforms. Training crops are on the fly; optional decoded-image caching must have a byte limit. Do not materialize millions of patches or duplicate full archives by default.

## Training and losses

One single-device trainer. A step means one successful optimizer update; microsteps count separately. Default optimizer AdamW, lr 2e-4, betas (0.9,0.999), weight_decay 0 for CNN/project SR runs; Restormer project runs may use 3e-4 and 1e-4, explicitly configured. Warm up for 100 successful updates, then cosine decay to 1e-6 over max_steps; scale warmup down for smoke runs. This is a common project recipe, not a paper reproduction. Clip global gradient norm at 1 after unscaling; record clipping and skipped-update counts. No EMA, mixup, compilation, multi-GPU or progressive patch schedule in the first baseline.

Initial CUDA profile: targets 128x128, microbatch 8, accumulation 2 (effective batch 16), fp16 autocast and CUDA GradScaler. First try microbatch 4 if profiling predicts pressure; never assume batch 16 fits a transformer. Local MPS: microbatch 2, accumulation 4, FP32 initially; optional reduced precision only after forward/backward and numerical checks. CPU: microbatch 1, FP32. For the two-view JPEG study, microbatch means clean target crops: concatenate both views, so the network sees 2B inputs and the effective count of degraded views is twice the clean-crop count. Profile this actual workload. Accumulation reduces activation memory; it does not accelerate a fixed effective batch or reproduce larger-batch BN statistics. Keep the same microbatch across the JPEG ablation arms.

Use `torch.amp.autocast` and `torch.amp.GradScaler("cuda")` for CUDA fp16; BF16 only with hardware support and validation, usually without scaling. Loss reductions in FP32. Validation defaults to FP32, model.eval(), no_grad/inference_mode. Explicit requested-device failure is fatal; automatic startup chooses CUDA, then MPS, then CPU and prints the decision. A research run requires an explicit device and does not change devices after OOM. Do not silently enable MPS CPU fallback. [PyTorch precision guidance](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html?highlight=torch+cuda+amp+autocast).

Loss API: `criterion(pred, target) -> (scalar_tensor, dict[str, float])`, matching spatial dimensions and finite values. The logging dictionary is detached. Predictions remain differentiable and unclamped.

- L1 baseline: `mean(abs(pred-target))` over B,C,H,W.
- Charbonnier ablation: `mean(sqrt((pred-target)^2+eps^2)-eps)`, eps=1e-3, FP32. The subtraction makes perfect reconstruction zero without changing gradients.
- Optional VGG feature term: average MSE feature distances at pre-ReLU VGG19 conv3_3/conv4_3/conv5_3, equal layer weights 1/3. Torchvision features indices 14/23/32 must be verified in the pinned model; disable in-place ReLUs so saved preactivation features are not mutated. Use ImageNet normalization mean (0.485,0.456,0.406), std (0.229,0.224,0.225), no classification resize/crop. Freeze all weights and keep eval mode even if the parent criterion is trained. Target features may use no_grad; prediction features must retain gradients. lambda in {0,0.01,0.05,0.1}, tuned only on development data with identical pixel-loss settings. No claim that this interval prevents hallucination.
- Optional LPIPS loss is a separate backend with exact network/version and normalization declared; do not mix its scale with VGG lambda without an ablation. No perceptual weights are needed when lambda=0. No random VGG/LPIPS substitute if weights are absent.
- MSE is permitted inside PSNR and a feature metric; pure pixel-MSE training remains prohibited under the existing project policy. Correct the rationale: L1 does not ensure exact detail recovery, and L2 is not universally scientifically invalid. Diffusion, if separately approved, uses its own objective.

## Focused candidate-method study

**Hypothesis, not a novelty claim:** balanced JPEG block-phase exposure plus agreement between differently compressed views of the same clean crop can improve worst-phase fidelity under a small training budget without changing inference architecture.

Candidate configuration name: `jpeg_phase_consistency_candidate`. Use the same dncnn_rgb_l1 architecture for every arm, trained from scratch on RGB images, Q20, target P=128. Do not use pretrained weights trained on the held-out images. The method adds a training objective only; inference remains one RGB tensor -> one restored tensor, one model pass.

Construct two JPEG views with the identical target content and different block origins. From a clean (P+32)x(P+32) context, target is `[16:16+P,16:16+P]`. For phase (u,v) in {0,...,7}², encode the (P+8)x(P+8) region starting at (16-u,16-v), decode, then crop `[u:u+P,v:v+P]`. Thus content aligns exactly while its JPEG grid changes. Decode RGB 4:4:4. Geometric augmentation occurs before this construction. A crop is not a torch.roll and must not wrap image edges. Reject too-small sources in this experiment.

Let p1=f(y1), p2=f(y2). The common pixel objective is 0.5*(L1(p1,x)+L1(p2,x)). The candidate adds eta*mean(abs(p1-p2)), eta=0.05 initially. Both branches retain gradients; no clamp or perceptual term. If peak memory requires sequential backward, combine the consistency graph correctly rather than detaching a branch. Reduce microbatch if necessary and retain it across arms.

Three ordered ablation arms, same clean crops, seeds, model initialization, optimizer schedule and number of two-view updates:

1. A: uniform-random phase pairs; pixel objective only.
2. B: phase-stratified pairs; pixel objective only.
3. C: phase-stratified pairs; pixel objective plus consistency.

For B/C, enumerate phase indices 0..63 in a deterministic shuffled cycle using sample ordinal; pair the second phase with index+32 modulo 64. A draws two independent phase indices. This tests stratification and then the extra loss separately. Count two forward passes for all arms; also compare elapsed GPU time. Avoid BN order effects by concatenating the two views in the same forward batch for all arms.

Screen for 1,000 updates on the small development subset; proceed to 5,000 updates and seeds 42/43/44 only if feasibility and novelty review justify it. Pilot eta choices {0,0.01,0.05} use dev data; freeze one nonzero eta before final seeds, report every pilot. Final Q20 test also reports Q10/Q40 as mismatch conditions, all 64 phases and fixed clean crops. Primary endpoint: paired difference in per-image worst-phase PSNR versus A. This is a patch-based phase-robustness endpoint, not a full-image benchmark. Also evaluate a fixed-phase full-image Q20 panel to check ordinary restoration quality. Report average PSNR, SSIM, edge-error, phase spread and inference cost to expose over-smoothing. Bootstrap at image level, not at correlated phase/crop level. Bootstrap intervals and three seeds do not cure a tiny sample.

**Novelty gate:** shift-invariant JPEG methods and equivariance regularization already exist: [Fu et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_JPEG_Artifacts_Reduction_via_Deep_Convolutional_Sparse_Coding_ICCV_2019_paper.pdf), [Bai et al., CVPR 2025](https://arxiv.org/abs/2505.19799). A focused nearest-prior-art comparison must identify a meaningful distinction before calling this a new method. No search result is evidence of absence. If it is already known, too incremental, or empirically unsuccessful, report an ablation/negative result and seek a new research decision; do not invent novelty. This remains a material unresolved academic risk.

D08 approves this optional non-adversarial consistency term. Novelty and empirical utility remain unproven; expensive experiments require separate bounded run authorization.

## Evaluation protocol

Keep internal comparisons and literature reproduction in different tables. Every result row records dataset/split/hash, task/scale, checkpoint origin, train data/steps/seed, operator version, channel, border, clamping, quantization, metric version and tile mode. No numerical benchmark results are asserted by this design.

| Protocol | Exact primary settings |
|---|---|
| project_rgb_float_v1 | Clamp prediction [0,1] at metric boundary; target already [0,1]; no quantization. FP64 MSE and PSNR with peak 1. RGB, border 0 for denoising/JPEG, border scale for SR. Report mean of per-image PSNR, never PSNR of pooled MSE. |
| project_ssim_v1 | Same image domain/border; Gaussian 11x11, sigma 1.5, population covariance, K1=.01/K2=.03, data_range=1; valid windows; compute each RGB channel and average, then average images. Images smaller than the valid window after crop fail metric validation. |
| swinir_sr_replay_v1 | Port a pinned upstream evaluator, including uint8 rounding, MATLAB-compatible YCbCr Y, scale border and reference SSIM. Verify fixture parity before any published comparison. Do not silently assume an EDSR DIV2K score uses this protocol; its official repository documents different border settings. |
| sidd_srgb_replay_v1 | Official public validation block ordering; RGB, border 0. Store clamped floating results and use the pinned official/reference scoring path. Quantization and aggregation must match the claimed target benchmark. A PNG export is not automatically the scoring input. |
| lpips_alex_v0.1 | Optional secondary full-image RGB score, no Y conversion; clamped float images transformed to [-1,1], net=alex, version=.1, spatial=false, eval/frozen; no artificial resizing. Images below configured minimum 64x64 fail this metric's validation. State any border separately. |

Reference correctness: the [official SwinIR metric code](https://github.com/JingyunLiang/SwinIR/blob/main/utils/util_calculate_psnr_ssim.py) fixes SSIM/Y/PSNR details; [official LPIPS](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py) fixes its normalization/version. Implement a single internal PSNR/SSIM backend and cross-check against reference fixtures; do not average library outputs.

Run numeric PSNR/SSIM on CPU after transferring predictions: NumPy float64 for MSE/PSNR and CPU torch.float64 Gaussian valid convolutions for SSIM. This avoids requesting unsupported float64 operations on MPS. LPIPS remains a separately provisioned neural metric and can run on the selected supported device.

Perfect reconstruction gives PSNR +infinity. CSV may use `inf`; strict JSON uses null plus `psnr_is_infinite=true`, including aggregate status. Do not silently drop infinite scores. Empty datasets, NaNs, failed images or missing LPIPS weights fail explicitly or mark a requested optional metric unavailable with a reason and count; never silently report a smaller sample.

Additional JPEG edge measure: mean absolute horizontal/vertical first-difference error versus clean target, averaged over both axes/channels. Record block-boundary and interior errors separately using the known phase. These are project diagnostics, not official benchmark metrics.

Selection: choose checkpoints only on fixed dev data. For ordinary tasks use highest dev mean PSNR, earliest step on ties; for the candidate use dev mean per-image worst-phase PSNR over a fixed development phase panel {0,2,4,6}². Final test uses all 64 phases once after selection. Do not tune on test outcomes. Record seed-level mean/std and image-grouped paired uncertainty; separate published values from reproduced values.

Latency: batch 1, fixed input shapes 256x256 and 512x512 (LR dimensions for SR), no self-ensemble, fixed precision, 10 warmups and 30 timed passes; synchronize CUDA/MPS before/after timing. Report median/p95 model-only time and separate I/O/end-to-end time. Count parameters at runtime. Report peak CUDA allocated/reserved memory including model; on MPS report available allocator/driver measurements explicitly as different quantities, not comparable VRAM. MACs/FLOPs are optional with tool/version and supported-op coverage; do not invent totals.

Full-image inference is the reference. Transformer tiling changes context (particularly Restormer's global channel statistics), so overlap does not guarantee equivalence. Optional tile=256/overlap=32 is labeled approximate, constrained to the model's grid; verify shapes, seam errors and score deltas against full images. Never silently tile after OOM or use tiled scores in a full-image literature table.

## Checkpoint, CLI and run contracts

Versioned checkpoint includes model state and full architecture spec; optimizer/scheduler/scaler; successful update and microstep counters; best dev metric/step; effective batch; resolved config; manifest/operator/protocol hashes; Python/NumPy/torch CPU/CUDA/MPS RNG states where supported; deterministic sampler cursor; dependency/hardware metadata; Git commit and dirty-tree/file hashes. Save atomically at optimizer boundaries, with zero accumulated gradients. State absent on a backend is recorded, not fabricated.

Exact CPU replay is tested in a fixed environment. Accelerators are tolerance-tested and limitations documented; [PyTorch does not guarantee cross-release/platform reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html). A weights-only initialization is distinct from resume. Resume rejects incompatible task/scale/architecture/data/schedule unless an explicit new run is created.

Run outputs: `experiments/runs/<unique_run_id>/` with config.yaml, environment.json, manifests.json, metrics.csv, summary.json, latest.pth, best_model.pth and selected qualitative PNGs. Metrics-only analysis artifacts go under versioned `reports/`; dataset files and weights stay ignored. Never overwrite an existing run or share best_model.pth across tasks.

Preserve existing CLI entry points. Extend them with explicit, validated flags:

```bash
python train.py --config configs/smoke.yaml --dry-run --epochs 1 --batch-size 2 --device cpu
python train.py --config configs/jpeg/phase_consistency.yaml --device cuda --max-steps 1000 --max-seconds 1800
python train.py --config configs/jpeg/phase_consistency.yaml --resume RUN/latest.pth --device cuda
python eval.py --checkpoint RUN/best_model.pth --data-dir DATA --task jpeg --protocol project_rgb_float_v1 --device cpu
python inference.py --input INPUT --output OUTPUT --model dncnn_rgb_l1 --weights RUN/best_model.pth --device mps
```

`--data-dir` either contains a documented paired layout or a manifest selected by config; ambiguity fails. Checkpoint metadata supplies model/task/scale for inference; conflicting CLI values fail. `--dry-run` is offline, procedural, two optimizer updates and one validation/checkpoint/reload pass, with no pretrained feature losses. Epochs remain supported as a bounded finite dataset traversal; do not combine explicit epochs and max_steps outside the documented dry-run compatibility case. max_seconds stops at an optimizer boundary, writes a resumable checkpoint and marks the run incomplete if the step target was not reached. These are required interfaces; implementation status is tracked separately and must not be inferred from this design.

## Dependencies and repository layout

Use a dedicated Python 3.11 environment rather than modifying the global Miniconda base. Candidate compatible pair: torch 2.13 / torchvision 0.28 from the [official compatibility matrix](https://github.com/pytorch/vision#installation); verify Mac ARM and CUDA wheels and pin exact validated patch versions in platform locks. This is a proposed compatibility target, not a tested installation. Do not spend the week compiling custom CUDA/MPS extensions. Record actual Colab preinstalled versions before choosing whether replacement is necessary.

Core: torch, NumPy, Pillow, PyYAML, tqdm; pytest for tests. TensorBoard optional but supported for training logs. Evaluation extras: SciPy/h5py for SIDD MAT formats, LPIPS+torchvision for learned metrics, OpenCV only for upstream replay. Matplotlib is enough for plots; Seaborn and TorchMetrics are unnecessary core dependencies. VGG weights are an optional explicitly provisioned resource. Requirements lock captures indirect versions; no network side effects on import or test execution.

```text
configs/                 # smoke, jpeg arms, denoising, sr, hardware overrides
data/{raw,processed}/    # ignored images; optional derived cache
data/manifests/          # versioned metadata, paths without image payloads
models/                  # __init__, common, baselines, restormer, swinir
losses/                  # l1_loss, composite, optional perceptual, consistency
utils/                   # io, dataset_loader, degradation, metrics
                         # config, runtime, reproducibility, checkpoint, logger
train.py / eval.py / inference.py
scripts/                 # prepare_manifest, profile, run_ablation
tests/                   # smoke + data, metrics, models, trainer/resume tests
notebooks/               # thin free-cloud launcher; no duplicate training logic
experiments/runs/        # ignored per-run artifacts
reports/                 # small versioned analysis/report artifacts
docs/architecture/       # review, proposal, roadmap, later approved design/ADRs/handoff
requirements.txt / requirements/locks/ / README.md
```

Keep the existing top-level packages; no disruptive package migration. Do not add diffusion.py as an empty promise. Add modules only in their implementing phase.

## Compute estimate and go/no-go rules

These are planning estimates and accounting formulas, not measured GPU timings or guaranteed free-tier capacity.

- Profile 10 warmup + 50 measured training updates for each chosen model/precision/microbatch, including degradation and backward. Use target peak CUDA allocated memory <=12 GiB on an approximately 15 GB device, leaving headroom; measure reserved memory too. Increase accumulation after reducing microbatch if needed, then freeze the study configuration before comparison.
- The nine final JPEG runs (3 arms x 3 seeds x 5,000 updates) total 45,000 two-view updates. At measured update time t seconds, training is 12.5*t GPU-hours: t=.2 implies 2.5 h; t=1 implies 12.5 h; t=2 implies 25 h. These t values are hypothetical. Validation, profiling, data/setup, transfers and repeated pilots are additional costs.
- Start with 3 x 1,000-update screening runs. Stop expansion if projected time exceeds available free allocation or improvements are still dominated by learning-curve slope. Shortening a schedule is a recorded reduced-budget study, not proof of convergence.
- Evaluate the 64-phase final panel on at most 16 test images for the pilot; expand only after measuring time. Report the subset and do not claim a full benchmark. Avoid running LPIPS every update or every development image; use PSNR/SSIM for frequent validation.
- A float32 image tensor alone costs 4*B*C*H*W bytes. Activations, saved backward intermediates, BN, optimizer state and attention dominate training memory; parameter count alone cannot predict fit. Standard FP32 Adam-family parameter/gradient/moment storage is approximately 16 bytes per parameter before activations and extra copies. Accumulation does not remove those costs.
- Full Restormer reproduction is not a week-one target. Its [Gaussian training configuration](https://github.com/swz30/Restormer/blob/main/Denoising/Options/GaussianColorDenoising_Restormer.yml) uses 300,000 updates, more training data than DIV2K alone and progressive patches. Approved pretrained inference or clearly labeled short fine-tuning is the practical alternative.
- Free Colab hardware and runtime duration are variable according to its [official FAQ](https://research.google.com/colaboratory/faq.html). No paid upgrade, AWS allocation or subscription is permitted. Use one T4; a second T4 can run an independent seed later, but memory is not pooled and multi-GPU infrastructure is unnecessary.

Main bottlenecks: environment/data acquisition, CPU JPEG encoding, full-image transformer memory, exhaustive phase evaluation, and interrupted free sessions. Cache a small fixed validation set, train from runtime-local storage, save periodic resumable checkpoints, and export run artifacts manually through the notebook. No claims of timeout immunity.

## Approval and decision register

D01–D10 are accepted by explicit user approval. Approval is a design gate, not blanket permission for dataset/weight downloads, paid compute or expensive runs.

| Decision | Selected approach / alternative / consequence | Validation and failure mode | Approval category |
|---|---|---|---|
| D01 Scope | One JPEG candidate study this week; retain SR/denoising milestones. Alternative: broad engineering comparison with no new-method claim. | Executable system and honest report; novelty may remain unmet. | Explicit milestone narrowing, academic tradeoff. |
| D02 Model whitelist | Add DnCNN-style RGB and EDSR baseline; NAFNet optional only after a named substitution decision. Alternative: only approved transformers. | Baseline contracts and task-specific comparisons; L1 adaptations labeled. | Changes section 2 approved-backbone restriction. |
| D03 LDM | Defer required LDM implementation. Alternative: separately approved non-adversarial generative study. | Provenance and fidelity review before any extension; no generative results implied. | Removes a provisional required component, preserves no-GAN rule. |
| D04 Loss default | L1/Charbonnier baseline; perceptual loss opt-in with lambda=0 by default. Alternative: mandatory composite. | Pixel/perceptual ablations; no unsupported texture guarantee. | Resolves conflicting section 2 and section 9; explicitly amend hard wording. |
| D05 Tensor domains | BCHW at model/batch boundary, CHW dataset samples; clean images [0,1], declared noisy exceptions, unrestricted predictions/features. | Range/shape tests, no training clamp, declared metric boundaries. | Clarifies/relaxes universal hard range/shape invariant. |
| D06 AMP | Current torch.amp API, CUDA AMP, FP32 CPU/MPS default. Alternative: legacy pinned reproduction environment. | Device/precision forward/backward tests and nonfinite detection. | Changes literal legacy-API hard requirement while preserving AMP support. |
| D07 Artifact task | RGB JPEG with fixed codec/quality/phase. Alternative: another explicitly scoped artifact dataset. | Operator golden fixtures; codec/version/phase documented. | Defines ambiguous project requirement; requires user acceptance. |
| D08 Candidate loss | Optional two-view phase consistency term, eta selected on dev. Alternative: pixel-only phase-sampling ablation. | Matched two-view controls and worst-phase fidelity tests; may over-smooth or duplicate prior art. | Extends allowed composite loss for this named experiment. |
| D09 Evaluation | Fixed project protocol, separate pinned benchmark replay. Alternative: one library's defaults without comparison claims. | Reference parity and no leakage; tiled/quantized mismatches excluded. | Protocol decision, no hard-constraint removal. |
| D10 Reuse/compute | Minimal licensed model ports, single GPU, profile before training, per-run state. Alternative: full upstream frameworks/reproduction. | Reference parity, measured memory/time, tested resume; some upstream convenience lost. | Engineering decision; downloads/runs still separately bounded. |

Unchanged: no GANs/discriminators/adversarial losses, no unapproved adversarially trained components, fidelity priority, RGB public image interfaces, no pure pixel-MSE training, no fabricated results, no training-output clamp, no leakage, device-agnostic support, and explicit authorization for large downloads/expensive runs.

Implementation follows implementation_handoff.md, which references this accepted v1.0 design. Later major design changes require a new approval record and consistent updates to agent instructions. Implementation and experimental completion are tracked separately.
