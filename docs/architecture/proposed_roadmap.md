# Ordered proposed implementation roadmap

Status: **PROPOSED — NOT AN APPROVED CODING HANDOFF**. Date: 2026-10-20, Asia/Kolkata.

Authority after approval will be a versioned approved_design.md, not this file. Read [the review](critical_review.md) and [the complete proposed contracts](proposed_architecture.md). No implementation work has started. The verification commands below are future acceptance commands; none are asserted to work in the current scaffold.

## Delivery sequence

Use the remaining work sessions this week: first session approval/environment/data/metrics; second baseline and profiling; third candidate screening and provenance/novelty review; fourth repeatable experiments and optional pretrained comparisons; final session frozen evaluation and reporting. Do not allocate the final session to adding models. Exact submission time and free-GPU allocation remain unknown.

Priorities: P0 scientific correctness and executable baseline; P1 focused research study; P2 task adapters and approved pretrained evaluation; P3 full task-specific training/extended research. No phase may claim later deliverables complete merely because its own tests pass.

## Phase 0 — Approve and establish execution authority (P0)

- **Input/dependency:** user response to decisions D01–D10, exact milestone scope, and acknowledgement that novelty remains unproven.
- **Objective/files:** create docs/architecture/approved_design.md, approval_record.md, adr/ records and implementation_handoff.md; reconcile AGENTS.md and Context.md. Retain this review/proposal as historical documents. Do not change implementation yet.
- **Contract:** handoff cites approved design revision and acceptance criteria. Record task support separately from completed experiments. No GAN/provenance exception is implied.
- **Acceptance/deliverable:** consistent instructions, explicitly approved model/loss/range/API amendments, ordered tasks with no unresolved major design choices for core implementation.
- **Verification:** inspect approval record; `rg -n 'PROVISIONAL|torch.cuda.amp|Composite Loss|exact structural|NAFNet' AGENTS.md Context.md docs/architecture` and classify every remaining occurrence as historical or operative. Do not indiscriminately delete history.
- **Risk/mitigation:** academic novelty cannot be promised. Approve a candidate experiment with a novelty gate rather than an asserted invention. Separate specific dataset/weight download approvals and bounded run authorization from design approval.

## Phase 1 — Runtime/configuration and trustworthy test entry point (P0)

- **Inputs:** approved contracts; local Python/hardware inventory. No dataset/weights required.
- **Files:** requirements.txt, requirements/locks/, configs/smoke.yaml, configs/hardware/{cpu,mps,t4}.yaml, utils/{config,runtime,reproducibility}.py, tests/{smoke_test,test_config,test_runtime}.py, README.md.
- **Interfaces:** load_config(path, overrides) returns validated configuration; select_device(requested) chooses only at startup; precision_context is a context manager. Reject unknown keys, invalid scales/batches/accumulation, unsupported model-task pairs and contradictory epoch/step budgets. Print resolved device and precision.
- **Expected behavior/edges:** dedicated environment; no downloads on import/test; CPU FP32 always supported. Requested unavailable device errors. Smoke errors exit nonzero; optional hardware tests skip with explicit reasons. Avoid interpreter-dependent imports by documenting `python -m pytest` and making the required direct smoke command import the repository correctly.
- **Acceptance/deliverable:** dependency lock for each validated platform, offline config/runtime tests, environment report template.
- **Verification:** `python -m pip check`; `python -m pytest -q tests/test_config.py tests/test_runtime.py`; `python tests/smoke_test.py` initially tests only implemented required contracts. It must not pretend missing later phases passed.
- **Risk/mitigation:** free-cloud environments vary. Inspect installed versions first, use a verified torch/vision pair, and avoid custom extensions. Do not modify global Miniconda base.

## Phase 2 — Data correctness and metrics before learned models (P0)

- **Inputs/dependencies:** Phase 1; approved dataset files when acquired; procedural fixtures otherwise. No model weights required.
- **Files:** utils/{io,dataset_loader,degradation,metrics}.py; scripts/prepare_manifest.py; data/manifests/; tests/{test_io,test_data,test_degradation,test_metrics}.py; tests/fixtures/ small authored numerical fixtures.
- **Interfaces:** manifest/sample/batch contracts and named operators from approved design. Metric functions return per-image scalars/status; validation does not change random state or inputs. Pair matching uses identifiers. Implement project RGB metrics first; upstream replay is an additional named protocol, not a default-library alias.
- **Expected behavior/edges:** aligned LR/HR crops; independent image/channel AWGN with sigma/255; declared clipping; Poisson units/transfer; measured SIDD bypass; exact codec/version/phase metadata; tiny sources, missing/duplicate partners, EXIF, alpha/grayscale, bad bit depth and empty splits fail clearly. Split hashes/parent IDs are checked before extracting patches. Test validation repeatability across worker counts.
- **Acceptance/deliverable:** deterministic manifests, offline operator examples and reference metric parity report. PSNR(identity)=infinity; fixed known-error PSNR matches analytic value. SSIM matches a pinned reference within 1e-6 on fixtures; negative values are not clipped. Test unequal-image aggregation to catch pooled-MSE bugs.
- **Verification:** `python -m pytest -q tests/test_io.py tests/test_data.py tests/test_degradation.py tests/test_metrics.py`; `python scripts/prepare_manifest.py --config configs/smoke.yaml --validate-only`. Missing real data must not be reported as validated.
- **Risk/mitigation:** source-codec confounding and leakage. Use DIV2K PNG for the primary JPEG experiment, immutable splits before crops, and no synthetic noise on SIDD. Keep MATLAB/codec replay golden fixtures distinct from framework-native fixture generation.

## Phase 3 — Small learned baseline and complete resumable loop (P0)

- **Inputs/dependencies:** Phases 1–2; procedural paired data for smoke, approved project split for actual training.
- **Files:** models/{baselines,common,__init__}.py; losses/{l1_loss,composite,__init__}.py; utils/{checkpoint,logger}.py; train.py, eval.py, inference.py; configs/jpeg/baseline.yaml; tests/{test_baselines,test_losses,test_training,test_checkpoint,test_cli}.py; extend tests/smoke_test.py.
- **Interfaces:** dncnn_rgb_l1 from design; criterion -> scalar plus detached float logs; trainer counters at optimizer boundaries; checkpoint schema and metadata-driven inference. Implement identity and bicubic references. eval accepts the documented paired data layout or manifest, otherwise fails.
- **Expected behavior/edges:** loss backpropagates through model parameters, outputs are not clamped in forward, predictions are clamped only for metrics/export. Handle odd/small dimensions, nonfinite loss/gradients, final incomplete accumulation group, optimizer-step skips and output path collisions. Per-update scheduler advances only on a real optimizer step. A partial wall-time-limited run remains incomplete.
- **Acceptance/deliverable:** offline two-update dry run with validation/checkpoint/reload/inference; finite gradients; a 20-update toy overfit loss decreases; CPU uninterrupted four-step execution matches two-step-save-resume-two-step execution, including optimizer/BN/scheduler state and next sample. Accelerator resume uses documented tolerances.
- **Verification:** `python -m pytest -q tests/test_baselines.py tests/test_losses.py tests/test_training.py tests/test_checkpoint.py tests/test_cli.py`; `python tests/smoke_test.py`; `python -m py_compile train.py eval.py inference.py models/*.py losses/*.py utils/*.py`; `python train.py --config configs/smoke.yaml --dry-run --epochs 1 --batch-size 2 --device cpu`.
- **Risk/mitigation:** BN microbatch sensitivity; gradient accumulation is not BN-equivalent. Fix the actual network batch across comparable arms and log both clean-crop and two-view counts. Do not mistake toy overfit for benchmark quality.

## Phase 4 — Feasibility profiling and bounded baseline experiment (P0/P1)

- **Inputs/dependencies:** working Phase 3; allocated free device; approved data acquisition and explicitly bounded run.
- **Files:** scripts/profile.py; configs/jpeg/{baseline,pilot}.yaml; notebooks/free_gpu.ipynb (thin command launcher); reports/feasibility.md. No duplicate notebook model/training code.
- **Interfaces:** profile emits machine-readable device, precision, microbatch, accumulation, measured update latency, throughput and peak allocator data. Run metadata includes source/dependency/data hashes and actual step target.
- **Expected behavior/edges:** T4 target <=12 GiB peak allocated, report reserved memory and headroom. Try one smaller microbatch before a run if necessary; freeze settings after profiling. MPS FP32 fallback is a separately identified run, not an invisible migration. Notebook installation, train, resume, evaluation and artifact export are ordinary explicit cells.
- **Acceptance/deliverable:** 10-warmup/50-measured-update profile, cost projection, bounded 1,000-update baseline with fixed development panel and latest/best checkpoints. A wall-time stop cannot be called a completed 1,000-step experiment.
- **Verification:** `python scripts/profile.py --config configs/jpeg/pilot.yaml --device cuda --warmup 10 --steps 50`; `python train.py --config configs/jpeg/pilot.yaml --device cuda --max-steps 1000 --max-seconds 1800`; inspect report and exit status, then run eval/inference with the produced checkpoint.
- **Risk/mitigation:** unavailable free GPU or slow JPEG encoding. Proceed with offline/local correctness work, profile a reduced local run, and report reduced experimental coverage. Cache fixed validation inputs and keep training images on runtime-local storage. No paid resource fallback.

## Phase 5 — Candidate JPEG method and discriminating ablations (P1)

- **Inputs/dependencies:** Phases 2–4; explicit D08 approval; a prior-art differentiation note. Scientific novelty is still a hypothesis.
- **Files:** extend utils/{degradation,dataset_loader}.py; losses/consistency.py; extend losses/composite.py and train.py for two-view batches; configs/jpeg/{random_phase,stratified_phase,phase_consistency,study}.yaml; scripts/run_ablation.py; tests/{test_jpeg_phase,test_consistency}.py; reports/novelty_assessment.md.
- **Interfaces:** phase-context construction, paired target alignment, 64-phase deterministic cycle, two-view objective and A/B/C definitions exactly as approved. Candidate uses the same RGB model and one-pass inference. Eta and checkpoint selection come only from development data.
- **Expected behavior/edges:** prove by array indices that both targets are identical; independent encoding must not alter crop alignment; RGB 4:4:4 phase is 8x8. No wraparound shifts, post-JPEG geometric augmentation, detached prediction branch or silent codec change. Every arm processes two views, concatenated consistently for BN; phase schedule and sample cursor survive resume.
- **Acceptance/deliverable:** eta=0 equals the common pixel objective; consistency is zero for identical predictions and gives finite nonzero gradients for different predictions; all 64 phases appear exactly once per cycle. A/B isolates sampling and B/C isolates consistency with matching data/model/update budgets. Report oversmoothing risk and all pilot settings, including failures.
- **Verification:** `python -m pytest -q tests/test_jpeg_phase.py tests/test_consistency.py tests/test_checkpoint.py`; `python scripts/run_ablation.py --config configs/jpeg/study.yaml --stage screening --device cuda`. Screening is three 1,000-update runs before a decision to expand.
- **Risk/mitigation:** ordinary consistency/phase augmentation may already be known or may harm pixel accuracy. Compare nearest primary prior art, not just titles. If novelty is not defensible, stop novelty claims and request a revised research decision; retain baseline/data/evaluation work. No automatic replacement by an unreviewed architecture.

## Phase 6 — Task-specific adapters and optional pretrained comparisons (P2)

- **Inputs/dependencies:** Phases 1–4; approved specific weight downloads, provenance/checksums and model licenses. This phase may run after candidate screening; it is not required before a baseline.
- **Files:** models/{restormer,swinir}.py; add EDSR in models/baselines.py; task-specific configs in configs/{sr,denoising}/; tests/{test_models,test_reference_parity}.py; reports/pretrained_provenance.md. Add perceptual loss and its gradient tests only if an approved ablation needs it.
- **Interfaces:** fixed model-task matrix, dynamic padding/unpadding and strict checkpoint loading. Validate task, scale, normalization, layernorm/head settings, training provenance and dataset exposure. No automatic permissive loading of mismatched weights.
- **Expected behavior/edges:** x2/x4 exact output dimensions; independent task checkpoints; transformer windows/masks work on 1x1, 7x9, 127x131 and 128x128. Compare CPU adapter outputs with pinned reference architecture on deterministic inputs (FP32 atol/rtol 1e-5 initially; investigate failures rather than loosening tolerance). Missing approved weights fail explicitly. Local architecture fixtures must not trigger downloads.
- **Acceptance/deliverable:** reference-parity report, per-model parameter count, a small real-data inference/evaluation panel and provenance-separated tables. Report out-of-domain task results as such; pretrained evaluation is not project training or methodological improvement.
- **Verification:** `python -m pytest -q tests/test_models.py tests/test_reference_parity.py`; required smoke/syntax/dry-run commands; `python eval.py --checkpoint WEIGHTS --data-dir DATA --task super_resolution --scale 2 --protocol swinir_sr_replay_v1 --device mps` only after weight wrapping and metric parity are implemented.
- **Risk/mitigation:** pretrained overlap and variant mismatch. Keep the candidate study from scratch on its protected split; do not use GAN-trained real-world SR checkpoints. No Restormer SR adaptation or LDM implementation. If time is short, finish the candidate report before adding another comparator.

## Phase 7 — Frozen evaluation, reproducible results and submission (P1)

- **Inputs/dependencies:** measured feasibility, approved final run budget, frozen configs/eta/dev selection; candidate training only if the novelty/feasibility gates justify it.
- **Files:** configs/jpeg/study.yaml; extend eval.py for phase panel aggregation; reports/{experiments,results,limitations}.md plus compact CSV/JSON artifacts; README.md; tests/test_report.py. Store actual checkpoints outside versioned report payloads.
- **Interfaces:** all A/B/C final runs use 5,000 updates and seeds 42/43/44 if the measured budget permits. Otherwise predeclare the reduced budget before test access. Per-image worst-phase endpoint and ordinary full-image quality are separate outputs; identical image lists and protocols across arms.
- **Expected behavior/edges:** no test tuning, missing-image drops or fabricated metrics. Infinity/NaN serialization obeys the design. Save identical qualitative crops and absolute-error visualizations with a shared scale. Include incomplete/failed runs and selected checkpoint step. Latency includes synchronized model timing and separate I/O overhead.
- **Acceptance/deliverable:** actual results with grouped paired uncertainty, seed variation, compute/parameter/memory figures, provenance and limitations. Explain support or rejection of the hypothesis. State whether novelty was established; an implementation alone does not satisfy publishability. Export code/configs/reports and a reproducible command sequence before the deadline.
- **Verification:** `python -m pytest -q`; required smoke/syntax/dry-run commands; `python scripts/run_ablation.py --config configs/jpeg/study.yaml --stage final --device cuda` only with final run authorization; `python -m pytest -q tests/test_report.py`; rerun evaluation from saved selected checkpoints and verify sample counts/hashes.
- **Risk/mitigation:** interrupted sessions and incomplete evidence. Resume at optimizer boundaries; preserve artifacts after each completed run. Stop scope expansion to leave time for a report. Never substitute paper-reported values for failed reproduction.

## Phase 8 — Complete original multi-task research scope (P3, later unless resources permit)

- **Inputs/dependencies:** validated week-one system, additional dataset/compute approvals and separate task-level experiment plans.
- **Files/modules:** finalize configs/{sr,denoising}/, SIDD MAT adapter in utils/dataset_loader.py, optional losses/perceptual.py and its gradient tests, dataset manifests and benchmark-replay tests. Optional NAFNet/BM3D require a named decision, not automatic inclusion.
- **Contract:** SR x2/x4 uses scale-specific EDSR/SwinIR; AWGN and SIDD use distinct DnCNN/Restormer runs; Poisson is a labeled RGB approximation. Loss and degradation ablations change one variable at a time. No joint inference cascade is implied.
- **Acceptance/deliverable:** real-data quality and compute reports for every required task, consistent evaluation/provenance, and documented training/reproduction limits. Passing shape tests is insufficient to mark a task scientifically completed.
- **Verification:** the complete offline gate plus each task's manifest validation, bounded dry run, verified checkpoint resume and frozen held-out evaluation. Record exact commands/config hashes in each experiment report.
- **Risk/mitigation:** this is much broader than one week's free-compute study. Prioritize unresolved fidelity questions over adding architectures. Any LDM, raw-sensor pipeline, custom Restormer-SR or new claimed method requires another architecture approval.

## Handoff creation rule

After approval, produce docs/architecture/implementation_handoff.md as a concise ordered list extracted from the accepted phases above, with one approved design revision, exact file lists, interfaces, acceptance commands and definitions of done. Mark optional/deferred tasks explicitly. Do not create an “approved” handoff by assuming approval or dispatch coding work from this proposal. Tasks can later be assigned to faster coding models without asking them to reconsider core architecture.
