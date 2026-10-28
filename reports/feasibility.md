# Measured feasibility and next research run

Hardware: Apple M5 MacBook Air, 16 GB unified memory ([sanitized hardware record](local_hardware.json)). PyTorch 2.12.0, MPS FP32. CUDA/T4 was not available or tested. All durations below are measured locally; no paper timing or hypothetical GPU result is presented as an observation.

## Actual training workload

[Release phase profile](mps_phase_profile_release.json): DnCNN RGB, **669,379 parameters**, depth 20/width 64, real DIV2K sources, target 128x128, 2 clean crops per microbatch, 4 accumulation microsteps, two degraded views per crop. One update therefore consumes 8 clean crops / 16 degraded views; BN sees 4 degraded images per forward. Ten warmups, 50 measured updates completed.

- Mean update: **0.9894 seconds**; median: **1.0002 seconds**.
- Largest sampled MPS driver allocation at optimizer boundaries: **1.059 GiB**. This is not a measured memory peak or CUDA VRAM capacity result. The end-of-run current allocation is zero after object release and is not an estimate of training requirements.
- Timing includes sample loading/JPEG, forward/backward, clipping, optimizer and scheduler. Validation, log/checkpoint I/O and initial manifest validation are excluded. The release profile ran after acceptance tests had finished; it performs no quality evaluation or checkpoint selection.

A separate ordinary single-view/full-source JPEG profile is retained in [baseline measurements](mps_baseline_profile_isolated.json), approximately 1.303 seconds/update. Its JPEG encoding context/workload differs from the phase study, and a short QA run overlapped part of acquisition. Treat it as a diagnostic baseline rather than a controlled speed comparison. The earlier two-view diagnostic is retained separately; use the release profile for budgeting.

## Inference latency

[Measured model-only latency](mps_latency_release.json): synthetic RGB input, random weights, batch 1, FP32, full-image forward, no ensemble; 10 warmups and 30 synchronized timed passes per shape. This measures architecture execution, with no restoration-accuracy claim.

| Input | Median | p95 |
|---|---:|---:|
| 256x256 | 29.97 ms | 31.67 ms |
| 512x512 | 115.13 ms | 116.17 ms |

## Concrete screening proposal

[Executable proposal](screening_run_proposal.json): three matched arms, seed 42, 1,000 successful updates each, 32 training sources / 8 development sources, fixed 16-phase development panel every 100 updates. Same MPS FP32 batch 2 / accumulation 4. No protected test scoring, weight/data downloads, paid service or novelty claim. Exact runner command is in the JSON and [runbook](../docs/runbook.md).

Measured training-only projection: **49.5 minutes total**, plus setup, development evaluation, checkpoint/log I/O and longer-run variation. Cap each arm at 1,800 seconds (three caps total 90 minutes); these are soft optimizer/validation boundaries, with setup/final I/O overhead. Incomplete runs remain resumable and stop the study runner. A stronger 3-seed x 5,000-update study is not included in this proposal and requires a new measured budget.

The user approved this screening, and all arms completed: A 18.56, B 19.14, C 18.74 minutes, **56.44 minutes combined** including in-run development validation/checkpoint overhead. Each completed 1,000 successful updates with zero skips, below its 30-minute soft cap. These actual costs supersede the training-only projection for this workload. [Results](results.md) show tiny gains and no benefit from the extra consistency term versus stratification; scientific novelty remains unestablished.

The user separately approved the [5,000-update B baseline diagnostic](proposals/baseline_learning_v1/proposal.json), which completed in **89.12 minutes** including in-run development/checkpoint overhead, zero skips, below the 100-minute cap. Post-run development/CLI/analysis checks completed in **111.23 seconds**. [Measured learning results](baseline_diagnostic.md) show useful improvement, not a novel-method claim. Do not extrapolate MPS measurements to T4 memory/throughput.

The proposed remaining matched A/C controls would add two 5,000-update runs: **178.24 minutes projected from actual B**, each capped at 100 minutes (200 combined), plus setup/final I/O. [Exact unapproved proposal/configs](proposals/matched5000_candidate_v1/proposal.json). Reuse completed B; no duplicate training. AGENTS.md retains specific approval for these additional long runs. None was launched.
