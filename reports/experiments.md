# Experiment ledger

| Activity | Data/workload | Executed | Interpretation |
|---|---|---|---|
| Offline acceptance | Authored procedural/temporary fixtures | 47 tests plus smoke/dry-run/CLI integration | Engineering verification only |
| DIV2K acquisition | 900 references, full/screening splits | Integrity/format/hash/leakage checks | Data preparation, no scores |
| MPS feasibility | Actual DIV2K train references, full DnCNN 128-pixel two-view workload | 10 warmups + 50 measured updates | [Measured compute](feasibility.md), no checkpoint selection or quality inference |
| Model latency | Synthetic 256/512 inputs, random weights | 10 warmups + 30 synchronized passes/shape | Execution cost only |
| Matched A/B/C screening | Actual 32 train / 8 dev, 1,000 updates/arm, seed 42, 16 dev phases | Approved and completed: 3,000 successful updates, 56.44 minutes, zero skips | [Selected development results](results.md): tiny gains; C trails B; no novelty claim |
| Full-image dev replay | Selected A/B/C, same eight dev sources, ordinary Q20 compression | All eight images/arm completed; independent input reference | Separate endpoint; B gains only +0.013309 dB |
| Weak-result diagnostics | Same eight dev images/16 phases, border-20 and residual inspection | Completed, no test data | Small corrections; padding alone does not explain weakness |
| Fixed-patch optimization QA | Original train sources 0001/0002; 200 updates, 60-second cap, different diagnostic LR | Completed 200 updates in 17.11 seconds | Not a research arm: learns but barely beats input; all parameter gradients finite/nonzero |
| Trained CLI | Selected B on fixed dev crop 0721 | Completed, exact direct-export PNG parity | Saved-model inference path verified |
| Longer B learning diagnosis | 5,000-update scratch baseline, same 32/8 split, 100-minute cap | Approved and completed: 89.12 minutes, zero skips; selected 4800 | [Results](baseline_diagnostic.md): +0.954915 dB worst-phase / +0.857950 dB full-image over dev input |
| Supplementary dev72 | Existing development IDs 0729–0800, unused for selection, 16 phases Q20 | Completed after selection, no test data; independently aggregated | +0.881471 dB worst-phase over input; supplementary, not confirmatory testing |
| Remaining matched 5,000-update A/C | Exact same B recipe except phase mode/agreement, seed 42 | Configs/proposal prepared; not approved or run | Reuse B; no longer-budget candidate claim yet |
| Final seeds/protected test | Conditional on better development evidence and freeze | Not run | No held-out scientific result available |

Profile checkpoints are explicitly tagged and cannot be substituted for selected research checkpoints. Earlier diagnostic profiles and acceptance records are retained; current release evidence is linked by implementation_status.md. No literature benchmark value is presented as a reproduced result.
