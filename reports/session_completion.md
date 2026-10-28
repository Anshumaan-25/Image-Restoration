# Current-session completion and recovery

The approved 5,000-update local baseline diagnostic and follow-up verification are complete. Training: 89.12 minutes, zero skipped updates, $0. Selected update 4,800; core pipeline unchanged. Development gains over JPEG input: +0.954915 dB worst-phase on the original eight selection images; +0.857950 dB full-image; +0.881471 dB worst-phase on 72 supplementary development sources unused for selection. No protected test was scored.

Results: reports/baseline_diagnostic.md. Audit: reports/final_audit.md. Selected weights: experiments/studies/baseline_learning_v1/B_seed42/best_model.pth, SHA256 db32a6373fad66a5d2e86bc6b2cc00ffc8e9935bad3aeb8b8e42ff553c0bdcb9. Training/optimizer/RNG/log artifacts remain preserved locally; source bundles exclude weights/data.

Recovery source/results bundle: dist/restoration_source_v2.zip; SHA256 8586abfe6fecaa0fa81c21411ffdae6428e243f6e319d9d97a5576a5e6b1483b. CRC, every indexed entry hash and the extracted bundle smoke test passed. The earlier v1 bundle remains retained. Post-run selected replay, independent aggregation, exact trained CLI output, supplementary identities, unmatched-budget rejection and 211 current local links passed. The unchanged core previously passed 47 tests; those tests were not falsely reported as rerun for this diagnostic.

Scientific novelty, matched longer-budget candidate controls, additional seeds and protected testing remain incomplete. The exact unapproved future A/C configs/budget and ordered handoff are in docs/architecture/research_checkpoint.md and reports/proposals/matched5000_candidate_v1/proposal.json. No additional training was launched.

The user requested normal Mac shutdown only after work finished. All current authorized work is saved/verified. Shutdown action state is retained in reports/studies/baseline_learning_v1/execution.json; a queued request is not a claim that power-off was independently observed. Normal macOS state saving is used; no forced document discard.
