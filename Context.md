# Project context

**Image Degradation Restoration Using Advanced Deep Learning Techniques**

B.Tech Minor Project, CSE–Data Science, Semester VII; Amity School of Engineering & Technology, Amity University Uttar Pradesh. Group 12. Guide: Dr. Rajat Kumar Arya. Team: Anshumaan Singh, Lakshay Teotia and Javin Dhaiya.

## Current authority

D01–D10 were approved on 2026-10-20. Follow [AGENTS.md](AGENTS.md), [approved design v1.0](docs/architecture/approved_design.md), [approval record](docs/architecture/approval_record.md) and [implementation handoff](docs/architecture/implementation_handoff.md). The [original context](docs/history/Context.pre-review.md) is historical; its scientific errors are addressed by the [independent review](docs/architecture/critical_review.md).

## Objective and constraints

Build faithful, non-adversarial SR x2/x4, real/synthetic RGB denoising and JPEG artifact removal. Begin with data/metrics and compact CNN baselines, then task-specific Restormer/SwinIR comparisons. LDMs are deferred. CNNs can capture broader context through composition/downsampling; NAFNet is a CNN. Neither diffusion nor perceptual loss guarantees faithful reconstruction.

Deadline: this week; exact submission timestamp unconfirmed. Budget: $0. Local M5/16 GB for development/light inference where MPS is available. Free cloud GPUs are opportunities, not confirmed allocations; memory is separate per device. Profile on one approximately 15 GB GPU with <=12 GiB peak allocated target and headroom.

The primary study tests JPEG phase-stratified sampling and optional two-view prediction consistency under matched controls. This is a candidate contribution, not established novelty. Prior-art differentiation and actual evidence are required before publishability claims.

L1 default; Charbonnier/perceptual ablations optional. Explicit RGB/domain/range contracts, unclamped training predictions, immutable source splits, fixed degradation and metric protocols. Separate pretrained evaluation, project training and exact reproduction. Architecture approval alone does not approve specific dataset/weight downloads, expensive runs, deletion or paid compute. Procedural dry runs are pipeline tests, not research results.
