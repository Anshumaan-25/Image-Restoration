# Baseline recipe cross-check during the approved diagnostic

The approved project model is a **DnCNN-style L1 variant**, not an exact DnCNN reproduction. No running setting was changed after this review.

The original paper uses a much larger patch-training procedure and an SGD recipe; its multi-task DnCNN-3 is initialized from DnCNN-B. Those experiments do not establish that this project's short scratch-trained, single-task RGB JPEG/L1/AdamW recipe should reach comparable fidelity. [Primary paper, section IV-A](https://arxiv.org/pdf/1608.03981).

The author's later hosted PyTorch Gaussian-denoising example uses orthogonal convolution initialization, BN epsilon 1e-4/momentum 0.95, Adam learning rate 1e-3 and batch 128. Our current project uses PyTorch-default convolution/BN initialization/settings, 2e-4 AdamW and four views per BN forward. This is a documented recipe difference, not proof of the weak JPEG result's cause. Copying the Gaussian example's BN settings or squared-error objective into this project would not be a validated fix. [Author-hosted PyTorch example](https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py).

The author's explanation ties a particular BN benefit to Gaussian residual statistics. JPEG residuals should not be assumed to share that property. This motivates testing optimization/initialization or no-BN alternatives only if needed through a separately reviewed design. It does not demonstrate BN is unsuitable for JPEG. [Author's architecture rationale](https://github.com/cszn/DnCNN#network-architecture-and-design-rationale).

Local evidence is stronger about execution consistency: the new run's first 100 warmup updates exactly match the earlier B run's logged losses, learning rates, pre-clip gradient norms and development endpoint in this environment. The cosine horizons differ afterward; no cross-platform or future bitwise guarantee follows. [Actual warmup replay](warmup_replay.json).

Source inspection: `models/baselines.py`, `train.py`, and the resolved config retained in the run directory. Retrieved primary sources 2026-10-21 Asia/Kolkata. No literature benchmark numbers are imported, no downloaded model weights, no architectural mutation and no new experiment beyond the approved diagnostic.
