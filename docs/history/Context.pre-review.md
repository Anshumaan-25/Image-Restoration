# MASTER PROJECT CONTEXT & AGENT INSTRUCTIONS
**Project Title:** Image Degradation Restoration Using Advanced Deep Learning Techniques  
**Repository Name:** `Image-Restoration`  
**Target Domain:** Machine Learning, Computer Vision, Deep Learning  
**Academic Context:** B.Tech Minor Project (CSE - Data Science, Semester VII), Amity School of Engineering & Technology, Amity University Uttar Pradesh  
**Group Number:** Group 12  
**Guide:** Dr. Rajat Kumar Arya  
**Team Members:**
* Anshumaan Singh (Enrollment No: A023167023097)
* Lakshay Teotia (Enrollment No: A0231670230120)
* Javin Dhaiya (Enrollment No: A0231670230118)

---

## SYSTEM PROMPT FOR CODE GENERATION AGENT
> You are an expert AI Computer Vision & Deep Learning Engineer. You are assigned to implement the B.Tech Minor Project titled **"Image Degradation Restoration Using Advanced Deep Learning Techniques"** (`Image-Restoration`). 
> Follow the architectural choices, directory layout, loss formulations, and coding standards defined in this document strictly. Do not use GANs. Write clean, modular, production-ready PyTorch code with docstrings, type hints, and hardware acceleration support (`torch.cuda.amp`).

---

## 1. Executive Summary & Problem Definition

### Problem Statement
Real-world visual data captured by digital sensors frequently suffers from severe quality degradation due to hardware constraints, atmospheric interference (fog, haze, rain), extreme low-light environments, and high ISO sensor noise. Traditional mathematical filtering techniques (e.g., Bilateral Filtering, BM3D, Total Variation Minimization) rely on rigid statistical priors, resulting in edge over-smoothing and loss of fine spatial textures.

Early deep learning approaches using standard Convolutional Neural Networks (CNNs) are limited by localized $3 \times 3$ receptive fields, making them incapable of modeling long-range spatial dependencies across an image. Legacy Generative Adversarial Networks (GANs) achieve visual sharpness but suffer from severe training instability, mode collapse, and artificial texture hallucination (generating false structures not present in the ground truth).

### Project Objective
To design, implement, and benchmark a modular, deep learning-based image restoration pipeline that eliminates adversarial training instabilities while capturing both local texture details and global spatial contexts. The architecture leverages **Vision Transformers (ViTs)** and **Latent Diffusion Models (LDMs)** to achieve high-fidelity image restoration across super-resolution, sensor denoising, and synthetic artifact removal.

---

## 2. Technical Architectural Pivot (Crucial Design Constraint)

### Abandoned Approach: Generative Adversarial Networks (GANs)
* **Architectures Evaluated:** SRGAN, ESRGAN, Pix2Pix, CycleGAN.
* **Reasons for Rejection:**
  1. **Adversarial Instability:** Minimax optimization between the Generator and Discriminator frequently leads to non-convergence or vanishing gradients.
  2. **Mode Collapse:** The Generator outputs limited repetitive patterns to fool the Discriminator.
  3. **Texture Hallucination:** GAN loss functions often create false high-frequency artifacts, rendering them unreliable for fidelity-critical computer vision tasks.

### Selected Approach: Vision Transformers & Latent Diffusion Models
* **Primary Backbones:** **SwinIR** (Shifted Window Transformer) and **Restormer** (Multi-Head Transposed Attention).
* **Generative Framework:** **Latent Diffusion Models (LDM)**.
* **Core Technical Advantages:**
  * **Global Receptive Field:** Self-attention mechanisms model contextual relationships across non-local image patches.
  * **Linear Computational Complexity:** Restormer's channel-wise attention (MDTA) reduces computational overhead from $O(N^2)$ spatial complexity to $O(N)$ channel complexity.
  * **Progressive Denoising:** Diffusion models execute step-by-step noise removal in a compressed latent space without adversarial dynamics, providing mathematical stability and exact structural retention.

---

## 3. Dataset Engineering & Preprocessing Specifications

### Benchmark Datasets Sourced
1. **DIV2K Dataset:**
   * **Size:** 1,000 high-resolution ($2\text{K}$) images (800 training, 100 validation, 100 testing).
   * **Primary Task:** High-resolution upscaling, super-resolution, and edge reconstruction.
2. **SIDD (Smartphone Image Denoising Dataset):**
   * **Size:** Thousands of real-world noisy-clean image pairs captured under varying ISO and lighting conditions across smartphone camera sensors.
   * **Primary Task:** Real sensor noise suppression (removing non-Gaussian, real-world camera noise).

### Preprocessing & Data Loading Pipeline
* **Patch Extraction:** Automated cropping of high-resolution images into uniform sub-image patches:
  * Small scale: $128 \times 128$ pixels (for rapid iteration and lightweight training).
  * Standard scale: $256 \times 256$ pixels (for full-scale feature learning).
* **Synthetic Degradation Engine (On-the-Fly / Pre-generated):**
  * **Additive White Gaussian Noise (AWGN):** $\sigma \in [10, 50]$.
  * **Poisson Noise:** Simulating low-light photon shot noise.
  * **Bicubic Downsampling:** Degradation factors of $\times 2$, $\times 4$ for super-resolution pipelines.
* **Data Augmentations:** Random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and $90^\circ$ spatial rotations to ensure rotation invariance.
* **PyTorch Loader:** Modular `Dataset` and `DataLoader` classes configured with multi-process prefetching (`num_workers=4`, `pin_memory=True`).

---

## 4. Model Architecture & Loss Function Formulations

### Network Design Architecture
1. **Shallow Feature Extractor:** A $3 \times 3$ convolutional layer mapping the input degraded image $I_{\text{deg}} \in \mathbb{R}^{H \times W \times C}$ to low-level feature space $F_0 \in \mathbb{R}^{H \times W \times C_{\text{feat}}}$.
2. **Deep Feature Extraction Backbone:**
   * **Residual Swin Transformer Blocks (RSTB):** Utilizing Shifted Window self-attention to capture cross-patch non-local features.
   * **Multi-Head Transposed Attention (MDTA) & Gated-Dconv Feed-Forward Network (GDFN):** For channel-wise attention with linear computational overhead.
3. **High-Resolution Reconstruction Head:** Upsampling sub-modules (PixelShuffle / Sub-pixel convolution) mapping deep features back to restored image space $\hat{I} \in \mathbb{R}^{H \times W \times C}$.

### Mathematical Loss Function Formulation
To optimize both direct pixel-level structural accuracy and visual perceptual quality, the model uses a **Composite Loss Function**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + \lambda_{\text{perceptual}} \cdot \mathcal{L}_{\text{perceptual}}$$

1. **Pixel Reconstruction Loss ($\mathcal{L}_1$):**
   $$\mathcal{L}_{1} = \frac{1}{N} \sum_{i=1}^{N} \Vert{}\hat{I}_{i} - I_{\text{GT}, i}\Vert{}_{1}$$
   Enforces exact numerical pixel-wise alignment without the blurriness caused by $\mathcal{L}_2$ (MSE) loss.

2. **Perceptual Feature Loss ($\mathcal{L}_{\text{perceptual}}$):**
   $$\mathcal{L}_{\text{perceptual}} = \sum_{l} \frac{1}{H_l W_l C_l} \Vert{}\phi_l(\hat{I}) - \phi_l(I_{\text{GT}})\Vert{}_{2}^{2}$$
   Where $\phi_l(\cdot)$ represents deep feature maps extracted from pretrained VGG-19 network layers (`conv3_3`, `conv4_3`, `conv5_3`) or LPIPS feature spaces. This forces the model to recover perceptually realistic textures and sharp edges.
3. **Weighting Hyperparameter:** $\lambda_{\text{perceptual}} = 0.01 \text{ to } 0.1$ (empirically tuned to prevent artifact synthesis).

---

## 5. Technology Stack & Development Infrastructure

* **Core Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch 2.x, `torchvision`
* **Computer Vision & Image Processing:** OpenCV (`cv2`), PIL (Pillow), `scikit-image`, SciPy
* **Compute Optimization:** Automatic Mixed Precision (`torch.cuda.amp.autocast`), PyTorch CUDA backend
* **Experiment Logging & Visualization:** TensorBoard, Matplotlib, Seaborn
* **Hardware Environment:** Cloud GPUs (Google Colab Pro T4/P100/A100 instances, AWS CUDA runtimes)
* **Version Control:** Git & GitHub (`Image-Restoration`)

---

## 6. Evaluation Metrics & Benchmarking Protocol

### Quantitative Evaluation Metrics
1. **Peak Signal-to-Noise Ratio (PSNR):**
   $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$
   Measures absolute pixel reconstruction fidelity in decibels ($dB$). Higher is better.
2. **Structural Similarity Index Metric (SSIM):**
   Evaluates perceptual quality based on luminance, contrast, and structural comparisons over localized image windows ($0 \le \text{SSIM} \le 1$). Higher is better.
3. **Learned Perceptual Image Patch Similarity (LPIPS):**
   Measures deep feature distance across deep neural layers. Correlates directly with human visual preference ($0 \le \text{LPIPS} \le 1$). Lower is better.

### Baseline Models for Benchmarking
* Classical baselines: Bilateral Filter, BM3D.
* Convolutional baselines: SRCNN, DnCNN, RCAN.
* Transformer baselines: NAFNet, SwinIR, Restormer.

---

## 7. Complete Project Directory Structure

```text
Image-Restoration/
│
├── data/
│   ├── raw/                  # Downloaded DIV2K and SIDD dataset archives
│   ├── processed/            # Extracted image patches (128x128, 256x256)
│   └── degradation/          # Pre-generated synthetic noisy/downscaled pairs
│
├── models/
│   ├── __init__.py
│   ├── swinir.py             # Swin Transformer restoration backbone implementation
│   ├── restormer.py          # Multi-Head Transposed Attention implementation
│   ├── diffusion.py          # Latent Diffusion Model & U-Net modules
│   └── common.py             # Convolutional blocks, residual blocks, upsamplers
│
├── losses/
│   ├── __init__.py
│   ├── l1_loss.py            # Pixel-level L1 loss module
│   ├── perceptual.py         # VGG-19 / LPIPS feature loss extraction module
│   └── composite.py          # Weighted composite loss calculation
│
├── utils/
│   ├── __init__.py
│   ├── dataset_loader.py     # Custom PyTorch Dataset and DataLoader implementations
│   ├── degradation.py        # Gaussian, Poisson, and bicubic degradation routines
│   ├── metrics.py            # Automated PSNR, SSIM, and LPIPS computation scripts
│   └── logger.py             # TensorBoard logging utilities and checkpoint manager
│
├── experiments/
│   ├── checkpoints/          # Saved model weights (.pth files)
│   └── logs/                 # TensorBoard event logs and loss curves
│
├── train.py                  # End-to-end training execution pipeline with AMP
├── eval.py                   # Test dataset evaluation and benchmarking script
├── inference.py              # Single/batch image restoration script for deployment
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation and usage guide


8. Dummy prompts for the Implementation
When building this project with an AI agent (Cursor, Claude, Copilot, ChatGPT), issue these targeted prompts in sequential order:
Phase 1: Environment & Preprocessing Engine
Prompt 1: "Based on MASTER_PROJECT_CONTEXT.md, generate utils/degradation.py and utils/dataset_loader.py. Write modular PyTorch scripts to take clean image datasets (DIV2K/SIDD), extract sub-image patches (256×256), apply synthetic degradation (Gaussian noise σ=15..50, Poisson noise, and bicubic downsampling ×2,×4), and return paired PyTorch Tensors (degraded, ground_truth) normalized in [0,1]."

Phase 2: Loss Function Engine
Prompt 2: "Generate losses/perceptual.py and losses/composite.py. Implement a composite loss class in PyTorch combining L 
1
​	
  pixel loss with VGG-19 perceptual feature loss extracted from layers conv3_3 and conv4_3. Ensure VGG-19 features are frozen (requires_grad=False) and handle GPU device mapping cleanly."

Phase 3: Model Architecture
Prompt 3: "Generate models/restormer.py and models/common.py. Implement the Restormer architecture using Multi-Head Transposed Attention (MDTA) and Gated-Dconv Feed-Forward Network (GDFN) blocks in PyTorch. Ensure input tensor shapes are verified during the forward pass and support dynamic spatial resolutions."

Phase 4: End-to-End Training Script
Prompt 4: "Generate train.py. Write a full PyTorch training loop using AdamW optimizer, CosineAnnealingLR scheduler, torch.cuda.amp.GradScaler() for mixed-precision training, TensorBoard logging (utils/logger.py), and automated checkpoint saving to experiments/checkpoints/ when validation PSNR improves."

Phase 5: Evaluation & Inference Engine
Prompt 5: "Generate eval.py and inference.py. Write an evaluation script that runs test images through trained checkpoints in torch.no_grad() mode, computes exact average PSNR, SSIM, and LPIPS metrics using scikit-image / torchmetrics, and outputs side-by-side visual comparison grids."
