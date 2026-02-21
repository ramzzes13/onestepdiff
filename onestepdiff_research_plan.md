# Research Plan: LoRA-Efficient Distribution Matching Distillation with Adaptive Score Weighting for One-Step Image Generation

**Working Title:** *AdaDMD: Adaptive, Memory-Efficient Distribution Matching Distillation via Low-Rank Score Estimation and Learned Density-Ratio Weighting*

**Based on:** Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024). *One-step Diffusion with Distribution Matching Distillation.* CVPR 2024. arXiv:2311.18828.

---

## 1. Introduction

### 1.1 Problem Statement

Diffusion models have become the dominant paradigm for high-fidelity image synthesis, powering systems such as Stable Diffusion, DALL-E, and Imagen. However, their iterative denoising procedure—typically requiring 20 to 1000 neural network forward passes—incurs prohibitive latency for interactive and real-time applications. Distilling these multi-step models into single-step generators is therefore a critical research direction for making generative AI practical in deployment scenarios such as creative tools, mobile applications, and real-time content creation.

### 1.2 Summary of the Original Work

Distribution Matching Distillation (DMD) [Yin et al., 2024] introduced an elegant framework for one-step distillation by minimizing an approximate KL divergence between the distributions of generated (fake) and real images. The key insight is that the gradient of this KL divergence can be expressed as the difference between two score functions: one for the real distribution (parameterized by the frozen pretrained diffusion model) and one for the fake distribution (parameterized by a dynamically updated diffusion model trained on the generator's outputs). This distribution-level matching is complemented by a regression loss (LPIPS) on pre-computed noise–image pairs, which prevents mode collapse and stabilizes training. DMD achieved remarkable results—2.62 FID on ImageNet 64×64 and 11.49 FID on zero-shot MS-COCO 30k—while reducing inference cost by 100× compared to multi-step Stable Diffusion.

### 1.3 Limitations of the Original Work

Despite its success, DMD has several notable limitations:

1. **Extreme memory requirements.** Training requires maintaining three full-size diffusion models in memory simultaneously (the frozen real score model, the dynamically updated fake score model, and the generator), making training feasible only on large GPU clusters (72 A100 GPUs for the LAION experiments).
2. **Dependence on pre-computed regression pairs.** The regression loss requires generating millions of noise–image pairs from the teacher model offline, which is both time-consuming and constrains the student to replicate the teacher's specific sampling trajectories rather than learning a potentially superior mapping.
3. **Fixed, heuristic timestep weighting.** While DMD proposes an improved weighting scheme (Eq. 8 in the paper), it remains a hand-designed heuristic that normalizes gradient magnitudes across noise levels without learning the optimal emphasis for each diffusion timestep.
4. **Limited architectural scope.** DMD was evaluated only on Stable Diffusion v1.5 (UNet-based). Modern generative models have shifted to transformer-based architectures such as MMDiT (Stable Diffusion 3, FLUX), whose distillation properties under distribution matching remain unexplored.
5. **Persistent quality gap.** A noticeable FID gap remains between the one-step generator and the multi-step teacher, particularly in fine-grained details and compositional accuracy.

### 1.4 Goal of the Proposed Work

This work proposes **AdaDMD** (Adaptive Distribution Matching Distillation), a memory-efficient and adaptive extension of DMD that addresses the above limitations through three synergistic innovations:

1. **LoRA-based fake score estimation** — replacing the full-copy fake diffusion model with a lightweight Low-Rank Adaptation (LoRA) module on the pretrained base model, reducing memory consumption by approximately 60% and enabling training on a single high-end GPU or a small cluster.
2. **Learned density-ratio weighting** — replacing the fixed timestep weighting heuristic with an adaptive weighting scheme derived from a lightweight density-ratio estimator that learns to allocate gradient emphasis optimally across noise levels during training.
3. **Regression-free training with density-ratio verification** — leveraging the learned density ratio as both a training signal to replace or reduce reliance on the expensive regression loss and as an inference-time sample verifier for quality-aware generation.

These contributions aim to democratize one-step diffusion distillation (making it accessible on limited hardware), improve generation quality through adaptive optimization, and generalize the framework to modern transformer-based diffusion architectures.

---

## 2. Literature Review

### 2.1 Distribution Matching Distillation and Its Evolution

**DMD** [Yin et al., 2024] pioneered the use of dual score functions for one-step distillation by interpreting the gradient of the KL divergence between real and fake distributions as the difference of denoising scores. This was complemented by a regression loss to prevent mode collapse. The method achieved 2.62 FID on ImageNet 64×64 and 11.49 FID on COCO-30k.

**DMD2** [Yin et al., 2024b] (NeurIPS 2024 Oral) significantly improved the original framework by: (a) eliminating the regression loss entirely through a two-time-scale update rule for the fake score model, (b) integrating a GAN discriminator to leverage real data and mitigate imperfect score estimation, and (c) introducing backward simulation for multi-step inference compatibility. DMD2 achieved 1.28 FID on ImageNet 64×64 and 8.35 FID on COCO-30k, surpassing the teacher model while maintaining a 500× inference speedup. However, DMD2's memory footprint remains substantial, as it adds a discriminator on top of the three-model architecture.

### 2.2 Unified Divergence Frameworks

**Uni-Instruct** [Zhang et al., 2025] (NeurIPS 2025) unified over 10 one-step distillation methods (including DMD, Diff-Instruct, SiD, and f-distill) within a theory-driven framework based on the diffusion expansion of f-divergence families. By deriving a tractable loss from the intractable expanded f-divergence, Uni-Instruct achieved record FIDs of 1.02 on ImageNet 64×64 and 1.38 on conditional CIFAR-10, demonstrating that the choice of divergence measure significantly impacts distillation quality.

**f-distill** [2025] similarly generalized distribution matching to arbitrary f-divergences, revealing trade-offs between mode coverage and training variance that the original KL-based DMD formulation does not address.

### 2.3 Teacher-Free and Simplified Approaches

**DiffRatio** [2025] proposed estimating the score difference directly as the gradient of a learned log density ratio between student and data distributions, eliminating teacher score supervision entirely. This reduces gradient estimation bias and replaces two full score networks with a single lightweight density-ratio network. DiffRatio also showed that the learned density ratio naturally serves as a sample verifier for inference-time quality scaling.

**Contrastive Energy Distillation (CED)** [Zhu et al., 2025] (NeurIPS 2025) introduced an unnormalized joint energy-based model optimized via Noise Contrastive Estimation, implicitly minimizing KL divergence between teacher and student without auxiliary score models or iterative multi-component training.

**Score Identity Distillation (SiD)** [Zhou et al., 2024] (ICML 2024) achieved exponentially fast distillation by leveraging score-related identities on semi-implicit distributions, requiring no real data. Its 2025 extension, **SiDA**, incorporated adversarial losses and demonstrated SOTA one-step generation on ImageNet 512×512 and SDXL at 1024×1024 resolution.

### 2.4 Multi-Student and Ensemble Approaches

**Multi-Student Distillation (MSD)** [NVIDIA et al., 2025] (ICML 2025) distilled a conditional diffusion teacher into multiple single-step student generators, each specializing in a subset of conditioning data. Using 4 students, MSD achieved 1.20 FID on ImageNet 64×64 and 8.20 on COCO-30k, suggesting that partitioned specialization can improve distillation quality.

### 2.5 Consistency Models and Flow Matching

**Flow-Anchored Consistency Models (FACM)** [2025] addressed training instability in consistency models through flow-matching anchoring, achieving 1.70 FID in 1 step and 1.32 in 2 steps on ImageNet 256×256, and scaling to 14B-parameter models for text-to-image generation.

**Stable Consistency Tuning (SCT)** [2025] reinterpreted consistency training through the lens of Temporal Difference learning and introduced variance-reduced training via score identity regularization.

**Truncated Consistency Models** [2024] focused on restricted time ranges, achieving better FID with 2× smaller networks by concentrating model capacity on generation rather than early denoising.

### 2.6 Adversarial Distillation Approaches

**Adversarial Diffusion Distillation (ADD)** [Sauer et al., 2023] combined score distillation with an adversarial loss to train SDXL-Turbo, achieving real-time single-step generation. **Latent Adversarial Diffusion Distillation (LADD)** [Stability AI, 2024] improved upon ADD by operating in latent space, enabling SD3-Turbo to generate high-quality images in 4 unguided steps.

### 2.7 Memory-Efficient Distillation

**RAPM** [2025] (Relative and Absolute Position Matching) demonstrated that effective diffusion distillation is achievable on a single GPU with a batch size of 1, using trajectory matching with two lightweight discriminators. While RAPM's quality at 4 steps was competitive, its one-step results lagged behind DMD2.

**LCM-LoRA** [Luo et al., 2023] showed that LoRA can serve as a universal acceleration module for Stable Diffusion variants, enabling few-step sampling with minimal fine-tuning. However, this operates on the sampling side rather than full distillation.

**IntLoRA** [2024] and related quantization-aware LoRA methods demonstrated that low-rank adaptations of diffusion models can maintain quality with dramatically reduced memory.

### 2.8 Extension to Video and 3D

DMD's distribution matching framework has been extended to video generation [Yin et al., CVPR 2025], achieving 9.4 FPS streaming video generation, and to text-to-3D synthesis via the VSD formulation [Wang et al., 2023]. These extensions highlight the generality of the dual-score framework but also its increasing memory demands in higher-dimensional settings.

### 2.9 Summary of Research Gaps

| Gap | Addressed by prior work? | Our contribution |
|-----|-------------------------|------------------|
| Memory efficiency of DMD | Partially (RAPM, LCM-LoRA) | LoRA-based fake score model |
| Adaptive timestep weighting | Not addressed | Learned density-ratio weighting |
| Regression loss dependency | DMD2 (via GAN loss) | Density-ratio verification |
| Modern architecture support | Not systematically studied | Experiments on SD v1.5 + SD3/FLUX |
| Single-GPU training | RAPM (trajectory matching only) | Full distribution matching on 1 GPU |

---

## 3. Proposed Methodology

### 3.1 Overview

AdaDMD retains the core dual-score distribution matching framework of DMD but introduces three key modifications: (1) a parameter-efficient fake score model, (2) a learned adaptive weighting module, and (3) an optional density-ratio-based regularizer that can replace the regression loss. The overall training objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DM}}^{\text{ada}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}^{\text{hybrid}}$$

where $\mathcal{L}_{\text{DM}}^{\text{ada}}$ is the adaptively-weighted distribution matching loss and $\mathcal{L}_{\text{reg}}^{\text{hybrid}}$ is a hybrid regularizer combining (optionally) a small regression component with a density-ratio-based mode coverage term.

### 3.2 LoRA-Based Fake Score Estimation

**Motivation.** In the original DMD, the fake score model $\mu_{\text{fake}}^{\phi}$ is a full copy of the pretrained diffusion model, initialized from $\mu_{\text{base}}$ and continuously updated to track the generator's changing distribution. This doubles the memory requirement beyond the already large generator and frozen real score model. However, the fake distribution evolves gradually from the real distribution (since the generator is initialized from the base model), suggesting that the difference between the real and fake score functions can be captured by a low-rank perturbation.

**Method.** We replace $\mu_{\text{fake}}^{\phi}$ with $\mu_{\text{base}} + \Delta_{\text{LoRA}}^{\phi}$, where $\Delta_{\text{LoRA}}^{\phi}$ is a set of LoRA adapter weights [Hu et al., 2022] applied to the attention layers (and optionally the convolutional/MLP layers) of the frozen pretrained model. The fake score denoising loss becomes:

$$\mathcal{L}_{\text{denoise}}^{\phi} = \left\| (\mu_{\text{base}} + \Delta_{\text{LoRA}}^{\phi})(x_t, t) - x_0 \right\|_2^2$$

where $x_0 = \text{stopgrad}(G_\theta(z))$ and $x_t \sim q_t(x_t | x_0)$.

**LoRA rank schedule.** At the beginning of training, the generator output closely resembles the base model's distribution, so a very low LoRA rank (e.g., $r=4$) suffices. As training progresses and the distributions diverge, we propose a rank-warming schedule that gradually increases the LoRA rank (e.g., $r=4 \to 16 \to 64$) by introducing new rank components initialized from the current adapter state. This ensures that the fake score model can capture increasing distributional complexity without sudden capacity shocks.

**Memory savings.** For Stable Diffusion v1.5 (860M parameters), a rank-64 LoRA adapter adds approximately 34M parameters (4% of the original). This reduces peak GPU memory from approximately 3× the base model size (real + fake + generator) to approximately 2.04× (real/generator shared backbone + LoRA), enabling training on a single 80GB A100 or even a 48GB A6000 with gradient checkpointing.

### 3.3 Learned Density-Ratio Weighting

**Motivation.** DMD's timestep weighting $w_t$ (Eq. 8) heuristically normalizes gradient magnitudes across noise levels. However, the optimal weighting should depend on where the real and fake distributions diverge most at each timestep—information that changes as training proceeds. Inspired by DiffRatio [2025], we propose learning this weighting explicitly.

**Method.** We introduce a lightweight density-ratio network $r_\psi(x_t, t)$ that estimates the log density ratio $\log \frac{p_{\text{fake},t}(x_t)}{p_{\text{real},t}(x_t)}$ for noised samples at each timestep. This network is trained via a binary classification objective (noise contrastive estimation):

$$\mathcal{L}_{\text{NCE}}^{\psi} = -\mathbb{E}_{x_t \sim p_{\text{real},t}} [\log \sigma(r_\psi(x_t, t))] - \mathbb{E}_{x_t \sim p_{\text{fake},t}} [\log(1 - \sigma(r_\psi(x_t, t)))]$$

where $\sigma$ is the sigmoid function, real noised samples are constructed by adding noise to images from the training set, and fake noised samples are constructed from generator outputs.

The learned density ratio provides two signals:

1. **Adaptive weighting:** The magnitude $|r_\psi(x_t, t)|$ indicates how distinguishable the real and fake distributions are at timestep $t$. We define the adaptive weight:

$$w_t^{\text{ada}} = \frac{\sigma_t^2}{\alpha_t} \cdot \frac{1}{\text{EMA}(|r_\psi(x_t, t)|) + \epsilon}$$

This downweights timesteps where the distributions are already well-aligned (small density ratio) and focuses gradient budget on timesteps with large distributional mismatch.

2. **Mode coverage signal:** The expected density ratio $\mathbb{E}_{x \sim p_{\text{fake}}}[r_\psi(x_t, t)]$ provides a per-sample signal of whether the generator is producing samples in low-density regions of the real distribution, which can be used to modulate the regression loss or serve as a diversity regularizer.

**Architecture.** The density-ratio network $r_\psi$ is implemented as a small UNet (e.g., 1/4 the channels of the base model) with timestep conditioning. Alternatively, it can share the base model's encoder with a lightweight classification head, further reducing overhead.

### 3.4 Hybrid Regularization: Reducing Regression Loss Dependence

**Motivation.** DMD2 showed that the regression loss can be removed if a GAN discriminator is added. However, this adds yet another full-resolution discriminator to the training pipeline. We propose a lighter alternative.

**Method.** We use the density-ratio network as a soft regularizer against mode collapse. The density-ratio verification loss is:

$$\mathcal{L}_{\text{DR}} = \mathbb{E}_{z \sim \mathcal{N}(0, \mathbf{I})} \left[ \max(0, -r_\psi(G_\theta(z), 0) + \delta) \right]$$

where $\delta$ is a margin hyperparameter. This loss penalizes the generator for producing samples that the density-ratio network confidently classifies as fake (i.e., far from the real distribution), providing mode coverage without requiring explicit noise–image pairs.

Our hybrid regularization combines a reduced regression component with the density-ratio term:

$$\mathcal{L}_{\text{reg}}^{\text{hybrid}} = \lambda_{\text{lpips}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{DR}} \mathcal{L}_{\text{DR}}$$

We hypothesize that the regression loss weight $\lambda_{\text{lpips}}$ can be reduced (or even set to zero in later training) when the density-ratio regularizer provides sufficient mode coverage.

### 3.5 Adaptation to Modern Architectures

For transformer-based diffusion models (SD3, FLUX), we adapt the LoRA-based fake score model by applying low-rank adaptations to the MMDiT attention layers (both self-attention and cross-attention with text embeddings). The density-ratio network is implemented as a lightweight transformer encoder with shared positional embeddings from the base model.

### 3.6 Multi-Step Extension via Backward Simulation

Following DMD2, we incorporate backward simulation to enable optional multi-step inference (2–4 steps) for applications requiring higher quality. The LoRA-based fake score model naturally supports this, as it can be composed with the base model at different noise levels during multi-step denoising.

### 3.7 Pseudocode

```
Algorithm: AdaDMD Training

Input: Pretrained diffusion model μ_base, (optional) paired dataset D = {z, y}
Output: Trained generator G_θ

1. Initialize G_θ ← copyWeights(μ_base), remove time conditioning
2. Initialize LoRA adapter Δ_LoRA^φ with rank r_init = 4
3. Initialize density-ratio network r_ψ

4. while training do
5.     // Generate fake samples
6.     Sample z ~ N(0, I), compute x_fake = G_θ(z)
7.
8.     // Update density-ratio network
9.     Sample x_real from dataset, sample t ~ U(T_min, T_max)
10.    Noise both: x_real_t, x_fake_t via forward diffusion at timestep t
11.    Update ψ via L_NCE(r_ψ, x_real_t, x_fake_t, t)
12.
13.    // Compute adaptive distribution matching loss
14.    Compute s_real(x_fake_t, t) via frozen μ_base
15.    Compute s_fake(x_fake_t, t) via μ_base + Δ_LoRA^φ
16.    Compute adaptive weight w_t^ada using r_ψ
17.    L_DM^ada = weighted_dm_loss(s_real, s_fake, w_t^ada)
18.
19.    // Compute hybrid regularization
20.    L_DR = density_ratio_regularizer(r_ψ, x_fake)
21.    L_LPIPS = lpips_loss(G_θ(z_ref), y_ref) if regression pairs available
22.    L_reg = λ_lpips * L_LPIPS + λ_DR * L_DR
23.
24.    // Update generator
25.    Update θ via L_DM^ada + L_reg
26.
27.    // Update LoRA fake score model
28.    x_t = forwardDiffusion(stopgrad(x_fake), t)
29.    L_denoise = ||(μ_base + Δ_LoRA^φ)(x_t, t) - stopgrad(x_fake)||²
30.    Update φ via L_denoise
31.
32.    // Rank warming (periodic)
33.    if iteration % rank_warm_interval == 0 and rank < r_max:
34.        Increase LoRA rank
35. end while
```

---

## 4. Experimental Design

### 4.1 Datasets

| Dataset | Resolution | Purpose | Size |
|---------|-----------|---------|------|
| CIFAR-10 [Krizhevsky, 2009] | 32×32 | Rapid prototyping, ablation studies | 50K train |
| ImageNet [Deng et al., 2009] | 64×64 | Class-conditional benchmarking | 1.28M train |
| MS-COCO 2014 [Lin et al., 2014] | 512×512 | Zero-shot text-to-image evaluation | 30K val prompts |
| LAION-Aesthetics 6.25+ [Schuhmann et al., 2022] | 512×512 | Text-to-image training | ~3M images |
| LAION-Aesthetics 6+ [Schuhmann et al., 2022] | 512×512 | Extended text-to-image training | ~12M images |
| GenEval [Ghosh et al., 2024] | 512–1024 | Compositional generation evaluation | 553 prompts |

CIFAR-10 and ImageNet 64×64 serve as computationally inexpensive benchmarks for rapid ablation. MS-COCO 2014 zero-shot 30k is the standard text-to-image benchmark. LAION subsets are used for text-to-image distillation training. GenEval is included to assess compositional understanding, a known weakness of one-step generators.

### 4.2 Baselines

**Original methods:**
- DMD [Yin et al., 2024] — the foundation this work extends
- DMD2 [Yin et al., 2024b] — the current best DMD variant

**Consistency-based:**
- Consistency Models (CM/iCT) [Song et al., 2023]
- Latent Consistency Models (LCM / LCM-LoRA) [Luo et al., 2023]
- Flow-Anchored Consistency Models (FACM) [2025]

**Score distillation methods:**
- Score Identity Distillation (SiD/SiDA) [Zhou et al., 2024–2025]
- DiffRatio [2025]
- Uni-Instruct [Zhang et al., 2025]

**Adversarial methods:**
- ADD / SDXL-Turbo [Sauer et al., 2023]
- LADD / SD3-Turbo [Stability AI, 2024]

**Memory-efficient methods:**
- RAPM [2025]
- LCM-LoRA [Luo et al., 2023]

**Other:**
- Progressive Distillation [Salimans & Ho, 2022]
- InstaFlow [Liu et al., 2023]
- Multi-Student Distillation (MSD) [2025]

### 4.3 Evaluation Metrics

| Metric | What it measures | Justification |
|--------|-----------------|---------------|
| **FID** (Fréchet Inception Distance) | Distribution-level image quality and diversity | Standard benchmark metric; enables direct comparison with all baselines |
| **CLIP Score** (OpenCLIP ViT-G) | Text–image alignment | Captures prompt adherence, critical for text-to-image models |
| **LPIPS** (Learned Perceptual Image Patch Similarity) | Perceptual similarity to teacher outputs | Measures how closely the student reproduces the teacher's mapping |
| **Precision / Recall** | Quality vs. diversity decomposition | Disentangles mode collapse (low recall) from quality (low precision) |
| **Inference latency** (ms/image) | Wall-clock generation speed | Critical for real-time applications; measured at batch size 1 on A100 |
| **Peak GPU memory** (GB) | Training resource requirements | Core contribution: demonstrates accessibility on limited hardware |
| **Training cost** (GPU-hours) | Total computational budget | Demonstrates feasibility for academic labs |
| **GenEval score** | Compositional generation accuracy | Evaluates whether distilled model preserves compositional abilities |

### 4.4 Implementation Details

**Hardware:**
- Primary experiments: 1–2 NVIDIA A100 (80GB) GPUs
- Scaling experiments: up to 8 A100s for LAION-scale training
- All experiments should also be profiled on a single A6000 (48GB) to demonstrate accessibility

**Framework:** PyTorch 2.x with `torch.compile`, Hugging Face Diffusers, PEFT library for LoRA

**Base models:**
- EDM [Karras et al., 2022] for CIFAR-10 and ImageNet 64×64
- Stable Diffusion v1.5 [Rombach et al., 2022] for text-to-image at 512×512
- Stable Diffusion 3 Medium [Esser et al., 2024] for MMDiT architecture validation

**Hyperparameters (initial settings, subject to tuning):**

| Parameter | CIFAR-10 | ImageNet 64 | SD v1.5 (LAION) |
|-----------|----------|-------------|-----------------|
| Learning rate (generator) | 5e-5 | 2e-6 | 1e-5 |
| Learning rate (LoRA φ) | 1e-4 | 5e-5 | 5e-5 |
| Learning rate (density ratio ψ) | 1e-4 | 1e-4 | 1e-4 |
| LoRA rank (initial → max) | 4 → 32 | 4 → 64 | 8 → 64 |
| LoRA target modules | attn q,k,v,o | attn q,k,v,o | attn q,k,v,o + ff |
| λ_lpips | 0.1 | 0.1 | 0.1 |
| λ_DR | 0.5 | 0.5 | 0.5 |
| Batch size | 64 | 48 | 16–32 |
| Optimizer | AdamW | AdamW | AdamW |
| T_min, T_max | 0.02T, 0.98T | 0.02T, 0.98T | 0.02T, 0.50T |
| Training iterations | 300K | 350K | 20K–50K |
| Gradient checkpointing | No | Yes | Yes |
| Mixed precision | FP16 | FP16 | FP16 |

### 4.5 Ablation Studies

Each ablation isolates one component while keeping all others fixed at their best configuration. All ablations are conducted on CIFAR-10 (for speed) and validated on ImageNet 64×64.

#### Ablation 1: LoRA Rank for Fake Score Model

**Purpose:** Determine the minimum LoRA rank that preserves distillation quality.

| Configuration | LoRA Rank | Expected FID (CIFAR-10) |
|--------------|-----------|------------------------|
| Full copy (DMD baseline) | N/A (860M params) | ~2.66 |
| LoRA r=4 | 4 (~1.7M params) | ~3.5–4.0 |
| LoRA r=16 | 16 (~6.8M params) | ~2.8–3.2 |
| LoRA r=32 | 32 (~13.6M params) | ~2.7–2.9 |
| LoRA r=64 | 64 (~27.2M params) | ~2.65–2.75 |
| LoRA rank warming (4→64) | 4→64 | ~2.60–2.70 |

**Expected outcome:** Rank warming achieves comparable quality to a full-rank copy while using far fewer parameters during the critical early training phase.

#### Ablation 2: Adaptive vs. Fixed Weighting

| Configuration | Weighting | Expected FID (CIFAR-10) |
|--------------|-----------|------------------------|
| σ_t / α_t [DreamFusion] | Fixed | ~3.60 |
| σ_t³ / α_t [ProlificDreamer] | Fixed | ~3.71 |
| Eq. 8 (DMD original) | Fixed, normalized | ~2.66 |
| Learned density-ratio weighting | Adaptive | ~2.40–2.55 |

**Expected outcome:** Adaptive weighting improves FID by 0.1–0.3 by allocating gradient emphasis to timesteps where the distributions diverge most.

#### Ablation 3: Regularization Strategy

| Configuration | Regression Loss | DR Loss | Expected FID (CIFAR-10) |
|--------------|----------------|---------|------------------------|
| Full LPIPS regression only (DMD) | λ=0.25 | 0 | ~2.66 |
| DR regularizer only | 0 | λ=0.5 | ~3.0–3.5 |
| Hybrid (reduced LPIPS + DR) | λ=0.1 | λ=0.5 | ~2.50–2.60 |
| Hybrid (no LPIPS + DR) | 0 | λ=0.5 | ~2.70–3.0 |

**Expected outcome:** The hybrid approach with reduced LPIPS and DR regularizer matches or improves upon full LPIPS regression while requiring fewer pre-computed pairs.

#### Ablation 4: LoRA Target Layers

| Configuration | Layers adapted | Params (SD v1.5) | Expected FID |
|--------------|----------------|-------------------|-------------|
| Attention only (q, k, v, out) | 4 per block | ~17M | Baseline |
| Attention + feedforward | 6 per block | ~34M | +0.1–0.3 better |
| All linear layers | All | ~68M | Diminishing returns |

#### Ablation 5: Density-Ratio Network Architecture

| Configuration | Architecture | Params | FID Impact |
|--------------|-------------|--------|------------|
| Small UNet (1/4 channels) | Independent | ~55M | Baseline |
| Shared encoder + classification head | Shared | ~5M extra | Slightly worse |
| Lightweight MLP on base model features | Feature-based | ~2M extra | Similar or slightly worse |

#### Ablation 6: Number of Regression Pairs

| # Pre-computed pairs | Original DMD requirement | With AdaDMD |
|---------------------|-------------------------|-------------|
| 0 (no regression) | Diverges | Stable (DR loss) |
| 10K | Poor quality | Competitive |
| 50K | Moderate | Near-best |
| 100K–500K | DMD standard | Marginal improvement |

**Expected outcome:** AdaDMD can achieve competitive results with 10× fewer regression pairs (or none at all).

### 4.6 Main Experiments

#### Experiment 1: Class-Conditional ImageNet 64×64

**Setup:** Distill EDM [Karras et al., 2022] pretrained model using AdaDMD.

**Expected results:**

| Method | # Steps | FID (↓) | GPU Memory | GPU-hours |
|--------|---------|---------|------------|-----------|
| DMD | 1 | 2.62 | ~45 GB | ~500 |
| DMD2 | 1 | 1.28 | ~55 GB | ~600 |
| Uni-Instruct | 1 | 1.02 | ~50 GB | ~800 |
| **AdaDMD (ours)** | **1** | **~1.5–2.0** | **~25 GB** | **~300** |

**Claim:** AdaDMD achieves competitive FID with approximately 50% less memory and fewer GPU-hours.

#### Experiment 2: Zero-Shot Text-to-Image on MS-COCO 30k

**Setup:** Distill Stable Diffusion v1.5 on LAION-Aesthetics 6.25+ using AdaDMD.

**Expected results:**

| Method | # Steps | Latency | FID (↓) | CLIP Score (↑) |
|--------|---------|---------|---------|----------------|
| SD v1.5 (teacher) | 50 | 2.59s | 8.78 | — |
| DMD | 1 | 0.09s | 11.49 | — |
| DMD2 | 1 | 0.09s | 8.35 | — |
| SwiftBrush v2 | 1 | 0.09s | 8.14 | — |
| **AdaDMD (ours)** | **1** | **0.09s** | **~8.5–9.5** | **~0.315–0.320** |

**Claim:** AdaDMD closely approaches DMD2's quality while training on a fraction of the computational budget (8 GPUs vs. 72 GPUs).

#### Experiment 3: Memory and Efficiency Analysis

**Setup:** Profile peak GPU memory and training throughput across methods on SD v1.5 distillation.

**Expected results:**

| Method | Peak GPU Memory (per GPU) | Min. GPUs Required | Training Time |
|--------|--------------------------|-------------------|---------------|
| DMD | ~65 GB | 72 × A100 | 36 hours |
| DMD2 | ~75 GB | 8 × A100 | 48 hours |
| RAPM | ~30 GB | 1 × A100 | 72 hours |
| **AdaDMD (ours)** | **~35 GB** | **1–2 × A100** | **~48–72 hours** |

**Claim:** AdaDMD is the first distribution-matching distillation method trainable on a single high-end GPU while maintaining competitive quality.

#### Experiment 4: Generalization to Transformer Architectures

**Setup:** Distill Stable Diffusion 3 Medium (MMDiT, 2B parameters) using AdaDMD with LoRA applied to MMDiT attention layers.

**Purpose:** Demonstrate that the LoRA-based fake score approach generalizes beyond UNet architectures. Compare against LCM-LoRA and SiDA adapted for SD3.

**Expected outcome:** Successful distillation with FID within 15% of the UNet-based results, confirming architectural generality.

#### Experiment 5: Inference-Time Density-Ratio Verification

**Setup:** Generate N candidate images per prompt and select the best according to the density-ratio verifier $r_\psi$.

| Strategy | N candidates | Latency | FID improvement |
|----------|-------------|---------|-----------------|
| No verification | 1 | 0.09s | Baseline |
| Density-ratio top-1 of 4 | 4 | 0.36s | ~0.5–1.0 better |
| Density-ratio top-1 of 8 | 8 | 0.72s | ~1.0–2.0 better |

**Claim:** The learned density ratio enables principled inference-time scaling that trades compute for quality, unlike arbitrary re-ranking schemes.

#### Experiment 6: Compositional Generation (GenEval)

**Setup:** Evaluate on GenEval benchmark to assess whether distillation preserves compositional understanding (counting, spatial relations, attribute binding).

**Expected outcome:** AdaDMD's density-ratio regularizer may improve compositional accuracy by preventing mode collapse on compositionally complex prompts.

### 4.7 Human Evaluation

Conduct a two-alternative forced choice (2AFC) study with at least 50 participants comparing:
- AdaDMD vs. DMD (direct improvement claim)
- AdaDMD vs. SD v1.5 50-step (quality parity claim)
- AdaDMD vs. DMD2 (competitive quality at lower cost claim)

Each participant evaluates 50 randomly selected prompts from COCO validation, judging both image quality and prompt adherence.

---

## 5. Discussion and Future Work

### 5.1 Potential Pitfalls

1. **LoRA capacity limitations.** If the fake distribution diverges significantly from the real distribution during training, a low-rank adapter may fail to capture the score function accurately, leading to poor gradient signals. The rank-warming schedule mitigates this, but extreme distributional shifts may require higher ranks, partially negating the memory savings.

2. **Density-ratio estimation instability.** Training the density-ratio network alongside the generator creates a moving-target problem. If the ratio network lags behind the generator, the adaptive weights will be stale. We address this with a faster learning rate for $r_\psi$ and EMA smoothing of the weights.

3. **Trade-off between memory and quality.** It is possible that the full-copy fake score model in DMD/DMD2 provides fundamentally better gradient signals than any low-rank approximation. In this case, our contribution would be primarily in the adaptive weighting and density-ratio verification components.

4. **Generalization to very large models.** Scaling to FLUX.1 (12B parameters) may require additional engineering (e.g., model parallelism), which could complicate the single-GPU narrative.

### 5.2 Alternative Interpretations

- The density-ratio weighting may implicitly perform a form of curriculum learning, gradually shifting focus from coarse structure (high noise) to fine details (low noise) as training proceeds. This connection to curriculum learning in generative models deserves theoretical investigation.
- The LoRA-based fake score model can be interpreted as a learned residual between the real and fake score functions, potentially relating to the score-difference formulation in DiffRatio.

### 5.3 Future Directions

1. **Theoretical analysis of LoRA expressiveness for score functions.** Under what conditions can a rank-$r$ perturbation of a trained score function adequately represent the score of a nearby distribution? This could connect to perturbation theory for diffusion processes.

2. **Joint architecture search for LoRA rank and density-ratio network.** Automated methods could find the optimal balance between fake score capacity and density-ratio network expressiveness for a given memory budget.

3. **Extension to video distillation.** Video diffusion models have even more severe memory constraints. LoRA-based distribution matching could enable single-GPU video distillation, a capability not currently achievable.

4. **Extension to 3D generation.** The density-ratio verification mechanism could improve text-to-3D generation (e.g., in ProlificDreamer-style optimization) by providing an adaptive learning signal that replaces fixed CFG weighting.

5. **Continual distillation.** As new base models are released, the LoRA-based approach could enable efficient incremental distillation by adapting existing LoRA weights rather than retraining from scratch.

6. **Combining with multi-student distillation.** The LoRA-based approach naturally supports multiple lightweight student-specific adapters, enabling a memory-efficient variant of MSD.

7. **Distillation-aware model design.** The finding that low-rank score differences suffice could inform the design of diffusion model architectures that are inherently more distillation-friendly.

---

## 6. Conclusion

This research plan proposes **AdaDMD**, an adaptive and memory-efficient extension of Distribution Matching Distillation for one-step image generation. The three core contributions—LoRA-based fake score estimation, learned density-ratio weighting, and density-ratio verification as a regularizer—address the most pressing limitations of the original DMD framework: prohibitive memory requirements, suboptimal fixed weighting, and dependence on expensive pre-computed regression pairs.

By reducing the memory footprint by approximately 50%, AdaDMD aims to make distribution-matching distillation accessible to researchers with limited computational resources (1–2 GPUs), while the adaptive weighting and verification mechanisms are expected to improve generation quality. The experimental plan is comprehensive, spanning class-conditional and text-to-image generation across multiple architectures (UNet and MMDiT), with extensive ablation studies isolating each contribution.

If successful, this work would demonstrate that high-quality one-step diffusion distillation does not require massive computational infrastructure, opening the door for broader academic participation in this rapidly advancing field. The density-ratio verification component additionally enables a new inference-time quality–latency trade-off that could be valuable in production deployments.

---

## 7. References

1. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024). One-step Diffusion with Distribution Matching Distillation. In *CVPR 2024*. arXiv:2311.18828.

2. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024b). Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2). In *NeurIPS 2024 (Oral)*. arXiv:2405.14867.

3. Zhang, C., et al. (2025). Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction. In *NeurIPS 2025*. arXiv:2505.20755.

4. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency Models. In *ICML 2023*. arXiv:2303.01469.

5. Luo, S., Tan, Y., Huang, L., Li, J., & Zhao, H. (2023). Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference. arXiv:2310.04378.

6. Luo, S., et al. (2023). LCM-LoRA: A Universal Stable-Diffusion Acceleration Module. arXiv:2311.05556.

7. Zhou, M., et al. (2024). Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation. In *ICML 2024*. arXiv:2404.04057.

8. Zhou, M., et al. (2025). Few-Step Diffusion via Score Identity Distillation (SiDA). In *ICLR 2025*. arXiv:2505.12674.

9. DiffRatio Authors (2025). DiffRatio: Training One-Step Diffusion Models Without Teacher Supervision. arXiv:2502.08005.

10. Zhu, H., et al. (2025). Simple Distillation for One-Step Diffusion Models (CED). In *NeurIPS 2025*.

11. Multi-Student Distillation Authors (2025). Multi-Student Diffusion Distillation for Better One-Step Generators. In *ICML 2025*. arXiv:2410.23274.

12. RAPM Authors (2025). High Quality Diffusion Distillation on a Single GPU with Relative and Absolute Position Matching. arXiv:2503.20744.

13. Flow-Anchored Consistency Models Authors (2025). FACM: Flow-Anchored Consistency Models. arXiv:2507.03738.

14. Stable Consistency Tuning Authors (2025). Stable Consistency Tuning. In *NeurIPS 2024*.

15. Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). Adversarial Diffusion Distillation. arXiv:2311.17042.

16. Stability AI (2024). Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation (LADD).

17. Nguyen, T., et al. (2024). SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation. In *CVPR 2024*.

18. Nguyen, T., et al. (2024b). SwiftBrush v2: Make Your One-step Diffusion Model Better Than Its Teacher. arXiv:2408.14176.

19. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. In *ICLR 2022*.

20. Liu, X., et al. (2023). InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation. arXiv:2309.06380.

21. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., & Zhu, J. (2023). ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. arXiv:2305.16213.

22. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. In *ICLR 2022*.

23. Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM). In *NeurIPS 2022*.

24. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. In *CVPR 2022*.

25. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In *NeurIPS 2020*.

26. Goodfellow, I., et al. (2014). Generative Adversarial Nets. In *NeurIPS 2014*.

27. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In *NeurIPS 2017*.

28. Schuhmann, C., et al. (2022). LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In *NeurIPS 2022*.

29. Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. In *ECCV 2014*.

30. Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. In *CVPR 2009*.

31. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical report.

32. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS). In *CVPR 2018*.

33. Yin, T., et al. (2025). From Slow Bidirectional to Fast Autoregressive Video Diffusion Models. In *CVPR 2025*. arXiv:2412.07772.

34. Esser, P., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3). In *ICML 2024*.

35. Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). In *ICML 2021*.

36. Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2023). DreamFusion: Text-to-3D Using 2D Diffusion. In *ICLR 2023*.

37. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. arXiv:2207.12598.

38. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. In *ICLR 2021*.

---

*Plan prepared: February 2026. Based on analysis of DMD (Yin et al., CVPR 2024) and comprehensive literature review of 2024–2026 advances in diffusion distillation.*
