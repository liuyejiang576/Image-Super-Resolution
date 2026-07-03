## Guideline

**Efficient Image Restoration for Mobile AI**

This project studies efficient image restoration models for mobile and edge devices. Instead of only optimizing image quality on desktop GPUs, the goal is to design a lightweight model that achieves a good trade-off between restoration quality and on-device efficiency. You can focus on one Mobile AI Challenge track, such as image super-resolution, image denoising, RGB photo enhancement, or learned smartphone ISP.

A simple baseline is to use a standard restoration model such as SwinIR, Real-ESRGAN, or a lightweight U-Net. The project can then improve efficiency through model pruning, knowledge distillation, quantization, lightweight convolution blocks, or mobile-friendly network design. The final model can be evaluated by both image quality metrics and computational efficiency.

Possible improvements: lightweight U-Net design, depthwise separable convolution, model pruning, quantization, knowledge distillation, or mobile-friendly attention modules.

Baseline: [SwinIR](https://github.com/JingyunLiang/SwinIR), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [lightweight U-Net](https://github.com/sgvaze/lightweight_unet), [MobileNet-style restoration model](https://arxiv.org/abs/1905.02244?utm_source=chatgpt.com) Dataset / Challenge: [Mobile AI 2026 Challenge](https://ai-benchmark.com/workshops/mai/2026/), such as [Image Super-Resolution](https://www.codabench.org/competitions/14041/), [Image Denoising](https://www.codabench.org/competitions/12934/).. Evaluation: PSNR, SSIM, LPIPS, runtime, memory usage, model size, and mobile inference speed.



## V1 Proposal

**Project Proposal: Mobile-Efficient 4 $\times$ Image Super-Resolution via Lightweight Architecture Design and Model Compression**

---

**1. Problem Statement and Motivation**

Modern image super-resolution (SR) models such as SwinIR have high quality but have too heavy computation load for mobile devices.  This project investigates the design of a lightweight 4 $\times$ SR model to do this. The core research question is: how do architecture choices, knowledge distillation, and structured compression affect the quality–efficiency trade-off, and can we identify an optimal operating point for mobile deployment?

This problem is interesting because mobile photography and real-time enhancement are ubiquitous applications, yet the gap between SOTA SR quality and on-device feasibility remains large. A systematic study of the Pareto frontier between quality and efficiency can guide practical model selection and inspire further lightweight design.

---

**2. Background and Related Work**

The project builds upon three main lines of work:

- **Classical lightweight SR:** simple but effective baselines. FSRCNN and ESPCN performed most computation in low-resolution (LR) feature space and using transposed convolution or sub-pixel convolution (PixelShuffle) for upsampling.
- **Modern efficient SR architectures:** RFDN and IMDN further improved lightweight SR performance through feature distillation connections and carefully designed residual blocks. Their design principles will inform the proposed model, especially LR-space processing, depthwise separable convolutions, and progressive upsampling.
- **Model compression for restoration:** Techniques such as knowledge distillation (using a large teacher to guide a small student), structured pruning, and quantization-aware training (QAT) have been successfully applied in high-level vision tasks, but their interaction in pixel-sensitive restoration tasks is less explored. SwinIR will be used as a pretrained teacher, following prior work on output-level distillation for SR.

---

**3. Data**

- **Training:** DIV2K (800 high-resolution images). Low-resolution input pairs will be generated using the official DIV2K bicubic downsampling, ensuring consistency with standard benchmarks.
- **Testing:** Set5, Set14, BSD100, and Urban100, well-established evaluation standards.
- **Data preprocessing:** All images are normalized to [0,1]. Training patches of size 256 $\times$ 256 (HR) and 64 $\times$ 64 (LR) will be randomly cropped on the fly with horizontal/vertical flips and 90° rotations for augmentation. No new data collection is required.

---

**4. Proposed Method**

The project consists of four main stages:

**Stage 1 — Pipeline Validation and Baselines**
Before training any model, the evaluation pipeline will be verified by computing bicubic PSNR/SSIM on Set5 (expected 28 dB). Then FSRCNN will be trained from scratch to establish a fair low-cost baseline. A pretrained SwinIR (classical SR, bicubic degradation) will serve as the quality upper bound and later as the distillation teacher.

**Stage 2 — Lightweight Model Design (MobileSRNet, Simple Core)**

Building blocks consist of depthwise separable convolutions combined with ReLU/ReLU6 activations for quantization friendliness.

The primary design rule is to keep the architecture simple and stable, avoiding broad architecture search. The goal is to keep implementation straightforward and focus project depth on rigorous analysis and contribution quality.

This design is inspired by RFDN and MobileNet, but simplified for SR. Existing implementations of depthwise separable convolutions and PixelShuffle in PyTorch will be used as building blocks.

**Stage 3 — Distillation and Contribution-Focused Analysis**
Two main strategies will be evaluated:
- **Knowledge distillation (KD):** Using the online teacher approach, each training batch will be fed to both the frozen SwinIR teacher and the MobileSRNet student. The loss will combine Charbonnier loss against ground-truth HR and a distillation term matching the teacher's output:  
  $ \mathcal{L} = \mathcal{L}_{\text{char}}(\hat{x}_S, x) + \lambda \mathcal{L}_{\text{char}}(\hat{x}_S, \hat{x}_T) $ with $ \lambda=0.2 $.  
- **Controlled ablation analysis:** A compact, hypothesis-driven ablation set (for example KD on/off, loss-weight sensitivity, and one compression setting at a time) will isolate which design choices produce robust gains and which add complexity without clear benefit.
- **Quantization (stretch goal):** FP16 and INT8 (via post-training quantization or QAT) will be explored to assess further deployment efficiency.

**Optional extension (only if core milestones are completed early):**
- **Single hybrid variant:** One lightweight transformer encoder + CNN decoder model will be tested as a bounded comparison point, not as the main project direction.

**Stage 4 — Deployment-Oriented Benchmarking**
All model variants will be profiled for parameter count, FLOPs, model file size, and batch-size-1 inference latency on an NVIDIA RTX 4060 (FP32 and FP16). These metrics serve as strong proxies for on-device efficiency without requiring actual mobile deployment.

**Use of existing implementations:** Standard PyTorch modules will be used for convolutions and PixelShuffle. FSRCNN will be reimplemented based on the original paper. The pretrained SwinIR model will be obtained from the official repository.

---

**5. Evaluation Plan and Expected Results**

**Evaluation Metrics:**
- **Reconstruction quality:** PSNR and SSIM computed on the Y channel of YCbCr color space, plus LPIPS on RGB for perceptual similarity.
- **Efficiency:** Parameters, FLOPs, model size (MB), inference latency (ms) at batch size 1.

**Experimental Protocol:**
A comprehensive results table will compare all variants (bicubic, FSRCNN, SwinIR, MobileSRNet core model, KD versions, and possibly quantized models) on all test sets. If time permits, one hybrid variant will be included as an extension. The central deliverables are a Pareto frontier plot and a concise set of evidence-based findings that explain which design choices are practically worthwhile.

**Expected Results:**
- The proposed MobileSRNet is expected to outperform bicubic interpolation and FSRCNN in PSNR/SSIM while using far fewer parameters and FLOPs than SwinIR.
- Knowledge distillation should noticeably narrow the quality gap between MobileSRNet and SwinIR, with minimal additional inference cost.
- Controlled ablations will reveal which design choices consistently improve the quality-efficiency trade-off and where diminishing returns begin, producing a clear Pareto curve with stronger interpretability.
- FP16 inference will demonstrate latency and memory reductions, and if successful, INT8 will further reduce model size with manageable quality loss.

Overall, this project expects to deliver a compact, deployable SR model (targeting <1M parameters and <20G FLOPs for 720p output) that achieves a favorable balance between efficiency and visual quality, along with a contribution-focused analysis that can inform future mobile restoration designs.
