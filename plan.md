# Early Project Plan (Contribution-First Scope)

Scope: only the first steps to get your SR project running and de-risked.  
Focus: simple lightweight model design plus deep analysis/contribution, environment, data pipeline, metric sanity check, and first baseline.

## Step 1 - Lock scope and success criteria (Day 1)

**I do**
- Convert the proposal into a concrete experiment checklist (task, datasets, metrics, baselines).
- Define a minimal "done for phase 1" target: reproducible PSNR/SSIM evaluation + one trained baseline.
- Provide a simple folder template and run order.

**You do**
- Confirm final task is **4x bicubic super-resolution**.
- Confirm evaluation sets: **Set5, Set14, BSD100, Urban100**.
- Confirm baseline order: **Bicubic -> FSRCNN -> SwinIR reference**.
- Confirm architecture policy: **keep model design simple**, and prioritize research depth through strong hypotheses, clean ablations, and clear contribution.

## Step 2 - Project skeleton and environment (Day 1-2)

**I do**
- Prepare the exact directory layout and starter files (`data/`, `src/`, `configs/`, `scripts/`, `results/`).
- Write a dependency list and one-command setup instructions.
- Add a quick verification script to confirm CUDA/PyTorch/device status.

**You do**
- Create and activate the Python environment.
- Install dependencies and run the verification script.
- Tell me your hardware/runtime output so I can tune defaults (batch size, workers, mixed precision).

## Step 3 - Data and preprocessing pipeline (Day 2-3)

**I do**
- Implement/prepare data loader logic for DIV2K with on-the-fly LR-HR patch generation (x4 bicubic).
- Add augmentation exactly as proposal states (flip/rotate) and normalization to [0,1].
- Provide a data sanity script (shape checks + sample patch export).

**You do**
- Download and place DIV2K + test benchmarks in the expected paths.
- Run the sanity script and confirm sample patches look correct.
- Report any path/read errors; I will adjust scripts immediately.

## Step 4 - Metric pipeline validation first (Day 3)

**I do**
- Implement PSNR/SSIM evaluation pipeline (Y-channel protocol) and logging format.
- Add bicubic evaluation script for Set5/Set14/BSD100/Urban100.
- Set a pass/fail check for expected Set5 bicubic range (around proposal expectation).

**You do**
- Run bicubic-only evaluation and save the output logs.
- Share the metric numbers with me.
- If numbers look off, rerun with my debug checklist (border crop/color conversion checks).

## Step 5 - First trainable baseline: FSRCNN (Day 4-7)

**I do**
- Set up FSRCNN training config (loss, optimizer, scheduler, checkpoints).
- Add train/validate scripts and result tracking format.
- Provide a minimal ablation plan (1-2 settings only) for stable first results.

**You do**
- Launch training runs using the provided config.
- Send me training curves/log snippets and validation outputs.
- Keep the best checkpoint and run full benchmark evaluation.

## Step 6 - Main model kickoff (Contribution-Focused MobileSRNet)

**I do**
- Build a simple MobileSRNet baseline without broad architecture search.
- Prepare a hypothesis-driven experiment matrix (KD on/off, loss-weight sensitivity, one compression choice at a time).
- Add a decision gate: only if core analysis is complete and schedule allows, run one hybrid transformer-encoder + CNN-decoder variant as bounded comparison.

**You do**
- Approve the core hypotheses and experiment priority order.
- Run the agreed compact ablation matrix.
- Share first Pareto points and key observations so I can decide whether to trigger the optional hybrid comparison.

## Phase-1 Exit Criteria

- Data pipeline is reproducible and validated.
- Bicubic metrics are trustworthy.
- FSRCNN baseline is trained and evaluated on all target test sets.
- Contribution-focused MobileSRNet experiment configs are ready, and the optional hybrid gate is clearly defined.
