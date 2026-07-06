# Experiment Plan (Training-First)

## Goal

Strengthen result credibility by testing whether current conclusions are limited by training design, undertraining, or measurement noise.  
This plan prioritizes controlled retraining and GPU-efficient execution before adding new model families.

## Core Principles

- Maximize useful throughput, not raw VRAM occupancy.
- Keep comparisons fair with equal update budgets.
- Change one major variable at a time.
- Use fixed seeds for ablations; use multiple seeds only for final candidates.
- Treat unstable or noisy validation as a design fault to fix.

## Current Risks to Resolve

- Models may be undertrained (late-stage validation PSNR still improving).
- Validation checkpointing uses only 20 images and may be noisy.
- KD effect is partially confounded with different training loss setup.
- FP16 latency comparison has suspicious values and needs audit.

## Experiment Pipeline

### Phase 0 - Freeze Existing Baseline Snapshot

- Keep current artifacts as immutable reference:
- `results/fsrcnn_fix_clean/*`
- `results/fsrcnn_small/*`
- `results/mobile_srnet/*`
- `results/mobile_srnet_kd/*`
- `results/swinir/*`
- Create a single summary file `results/exp_runs/baseline_snapshot.json` with PSNR/SSIM/LPIPS/FLOPs/latency.

### Phase 1 - GPU Throughput Tuning (Mini-Probe)

Purpose: find stable hyperparameters that use GPU efficiently without OOM.

- Target memory window: 6.8 GB to 7.2 GB (do not push to the hard limit).
- Select config by best `steps/sec` inside safe memory, not by highest memory alone.
- Probe duration per run: 20 to 40 train steps.
- Keep validation tiny during probes (`val_max_images=5`) to reduce overhead.

Probe grid:

- MobileSRNet (`configs/train_mobile_srnet.yaml`):
- batch size candidates: 24, 28, 32
- `amp: true`
- worker candidates: 8, 10

- FSRCNN clean (`configs/train_fsrcnn_fix_clean.yaml`):
- batch size candidates: 8, 10, 12, 16
- test both `amp: false` and `amp: true`

- KD (`configs/train_mobile_srnet_kd.yaml`):
- batch size candidates: 16, 20, 24
- switch `amp` from false to true and compare
- note: KD script currently has no max-step arg; use 1-2 epochs for quick probes

Expected output:

- `results/exp_runs/gpu_probe.csv` with columns:
- run_id, model, batch_size, amp, workers, steps_per_sec, peak_mem_gb, oom, notes

### Phase 2 - Fair Budget Retraining (Undertraining Check)

Purpose: test if quality gains come from better training budget rather than architecture luck.

- Use equal update budgets across models.
- Suggested budgets: 5k, 10k, 20k updates.
- Compute epochs from `steps_per_epoch = floor(800 / batch_size)` (dataset has 800 training images, drop_last=True).
- For each budget, set `epochs = ceil(target_updates / steps_per_epoch)`.

Scheduler fairness rule:

- Current scripts use epoch-based milestones.
- For each run, set milestones as percentages of total epochs:
- milestone1 = round(0.60 * epochs)
- milestone2 = round(0.85 * epochs)

Runs required:

- `fsrcnn_fix_clean` at 10k and 20k updates
- `mobile_srnet` at 10k and 20k updates
- optional quick 5k runs only if needed to plot learning curves

Expected output:

- `results/exp_runs/fair_budget_runs.json`
- one curve figure: val PSNR vs updates for FSRCNN vs MobileSRNet

### Phase 3 - KD Deconfounding

Purpose: isolate true KD contribution.

- Use `train_mobile_srnet_kd.py` for both runs so base loss is identical (Charbonnier).
- Run pair A: `lambda_kd = 0.0` (Charbonnier-only student).
- Run pair B: `lambda_kd = 0.2` (current KD setup).
- Use same batch size, same epochs, same scheduler, same seed.
- Optional robustness: add `lambda_kd = 0.1` and `0.3` only if pair A/B is stable.

Initialization policy:

- Option 1 (cleanest): train both from scratch with identical setup except lambda.
- Option 2 (faster): initialize both from same checkpoint and continue equal updates.

Expected output:

- `results/exp_runs/kd_isolation.csv` with per-dataset PSNR/SSIM/LPIPS deltas.

### Phase 4 - Validation Reliability Upgrade

Purpose: reduce checkpoint selection noise.

- Increase validation subset from 20 images to a larger fixed subset (recommended 80-100).
- Keep this subset fixed across all runs.
- For final model selection, run one full validation pass over all DIV2K-valid images.

Expected output:

- `results/exp_runs/validation_protocol.md` documenting selection rule.

### Phase 5 - Final Candidate Selection and Full Evaluation

- Select 2-3 candidates:
- best FSRCNN variant
- best MobileSRNet no-KD
- best MobileSRNet KD
- Run full benchmark scripts on Set5/Set14/BSD100/Urban100.
- Re-run profiling and audited latency protocol.
- Export final metrics tables and plots.

Expected output:

- `results/exp_runs/final_candidates.json`
- updated `results/final_report.md` inputs

## Decision Gates

- Gate 1 (after Phase 1): proceed only with settings that are OOM-free and throughput-optimal.
- Gate 2 (after Phase 2): if both models plateau early, stop long retraining and focus on analysis.
- Gate 3 (after Phase 3): if KD effect is inconsistent, keep KD as secondary result, not headline.
- Gate 4 (after Phase 5): lock final narrative only after audited latency and fair-budget curves are complete.

## Practical GPU Usage Rules

- Keep 10-15 percent VRAM headroom.
- Prefer AMP when numerically stable.
- Use largest safe batch only when it improves steps/sec.
- Do not compare runs with different effective update counts.
- Record memory and throughput for every run in a single CSV.

## Minimal Run Manifest Template

Use one line per run in `results/exp_runs/run_manifest.csv`.

`run_id,model,config,seed,batch_size,amp,workers,updates_target,epochs,milestones,lambda_kd,val_subset,status,best_ckpt`

## Success Criteria

- Training-budget curves clearly show whether current models were undertrained.
- KD effect is isolated from base-loss changes.
- Best model choice is robust to validation subset noise.
- GPU utilization strategy is reproducible and justified by throughput data.
- Final claims are supported by controlled experiments, not one-off settings.
