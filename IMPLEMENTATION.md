# Implementation rules (lab)

> **控制面 / 主循环：** `../progress/README.md`（状态只改 `../progress/`）。  
> 本文件是**数据面**：训练、fair-budget、pause/resume、并行、已知漏洞。

**Python interpreter (non-negotiable):** every command below that says `python` means  
`/home/hyb/miniforge3/envs/cv_env/bin/python`  
(or `conda activate cv_env` first). Cold-shell `python` / `python3` is usually base conda → CPU torch / missing deps. Progress gate: **G-lab**.

Single source of truth for how training, fair-budget, and pause/resume actually work in this repo.
Code names: **MobileSRNet** (Base / Plus). Report may say SepResSR — same models (`feat=40,N=6` Base; `feat=64,N=8` Plus).

---

## 1. Active control planes

There are **eight** separate launch systems. Do not mix them.

| Control | Pause / resume / watch | Launcher | Manifest | Models |
|---|---|---|---|---|
| Fair-budget MSE+KD (historical) | `exp_status.py` (status only; no pause CLI) | `run_exp_parallel.py` or `run_exp_pipeline.py` | `results/exp_runs/fair_budget_manifest.json` | FSRCNN, Base, KD λ=0/0.2 — **Plus is not in this manifest** |
| Plus 20k | `plus_20k.py pause\|resume\|watch` | `run_plus_20k.py` | `plus_20k_manifest.json` | Plus only |
| Arch 30k continuation | `arch_30k.py pause\|resume\|watch` | `run_arch_30k.py` | `arch_30k_manifest.json` | FSRCNN + Base + Plus |
| **ECBSR 20k (B3)** | `ecbsr_20k.py pause\|resume\|watch` | `run_ecbsr_20k.py` | `ecbsr_20k_manifest.json` | ECBSR (M10C16; report display **ECBSR**) |
| **A1a FSRCNN bs24** | `a1a_20k.py pause\|resume\|watch` | `run_a1a_20k.py` | `a1a_20k_manifest.json` | FSRCNN equal-batch 20k |
| **SepResV2 20k (B4 r1)** | `sepres_v2_20k.py pause\|resume\|watch` | `run_sepres_v2_20k.py` | `sepres_v2_20k_manifest.json` | default **v2_b C16N10**; a/c via `--run-id` after gates |
| **Dual/Plain 2k (B4 r2)** | `dual_plain_2k.py pause\|resume\|watch` | `run_dual_plain_2k_probes.py` | `dual_plain_2k_manifest.json` | **DualStream-C20N5** then **Plain-C20N5** |
| **B5 unified (B5a+seeds+KD)** | `b5_train_20k.py pause\|resume\|watch` | `run_b5_train_20k.py` | `b5_train_20k_manifest.json` | P0 + PECSR/ECBSR×3 seeds + **KD Stage-B 2k** (pixel 0/0.2; VGG 0/0.01; MSE 3-wide) |

Legacy thin wrappers (redirect → `b5_train_20k.py`): `b5a_20k.py` / `b5_confirm_20k.py` / `pecsr_kd_20k.py`. Do not start their old `run_*` launchers alongside unified.

**Day-to-day for architecture work:**

```bash
# Plus 20k
python scripts/plus_20k.py resume
python scripts/plus_20k.py watch --interval 60
python scripts/plus_20k.py pause

# 30k continuation (from 20k latest)
python scripts/arch_30k.py resume
python scripts/arch_30k.py watch --interval 60
python scripts/arch_30k.py pause

# ECBSR fair 20k (B3) — geometry M10C16; report name ECBSR
python scripts/ecbsr_20k.py resume
python scripts/ecbsr_20k.py watch --interval 60
python scripts/ecbsr_20k.py pause

# A1a FSRCNN bs=24 / 20k (equal batch+steps vs Mobile)
python scripts/a1a_20k.py resume
python scripts/a1a_20k.py watch --interval 60
python scripts/a1a_20k.py pause

# SepResSR-v2 fair 20k (B4 round-1) — default = v2_b only
python scripts/check_sepres_v2_fuse.py          # Gate-0 model/fuse/budget
python scripts/sepres_v2_20k.py resume          # after pre-screen + envelope
python scripts/sepres_v2_20k.py watch --interval 60
python scripts/sepres_v2_20k.py pause
# After train (Gate-2): fuse → NCNN → paired phone → b4_v2_compare.json
python scripts/run_b4_v2_posttrain.py --wait
# Mid-train pipeline smoke (not official):
# python scripts/run_b4_v2_posttrain.py --checkpoint …/latest.pt --smoke --skip-eval --sessions 0
# Gated a/c (only after B4 gates):
# python scripts/sepres_v2_20k.py resume --run-id sepres_v2_c16n8_20k

# DualStream / Plain C20N5 2k probes (B4 round-2) — D18 gate before 20k
python scripts/check_dual_plain_fuse.py         # Gate-0 Dual→Plain fuse + budget
python scripts/dual_plain_2k.py resume          # Dual then Plain (~61 ep each)
python scripts/dual_plain_2k.py watch --interval 60
python scripts/dual_plain_2k.py pause
# Single probe:
# python scripts/dual_plain_2k.py resume --run-id dual_stream_c20n5_2k

# B5 unified — B5a P0 + multi-seed MSE (3-wide) + gated KD
python scripts/check_b5a_plain_fuse.py         # Gate-0 once
python scripts/b5_train_20k.py resume          # --max-parallel 3; adopts live trains
python scripts/b5_train_20k.py watch --interval 60
python scripts/b5_train_20k.py pause
# KD starts automatically after all MSE done: Stage-B 2k (pixel then VGG); never mixed (§9)
# Optional: resume --mse-only to stop before KD
```
Generic trainers (called by launchers):

- `scripts/train_fsrcnn.py`
- `scripts/train_mobile_srnet.py`
- `scripts/train_mobile_srnet_kd.py`
- `scripts/train_ecbsr.py` (also used by Dual/Plain via `aux_weight`; Dual: `return_aux=True`)
- `scripts/train_sepres_v2.py` (thin wrapper → same loop as `train_ecbsr.py`, `type: sepres_v2`)
- `scripts/train_dual_plain.py` (thin wrapper → `train_ecbsr.py`; `type: dual_stream_sr` / `plain_sr`)
- B5 unified (`b5_train_20k` / `run_b5_train_20k`) follows the same pause|resume|watch + manifest pattern as `sepres_v2_20k` / `arch_30k` (dispatches `train_script` per entry)

---

## 2. Pause / resume / watch — contract

### What pause does

1. Writes a pause-state JSON (`plus_20k_paused.json` / `arch_30k_paused.json` / `ecbsr_20k_paused.json` / `a1a_20k_paused.json` / `sepres_v2_20k_paused.json` / `dual_plain_2k_paused.json` / `b5_train_20k_paused.json`).
2. SIGTERM matching train PIDs (matched by `train_*.py --config <cfg>` in `pgrep`).
3. Waits, then SIGKILL if needed.
4. Stops launcher + watch processes.

**Checkpoints are epoch-granular only.** Trainers save `latest.pt` / `best.pt` at **end of each epoch**. Mid-epoch kill loses that epoch’s steps.

### What resume does

1. Skips runs already at target epoch (log last epoch ≥ target) with `best.pt` present.
2. If `latest.pt` exists and run incomplete → launch with `--resume-from …/latest.pt`.
3. For 30k first start: bootstrap from `resume_from_run_id`’s `latest.pt`, copy 20k `train_log.jsonl` into the 30k run dir, then continue epochs.

### What watch does

Live table from `train_log.jsonl`: epoch, val_psnr, best, ETA, state (`running` / `paused` / `done` / `pending`).

### Resume loads (trainers)

From checkpoint:

- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `best_psnr`, `epoch` → `start_epoch = epoch + 1`
- `global_step` **recomputed** as `epoch * len(train_loader)` (not stored)

**Not saved / not restored:**

- AMP `GradScaler` state
- Python / NumPy / Torch RNG state
- DataLoader shuffle / worker RNG
- Mid-epoch optimizer step count

So resume is **optimizer-continuous and approximately step-aligned**, not bit-exact reproducibility.

---

## 3. Fair-budget — how it is defined in code

### Intended definition

Equal **optimizer update count** (e.g. 20 000 steps), not equal wall time, not equal number of image patches.

Generator: `scripts/generate_exp_configs.py`

```text
TRAIN_IMAGES = 800                    # DIV2K train HR count
steps_per_epoch = 800 // batch_size   # integer division
epochs = ceil(target_updates / steps_per_epoch)
milestones ≈ 60% and 85% of epochs
```

DataLoader uses `drop_last=True`, so each epoch actually runs `floor(800 / batch_size)` steps — same as `//`.

### Locked 20k configs (current on disk)

See also **[`configs/README.md`](configs/README.md)** (exp vs templates vs inactive).

| Run | batch | steps/epoch | epochs | updates | patches seen (updates × bs) |
|---|---:|---:|---:|---:|---:|
| `fsrcnn_fix_clean_20k` | 8 | 100 | 200 | 20 000 | **160 000** |
| `mobile_srnet_20k` | 24 | 33 | 607 | 20 031 | **480 744** |
| `mobile_srnet_plus_20k` | 24 | 33 | 607 | 20 031 | **480 744** |
| `mobile_srnet_kd0_20k` | 16 | 50 | 400 | 20 000 | **320 000** |
| `mobile_srnet_kd02_20k` | 16 | 50 | 400 | 20 000 | **320 000** |

Batch sizes come from `results/exp_runs/gpu_probe_recommendations.json` (throughput under VRAM cap), **not** from equalizing patch exposure.

### 30k continuation

Not a fresh 30k-from-scratch recipe. It resumes 20k `latest.pt` and trains to a higher epoch count (~+10k updates) under a new config whose milestones are written for the full span; **scheduler state is restored from the checkpoint**, so LR continues from the decayed 20k schedule (low-LR fine-tune in practice).

Verified on disk: 30k logs are continuous copies of 20k logs then epochs 201… / 608….

### Manifest split (easy to miss)

- `fair_budget_manifest.json` — FSRCNN, Base, KD only.
- `plus_20k_manifest.json` — Plus alone.
- Regenerating via `generate_exp_configs.py` **does not emit Plus** and can overwrite `configs/exp/*` from probe defaults.

---

## 4. Data / metric conventions (code reality)

| Stage | What happens |
|---|---|
| Train LR | On-the-fly PIL bicubic downsample of random HR crop (`DIV2KPatchDataset`) — **not** official DIV2K LR folders |
| Val during train | First `validation.max_images` (100) full DIV2K-valid images; **RGB** PSNR from tensor MSE (no Y, no 4px shave) |
| Checkpoint pick | Max of that RGB val PSNR → `best.pt` |
| Report / `eval_sr.py` | Benchmarks; **Y-channel** PSNR/SSIM + 4px shave; optional RGB LPIPS |

Train-time val and report metrics are **different spaces**. Ranking by train val is still OK for selection if consistent; do not treat train `val_psnr` as comparable to table PSNR.

---

## 5. KD path

- Script: `train_mobile_srnet_kd.py` (Charbonnier GT + optional teacher term).
- Fair KD pair: `kd0` (λ=0) vs `kd02` (λ=0.2), same recipe — correct isolation **within KD script**.
- Main MSE Base (`train_mobile_srnet.py`, MSE) is **not** a valid KD-off control for Charbonnier KD runs.
- Teacher: frozen SwinIR classical ×4.

One-off Stage B / λ-sweep / parallel KD probes live under `scripts/_inactive/` (see §8).

---

## 6. Known holes / traps

### Protocol (affects claims)

1. **Unequal patch exposure** under “fair” 20k steps (FSRCNN sees ~⅓ the patches of Base/Plus). Do not claim “architectural not tuning” from this alone.
2. **PIL on-the-fly bicubic** ≠ canonical aligned DIV2K LR–HR pairs.
3. **Train RGB val ≠ report Y PSNR**.
4. **Phone deploy used 30k `best.pt`** while many quality tables are 20k — pair checkpoint identity before Pareto language (`results/MANIFEST.md` already notes this).
5. **FSRCNN probe `amp: false`** but exp configs set `amp: true` — probe and train disagree.

### Resume / launcher

6. **No mid-epoch resume** — pause only safe between epochs (in practice: after `latest.pt` write).
7. **AMP scaler not checkpointed**.
8. **`global_step` on resume** ignores partial epochs / `max_train_steps_per_epoch`.
9. **30k bootstrap does not copy `best.pt`**. If continuation never beats resumed `best_psnr`, `best.pt` might be missing until an improvement — `is_done` requires `best.pt`. (Current 30k runs did write `best.pt`.)
10. **`run_exp_parallel` / `pgrep` matching** can false-positive if another process has the same config path string.
11. **Hardcoded Python** in some launchers: `/home/hyb/miniforge3/envs/cv_env/bin/python` (falls back to `sys.executable` in Plus/30k).

### Config generation

12. Re-running `generate_exp_configs.py` can **clobber** hand-maintained exp YAMLs; Plus is outside its loop.
13. `epochs_for_updates` uses `800 // bs`; with `drop_last`, leftover images (`800 % 24 = 8`) are dropped every epoch for Base/Plus.

### Naming / docs drift

14. README still says `run_exp_pipeline.py --config …` — pipeline does **not** take that flag; it always trains from `fair_budget_manifest.json`.
15. Codebase still says MobileSRNet; planning docs say SepResSR / Lite / Plus.

---

## 7. Active vs inactive (after cleanup)

### Keep active

```
# core — research spine
scripts/train_*.py
scripts/plus_20k.py, run_plus_20k.py
scripts/arch_30k.py, run_arch_30k.py
scripts/run_exp_parallel.py, run_exp_pipeline.py, exp_run_utils.py, exp_status.py
scripts/generate_exp_configs.py, gpu_probe.py, snapshot_baselines.py
scripts/eval_*.py, eval_exp_runs.py
scripts/export_*.py, convert_deployment.py, bench_mobile.py, deploy_smoke.py
scripts/profile_model.py, profile_b2.py
scripts/fuse_deploy_compare.py, check_ecbsr_fuse.py, parse_ncnn_blobs.py
# figures / CUDA latency → ../report/plot/ (not scripts/)

# ops — host utilities (NOT inactive)
scripts/keep_awake.sh, win_keep_awake.ps1, win_lock_power.ps1
scripts/check_env.py, download_datasets.sh, sanity_check_data.py
```

### Inactive (`scripts/_inactive/`)

| Path | Why |
|---|---|
| `stage_b/*` | Plus 2k / VGG3 probes finished — **do not re-run as current workflow** |
| `kd_probes/*` | λ-sweep / KD parallel probe finished |
| `kd_diag/*` | gate / diagnose / per-image KD — regen RQ3 evidence only |
| `lab_plot_dup/*` | Lab plot/audit copies; **canonical = `report/plot/`** |
| `profile_fsrcnn.py`, `compare_metrics.py` | Thin / early utilities |
| `kd_sweep.sh` + related shell | superseded; keep_awake no longer calls it |
| `quantize_benchmark.py` | CPU proxy; not in current deploy spine |
| report builders / old postprocess shells | superseded by `report/` |

**Tag rule:** `_inactive` = should not run now. **ops ≠ inactive.** Overnight `win_*.ps1` stay next to `keep_awake.sh`.

---

## 8. Minimal reproduce paths

```bash
# Eval locked checkpoint (report metrics)
python scripts/eval_sr.py \
  --checkpoint results/exp_runs/mobile_srnet_plus_20k/checkpoints/best.pt \
  --save-json results/exp_runs/mobile_srnet_plus_20k/benchmark_metrics.json \
  --compute-lpips

# Status of fair-budget manifest runs
python scripts/exp_status.py

# Do NOT blindly regenerate configs unless you intend to rewrite configs/exp/
# python scripts/generate_exp_configs.py
```

Deploy: see `deploy/DEPLOY.md`.

Report sync: `../report/SYNC.md`.

---

## 9. Parallel training, time estimates, GPU utilization

### Metric: total calendar time（总日历）

**要不要并行，比的是「两件事都做完要等多久」**，不是「每个任务自己变慢多少」。

对两个任务 A、B：

```text
T_seq  = T_A_solo + T_B_solo
T_par  = max(T_A_para, T_B_para)
并行划算  ⟺  T_par < T_seq
```

令慢化比 `s = (并行时 s/ep) / (solo s/ep)`（或等价地 `solo_ep_per_min / para_ep_per_min`）：

```text
T_par ≈ max(s_A · T_A_solo, s_B · T_B_solo)
```

**两个差不多大的 MSE 任务：** 只要两边的 `s < 2`，并行总日历就更短。  
本机 `arch_30k` 用的门槛「并行 ep/min < 55% solo」≈ `s > ~1.82`，接近两人任务的盈亏平衡；再差就该减并行。

**一长一短：** 设长任务 solo 为 `L`、短为 `S`。并行时若长任务整段都被拖慢到 `s·L`，则

```text
T_par ≈ s·L
T_seq = L + S
并行更慢  ⟺  s > 1 + S/L
```

短任务只占长任务 10% 时，`s > 1.1` 就会让总日历变差——别为了「吃满 GPU」去叠一个几乎立刻结束的小活在长训上。

### 什么时候总日历更快

| 情况 | 为何 |
|---|---|
| 两路 **MSE 小 CNN**，GPU 常吃不饱（val 重、util 常 &lt;50%、VRAM 富余） | `s` 接近 1；`T_par ≈ max(T_A,T_B) ≪ T_A+T_B` |
| 两路体量接近，实测 `s ≲ 1.5–1.8` | 仍满足 `s &lt; 2` |
| **本机实例（2026-07-10）** A1a FSRCNN bs24 + B3 ECBSR 并行 | A1：solo≈32.6s/ep → 并行≈36.6s/ep（**s≈1.12**）；剩余总日历估约 **0.6×** 分开训 |

### 什么时候总日历更慢（或别并行）

| 情况 | 为何 |
|---|---|
| 争用很重：`s ≥ 2`（或 ep/min &lt; ~50% solo） | `T_par ≥ T_seq`（等长任务时） |
| **KD（SwinIR teacher）** 与任何东西混跑 | 历史约 **~2.9×** per-job；两人任务已 `s&gt;2`，总日历通常更差 |
| 一长一短，且 `s > 1 + S/L` | 长任务被拖整段，短任务省不下多少 |
| 需要 **solo 墙钟/ep/min 作论文或 probe 基线** | 比的不是日历，是测量干净；此时必须 solo |
| VRAM 顶满 / 两活互相 OOM 重试 | 墙钟崩盘，直接减并行 |

### When to parallelize（任务类型）

| Job mix | Parallel? | Why |
|---|---|---|
| MSE-only small CNNs (FSRCNN / Base / Plus / ECBSR) | **通常 Yes，2 路**；3 路仅过夜且盯观察 | VRAM 小；总日历常赢（见上表）。用 ep/min 与 `s` 复核 |
| Any KD job (SwinIR teacher) | **No mixing**；KD 单独 | ~2.9× slowdown → 总日历常输 |
| MSE + KD together | **Avoid** | 同上 |
| Export / phone bench / profiling | 不占「训并行」名额 | 与训练槽分开 |

Launchers:

- `run_arch_30k.py` — MSE-only, default `--max-parallel 3`, refuses KD.
- `run_exp_parallel.py` — `--max-parallel 3 --max-kd-parallel 2`.
- `run_plus_20k.py` / `run_ecbsr_20k.py` — sequential single job via control CLI.
- `ecbsr_20k.py` / `plus_20k.py` — pause\|resume\|watch.

**操作口诀（E5）：** 先看人时/卡时谁紧 → 若要并行，用 **总日历** `T_par ? T_seq` 判断，并用 live `s`（或 ep/min vs solo）复核；不要用「单个 ETA 变长」当否决理由。

### How the launchers judge GPU health

`run_arch_30k.py` polls and writes `results/exp_runs/arch_30k_observe.json`:

| Signal | Threshold in code | Meaning |
|---|---|---|
| GPU util | `< 25%` for **3 consecutive polls** while ≥2 jobs | Likely **CPU / dataloader** bound, not “need more jobs” |
| VRAM used | `≥ 6500 MiB` | Reduce `--max-parallel` |
| ep/min vs solo | `< 55%` of `SOLO_EP_PER_MIN`（≈ `s > ~1.82`） | 两人等长任务接近总日历盈亏平衡 → 减到 2 或 `--sequential` |

Hardcoded solo baselines in `run_arch_30k.py` (from probe era):

```text
fsrcnn ≈ 1.78 ep/min
base/plus ≈ 3.10 ep/min   # same batch recipe; Plus is heavier so real solo is lower
```

These are **approximate**. Prefer measuring your own solo ep/min once per machine.

### How to estimate training time

**Method A — from probe (train steps only, no val):**

```text
steps/sec  ← gpu_probe_recommendations.json
train_hours ≈ target_updates / steps_per_sec / 3600
```

Current probe (solo):

| Model | bs | steps/s | 20k train-only |
|---|---:|---:|---:|
| FSRCNN | 8 | ~2.97 | ~1.9 h |
| Base | 24 | ~1.71 | ~3.3 h |
| KD | 16 | ~0.97 | ~5.7 h |

**Method B — from real logs (includes validation each epoch) — use this for ETA:**

```text
hours ≈ median(elapsed_sec per epoch) × epochs / 3600
```

Measured wall times on this machine (from `train_log.jsonl`):

| Run | med s/ep | epochs | wall hours | Notes |
|---|---:|---:|---:|---|
| FSRCNN 20k | ~47 | 200 | **~2.7 h** | |
| Base 20k | ~32 | 607 | **~5.5 h** | Likely contended (slower than Plus) |
| Plus 20k | ~13 | 607 | **~2.3 h** | Solo via `plus_20k` — cleaner reference |
| KD0 / KD02 20k | ~161 | 400 | **~20 h each** | Teacher tax |
| 30k continuation (each) | — | +100 / +303 ep | **~1.4 / ~2.8 h** | After 20k |
| A1a FSRCNN bs24（solo 段） | ~33 | 607 | **~5.5 h** | 2026-07-10 |
| A1a + B3 并行时的 A1a | ~37 | — | s≈1.12 | 同日叠 ECBSR |

**Plus faster than Base is a red flag for parallel contention**, not “Plus is cheaper.” For planning, use **Plus solo ~2.3 h / 20k** and **FSRCNN ~2.7 h**; assume Base solo closer to Plus×(compute ratio) or re-measure one solo Base run (~3–4 h expected).

**Method C — quick live ETA while training:**

```text
ETA ≈ median(last 5 epoch elapsed) × (target_epoch − current_epoch)
```

`plus_20k.py watch` / `arch_30k.py watch` / `ecbsr_20k.py watch` already do this.

**Method D — 并行 vs 分开（总日历）:**

```text
# 先有一段 solo 或历史 solo s/ep
s = median_para_s_per_ep / median_solo_s_per_ep
T_par ≈ max(s * remaining_A_solo, s * remaining_B_solo)   # 若两活对称慢化
T_seq ≈ remaining_A_solo + remaining_B_solo
并行 若 T_par < T_seq
```

### Practical parallel decision tree

```text
Need fair solo timing for a paper/probe table?
  → SOLO (max-parallel 1) — 不是日历问题

两路 MSE，人时在等结果，VRAM 富余？
  → 估 s（或先开 2 路看 ep/min）
  → s < ~1.8 且两活体量接近 → 并行（总日历通常赢）
  → s ≥ ~2 或一长一短且 s > 1+S/L → 减并行 / 改顺序

KD involved?
  → KD alone (or after MSE queue). Never mix for calendar wins.

Util <25% with 2+ jobs?
  → fewer workers or fewer jobs; do NOT add more train jobs

Util high, ep/min << solo (s 大)?
  → SM contention; reduce parallel count
```

---

## 10. How broken is the current training? What is still useful?

### Verdict in one line

**Infrastructure and within-family comparisons are useful; the FSRCNN-vs-Mobile “fair” gap is not.**  
This is a **protocol fairness** problem, not “weights are garbage / direction is wrong.”

### Still useful (keep; do not throw away)

| Asset | Why it survives |
|---|---|
| Model code + train/eval/export/NCNN path | Reusable as-is |
| **Base vs Plus** under same bs=24, same 20k recipe | Same patch exposure → capacity scaling signal is internally valid |
| **KD0 vs KD02** (same Charbonnier script) | Valid narrow KD-on/off |
| Phone latency ordering / FLOPs–latency mismatch | Mostly graph/runtime; weakly depends on which 20k/30k weights |
| Lite real-time median point | Deployment fact |
| 30k continuation logs | Honest low-LR fine-tune evidence; just don’t mix with 20k quality in one Pareto cell |
| Probe + pause/resume tooling | Operational |

### Compromised (do not overclaim)

| Claim / table cell | Problem |
|---|---|
| Mobile beats FSRCNN by ~0.3–0.5 dB “fairly” | FSRCNN saw **~⅓ the patches** |
| “Architectural not tuning-related” | Single unequal recipe |
| One quality–latency Pareto using 20k PSNR + 30k phone ckpt | Identity mismatch |
| Train `val_psnr` vs report Y-PSNR | Different metric |

### If only fixing problems (no v2 / no ECBSR) — retrain budget

Three tiers. Pick one; do not invent a fourth mid-flight.

#### Tier 0 — **0 GPU train hours** (evidence hygiene only)

- Re-export + re-bench phone from **20k** `best.pt` for FSRCNN/Base/Plus.
- Strip overclaims; footnote unequal patch exposure.
- Keep Base vs Plus as the main architecture story; demote FSRCNN to “historical reference, under-exposed.”

**Enough for a careful course report. Thin for arXiv “fair modern comparison.”**

#### Tier 1 — **minimal fair FSRCNN** (~3–8 h solo)

Fix the worst hole only: give FSRCNN equal exposure **or** equal batch.

| Option | What to run | Est. wall |
|---|---|---|
| **1a** Same updates + same bs as Mobile | FSRCNN `bs=24`, 20k steps (~607 ep) | ~3–4 h (re-probe first) |
| **1b** Same patch count as Mobile | FSRCNN `bs=8`, ~60k steps | ~8 h |

Keep existing Base/Plus 20k. Re-eval FSRCNN; rewrite gap.  
**Still no modern baseline** — only repairs the old reference.

#### Tier 2 — **lock one protocol, retrain all three MSE** (~8–12 h solo sequential)

One batch size (recommend **24** if VRAM OK), 20k steps, same loss/seed/val rule, official or documented LR rule. Retrain FSRCNN + Base + Plus.  
KD optional later (~20 h each) — not required to fix the architecture table.

| Work | Hours |
|---|---:|
| FSRCNN + Base + Plus @ locked 20k | ~8–12 h sequential |
| Optional: parallel 2-wide overnight | wall ~5–7 h calendar |
| Re-export phone @ new 20k best | <1 h machine + phone time |

### What you do **not** need to retrain just to “fix problems”

- Full 30k again (unless you insist deploy==train budget).
- KD λ-sweep / VGG probes.
- Multi-seed (that’s confirmation for a future frozen design, not a protocol bugfix).

### Bottom line for planning

- **大问题？** 对「FSRCNN 公平对照 / 架构优越措辞 / 20k–30k 混用」——是。对「模型能不能跑、Base/Plus 相对关系、手机延迟现象」——不是。  
- **有用部分？** 大半流水线 + Base/Plus + 部署信号；废的是不公平的跨模型绝对差距叙事。  
- **只修正、不管未来：** Tier 0 = 0 训；Tier 1 ≈ 半个工作日；Tier 2 ≈ 一昼夜卡时。  
- 若目标仍是 arXiv 级贡献，这些只是**还债**；A1a/B1–B3 已完成，B4 v2 现为主研究路径。

---

## 11. B4 SepResSR-v2 — implementation and measurement contract

`progress/track_b.md` owns the research decision and candidate gates. This section owns the mechanism: how a candidate is built, screened, measured, and recorded.

### Canonical model contract

`sepres_v2` must implement:

```text
Conv3x3(3→C, no act)
→ N × ECB(C→C, PReLU, with_idt=True, depth_multiplier=2.0)
→ Conv3x3(C→48, no act)
→ PixelShuffle(4)
```

- `N` counts body ECBs only. Fused deploy graph must contain exactly `N+2` dense 3×3 convolutions.
- There is no global LR RGB repeat/add shortcut.
- Head and tail are plain convolutions, not ECBs. Do not instantiate or wrap `ECBSR`.
- Training recipe is copied from `configs/exp/ecbsr_m10c16_20k.yaml`: DIV2K, MSE, Adam, seed 42, AMP, bs=24, 607 epochs, milestones `[364,516]`.
- Candidate configs are `C16N8`, `C16N10`, and `C20N6`. The superseded `C24N6` is over budget and must not be generated.

Fused-count audit at LR 180×180:

| Candidate | Conv layers | Fused params | Conv MACs |
|---|---:|---:|---:|
| `c16n8` | 10 | 26,096 | 835,142,400 |
| `c16n10` | 12 | 30,768 | 984,441,600 |
| `c20n6` | 8 | 31,088 | 997,272,000 |

Count params as Conv weight+bias plus body PReLU parameters. Count MACs for Conv only; do not count PReLU, PixelShuffle, or elementwise add.

### Gate 0 — model/fuse/export audit

Before any 20k run:

1. Build each candidate with a fixed initialization seed and assert output shape `(B,3,4H,4W)`.
2. Compare unfused `eval()` with `fuse_sepres_v2()` on a fixed random input. Require `max_abs ≤ 1e-5`.
3. Assert fused Conv count, parameter count, and Conv MACs against the table above.
4. Export the fused random-initialized model through TorchScript → PNNX → NCNN at the `deploy_720p` preset.
5. Run a short phone smoke to catch unsupported operators and gross latency regressions. This value is labelled `graph_smoke`, never `official`.

Write one aggregate file after model/fuse/budget audit:

```text
results/exp_runs/b4_v2_fuse_smoke.json   # from scripts/check_sepres_v2_fuse.py
```

Then complete export + phone smoke into:

```text
results/exp_runs/b4_v2_prescreen.json
```

For each candidate record config identity, seed, shape, fuse max error, fused Conv/params/MACs, export paths/status, NCNN bytes, phone smoke protocol, median/p90, and pass/fail reason.

Random-initialized latency is valid for exportability and graph-order checks because tensor shapes/operators are fixed. It is not valid for close ranking, the 33.3 ms claim, or architecture freeze: thermal/session variation is larger than the existing ECBSR–Lite median gap.

### Gate 1 — phone measurement envelope

Before using phone latency to decide B4:

1. Bench Lite-sep and ECBSR-fused in at least three paired sessions on the same device/backend.
2. Alternate order (`Lite→ECBSR`, then `ECBSR→Lite`) and use the official `warmup=50`, `iters=300`, LR `320×180`, NCNN Vulkan FP16 protocol.
3. Record every session separately; do not overwrite raw values with only an average.
4. Define `E_med` and `E_p90` as the larger same-model cross-session range observed across the two references. A candidate/reference difference inside the corresponding envelope is a tie.

Write:

```text
results/exp_runs/b4_measurement_envelope.json
```

The file must include device/build/protocol identity, run order, per-session med/p90, ranges, and the resulting `E_med`/`E_p90`. If device temperature or background load invalidates a session, retain it with an exclusion reason rather than deleting it silently.

For one-seed validation screening, compute:

```text
E_val = max(
  2 × std(last 20 val_psnr of candidate),
  2 × std(last 20 val_psnr of ECBSR)
)
```

This only separates late-run jitter from an apparent difference. It does not estimate seed variance; B5 remains responsible for multi-seed confirmation.

### Gate 2 — staged training and official deployment

Training order is dependency-driven:

1. Train `sepres_v2_c16n10_20k` first.
2. Use its val result plus graph pre-screen to decide whether `c16n8` can buy meaningful speed/size and whether `c20n6` can test a viable width–depth shape.
3. Launch a/c only when the result can change freeze/Exit. Do not launch all three merely because configs exist.

Checkpoint selection remains DIV2K-valid RGB PSNR only. Benchmark sets never select a candidate.

For every promoted `best.pt`, repeat fuse/export and run official phone sessions paired with ECBSR. Write timestamped phone JSON under `deploy/artifacts/results/`; only a frozen model may become a canonical `deploy/models.json` entry.

One-shot Gate-2 driver:

```bash
python scripts/run_b4_v2_posttrain.py --wait
# or when best.pt already exists:
python scripts/run_b4_v2_posttrain.py
```

Flags: `--sessions 3` (default), `--skip-eval`, `--smoke` (writes `b4_v2_posttrain_smoke.json`, not official). Does **not** edit `deploy/models.json`.

Final B4 aggregate:

```text
results/exp_runs/b4_v2_compare.json
```

It must distinguish `graph_smoke` from `trained_official`, include `E_val/E_med/E_p90`, use actual NCNN bytes, and store the envelope-aware dominance decision. Differences inside an envelope are ties. Median `≤33.3 ms` is the real-time budget; p90 is reported separately and is required for a stability claim.

---

## 12. Post-train pipeline warmup（agent 合同，非 resume 挂钩）

**Why it used to look different every time:** each family (Plus / ECBSR / SepResV2) got an ad-hoc dry-run script invented in-chat. That was process debt, not a real need for different mechanics.

**Contract (principles E12):**

| Rule | Meaning |
|---|---|
| **Once per model family** | First time a new deployable family starts training, prepare the post-train path once (script exists, export smoke with `latest.pt` if useful, known official command). |
| **Not every resume** | Later `pause`/`resume` of the same family does **not** re-trigger warmup. |
| **Agent-driven, not script-auto** | Do **not** hook phone/posttrain into `*_20k.py resume`. Phone may be unplugged; auto-bench would fail or block. |
| **Shared steps** | Always the same skeleton: fuse → TorchScript/PNNX/NCNN → (when phone up) paired bench → eval → compare JSON. Family-specific = model/`fuse_*` + output names only. |

**Done markers (skip if already present for that family):**

| Family | Warmup done when |
|---|---|
| SepResV2 | `scripts/run_b4_v2_posttrain.py` exists + `b4_v2_posttrain_smoke.json` (or Gate-0 export already proved the graph) |
| ECBSR | `dryrun_ecbsr_deploy.py` / official mobile bench path already used |
| MobileSRNet Lite/Plus | A0 `convert_deployment` + `bench_mobile` path already proven |
| **DualStream / Plain C20N5** | `check_dual_plain_fuse.py` PASS + `b4_dual_plain_prescreen.json` + `run_dual_plain_posttrain.py` + `b4_dual_plain_posttrain_smoke.json` |
| **B5a P0 plain3x3** | Same deploy path as SepResV2 (`fuse_sepres_v2` / convert); Gate-0 `b5a_plain_fuse_smoke.json` PASS — **not** a new export family |

**Official after train (human/agent when phone is available):**

```bash
# SepResV2 Gate-2
python scripts/run_b4_v2_posttrain.py --wait   # or without --wait if best.pt ready

# Dual/Plain after D18 val gate (paired phone path: extend sessions when ready)
python scripts/run_dual_plain_posttrain.py --wait
```

Smoke during train (optional, CPU/export only, no phone):

```bash
python scripts/run_b4_v2_posttrain.py \
  --checkpoint results/exp_runs/<run>/checkpoints/latest.pt \
  --smoke --skip-eval --sessions 0

python scripts/run_dual_plain_posttrain.py \
  --checkpoint results/exp_runs/dual_stream_c20n5_2k/checkpoints/latest.pt \
  --smoke --skip-eval --sessions 0
```

---

## 13. B4 round-2 DualStream / Plain C20N5 — contract

`progress/track_b.md` **B4 round-2** owns D18 gates. This section owns mechanism.

### Model contract

- Train Dual: `type: dual_stream_sr` — detail 17 + low 3, `num_mid=5`, YAML `aux_weight=0.5`; trainer `return_aux=True`.
- Train Plain: `type: plain_sr` — identical deploy geometry, `aux_weight=0`.
- Fuse: `fuse_dual_stream_sr(DualStreamSR) → PlainSR`, FP32 `max_abs ≤ 1e-5`.
- Deploy budget @ LR 180²: **27,348** params, **7** Conv, **880,632,000** MACs (~0.881G).

### Control plane

```bash
python scripts/check_dual_plain_fuse.py
python scripts/prescreen_dual_plain.py --skip-bench
python scripts/dual_plain_2k.py resume|watch|pause
python scripts/run_dual_plain_posttrain.py --smoke --sessions 0
```

Python interpreter: `/home/hyb/miniforge3/envs/cv_env/bin/python` (as in other B3/B4 launchers).

### Artifacts

| File | Role |
|---|---|
| `results/exp_runs/b4_dual_plain_fuse_smoke.json` | Fuse + analytic budget |
| `results/exp_runs/b4_dual_plain_prescreen.json` | Random-init export (`graph_smoke`) |
| `results/exp_runs/b4_dual_plain_posttrain_smoke.json` | Trained `latest.pt` export (E12) |
| `results/exp_runs/dual_plain_2k_manifest.json` | Launcher registry |

Reuse Gate-1 `b4_measurement_envelope.json` for phone ties. Do not edit `deploy/models.json` until freeze.
---

## 14. B5 unified — B5a / multi-seed / PECSR KD

`progress/track_b.md` B5a/B5 owns research gates. This section owns mechanism.

**One control plane:** `b5_train_20k.py` / `run_b5_train_20k.py` / `b5_train_20k_manifest.json`.  
Legacy `b5a_20k` / `b5_confirm_20k` / `pecsr_kd_20k` are thin redirects — do not run their old `run_*` alongside unified.

### B5a P0 (`body_kind=plain3x3`)

- Model: `SepResV2(..., body_kind="plain3x3")` — `PlainBodyBlock` = Conv3×3+PReLU (ECBSR paper plain baseline).
- Fuse: weight copy into `FusedECB`; fused budget **must** match PECSR C16N10 (30,768 / 0.984G / 12 conv).
- Phone: **graph-identity only** (do not multi-session rank vs PECSR).
- Gate-0: `check_b5a_plain_fuse.py`. Manifest lane=`mse`, variant=`P0_plain`.

### B5 multi-seed (MSE lane)

- Seeds 42/123/2026 for PECSR + ECBSR; seed-42 rows reuse existing runs (`skip-done`).
- FSRCNN×3 and VGG KD: **default off**.
- Launcher default **`--max-parallel 3`** (overnight keep 3-wide until MSE queue empty). Adopts already-live trains.
- Calendar: IMPLEMENTATION §9.

### PECSR matched KD (KD lane — Stage-B 2k)

- Student: `type: sepres_v2` C16N10 via `train_mobile_srnet_kd.py`.
- **Screen only (~2k updates, bs=16 / 40 ep).** Not a full-budget PECSR KD claim.
- Method-local λ (do **not** share one λ grid):
  - Pixel: `pecsr_pixel_kd0_2k` (λ=0) vs `pecsr_pixel_kd02_2k` (λ=0.2)
  - VGG relu3: `pecsr_vgg3_kd0_2k` (λ=0) vs `pecsr_vgg3_kd01_2k` (λ=**0.01**, gate-equalized)
- Promote any pair to 20k **only if** Stage-B Δ clears noise. Superseded 20k configs live under `configs/_inactive/exp/pecsr_kd*_20k.yaml`.
- Unified launcher starts KD **only after every MSE entry is done**, sequential (`max_parallel=1`). Never mix with MSE (§9). Optional `--mse-only`.
- **Won’t:** pruning; INT8 / BF16. FP32/FP16 = deploy precision reporting (RQ deploy), not a quant study.

### Commands

```bash
python scripts/check_b5a_plain_fuse.py
python scripts/b5_train_20k.py resume          # --max-parallel 3
python scripts/b5_train_20k.py watch --interval 60
python scripts/b5_train_20k.py pause
```

Python: `/home/hyb/miniforge3/envs/cv_env/bin/python`.

