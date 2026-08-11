# Evaluation of Agent Suggestions (sug_1 – sug_4)

**Basis:** objective reasoning cross-checked against the local files — `proposal.md` (guideline + V1 proposal), `codebase.md` (registry), `report/latex/report.tex` (current report), and the actual source under `Image-Super-Resolution/` (verified, not just the manifest).

**Scope note:** `sug_1` is the "next experiments" plan itself; `sug_2`–`sug_4` are critiques of it. All four are evaluated on their own merits.

---

## 1. Ground truth from the local files

### Guideline requirements (`proposal.md`)

- Eval metrics explicitly listed (`proposal.md:11`): **PSNR, SSIM, LPIPS, runtime, memory usage, model size, and mobile inference speed.**
- Motivation (`proposal.md:5`): "mobile photography and **real-time enhancement**."
- Possible improvement methods (`proposal.md:9`): lightweight U-Net, **depthwise separable convolution, model pruning, quantization, knowledge distillation, mobile-friendly attention modules.**
- Challenge alignment (`proposal.md:11`): links the **Mobile AI 2026 Challenge** tracks.
- V1 proposal itself (`proposal.md:83`) planned to report "Parameters, FLOPs, **model size (MB)**, inference latency."

### What the current report actually delivers (`report.tex`)

- **Delivered:** PSNR, SSIM, LPIPS, params, FLOPs, FP32/FP16 RTX-4060 latency (Table `tab:efficiency`, `report.tex:283-293`).
- **Missing vs. guideline:** memory usage (nowhere), mobile inference speed (RTX proxy only), and model size in MB is **not in Table 4** — even though `profile_model.py:105` already computes `model_size_fp32_mb`. So model size is a *reporting* gap, not a tooling gap.
- **Non-convergence is explicit:** best checkpoint at the final epoch (607), not plateaued (`report.tex:298-299`, `431-433`).
- **RTX-proxy limitation is explicit:** "on-device mobile inference (NPU, INT8) is not measured" (`report.tex:443-444`).
- **KD is thoroughly documented** as RQ2 (`report.tex:315-360`): gates table (`tab:gates`), lambda sweep (Fig. 3), VGG Stage-B probe (Fig. 6), and a KD summary table (`tab:kd`). KD is **not** omitted.
- **Operator finding already in the report:** "depthwise-separable convolutions are memory-bandwidth bound rather than compute bound on the RTX 4060" (`report.tex:370`).
- **ReLU6 chosen for quant friendliness:** "ReLU6 bounds activations for quantization stability" (`report.tex:138`).

### What the codebase actually supports (verified in source)

- **20k checkpoints exist for all three models** (`codebase.md:201-203`): `fsrcnn_20k.pt`, `mobile_srnet_base_20k.pt`, `mobile_srnet_plus_20k.pt`. Mobile export can start **now**.
- **Zero mobile-deployment tooling exists.** No ONNX/NCNN/MNN/TFLite export script anywhere; `profile_model.py` and `quantize_benchmark.py` are PyTorch/CUDA/CPU only. Mobile is the **riskiest, least-developed** path.
- **`quantize_benchmark.py` is exactly the "PyTorch-only dynamic INT8"** that `sug_1` deprioritizes: `torch.ao.quantization.quantize_dynamic` on Conv2d, with CPU latency as a "deployment proxy" (`quantize_benchmark.py:61-67,86`). So existing INT8 numbers are a CPU proxy, not a deployment result.
- **`mobile_srnet.py` is pure depthwise-separable + ReLU6 + PixelShuffle — no attention, no pruning.** The guideline items "mobile-friendly attention modules" and "model pruning" are both unaddressed in code and report.
- **Memory is measured nowhere** — neither in tooling nor in the report. This is a genuine tooling gap (unlike model size).

---

## 2. Per-suggestion verdicts

### sug_1 — the next-experiments plan (the base)

**Objectively correct:**

- **30k continuation** — justified by non-convergence (epoch 607, not plateaued). The honest labeling ("20k training + ~10k low-LR fine-tuning, not a fresh 30k recipe") is scientifically sound.
- **Real mobile benchmark is required** — the guideline mandates "mobile inference speed" and the report only has an RTX proxy. Correct.
- **Keep KD closed** — the gate-then-probe protocol + Stage-B failure justify this; reopening risks "try until something sticks."
- **Deprioritize PyTorch-only dynamic INT8** — *verified correct*: `quantize_benchmark.py` is precisely that. Real INT8 must go through a deployment backend.
- **Three RQs, anti-sprawl discipline** — sound scope management; the report's compact story (architecture wins / KD fails / FLOPs ≠ latency) is worth protecting.

**Weaknesses (against the ground truth):**

- **Sequences mobile *after* 30k.** With 20k checkpoints already on disk and zero mobile tooling, this wastes the 30k training window and risks discovering export breakage late. Should run in parallel.
- **Under-emphasizes memory usage** — a guideline-required metric absent from both report and tooling.
- **Does not surface model size MB** — the number already exists in `profile_model.py:105`; it just isn't in Table 4.
- **ECBSR framed as "spare time."** The report already supplies the motivating evidence (`report.tex:370`, depthwise is memory-bandwidth bound). ECBSR should be *evidence-triggered*, not spare-time.
- **"What previous work already proves"** language is too strong for non-converged, single-seed (42), proxy-only results. Should be "supports/suggests."
- **Two guideline items entirely unaddressed:** mobile-friendly attention modules and pruning (the latter not even acknowledged as an omission).
- **No flag on Mobile AI Challenge alignment** — a rubric risk if the course expects challenge relevance.

### sug_2 — critique

**The strongest of the four.** Best grounded in the local files; catches the most real issues.

- **Mobile non-negotiable + start immediately with 20k checkpoints (parallel):** *verified correct*. 20k checkpoints exist (`codebase.md:201-203`), zero mobile tooling exists, and mobile export is known-fiddly (PixelShuffle, depthwise kernels, FP16 handling). Parallel start is objectively the right risk management — this is the single most important correction to `sug_1`.
- **Report memory + model size:** *verified guideline gap*. Nuance worth noting: model size is a *reporting* fix (data already in `profile_model.py`); memory needs *new tooling*. The proposed table shape (params, model size, FLOPs, backend, precision, median, p90, peak memory) is the right one.
- **Don't overstate as mature:** *verified correct*. Non-converged + single seed + proxy ⇒ "supports/suggests," not "proves." A real rigor point the others missed.
- **ECBSR triggered by mobile evidence, not spare time:** well-grounded — `report.tex:370` already shows the operator/backend mismatch that ECBSR targets.
- **One-phone claims must be limited to the tested stack:** scientifically correct; the report already found FLOPs ≠ RTX latency, so mobile backends will vary at least as much.
- **INT8 elevated slightly:** defensible — ReLU6 was chosen for quant friendliness (`report.tex:138`) and the existing INT8 work is only a PyTorch CPU proxy, so a real-backend INT8 is the genuine missing piece. Not an overreach.
- **Challenge alignment (conditional on rubric):** valid — guideline links Mobile AI 2026 tracks; report/proposal use DIV2K classical bicubic only.

**Minor:** none significant. The INT8 elevation is the most debatable point and still defensible.

### sug_3 — critique

**Solid on metrics, but contains a factual misread of the report and some padding.**

- **Memory usage missing:** *verified* ✓ (agrees with sug_2).
- **30k, mobile (strongly), ECBSR conditional, INT8:** all correct ✓.
- **FPS / real-time framing:** reasonable but *soft* — the guideline's eval list says "runtime" / "mobile inference speed," from which FPS is directly derivable. Valid as a presentation choice, not a missing metric. Low priority.
- **Pruning — "acknowledge omission + justify":** *this is the objectively correct pruning stance* (see cross-cutting C below). sug_3 is better than sug_4 here.

**Weakness / error:**

- **Misreads the report on KD.** sug_3 says KD should be "documented rather than simply omitted." But `report.tex` already has a full RQ2 (`report.tex:315-360`) with the gates table, lambda sweep, VGG Stage-B probe, and a KD summary table. KD is one of the three RQs — it is **not** omitted. This point addresses a non-issue and is a factual error about the current report.
- **Misses the biggest risk-management point** (parallel mobile export) that sug_2 makes.
- **Misses the "proves" language issue** and **challenge alignment**.
- The mermaid diagram is decorative padding (and uses emoji node labels), adding no analytical content.

### sug_4 — critique

**The most intellectually careful/nuanced; catches two things the others miss, but overstates pruning.**

- **Mobile benchmark, 30k, KD closed, INT8, three RQs, philosophy:** all agreed ✓.
- **Mobile-friendly attention modules never tested:** *verified* — `mobile_srnet.py` has none; the guideline lists them (`proposal.md:9`). A real guideline item that sug_1/2/3 all missed. Good catch. *However*, the proposed experiment is low-value: SE/CA attention adds more pointwise (memory-bandwidth-bound) ops — exactly the bottleneck the report identified (`report.tex:370`). So it is a correct *gap identification* but a questionable *experiment*. sug_4 hedges this honestly ("risk of missing it is low").
- **ECBSR risks diluting the core message — keep as a separate sub-study, do not merge into the main architecture table:** a good methodological guard that sug_2/3 did not make. Complements sug_2's "trigger by evidence" framing.
- **Single-device generalizability stated in limitations:** ✓ (agrees with sug_2).

**Weakness:**

- **Pruning as "missed opportunity / stretch goal" is overstated at this scale.** Plus is 67K params, Base 30K. Pruning a 30–67K-param depthwise model yields almost nothing, and the report's own RQ3 finding (memory-bandwidth bound, kernel-launch overhead dominates at batch 1) predicts pruning will not improve latency. sug_3's "acknowledge omission" is the better stance.
- **Misses** the parallel-mobile-export point, the "proves" language issue, memory usage, and challenge alignment.

---

## 3. Cross-cutting verdicts (who is objectively right)

| Disagreement | sug_1 | sug_2 | sug_3 | sug_4 | Objective verdict |
|---|---|---|---|---|---|
| **A. Mobile sequencing** (after 30k vs parallel) | after 30k | parallel | — | — | **sug_2.** 20k checkpoints exist; zero mobile tooling; export is the riskiest path. |
| **B. Memory + model size reporting** | weak | require both | require memory | — | **sug_2 ≈ sug_3.** Guideline gap verified. Model size = reporting fix (data exists); memory = new tooling. |
| **C. Pruning** | silent | — | acknowledge omission | missed opportunity | **sug_3.** 30–67K params + memory-bandwidth-bound regime ⇒ pruning is low-value; acknowledge, don't run. |
| **D. Attention modules** | silent | — | — | flag gap (weak experiment) | **sug_4** correctly flags the guideline gap; experiment is low-value given `report.tex:370`. Mention in limitations, don't run. |
| **E. ECBSR trigger** | spare time | evidence-triggered | conditional | keep separate sub-study | **sug_2 + sug_4** (complementary). `report.tex:370` already supplies the evidence. |
| **F. "Proves" language** | too strong | weaken to "supports" | — | — | **sug_2.** Non-converged, single seed, proxy-only ⇒ "supports." |
| **G. Challenge alignment** | silent | conditional flag | — | — | **sug_2.** Valid, conditional on rubric. |
| **H. FPS / real-time framing** | — | — | add FPS | — | **sug_3**, but soft/low priority (derivable from latency). |
| **I. KD documentation** | — | — | "document, don't omit" | — | **Non-issue.** Report already documents KD thoroughly; sug_3 misreads. |

---

## 4. Overall ranking

1. **sug_2** — most rigorous and best-grounded in the local files; catches the most real issues (parallel mobile export, memory + model size, epistemic language, challenge alignment, ECBSR trigger). The strongest.
2. **sug_4** — the most nuanced; uniquely catches the attention-module guideline gap and the ECBSR-dilution guard; weakest on pruning. Intellectually careful even where less comprehensive.
3. **sug_3** — solid on memory and has the correct pruning stance, but padded and contains a factual misread of KD documentation; misses the biggest risk-management point.
4. **sug_1** — the base plan; sound core priorities and good anti-sprawl discipline, objectively justified on 30k / mobile / KD-closed, but improvable on sequencing, metric completeness, language, and two unaddressed guideline items.

---

## 5. Recommended synthesis (what to actually do)

Assembled from the objectively-correct points across all four:

1. **Run 30k continuation for FSRCNN / Base / Plus AND start mobile export smoke tests on the existing 20k checkpoints in parallel.** Do not gate mobile on 30k. *(sug_1 + sug_2)*
2. **Mobile benchmark:** one device, one runtime (NCNN Vulkan FP16 first), report median + p90 ms, precision, backend, device/SoC/OS/runtime metadata, **and peak memory + model size MB**. Limit all claims to the tested stack. *(sug_2 + sug_3)*
3. **Surface `model_size_fp32_mb`** (already produced by `profile_model.py:105`) into report Table 4 now — trivial reporting fix; add a memory measurement to the mobile protocol. *(sug_2 + sug_3)*
4. **INT8 PTQ on one finalist through the SAME mobile backend** (not PyTorch `quantize_dynamic`). Report PSNR/SSIM/LPIPS drop vs latency/model-size gain. If PTQ fails badly, report that as a deployment finding. *(all agree)*
5. **ECBSR:** only if mobile evidence shows depthwise maps poorly to the chosen backend; keep it as a clearly separated "operator-aware design" sub-study, **not** merged into the main architecture table. *(sug_2 + sug_4)*
6. **KD:** stay closed; the report already documents the null thoroughly (no further sweeps). *(all agree)*
7. **Pruning + attention:** do **not** run either. Acknowledge both as deliberately omitted guideline items in Limitations, with the objective justification — params already minimal (30–67K), and attention adds memory-bandwidth-bound pointwise ops, which is the exact bottleneck identified in `report.tex:370`. *(sug_3 stance + sug_4 catch)*
8. **Language:** downgrade "proves" → "supports/suggests" throughout the report and the next-steps doc. *(sug_2)*
9. **Challenge alignment:** if the rubric expects it, add one sentence (e.g., "We focus on classical bicubic 4× SR as a controlled mobile-oriented setting rather than a specific challenge track"). *(sug_2)*

**Bottom line:** `sug_1`'s direction is right (stabilize architecture, validate on real hardware, don't reopen KD, don't sprawl). The corrections that materially matter, in priority order: (i) start mobile export **now, in parallel** with 30k; (ii) add **memory + model size** to the deployment table; (iii) fix the **"proves"** register; (iv) treat **ECBSR as evidence-triggered**, not spare-time; (v) **acknowledge** (don't run) pruning and attention in Limitations. With those, the plan satisfies the guideline better than the original proposal did.
