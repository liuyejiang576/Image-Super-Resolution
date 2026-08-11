# Next experiments

Aligned with frozen baseline in [SNAPSHOT.md](SNAPSHOT.md). Promote results via [SYNC.md](SYNC.md).

---

## Philosophy (unchanged, stated clearly)

> **Train expensively if useful; deploy cheaply by design; verify efficiency on the actual inference path.**

- SwinIR at training time: acceptable (offline teacher).
- MobileSRNet at inference: required.
- Efficiency = quality + params + FLOPs + **precision + backend latency** — not FLOPs alone.

The project is a **controlled study of quality–efficiency trade-offs in mobile-oriented SR**, not SOTA chasing, not a mobile app, not “try every compression trick.”

---

## Do we change report focus?

**Reframe, not replace.**

| Keep as foundation | Extend under deploy emphasis |
|---|---|
| RQ1: fair-budget architecture (Plus > FSRCNN @ lower FLOPs) | 30k stability table (all three models) |
| RQ2: KD null (pixel + VGG gates; capacity scaling only win) | Do **not** reopen KD as centerpiece |
| RQ3: RTX proxy latency; FLOPs ≠ ms; FP16 asymmetry | **Real device** latency for same checkpoints |

- **Keep three RQs.** Do not expand to five RQs (ECBSR / INT8 / HF-KD as separate research questions) until data exists.
- **Do not rewrite `report.tex` structure before experiments.** Update intro framing + limitations now; add deployment tables when numbers land.
- Previous 20k experiments remain **highly valuable** — they motivate *why* device validation matters. Mobile bench **extends** RQ3; it does not invalidate fair-budget or KD-null evidence.

---

## Priority order (do in this sequence)

```text
1. Continue FSRCNN / Base / Plus → 30k          [required]
2. Real mobile benchmark (one device, one runtime) [required for deploy story]
3. ECBSR block as one Plus variant              [optional — only if 1–2 done]
4. INT8 PTQ through same mobile backend         [optional — only after 2]
```

**Deprioritized / skip unless spare time:**

- Wavelet / high-frequency KD — already gated; pixel KD redundant (0.924); VGG failed Stage B. Another KD variant risks “try until something sticks” and adds **zero** inference benefit. Not worth reviving for a deploy-focused arc.
- More λ-sweeps, transformers, diffusion, GAN SR, arbitrary model sizes.
- PyTorch-only dynamic INT8 without deployment export.
- Second RTX re-audit at 320×180 — already in `latency_audit_320x180.json` unless figures are wrong.

---

## Experiment 1 — 30k continuation

**Priority: highest**

All three main models, not Plus alone:

```text
FSRCNN-fix-clean
MobileSRNet-Base
MobileSRNet-Plus
```

**Honest labeling:** original recipe decayed LR at 60% / 85% of 20k. Continuing to 30k is **20k training + ~10k low-LR fine-tuning**, not a fresh “fair 30k from scratch” recipe. Say that in the report.

**Note:** 20k does **not** invalidate current claims — all models peaked at epoch 607 under the **same** budget. 30k **strengthens** ranking stability; it does not fix a broken comparison.

### Deliverables

| Model | 20k Avg PSNR | 30k Avg PSNR | Δ | Best step | FLOPs |
|---|---:|---:|---:|---:|---:|
| FSRCNN | (frozen) | … | … | … | 7.41G |
| Base | (frozen) | … | … | … | 0.98G |
| Plus | (frozen) | … | … | … | 2.16G |

Plot: DIV2K-val PSNR vs steps, 0–30k (all three).

### Success

- **Best:** Plus > Base > FSRCNN at 30k benchmarks.
- **Acceptable:** All improve; Plus still best quality–FLOP point; or “MobileSRNet wins at equal/lower training budget; long-run gap narrows.”

---

## Experiment 2 — Real mobile benchmark

**Priority: highest new evidence** (closes the “mobile-efficient” gap vs RTX proxy)

### Scope (calibrated ambition)

- **One phone** + **one runtime** beats a two-chip study that never ships.
- Start with **Snapdragon 8s Gen 3** if available (NCNN/MNN/TFLite/ONNX Runtime Mobile tooling is more documented than chasing multiple NPUs).
- Dimensity 9400 is a bonus, not a requirement.

### Backend

Pick **one** stack for all models; do not spend weeks on export plumbing.

Suggested order to try:

```text
NCNN Vulkan FP16  →  MNN  →  TFLite FP16  →  ONNX Runtime Mobile
```

PixelShuffle / depthwise blocks: NCNN or MNN often easier than some TFLite paths.

### Models

```text
FSRCNN, Base, Plus   (30k checkpoints when ready; 20k acceptable for first pipeline test)
```

No KD variants on device — same graph as non-KD student.

### Input

Primary: **LR 320×180 → HR 1280×720** (meaningful deploy size).

Optional cross-check: LR 180×180 (match RTX audit).

### Protocol

```text
Warmup: 50
Timed: 300
Report: median + p90 ms, batch=1, backend, precision, threads (if CPU)
```

### Success (all are publishable)

| Outcome | Story |
|---|---|
| Plus faster than FSRCNN on device **and** higher PSNR | Strong deploy win |
| Plus lower FLOPs but similar/slower latency | **Still strong** — confirms RQ3: operator/backend matter |
| Ranking differs from RTX | **Still strong** — proxy ≠ device; report both |

---

## Experiment 3 — ECBSR block (optional)

**Priority: after 1–2, only if time remains**

Hypothesis: depthwise-separable blocks cut FLOPs but may map poorly to some backends; reparameterizable / edge-oriented blocks may improve **measured** latency.

**Scope guard:**

- Replace **one** block type in Plus → `MobileSRNet-ECB-Plus` (single variant).
- Do **not** reproduce full ECBSR paper or start a new model family.
- This is a **follow-up hypothesis**, not required to satisfy the course deploy narrative.

Compare: ECB-Plus vs Plus @ 30k on benchmarks + RTX + (if possible) same mobile runtime.

---

## Experiment 4 — INT8 PTQ (optional)

**Priority: only after mobile export works**

- PTQ through the **same** deployment backend as Exp 2 (not PyTorch `quantize_dynamic` alone).
- Quantize **one** finalist (likely Plus or ECB-Plus).
- Report PSNR drop vs FP16 latency gain.

If PTQ fails badly, that is still a valid deployment finding.

---

## Minimal experiment matrix (anti-sprawl)

```text
Core:     FSRCNN-30k | Base-30k | Plus-30k
Optional: Plus-ECB-30k
Deploy:   {FSRCNN, Base, Plus} × {FP32, FP16} on mobile; + INT8 on one model
KD:       none new
```

Do **not** combine: ECBSR + HF-KD + INT8 + QAT + multi-λ in one run.

---

## Report updates (when data exists)

| When | Change |
|---|---|
| Now | Intro/limitations: “proxy validated; device TBD” |
| After 30k | One table + curve; one paragraph on stability |
| After mobile | RQ3 subsection: device table + proxy vs device discussion |
| After ECBSR/INT8 | Short subsubsection each; only if actually run |

**Thesis sentence** (for intro, when revising):

> We separate training-time resources from deployment-time constraints: offline supervision and fair-budget training design compact students whose efficiency we evaluate through arithmetic cost, precision mode, and measured latency on proxy and mobile backends.

---

## What previous work already proves (do not re-litigate)

- Plus beats FSRCNN at ~3.4× lower FLOPs under fair 20k MSE recipe.
- Capacity scaling Base→Plus is the only confirmed quality lever.
- Pixel Charbonnier KD is gradient-redundant; VGG feature KD failed Stage B.
- FLOPs ranking ≠ CUDA latency ranking; FP16 helps depthwise models differently than FSRCNN on RTX 4060.

Next experiments **test stability and external validity**, not whether to throw this away.

---

## One paragraph for the team

> Our project is **deployment-aware efficient SR**: allow expensive offline training, deploy only the student, and judge efficiency on real execution paths—not FLOPs alone. Existing 20k lab work stays; we extend with 30k stability (all three models) and one mobile latency study. ECBSR and INT8 are optional follow-ups tied to the same backend. We are **not** running more KD sweeps unless a new gate proves non-redundant signal—and HF/wavelet KD is deprioritized. Every new run must stabilize architecture claims, validate deployment, or test operator-aware design; otherwise skip it.
