# B2 profile note — desktop module timing

- Device: `cpu` @ LR 320×180
- Warmup/runs: 5/30
- Timestamp: 2026-07-10T19:40:55.652445+08:00
- NCNN layer timing: **skipped** (`sr_bench` has no per-layer mode).

## Conclusions

- Desktop PyTorch (Base block0): DW=8.677ms vs PW=6.046ms (DW heavier on this backend).
- Plus vs Base desktop full: 235.07 vs 91.17 ms (feat 64/40, blocks 8/6, convs 18/14).
- Body dominates gap: Base body 68.30ms, Plus body 227.52ms (head 2.32/1.58, tail 19.90/14.00).
- Desktop fuse Base: 91.17 → 91.20 ms (convs 14 → 8).
- Desktop fuse Plus: 235.07 → 257.20 ms (convs 18 → 10).
- Phone B1 already showed fused slower on NCNN Vulkan; desktop result is supporting context only (PyTorch ≠ NCNN).
- B4 guidance: do not chase DW→dense fuse for speed; prefer capacity (width/depth) or backend-friendly blocks (ECBSR-style) over folding sep.

## Phone context (already measured)

- mobile_srnet_base: sep median 34.0097 → fused 48.9565 ms (phone)
- mobile_srnet_plus: sep median 48.8737 → fused 76.1955 ms (phone)
- A0 20k phone medians:
  - fsrcnn: 46.31 ms (p90 51.91)
  - mobile_srnet_base: 30.94 ms (p90 37.52)
  - mobile_srnet_plus: 51.08 ms (p90 54.83)

## Desktop timing table (median ms)

| model | variant | full | head | body | tail | convs |
|---|---|---:|---:|---:|---:|---:|
| fsrcnn | native | 61.33 | — | — | — | 7 |
| mobile_srnet_base | native | 91.17 | 2.32 | 68.30 | 19.90 | 14 |
| mobile_srnet_base | fused | 91.20 | 2.84 | 102.38 | 13.59 | 8 |
| mobile_srnet_plus | native | 235.07 | 1.58 | 227.52 | 14.00 | 18 |
| mobile_srnet_plus | fused | 257.20 | 1.58 | 182.35 | 20.06 | 10 |
