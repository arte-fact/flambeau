# Cumulative opt-in test — Qwen3.6-27B-Q4_0 / pp2tp2 / N=[1,2,4,8]

- **Generated:** 2026-05-06T14:59:26+00:00
- **Profile:** `bench/profiles/optimized.toml` (5 baseline keys)
- **Cumulative env:** 8 opt-in gates set simultaneously
- **Runs/cell:** 3 measured + 2 warmup, greedy / temp=0 / seed=0
- **Noise floor:** ±2.0%

**Hypothesis.** Each opt-in individually certed null (≤ ±2%) at N=1. If they're additive, setting all 7 together should give a measurable cumulative gain.

**Cumulative env applied:**

- `FLAMBEAU_AR_FUSE_Q8_1=on`
- `FLAMBEAU_Q4_0_GU_T128=on`
- `FLAMBEAU_Q4_0_GU_WARPCOOP=on`
- `FLAMBEAU_BATCHED_MMVQ=1`
- `FLAMBEAU_SSM_OUT_F16_DST=on`
- `FLAMBEAU_VARIANT=fused`
- `FLAMBEAU_Q8_0_MMVQ_T128_VDR2=on`
- `FLAMBEAU_Q8_0_GU_T128_VDR2=on`

## Measured cells

| N | label | n | prefill ms | decode ms | aggregate tg t/s | Δ vs baseline | correct? | VRAM peak GB | err |
|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| 1 | baseline | 3 | 11321 | 2729 | 23.45 | — | — | 20.92 |  |
| 1 | cumulative | 3 | 11351 | 2754 | 23.23 | -0.9% | match | 20.92 |  |
| 2 | baseline | 3 | 17893 | 3411 | 31.35 | — | — | 21.37 |  |
| 2 | cumulative | 3 | 17937 | 3460 | 30.49 | -2.8% | match | 21.37 |  |
| 4 | baseline | 3 | 31033 | 4025 | 61.32 | — | — | 22.40 |  |
| 4 | cumulative | 3 | 31059 | 4118 | 60.90 | -0.7% | match | 22.35 |  |
| 8 | baseline | 3 | 57096 | 4036 | 119.29 | — | — | 24.03 |  |
| 8 | cumulative | 3 | 57142 | 4136 | 120.54 | +1.1% | match | 24.04 |  |

## Triage

- N=1 → **null (-0.9%)**
- N=2 → **loss (-2.8%)**
- N=4 → **null (-0.7%)**
- N=8 → **null (+1.1%)**

## Disposition

- **CUMULATIVE-LOSS** — 1/4 N values regress; the kitchen-sink hurts overall
