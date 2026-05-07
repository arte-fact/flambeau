# Env-impact cert: `FLAMBEAU_INFLIGHT_SLOTS`

- **Class:** C
- **Default state:** `4`
- **Generated:** 2026-05-07T17:23:27+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Phase 3a — perf curve across slot counts at N=1

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `1` | 3 | 11382 | 2742 | 23.34 | — | — | 20.92 |  |
| `4` | 3 | 11387 | 2740 | 23.36 | +0.1% | match | 22.26 |  |
| `8` | 3 | 11478 | 2745 | 23.32 | -0.1% | match | 24.03 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `1` | 3 | 3918 | 1433 | 44.65 | — | — | 23.05 |  |
| `4` | 3 | 3910 | 1441 | 44.42 | -0.5% | match | 24.34 |  |
| `8` | 3 | 3900 | 1434 | 44.62 | -0.1% | match | 25.82 |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=1] `4` → **null** (+0.1%)
- qwen36-27b-q4_0/pp2tp2 [N=1] `8` → **null** (-0.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `4` → **null** (-0.5%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=1] `8` → **null** (-0.1%)

## Disposition

- **CANDIDATE-DELETE** — no measured impact on any of 4 cells; bake default and remove gate

## Notes from spec

Confirms the slots=1 → slots=8 throughput cliff at N=1.
