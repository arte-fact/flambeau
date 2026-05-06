# Env-impact cert: `FLAMBEAU_Q4_0_GU_T128`

- **Class:** B
- **Default state:** `T256`
- **Generated:** 2026-05-05T20:56:25+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Q4_0 gate_up tile size (T=128 vs T=256)

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11442 | 2719 | 23.54 | — | — |  |
| `on` | 3 | 11432 | 2734 | 23.41 | -0.5% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11444 | 3000 | 21.33 | — | — |  |
| `on` | 3 | 11519 | 3001 | 21.33 | -0.0% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3933 | 1447 | 44.24 | — | — |  |
| `on` | 3 | 3909 | 1425 | 44.92 | +1.5% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3779 | 1275 | 50.18 | — | — |  |
| `on` | 3 | 3795 | 1260 | 50.79 | +1.2% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `on` → **null** (-0.5%)
- qwen36-27b-q4_0/tp2 `on` → **null** (-0.0%)
- qwen36-35b-a3b-q4_0/pp2tp2 `on` → **null** (+1.5%)
- qwen36-35b-a3b-q4_0/tp2 `on` → **null** (+1.2%)

## Disposition

- CANDIDATE-DELETE — no measured impact; bake default and remove gate

## Notes from spec

Read in models/dense_ffn_tp.rs:165. C6 cert exists; re-measure.
