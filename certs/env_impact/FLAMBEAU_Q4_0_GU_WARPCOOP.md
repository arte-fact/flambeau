# Env-impact cert: `FLAMBEAU_Q4_0_GU_WARPCOOP`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T21:05:43+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Q4_0 gate_up wave64 warpcoop variant (C6 opt-in)

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11445 | 2737 | 23.38 | — | — |  |
| `on` | 3 | 11449 | 2711 | 23.61 | +1.0% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11422 | 3003 | 21.31 | — | — |  |
| `on` | 3 | 11509 | 2961 | 21.61 | +1.4% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3919 | 1433 | 44.68 | — | — |  |
| `on` | 3 | 3893 | 1448 | 44.20 | -1.1% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3787 | 1267 | 50.53 | — | — |  |
| `on` | 3 | 3789 | 1260 | 50.81 | +0.6% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `on` → **null** (+1.0%)
- qwen36-27b-q4_0/tp2 `on` → **null** (+1.4%)
- qwen36-35b-a3b-q4_0/pp2tp2 `on` → **null** (-1.1%)
- qwen36-35b-a3b-q4_0/tp2 `on` → **null** (+0.6%)

## Disposition

- CANDIDATE-DELETE — no measured impact; bake default and remove gate

## Notes from spec

Read in models/dense_ffn_tp.rs:163.
