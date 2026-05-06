# Env-impact cert: `FLAMBEAU_Q8_0_MMVQ_T128`

- **Class:** B
- **Default state:** `T256`
- **Generated:** 2026-05-05T21:17:48+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Q8_0 MMVQ tile T=128 (latency-hiding variant)

## Measured cells

### qwen36-27b-q8_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 36678 | 3906 | 16.39 | — | — |  |
| `on` | 3 | 36617 | 4101 | 15.60 | -4.8% | match |  |

### qwen36-27b-q8_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19474 | 3092 | 20.70 | — | — |  |
| `on` | 3 | 19885 | 3323 | 19.26 | -7.0% | match |  |

## Triage

- qwen36-27b-q8_0/pp4 `on` → **loss** (-4.8%)
- qwen36-27b-q8_0/tp2 `on` → **loss** (-7.0%)

## Disposition

- KEEP-DEFAULT — current default wins, alternate is dead path

## Notes from spec

Read in ops/qmatmul.rs:988. C9 cert exists; re-measure post-recent commits.
