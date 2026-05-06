# Env-impact cert: `FLAMBEAU_Q8_0_GU_T128_VDR2`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T21:45:14+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Q8_0 gate_up T=128 + VDR2 (decoder MoE up/down path)

## Measured cells

### qwen36-27b-q8_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19234 | 2797 | 22.88 | — | — |  |
| `on` | 3 | 19199 | 2803 | 22.83 | -0.2% | match |  |

### qwen36-27b-q8_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 19979 | 3153 | 20.30 | — | — |  |
| `on` | 3 | 19999 | 3154 | 20.29 | -0.0% | match |  |

## Triage

- qwen36-27b-q8_0/pp2tp2 `on` → **null** (-0.2%)
- qwen36-27b-q8_0/tp2 `on` → **null** (-0.0%)

## Disposition

- CANDIDATE-DELETE — no measured impact; bake default and remove gate

## Notes from spec

Read in ops/qmatmul.rs (within GU dispatch). c9_followup2 cert exists.
