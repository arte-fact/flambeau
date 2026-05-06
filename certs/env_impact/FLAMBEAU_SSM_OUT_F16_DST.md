# Env-impact cert: `FLAMBEAU_SSM_OUT_F16_DST`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T20:47:11+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** F16-dest fusion on SSM output (GDN path, requires VARIANT != baseline)

**Preconditions:** `FLAMBEAU_VARIANT=fused`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11447 | 2747 | 23.30 | — | — |  |
| `on` | 3 | 11449 | 2750 | 23.27 | -0.1% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11430 | 2989 | 21.41 | — | — |  |
| `on` | 3 | 11524 | 3024 | 21.16 | -1.2% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3910 | 1431 | 44.72 | — | — |  |
| `on` | 3 | 3912 | 1448 | 44.21 | -1.1% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3786 | 1269 | 50.42 | — | — |  |
| `on` | 3 | 3795 | 1268 | 50.49 | +0.1% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `on` → **null** (-0.1%)
- qwen36-27b-q4_0/tp2 `on` → **null** (-1.2%)
- qwen36-35b-a3b-q4_0/pp2tp2 `on` → **null** (-1.1%)
- qwen36-35b-a3b-q4_0/tp2 `on` → **null** (+0.1%)

## Disposition

- CANDIDATE-DELETE — no measured impact; bake default and remove gate

## Notes from spec

Read in models/gdn_tp.rs:609. Conditional — the precondition matters.
