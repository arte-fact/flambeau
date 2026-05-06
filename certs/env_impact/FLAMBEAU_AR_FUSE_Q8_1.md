# Env-impact cert: `FLAMBEAU_AR_FUSE_Q8_1`

- **Class:** B
- **Default state:** `off`
- **Generated:** 2026-05-05T20:37:56+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** AllReduce + fused Q8_1 KV quantize on TP boundary

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11430 | 2739 | 23.37 | — | — |  |
| `on` | 3 | 11410 | 2772 | 23.09 | -1.2% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11440 | 3005 | 21.30 | — | — |  |
| `on` | 3 | 11539 | 3038 | 21.07 | -1.1% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3925 | 1439 | 44.47 | — | — |  |
| `on` | 3 | 3915 | 1453 | 44.05 | -0.9% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3789 | 1272 | 50.30 | — | — |  |
| `on` | 3 | 3793 | 1270 | 50.41 | +0.2% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `on` → **null** (-1.2%)
- qwen36-27b-q4_0/tp2 `on` → **null** (-1.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 `on` → **null** (-0.9%)
- qwen36-35b-a3b-q4_0/tp2 `on` → **null** (+0.2%)

## Disposition

- CANDIDATE-DELETE — no measured impact; bake default and remove gate

## Notes from spec

Read in models/tp.rs:1840,2610. If on >> off, promote to default.
If shape-dependent across (model, topology), candidate for dispatch-table
row. If null, delete.
