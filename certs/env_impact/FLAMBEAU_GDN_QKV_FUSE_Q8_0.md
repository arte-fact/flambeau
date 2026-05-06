# Env-impact cert: `FLAMBEAU_GDN_QKV_FUSE_Q8_0`

- **Class:** B
- **Default state:** `fused`
- **Generated:** 2026-05-05T20:19:25+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Q8_0 QKV fusion in TP-GDN path (hybrid arch only)

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11446 | 2751 | 23.26 | — | — |  |
| `off` | 3 | 11425 | 2742 | 23.34 | +0.3% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11547 | 3024 | 21.17 | — | — |  |
| `off` | 3 | 11568 | 3043 | 21.03 | -0.6% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3917 | 1439 | 44.47 | — | — |  |
| `off` | 3 | 3915 | 1447 | 44.23 | -0.5% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3784 | 1268 | 50.46 | — | — |  |
| `off` | 3 | 3789 | 1271 | 50.37 | -0.2% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `off` → **null** (+0.3%)
- qwen36-27b-q4_0/tp2 `off` → **null** (-0.6%)
- qwen36-35b-a3b-q4_0/pp2tp2 `off` → **null** (-0.5%)
- qwen36-35b-a3b-q4_0/tp2 `off` → **null** (-0.2%)

## Disposition

- CANDIDATE-DELETE — no measured impact; bake default and remove gate

## Notes from spec

Read in models/gdn_tp.rs:229. Standalone delta unknown.
