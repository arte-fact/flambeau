# Env-impact cert: `FLAMBEAU_KV_F16_DST`

- **Class:** B
- **Default state:** `on`
- **Generated:** 2026-05-05T20:28:43+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Fused Q4_0 KV-quantize with F16 destination

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11435 | 2743 | 23.33 | — | — |  |
| `off` | 3 | 11436 | 2787 | 22.96 | -1.6% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11396 | 2995 | 21.37 | — | — |  |
| `off` | 3 | 11523 | 3082 | 20.76 | -2.8% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3902 | 1431 | 44.72 | — | — |  |
| `off` | 3 | 3916 | 1439 | 44.49 | -0.5% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3783 | 1272 | 50.33 | — | — |  |
| `off` | 3 | 3778 | 1273 | 50.26 | -0.1% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `off` → **null** (-1.6%)
- qwen36-27b-q4_0/tp2 `off` → **loss** (-2.8%)
- qwen36-35b-a3b-q4_0/pp2tp2 `off` → **null** (-0.5%)
- qwen36-35b-a3b-q4_0/tp2 `off` → **null** (-0.1%)

## Disposition

- **CONTEXT-DEPENDENT** — 1/4 cells loss (qwen36-27b-q4_0/tp2: -2.8%
  when fusion off), rest null. Q4_0 KV-fuse helps materially on
  27b/tp2 but is null on 27b/pp2tp2 and on 35b. Default-on stays;
  the fusion is the right default. Either keep the gate as
  TP2-only-tunable, or migrate to dispatch table for the affected
  cell. The `off` value is not a global dead path.

_(Disposition aggregate added post-hoc on 2026-05-05.)_

## Notes from spec

This run measures Q4_0 only — the Q4_1 cell is already certed null.
