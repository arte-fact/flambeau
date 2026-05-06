# Env-impact cert: `FLAMBEAU_QKV_FUSED`

- **Class:** B
- **Default state:** `fused`
- **Generated:** 2026-05-05T20:00:12+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Fused QKV projection in full-attn layers (default fused)

## Measured cells

### qwen35-9b-q4_1 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 4236 | 1274 | 50.24 | — | — |  |
| `0` | 3 | 4245 | 1248 | 51.29 | +2.1% | match |  |

### qwen35-9b-q4_1 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 5951 | 1456 | 43.95 | — | — |  |
| `0` | 3 | 6482 | 1510 | 42.38 | -3.6% | match |  |

### qwen35-9b-q4_1 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 4042 | 1194 | 53.61 | — | — |  |
| `0` | 3 | 4071 | 1200 | 53.31 | -0.5% | match |  |

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11422 | 2720 | 23.53 | — | — |  |
| `0` | 3 | 11423 | 2742 | 23.34 | -0.8% | match |  |

### qwen36-27b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 18991 | 3812 | 16.79 | — | — |  |
| `0` | 3 | 19986 | 3843 | 16.65 | -0.8% | match |  |

### qwen36-27b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11340 | 2967 | 21.57 | — | — |  |
| `0` | 3 | 11445 | 3018 | 21.20 | -1.7% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3910 | 1444 | 44.33 | — | — |  |
| `0` | 3 | 3904 | 1438 | 44.52 | +0.4% | match |  |

### qwen36-35b-a3b-q4_0 / pp4

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 5380 | 1574 | 40.66 | — | — |  |
| `0` | 3 | 6008 | 1564 | 40.93 | +0.7% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3762 | 1260 | 50.79 | — | — |  |
| `0` | 3 | 3765 | 1272 | 50.32 | -0.9% | match |  |

## Triage

- qwen35-9b-q4_1/pp2tp2 `0` → **win** (+2.1%)
- qwen35-9b-q4_1/pp4 `0` → **loss** (-3.6%)
- qwen35-9b-q4_1/tp2 `0` → **null** (-0.5%)
- qwen36-27b-q4_0/pp2tp2 `0` → **null** (-0.8%)
- qwen36-27b-q4_0/pp4 `0` → **null** (-0.8%)
- qwen36-27b-q4_0/tp2 `0` → **null** (-1.7%)
- qwen36-35b-a3b-q4_0/pp2tp2 `0` → **null** (+0.4%)
- qwen36-35b-a3b-q4_0/pp4 `0` → **null** (+0.7%)
- qwen36-35b-a3b-q4_0/tp2 `0` → **null** (-0.9%)

## Disposition

- **SHAPE-DEPENDENT** — wins on 1, loses on 1, null on 7 (of 9);
  migrate to dispatch table per (model, topology). Specifically:
  fused wins on qwen35-9b-q4_1 / pp2tp2 (+2.1%), loses on
  qwen35-9b-q4_1 / pp4 (-3.6%), null elsewhere. The `=0` opt-out
  isn't a global dead path; it pays off on small-model PP4.

_(Disposition aggregate added post-hoc on 2026-05-05; original cert
listed three contradictory candidates from the pre-patch buggy logic.)_

## Notes from spec

Read in models/gdn.rs:1175. Last cert is TP-4d (2026-04-29). If fused
wins ≥ noise_floor on every cell, default-bake and delete the var.
