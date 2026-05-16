# Env-impact cert: `FLAMBEAU_BASELINE_27B_PP2TP2`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-06T22:37:12+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Baseline characterization of Qwen3.6-27B on pp2tp2 across N=[1,4,8]

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11331 | 2724 | 23.50 | — | — | 24.03 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 30853 | 4006 | 62.57 | — | — | 24.03 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=8

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 57055 | 4097 | 121.67 | — | — | 24.33 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Quick confirmation bench at N=1/4/8 on 27B/pp2tp2 PRE-COLLAPSE (commit 27cdabc, fast-path active).
