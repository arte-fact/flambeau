# Env-impact cert: `FLAMBEAU_BASELINE_35B_PP2TP2`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-07T12:37:51+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Smoke bench 35B-A3B/pp2tp2 N=1/4

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 3897 | 1449 | 44.18 | — | — | 25.90 |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11036 | 2162 | 114.93 | — | — | 26.13 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Smoke verification gate between cleanup slices.
