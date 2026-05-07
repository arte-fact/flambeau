# Env-impact cert: `FLAMBEAU_BASELINE_27B_PP2TP2`

- **Class:** X
- **Default state:** `n/a`
- **Generated:** 2026-05-07T10:06:19+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Smoke bench 27B/pp2tp2 N=1/4 (S2-style verification)

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=8`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 11342 | 2709 | 23.63 | — | — | 24.03 |  |

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | VRAM peak GB | err |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `unset` | 3 | 31014 | 4028 | 60.52 | — | — | 24.03 |  |

## Triage

_No comparison rows produced._

## Disposition

- INSUFFICIENT-DATA

## Notes from spec

Smoke verification gate between cleanup slices.
