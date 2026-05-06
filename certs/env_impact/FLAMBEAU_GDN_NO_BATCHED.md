# Env-impact cert: `FLAMBEAU_GDN_NO_BATCHED`

- **Class:** B
- **Default state:** `batched`
- **Generated:** 2026-05-05T22:03:19+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force per-token GDN (disable batched-GDN inner loop)

**Preconditions:** `FLAMBEAU_BATCHED_DECODE=1`, `FLAMBEAU_INFLIGHT_SLOTS=2`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11427 | 2718 | 23.55 | — | — |  |
| `1` | 3 | 11420 | 2734 | 23.41 | -0.6% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3920 | 1449 | 44.17 | — | — |  |
| `1` | 3 | 3908 | 1437 | 44.54 | +0.8% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 `1` → **null** (-0.6%)
- qwen36-35b-a3b-q4_0/pp2tp2 `1` → **null** (+0.8%)

## Disposition

- **CANDIDATE-DELETE** — no measured impact on any of 2 cells; bake default and remove gate

## Notes from spec

Per-slot GDN ceiling is the documented 3× hybrid-throughput blocker
(P2.9b-i2-F memory). This run measures whether disabling batched-GDN
matters at N=1 single-slot decode (likely NULL — gate ought to die).
