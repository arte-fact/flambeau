# Env-impact cert: `FLAMBEAU_BATCH_MAX`

- **Class:** C
- **Default state:** `unbounded`
- **Generated:** 2026-05-05T22:32:10+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Max batch size cap on scheduler dispatch

**Preconditions:** `FLAMBEAU_BATCHED_DECODE=1`, `FLAMBEAU_INFLIGHT_SLOTS=8`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3885 | 1433 | 44.67 | — | — |  |
| `2` | 3 | 3876 | 1450 | 44.15 | -1.2% | match |  |
| `4` | 3 | 3874 | 1431 | 44.72 | +0.1% | match |  |
| `8` | 3 | 3886 | 1446 | 44.26 | -0.9% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 `2` → **null** (-1.2%)
- qwen36-35b-a3b-q4_0/pp2tp2 `4` → **null** (+0.1%)
- qwen36-35b-a3b-q4_0/pp2tp2 `8` → **null** (-0.9%)

## Disposition

- **CANDIDATE-DELETE** — no measured impact on any of 3 cells; bake default and remove gate

## Notes from spec

Tunable that affects per-dispatch overhead. Find the inflection point
where bigger batches stop helping. Result becomes a documented default
in SchedulerConfig::batch_max: Option<usize>.
