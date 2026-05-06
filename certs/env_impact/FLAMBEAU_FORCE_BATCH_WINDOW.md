# Env-impact cert: `FLAMBEAU_FORCE_BATCH_WINDOW`

- **Class:** C
- **Default state:** `off`
- **Generated:** 2026-05-05T22:28:51+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force batch-window timeout even when batch is empty

**Preconditions:** `FLAMBEAU_BATCHED_DECODE=1`, `FLAMBEAU_INFLIGHT_SLOTS=4`, `FLAMBEAU_BATCH_WINDOW_US=1500`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3901 | 1427 | 44.84 | — | — |  |
| `1` | 3 | 3895 | 1448 | 44.19 | -1.4% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 `1` → **null** (-1.4%)

## Disposition

- **CANDIDATE-DELETE** — no measured impact on any of 1 cells; bake default and remove gate

## Notes from spec

Sensitivity test for the scheduler edge case. Measure under N=4 mixed
load. If on/off both within noise, default to off and absorb into
SchedulerConfig::force_window: bool (default false).
