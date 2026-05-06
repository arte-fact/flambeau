# Env-impact cert: `FLAMBEAU_FORCE_BATCH_WINDOW`

- **Class:** C
- **Default state:** `off`
- **Generated:** 2026-05-06T07:58:22+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force batch-window timeout even when batch is empty

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=4`, `FLAMBEAU_BATCH_WINDOW_US=1500`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 11370 | 2215 | 112.55 | — | — |  |
| `1` | 2 | 9930 | 2213 | 99.01 | -12.0% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **loss** (-12.0%)

## Disposition

- **KEEP-DEFAULT** — current default wins on all 1 cells; alternate is a dead path, migrate to tracing or delete

## Notes from spec

Scheduler edge case: at N=1 the empty-batch timeout never fires. At N=4
the window gates how aggressively the scheduler coalesces — if forcing
the timeout helps tail latency or hurts throughput, the gate is real.
