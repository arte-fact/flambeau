# Env-impact cert: `FLAMBEAU_NO_FAST_PATH`

- **Class:** D
- **Default state:** `fast-path-on`
- **Generated:** 2026-05-05T19:29:12+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Disable scheduler fast path (force full coalescing logic)

**Preconditions:** `FLAMBEAU_BATCHED_DECODE=1`, `FLAMBEAU_INFLIGHT_SLOTS=2`

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3881 | 1418 | 45.14 | — | — |  |
| `1` | 3 | 3881 | 1448 | 44.21 | -2.0% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 `1` → **loss** (-2.0%)

## Disposition

- KEEP-DEFAULT — current default wins, alternate is dead path

## Notes from spec

If fast path is materially faster, delete the gate (the slow path is
dead). If indistinguishable, the gate is purely debug — migrate to
RUST_LOG=flambeau::scheduler::fast_path=trace and delete the env.
