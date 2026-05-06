# Env-impact cert: `FLAMBEAU_GDN_NO_BATCHED`

- **Class:** B
- **Default state:** `batched`
- **Generated:** 2026-05-06T07:46:15+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force per-token GDN (disable batched-GDN inner loop)

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=4`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 40298 | 5478 | 39.93 | — | — |  |
| `1` | 3 | 31270 | 4103 | 59.41 | +48.8% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=4

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 12354 | 2369 | 102.97 | — | — |  |
| `1` | 3 | 11086 | 2190 | 112.61 | +9.4% | match |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=4] `1` → **win** (+48.8%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=4] `1` → **win** (+9.4%)

## Disposition

- **CANDIDATE-DEFAULT** — non-default wins on all 2 cells; flip the default and delete the gate

## Notes from spec

Per-slot GDN ceiling is the documented 3× hybrid-throughput blocker
(P2.9b-i2-F memory). At N=1 the GDN-batched path is a no-op; at N=4
it should be the dominant cost on hybrid arch. If N=4 shows the
=1 path significantly slower, the gate is critically important —
do NOT delete. If still null, delete.
