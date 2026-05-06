# Env-impact cert: `FLAMBEAU_GDN_NO_BATCHED`

- **Class:** B
- **Default state:** `batched`
- **Generated:** 2026-05-06T07:44:18+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** Force per-token GDN (disable batched-GDN inner loop)

**Preconditions:** `FLAMBEAU_INFLIGHT_SLOTS=2`

## Measured cells

### qwen36-27b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 24001 | 5855 | 19.05 | — | — |  |
| `1` | 3 | 23002 | 5828 | 19.69 | +3.3% | match |  |

### qwen36-35b-a3b-q4_0 / pp2tp2 / N=2

| value | n | prefill ms | decode ms | aggregate tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 7174 | 2308 | 47.21 | — | — |  |
| `1` | 3 | 6330 | 1833 | 57.55 | +21.9% | DIVERGENT |  |

## Triage

- qwen36-27b-q4_0/pp2tp2 [N=2] `1` → **win** (+3.3%)
- qwen36-35b-a3b-q4_0/pp2tp2 [N=2] `1` → **divergent** (+21.9%)

## Disposition

- **HALT-DIVERGENT** — 1/2 cells produce different output at greedy/fixed-seed; correctness bug, not a perf gate — file before migration

## Notes from spec

Per-slot GDN ceiling is the documented 3× hybrid-throughput blocker
(P2.9b-i2-F memory). At N=1 the GDN-batched path is a no-op; at N=2+
it should be the dominant cost on hybrid arch. If N=2 shows the
=1 path significantly slower, the gate is critically important —
do NOT delete. If still null, delete. (N=4 attempted but stage 0
OOMs on pp2tp2/27B — flagged as separate VRAM-bloat finding.)
