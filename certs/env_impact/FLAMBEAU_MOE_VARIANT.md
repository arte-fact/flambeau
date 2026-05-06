# Env-impact cert: `FLAMBEAU_MOE_VARIANT`

- **Class:** B
- **Default state:** `tile8`
- **Generated:** 2026-05-05T22:26:49+00:00
- **Spec runs/cell:** 3 (+ 2 warmup)

**Description.** MoE prefill kernel selector — picks completely different code paths

## Measured cells

### qwen36-35b-a3b-q4_0 / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3900 | 1424 | 44.93 | — | — |  |
| `tile8` | 3 | 3897 | 1448 | 44.20 | -1.6% | match |  |
| `sorted` | 3 | 3902 | 1432 | 44.68 | -0.6% | match |  |
| `r4` | 3 | 3919 | 1443 | 44.34 | -1.3% | match |  |
| `turbo` | 3 | 3916 | 1432 | 44.68 | -0.6% | match |  |

### qwen36-35b-a3b-q4_0 / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 3766 | 1266 | 50.57 | — | — |  |
| `tile8` | 3 | 3768 | 1261 | 50.76 | +0.4% | match |  |
| `sorted` | 3 | 3780 | 1273 | 50.26 | -0.6% | match |  |
| `r4` | 3 | 3780 | 1271 | 50.35 | -0.4% | match |  |
| `turbo` | 3 | 3783 | 1275 | 50.19 | -0.8% | match |  |

### qwen36-35b-a3b-q4ks / pp2tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 7970 | 1371 | 46.69 | — | — |  |
| `tile8` | 3 | 7951 | 1361 | 47.01 | +0.7% | match |  |
| `sorted` | 3 | 7920 | 1362 | 46.99 | +0.6% | match |  |
| `r4` | 3 | 7934 | 1376 | 46.51 | -0.4% | match |  |
| `turbo` | 3 | 7932 | 1359 | 47.09 | +0.9% | match |  |

### qwen36-35b-a3b-q4ks / tp2

| value | n | prefill ms | decode ms | tg64 tok/s | Δ vs default | correct? | err |
|---|---:|---:|---:|---:|---:|---|---|
| `unset` | 3 | 7911 | 1170 | 54.72 | — | — |  |
| `tile8` | 3 | 7975 | 1197 | 53.46 | -2.3% | match |  |
| `sorted` | 3 | 8041 | 1181 | 54.21 | -0.9% | match |  |
| `r4` | 3 | 8044 | 1189 | 53.82 | -1.7% | match |  |
| `turbo` | 3 | 8094 | 1196 | 53.53 | -2.2% | match |  |

## Triage

- qwen36-35b-a3b-q4_0/pp2tp2 `tile8` → **null** (-1.6%)
- qwen36-35b-a3b-q4_0/pp2tp2 `sorted` → **null** (-0.6%)
- qwen36-35b-a3b-q4_0/pp2tp2 `r4` → **null** (-1.3%)
- qwen36-35b-a3b-q4_0/pp2tp2 `turbo` → **null** (-0.6%)
- qwen36-35b-a3b-q4_0/tp2 `tile8` → **null** (+0.4%)
- qwen36-35b-a3b-q4_0/tp2 `sorted` → **null** (-0.6%)
- qwen36-35b-a3b-q4_0/tp2 `r4` → **null** (-0.4%)
- qwen36-35b-a3b-q4_0/tp2 `turbo` → **null** (-0.8%)
- qwen36-35b-a3b-q4ks/pp2tp2 `tile8` → **null** (+0.7%)
- qwen36-35b-a3b-q4ks/pp2tp2 `sorted` → **null** (+0.6%)
- qwen36-35b-a3b-q4ks/pp2tp2 `r4` → **null** (-0.4%)
- qwen36-35b-a3b-q4ks/pp2tp2 `turbo` → **null** (+0.9%)
- qwen36-35b-a3b-q4ks/tp2 `tile8` → **loss** (-2.3%)
- qwen36-35b-a3b-q4ks/tp2 `sorted` → **null** (-0.9%)
- qwen36-35b-a3b-q4ks/tp2 `r4` → **null** (-1.7%)
- qwen36-35b-a3b-q4ks/tp2 `turbo` → **loss** (-2.2%)

## Disposition

- **CONTEXT-DEPENDENT** — 2/16 cells loss, rest null; consider dispatch-table row for the affected cells

## Notes from spec

Read in moe.rs:984. 4-way selector: tile8 (default, MMQ 4-warp LDS-tiled)
| sorted (MMVQ r2 sorted-block) | r4 (raw MMVQ) | turbo (DS4 Q8_1 + MMQ
turbo). Each is a distinct kernel pipeline with different VRAM (Q8_1 vs
Q8_1_MMQ activation packing). Migration target: dispatch table row per
(dtype, n_tokens) shape — likely tile8 for prefill, sorted for some
decode paths. Q4_K only supports tile8 + sorted + turbo for some sub-ops;
Q4_0/Q5_0/Q5_1/Q8_0 only support tile8 in current code.
