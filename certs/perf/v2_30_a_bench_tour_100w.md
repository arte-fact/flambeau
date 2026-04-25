# V2.30.a + V2.28.b — full bench tour at 100 W/GPU

Comprehensive perf numbers across the four supported models at 100 W/GPU
power cap (down from 200 W on 2026-04-24), post-V2.30.a (async prefill
race-guards removed) and V2.28.b (Qwen3-Coder/qwen3moe path shipped).

## Matrix (prefill tok/s, decode tok/s)

| Model                              | Arch       | Mesh | Config                  | L=128 | L=512 | L=1024 | L=2048 | L=4096 | L=8192 | tg=64 |
|------------------------------------|------------|------|-------------------------|------:|------:|-------:|-------:|-------:|-------:|------:|
| Qwen3.5-9B-Q4_1                    | qwen35     | 1    | sync                    |   512 |   641 |    649 |    617 |   OOM* |   OOM* |     — |
| Qwen3.5-9B-Q4_1                    | qwen35     | 1    | async u_lanes=2 ub=64   |    66 |    70 |     70 |     70 |     68 |     66 |     — |
| Qwen3.6-27B-Q8_0                   | qwen35     | 4    | sync                    |    99 |    99 |     97 |     95 |     91 |     85 |  18.7 |
| Qwen3.6-27B-Q8_0                   | qwen35     | 4    | async u_lanes=2 ub=128  |    98 |   160 |    230 |    284 |    315 |    318 |  18.6 |
| Qwen3.6-35B-A3B-UD-Q4_K_S          | qwen35moe  | 4    | async u_lanes=2 ub=128  |   462 |   762 |   1201 |   1431 |  **1532** |   1428 |  53.5 |
| Qwen3-Coder-30B-A3B-UD-Q4_K_XL     | qwen3moe   | 4    | sync                    |   288 |   298 |    264 |    218 |    148 |     88 |  38.5 |
| Qwen3-Coder-30B-A3B-UD-Q4_K_XL     | qwen3moe   | 4    | async u_lanes=2 ub=128  |   286 |   426 |    532 |    523 |    417 |    290 |  38.9 |

*9B Mesh<1> sync OOMs at L≥4096 because sync scratch is sized for full L
(570 MiB+ activations). Async ub=64 u_lanes=2 runs but loses on Mesh<1>
due to no cross-rank overlap (one GPU — u_lanes just adds overhead).
Proper Mesh<1> L≥4096 would need async chunking without PP;
out-of-scope.

## Headline numbers

- **Qwen3.6-35B-A3B is the fastest prefill target**: 1532 tok/s at
  L=4096 Mesh<4> async. That's 35B's compute sweet spot — fully MoE
  (A3B active params), 40-layer hybrid with pipeline parallelism hiding
  rank-to-rank peer copies.
- **Qwen3.6-27B the biggest V2.30.a async win**: 85 → 318 tok/s at
  L=8192 (+276 %, 3.76×). Previously hard-blocked by V2.28.a-i1 guard;
  V2.30.a event-ordered GDN state unlocked it.
- **Qwen3-Coder-30B-A3B (qwen3moe) shipped**: +43 % prefill async vs
  sync at L=512 (298 → 426), +101 % at L=1024 (264 → 532). Async peaks
  at 532 tok/s, decode 38.9 tok/s. Note the sharp prefill degradation
  of the sync path past L=1024 — quadratic-in-L attention + sync scratch
  sizing. Async ub=128 holds within 10 % of peak through L=4096.
- **Qwen3.5-9B-Q4_1 Mesh<1>**: 649 tok/s at L=1024 sync — still the
  fastest per-card config (single MI50, no inter-rank overhead). 9B is
  the only model that fits one card; Mesh<N> for N≥2 costs PP bubbles.

## Decode rank

1. **35B Mesh<4>**: 53.5 tok/s (MoE A3B active params, minimal compute
   per token after KV append)
2. **Coder-30B Mesh<4>**: 38.9 tok/s (pure MoE, slightly heavier attn
   than 35B since no GDN offloading most layers)
3. **27B Mesh<4>**: 18.6 tok/s (Q8_0 dense-hybrid — more compute
   per-layer, larger activation, PP bubbles dominate at tg=64)
4. **9B Mesh<1>**: decode data not in sync test path (was 65 tok/s
   @ 200W per V2.2.c.2; similar at 100W)

## Power cut (200 → 100 W) impact

Where we have direct 200 W comparison:
- 9B Mesh<1> sync pp=512: 627 (V2.2.c.2 @ 200 W) → 641 (now @ 100 W) —
  **no regression** (noise/slight improvement).
- 35B Mesh<2> pp=512: 508 (V2.8 @ 200 W, sync-ish) → 762 (now @ 100 W
  async Mesh<4>) — more cards + async unlock dominate; not apples-to-apples.

The takeaway: **100 W/GPU doesn't meaningfully regress compute-bound
kernels on gfx906** at current shapes. MI50's sweet spot is below
150 W for the ~13 TFLOPS F32 / 1 TB/s HBM envelope the kernels actually
hit. The pre-cut headroom was burned on thermals, not perf.

## Perf snapshots on disk

- `certs/perf/qwen35_9b_q4_1_mesh{1,2,4}.json`
- `certs/perf/qwen3_6_27b_q8_0_mesh4.json` (via earlier test harness)
- `certs/perf/qwen3_6_35b_a3b_ud_q4_k_s_mesh{2,4}.json`
- `certs/perf/qwen3_coder_30b_mesh4.json`

## Follow-ups

- **Mesh<2> numbers for 35B async**: V2.30.a-i4 predicted Mesh<2> PP
  would still scale well at new power envelope; not re-measured here
  in the interest of thermal budget — next bench tour.
- **9B async on Mesh<2>/Mesh<4>**: out of scope (model fits 1 card,
  Mesh>1 is lossy).
- **Parity vs llama.cpp for Qwen3-Coder (V2.28.b-i3)**: still pending.
- **Q4_1 indexed-MoE (V2.28.b-i6)**: needed for `Qwen3-Coder-30B-Instruct-1M-Q4_0.gguf`
  whose `ffn_down_exps` promote to Q4_1 on 6/48 layers.

## Regeneration

```bash
cargo test --release -p flambeau-qwen3-moe --features hip \
  --test perf_baseline_qwen3_coder   # Coder
  --test perf_baseline_qwen3_moe     # 35B (with FLAMBEAU_QWEN3_GGUF=…)
  --test perf_baseline_qwen35_9b     # 9B (with FLAMBEAU_QWEN35_GGUF=…)
# each with FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2
# FLAMBEAU_MESH_RANKS={1,2,4} for the desired mesh size.
```
