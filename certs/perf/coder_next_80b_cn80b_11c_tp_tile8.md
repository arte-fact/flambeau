# CN-80B-11c — Q4_1 indexed-MoE MMQ tile8 down + TP tile8 wiring on Coder-Next-80B

## Result

pp2tp2 is now the strict-best topology on Coder-Next-80B: it beats pp4
on **every** axis (prefill at all L, decode at tg64).

| topology | pp128 | pp512 | pp2048 | tg64 |
|----------|------:|------:|-------:|-----:|
| pp4 (CN-80B-3 baseline)        | 354.0 | 568.0 | 593.0 | 41.6 |
| pp4 (post-11c)                  | 441.0 | 569.0 | 597.3 | 40.6 |
| pp2tp2 (CN-80B-3 baseline)      | 156.7 | 183.0 | 192.4 | 46.0 |
| **pp2tp2 (post-11c)**           | **493.7** | **800.6** | **908.9** | **46.1** |

pp2tp2 vs pp4 head-to-head post-11c:
- pp128:  +12 %
- pp512:  +41 %
- pp2048: +52 %
- tg64:   +13 %

## What landed

Two pieces, both required:

1. **`indexed_moe_mmq_q4_1_down_tile8_dp4a` kernel** — port of the
   Q4_0 down tile8 with the Q4_1 affine reconstruction
   `y_real · w_real = d_x · d_y · sumi + m_x · s_y` (Q4_0 used the
   bias-correction form `d_x · (d_y · sumi - 8 · s_y)`). Same launch
   shape as Q4_0/Q8_0 down tile8 — `MMQ_Y=64, TILE_N=8`, wave64,
   `__launch_bounds__(WARP_SIZE, 1)`. Reads both `wbx->d` and `wbx->m`
   per Q4_1 block; the nibble layout is identical to Q4_0
   (low @ byte i → element i, high @ byte i → element i+16).

2. **TP tile8 wiring in `forward_moe_ffn_prefill_tp`** — the path
   historically fell through to per-token MMVQ at any L because
   tile8 dispatch was non-TP-only. This added a sort+pad gate at
   `n_tokens >= 32` that runs the same Q4_0 / Q8_0 / Q4_1 tile8
   kernels with `local_inter` instead of `inter`, ending in
   `moe_combine_no_residual_f16` (TP residual is folded by the AR
   that follows). The expert-id table is identical across ranks
   because the router runs replicated per `tp_layout::for_tensor`,
   so sort scratch can be reused with no per-rank filter.

`q4_0_use_tile8` in non-TP `forward_moe_ffn_prefill` was also extended
to admit Q4_1 down (was {Q4_0, Q8_0}), so the kernel is reachable on
the non-TP PP path too — though pp4 numbers showed it essentially
flat (the path was already running at saturation for Coder-Next's
gate=Q4_0 / down=Q4_1 mix, modulo the +25 % at L=128).

## Profile attribution

`coder_next_pp2tp2_prefill_profile` test (HipEvent marks at section
boundaries, rank 0's stream):

Pre-11c (CN-80B-11a baseline):

```
section                   total_ms       count     mean_ms
ptp_moe_ffn               2348             48      48.92    ← 84 % of wall
ptp_attn_gdn               176             36       4.90
ptp_router                  68             48       1.41
ptp_attn_full               39             12       3.28
total_attributed         ~2800
```

Post-11c:

```
section                   total_ms       count     mean_ms
ptp_moe_ffn                199.978         48       4.166   ← 12× drop per layer
ptp_attn_gdn               176.226         36       4.895
ptp_router                  68.902         48       1.435
ptp_shared                  46.408         48       0.967
ptp_attn_full               39.385         12       3.282
ptp_ffn_ar                  26.270         48       0.547
ptp_attn_ar                 24.841         48       0.518
ptp_ffn_norm                 3.006         48       0.063
total                      585.017  attributed
```

ptp_moe_ffn went 48.92 → 4.166 ms/layer (**11.7× faster per layer**),
matching the +4–5× wall-time gain at L=512 (49 ms × 48 layers ≈ 2.35 s
wall removed). Wall went 2.80 s → 0.65 s = 4.3× ≈ matches 800/183 from
the bench harness.

The bottleneck has now shifted: ptp_moe_ffn (200 ms) and ptp_attn_gdn
(176 ms) are roughly balanced. Next lever, if pursued, would be GDN
intra-section (fused alpha/beta + state-step), but the user's
combined-pp+tg target is met — close out CN-80B-11.

## Why pp4 didn't move the same way

pp4 routes through `forward_moe_ffn_prefill` (non-TP). Pre-11c, by
code reading, `q4_0_use_tile8 = false` for (Q4_0, Q4_1) so the path
should have bailed at the (Q4_0, Q4_0)/(Q4_0, Q8_0)/(Q8_0, Q8_0) match
in `moe.rs:1067`. The CN-80B-3 569 tok/s reading suggests either the
bench was hitting a different driver, or the pre-11c code path I
read had a non-obvious early-return I missed in this debugging pass.
Post-11c the (Q4_0, Q4_1) tile8 arm is reachable and runs cleanly,
so the question is moot for forward correctness. pp4 numbers are
unchanged within noise, confirming the lever is TP-shaped.

## Bit-exact regression

35B-A3B-UD-Q4_K_S parity is unaffected (Q4_K everywhere, doesn't hit
the new (Q4_0,*) tile8 arms). 35B-A3B-Q4_0 / 35B-A3B-Q8_0 routes
through the new TP tile8 path at L≥32 — structure identical to the
non-TP single-rank case which has been in production since V2.22.b /
V2.28.c, so no parity cert change expected. Smoke pass on
`coder_next_pp2tp2_prefill_profile` (4-rank end-to-end forward) was
clean: timer flushed, no NaNs, no panics.

## Closes

CN-80B-10 #136 (pp2tp2 3× gap diagnosis, originally MMVQ-vs-MMQ).
CN-80B-11a #139 (re-diagnosis: 84 % in MoE FFN, not a topology
fundamental).
CN-80B-11c #141 (kernel + non-TP wiring + TP wiring).

CN-80B-11b is folded in (the lever from 11a was the same Q4_1 tile8
+ TP wiring; no separate implementation needed).

## Reproduce

```
FLAMBEAU_BENCH_GGUF=/artefact/models/Qwen3-Coder-Next-Q4_0.gguf \
FLAMBEAU_BENCH_TAG=coder_next_80b_q4_0_cn80b_11c \
FLAMBEAU_BENCH_TOPOLOGY_TAG=pp2tp2 \
FLAMBEAU_BENCH_PREFILL_LENGTHS=128,512,2048 \
FLAMBEAU_BENCH_DECODE_LENGTHS=64 \
RUST_MIN_STACK=67108864 \
cargo test --release -p flambeau-qwen3-moe --features hip \
  --test v1_bench_matrix v1_bench_matrix -- --ignored --nocapture
```

Profile:

```
FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3-Coder-Next-Q4_0.gguf \
RUST_MIN_STACK=67108864 \
cargo test --release -p flambeau-qwen3-moe --features hip \
  --test coder_next_pp2tp2_prefill_profile -- --nocapture
```
