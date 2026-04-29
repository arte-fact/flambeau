# Qwen3.6-35B-A3B-Q4_0 — TP2 kernel-trace profile — 2026-04-29

`rocprofv3 --kernel-trace --stats` over `profile_point_tp` (warmup +
prefill L=512 + decode tg=64) on TP=2 (devices 0,1), 100 W cap, ROCm 7.1.1.

Wall: prefill **921 tok/s** (555 ms for L=512), decode **26.2 tok/s** (synthetic
short context — chat workload at L≈700 measured 36 tok/s in
`qwen36_family_2026_04_29.md`).

## Top kernels by total GPU time (sum of both ranks)

| rank | kernel | n_calls | total_ms | avg_us | pct |
|---:|---|---:|---:|---:|---:|
| 1 | `flambeau_gdn_state_step_alphabeta_f32_s128`             | 4020  | 297.98 |  74.12 | **11.70%** |
| 2 | `flambeau_indexed_moe_mmvq_q4_0_q8_1`                    | 4620  | 261.26 |  56.55 | **10.26%** |
| 3 | `flambeau_mmvq_q4_0_gate_up_dp4a_q8_1`                   | 3960  | 126.69 |  31.99 |   4.98% |
| 4 | `flambeau_indexed_moe_mmq_q4_0_gate_up_tile8_dp4a_q8_1`  |   80  | 112.87 |1410.85 |   4.43% |
| 5 | `flambeau_moe_sort_scatter_det`                          |   80  |  95.41 |1192.66 |   3.75% |
| 6 | `flambeau_mmq_q4_0_4warp_lds_q8_1`                       |  182  |  94.31 | 518.19 |   3.70% |
| 7 | `flambeau_topk_softmax_f32`                              | 5360  |  89.83 |  16.76 |   3.53% |
| 8 | `flambeau_p2p_allreduce_residual_tp2`                    | 5440  |  88.65 |  16.30 |   3.48% |
| 9 | `flambeau_mmq_q8_0_wave64_tile16_q8_1`                   |  308  |  85.83 | 278.67 |   3.37% |
| 10 | `flambeau_cast_f32_f16`                                 |20100  |  80.44 |   4.00 |   3.16% |
| 11 | `flambeau_indexed_moe_mmvq_q4_0_gate_up_dp4a_q8_1`      | 5280  |  79.91 |  15.13 |   3.14% |
| 12 | `flambeau_quantize_row_f16_q8_1`                        |12300  |  74.22 |   6.03 |   2.91% |
| 13 | `flambeau_rmsnorm_q8_1_fused`                           | 5347  |  71.71 |  13.41 |   2.82% |
| 14 | `flambeau_attention_decode_f16`                         | 1320  |  64.34 |  48.75 |   2.53% |
| 15 | `flambeau_p2p_allreduce_residual_rmsnorm_tp2`           | 5280  |  60.55 |  11.47 |   2.38% |
| 16 | `flambeau_mmvq_q6_k_dp4a_q8_1`                          |  727  |  59.84 |  82.32 |   2.35% |
| 17 | `flambeau_swiglu_f32_to_q8_1`                           |14520  |  56.99 |   3.92 |   2.24% |
| 18 | `flambeau_mmvq_q8_0_t128_vdr2_q8_1`                     | 7128  |  54.17 |   7.60 |   2.13% |
| 19 | `flambeau_mmvq_q5_0_q8_1`                               | 7920  |  48.87 |   6.17 |   1.92% |
| 20 | `flambeau_dense_gemv_f16_f16_batched`                   |   80  |  47.19 | 589.85 |   1.85% |

Total kernel time across both ranks: **2546 ms**.
Memcopy (DtoD/HtoD/DtoH) traffic: **0** events — TP-only path is fully
GPU-side, AR via BAR1 P2P. (pp+tp would show host-bounce DtoH/HtoD here.)

## Lever map

### Bandwidth-bound (decode hot path)
- **GDN state-step (11.7%)** is the largest single bucket. Already the fused
  C10 kernel (alphabeta absorbed). 4020 calls = 30 GDN layers × 134 forwards
  (1 prefill + 64 decode + warmup). Avg 74 µs is dominated by the per-rank
  state read+write for `local_num_v_heads = 16, head_k=128, head_v=128`.
  Lever: **fuse swiglu + state read into the kernel** to cut one HBM round-trip;
  estimate +2-4% overall on this rig.
- **Indexed MoE MMVQ Q4_0 ffn_down_exps (10.3%)** is the second-largest. 4620
  calls = at least 2 (gate/up shared sort) × 30+ ranks per macro × 65 forwards
  + prefill. Already the multi-row DPP-reduce kernel. Lever: **r4 multi-row
  variant** is a confirmed dead lever on gfx906 per memory note
  `v2_31_b_e_multirow_dead_lever.md` — don't re-propose. Lever: **MoE expert
  fusion** (combine gate+up+down for the active expert at each token via a
  block-fused kernel) — would need a redesign; ~5-8% if it lands.
- **Fused gate+up Q4_0 dp4a (5.0%)** is GDN's `attn_qkv + attn_gate`
  projection. With `kq_replicated=true` (TP-4d-i3) each rank computes the FULL
  output. Lever: this is now the same wall as TP=1 since K/Q are replicated;
  TP=2 doesn't help this kernel. Acceptable cost of the rep_outer fix.

### Compute-bound (prefill-only)
- **Indexed MoE MMQ tile8 + MoE sort_scatter (combined 8.2%, 80 calls each)**
  is the prefill MoE kernel chain. Already includes the V2.31.a tile8 win.
  Lever: **bigger expert-tile (16 or 32)** is at-floor on VGPR per
  `v2_31_d_tile32_null.md` — don't propose. Lever: **stream-K** for the MoE
  variant matching the dense Q4_0 4warp_lds kernel — possible +5% prefill.
- **Q4_0 4warp_lds attn MMQ (3.7%)** is full-attn `attn_q` (fused [Q|gate])
  prefill. Already at the gold-standard 4-warp LDS-tiled path.

### Latency-bound (small kernels, high count)
- **cast_f32_f16 (3.2%, 20100 calls @ 4 µs)** is everywhere — every F32 op
  result cast back to F16 for residual stream. Lever: **dual-output kernels**
  that produce both F32 and F16 in one launch where the next op needs F16.
  Cumulative ~1-2% achievable; high audit cost.
- **AR primitives (3.5% + 2.4% = 5.9%)** are nearly at the ceiling for BAR1
  P2P over 2-rank residual traffic. Already fused with rmsnorm. Lever: only
  cross-rank topology changes (not applicable on this rig).

## What this profile isn't

- **No PMC counters** (`--pmc` would trip the multi-rank SIGABRT under ROCm
  7.1.1 per the V2.30 memory note). VGPR pressure / MemBusy / VALUBusy must
  be sampled per-kernel via single-rank harness.
- **TG=64 over short synthetic prompt** — KV walks are short, so the
  attention-decode kernel (#14, 2.5%) is under-represented vs realistic chat.
  At 4K position the attention kernel's share rises to ~6-8% (extrapolating
  linearly).

## Reproducer

```sh
cd /tmp/qwen36_profile_tp2_35BA3B
PATH=/opt/rocm-host/bin:$PATH \
FLAMBEAU_PROFILE_GGUF=/artefact/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
FLAMBEAU_PROFILE_TP_DEVICES=0,1 \
FLAMBEAU_PROFILE_L=512 \
FLAMBEAU_PROFILE_TG=64 \
FLAMBEAU_TP_BATCHED=1 \
rocprofv3 --kernel-trace --stats -d kt_out -- \
  /artefact/flambeau/target/release/deps/profile_point_tp-161f44180c952115 \
  --nocapture profile_point_tp
```

Per-kernel summary extracted from the `rocpd_kernel_dispatch` table of the
SQLite results db.
