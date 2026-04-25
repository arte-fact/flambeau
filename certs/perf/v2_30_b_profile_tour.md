# V2.30.b — full-kernel profile tour @ 100 W/GPU — pp/tg leverage map

`rocprofv3 --kernel-trace --stats` on each supported model at prefill
L=512 + decode tg=64. Numbers in absolute ms on GPU, rounded.

## Prefill L=512 — top kernels

### Qwen3.5-9B-Q4_1 · Mesh<1> sync · wall 801 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`mmq_q4_1_4warp_lds`**            |  482 | **60** | 176 |
| `mmq_q5_K_wave64`                   |  112 | 14 |    24 |
| `gdn_state_step_f32_s128`           |   57 |  7 |    48 |
| `quantize_row_f16_q8_1`             |   37 |  5 |   208 |
| `mmq_q8_0_wave64_tile16`            |   29 |  4 |    48 |

### Qwen3.6-27B-Q8_0 · Mesh<4> async ub=128 · wall 8087 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`mmq_q8_0_wave64_tile16`**        | **7002** | **87** | 1984 |
| `gdn_state_step_f32_s128`           |  209 |  3 |   240 |
| `attn_prefill_flash_tile_d256_br8`  |  157 |  2 |    64 |
| `rmsnorm_f16` + others              | ≤ 160 |  2 |   800 |

### Qwen3.6-35B-A3B-UD-Q4_K_S · Mesh<4> async ub=128 · wall 1099 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`mmq_q8_0_wave64_tile16`**        |  411 | 37 |  1240 |
| `indexed_moe_mmq_q4_k_gate_up_tile8`|  243 | 22 |   200 |
| `indexed_moe_mmq_q4_k_down_tile8`   |  131 | 12 |   185 |
| `dense_gemv_f32_f16`                |  104 |  9 | 20520 |
| `gdn_state_step_f32_s128`           |   61 |  6 |   150 |

### Qwen3-Coder-30B-A3B-UD-Q4_K_XL · Mesh<4> sync · wall 2026 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`indexed_moe_mmvq_q5_k`**         |  **640** | **32** | **65** |
| `indexed_moe_mmq_q4_k_gate_up_tile8`|  375 | 19 |   240 |
| `attn_prefill_flash_tile_d128`      |  278 | 14 |   192 |
| `mmq_q4_K_wave64`                   |  216 | 11 |   640 |
| `indexed_moe_mmq_q4_k_down_tile8`   |  160 |  8 |   175 |

## Decode tg=64 — top kernels

### Qwen3.5-9B · Mesh<1> · wall 1220 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`mmvq_q4_1_t128`**                |  538 | **44** | 11616 |
| `mmvq_q5_k_r2`                      |  212 | 17 |  1584 |
| `mmvq_q6_k_dp4a`                    |  110 |  9 |    66 |
| `rmsnorm_q8_1_fused`                |   59 |  5 |  2114 |

### Qwen3.6-27B-Q8_0 · Mesh<4> · wall 3543 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`mmvq_q8_0_gate_up_dp4a`**        | 1658 | **47** | 11264 |
| **`mmvq_q8_0_dp4a_vdr2`**           | 1213 | **34** | 10274 |
| `rmsnorm_q8_1_fused`                |  119 |  3 |  4162 |
| `gdn_state_step_f32_s128`           |   77 |  2 |  3168 |

### Qwen3.6-35B-A3B · Mesh<4> · wall 1218 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| `mmvq_q8_0_gate_up_dp4a`            |  239 | 20 |  7040 |
| `mmvq_q8_0_dp4a_vdr2`               |  116 | 10 |  6380 |
| `indexed_moe_mmvq_q4_k_gate_up_r4`  |  110 |  9 |  2560 |
| `indexed_moe_mmvq_q4_k_r2`          |   84 |  7 |  2368 |
| `gdn_state_step_f32_s128`           |   82 |  7 |  1980 |
| `topk_softmax_f32`                  |   42 |  3 |  2640 |

### Qwen3-Coder-30B · Mesh<4> · wall 1752 ms

| kernel                              |   ms |  % | calls |
|-------------------------------------|-----:|---:|------:|
| **`mmvq_q4_k_r2`**                  |  526 | **30** | 10560 |
| `indexed_moe_mmvq_q4_k_gate_up_r4`  |  216 | 12 |  3072 |
| `indexed_moe_mmvq_q4_k_r2`          |  182 | 10 |  2240 |
| `attention_decode_f16`              |  159 |  9 |  3072 |
| `indexed_moe_mmvq_q5_k`             |   75 |  4 |   858 |

## Levers ranked by expected gain

### Prefill

1. **Coder-30B: port `indexed_moe_mmq_q5_k_down_tile8_dp4a`.** MMVQ on prefill
   is a known structural bug (V2.8.b fixed same pattern for Q6_K on 35B). At
   L=512 we're paying 640 ms / 2026 ms = 32 % on MMVQ where a tile8 MMQ would
   cut ~90 % per V2.8.b precedent. **Expected Coder prefill lift: +25–30 %.**

2. **27B: investigate `mmq_q8_0_wave64_tile16` TILE_N=32.** Q8_0 eats 87 % of
   wall (7000 ms). TILE_N=16 was already a win (V2.7); going to 32 pushes
   each decoded weight block over 2× activation reuse. VGPR headroom is the
   constraint — V2.7 hit VGPR=112 at TILE_N=16, 2 waves/SIMD. A 32-wide tile
   may hit VGPR=224 and spill. **Expected 27B prefill lift: +10–20 %** if
   occupancy holds, 0 % otherwise. Needs PMC probe first.

3. **35B: port `indexed_moe_mmq_q4_k_{gate_up,down}_tile16`.** tile8 is
   22+12=34 % of wall (374 ms). The same TILE_N doubling that worked for
   Q8_0 (V2.7 tile16) and helped Q6_K (V2.8) would apply. **Expected 35B
   prefill lift: +8–12 %.**

4. **9B: revisit Q4_1 MMQ variants.** `4warp_lds` takes 60 %. V2.29.e at
   200 W said `wave64` and `tile16` variants regressed; but those runs
   used 200 W envelope. Worth a fresh A/B at 100 W — power cut may have
   shifted the sweet spot. **Expected 9B prefill lift: unknown, re-test.**

### Decode

5. **27B: multi-row Q8_0 MMVQ (r2 or r4).** Q8_0 MMVQ gate_up + single-row
   are **81 %** of decode wall (2871 ms of 3543 ms). Currently single-row
   or 128-thread; r2/r4 would amortise launch overhead 2-4× for the same
   per-block DP4A work. This is the pattern V2.4 proved for Q4_K MoE.
   **Expected 27B decode lift: +25–40 %** — single biggest decode win on
   any model.

6. **Coder-30B: multi-row Q4_K MMVQ for dense attention.** `mmvq_q4_k_r2`
   takes 30 % (526 ms). Writing a `mmvq_q4_k_r4` (or r8) for the plain
   dense attention path would cut 25-50 % of that call's cost.
   **Expected Coder decode lift: +10–15 %.** Note: MoE path already has
   r4 (via `indexed_moe_mmvq_q4_k_gate_up_r4`); this lever is only about
   the non-MoE Q4_K attention projections.

7. **9B: multi-row Q4_1 MMVQ.** `mmvq_q4_1_t128` at 44 %. Same r2/r4
   lever as 27B. If V2.29.c / V2.29.d didn't already try this, low-risk.
   **Expected 9B decode lift: +10–15 %.**

## Levers that are NOT worth pursuing

- **GDN state_step optimisation** — 2–7 % of decode wall per model, and the
  V2.30.a event-ordering changes the profile anyway. Already async-safe.
- **`rmsnorm_q8_1_fused` tuning** — 3–5 % per model; already fused.
- **`cast_f32_f16`, `quantize_row_f16_q8_1` consolidation** — cumulative
  5–10 % on 35B/Coder, but individual call overhead is < 50 µs. A fused
  "cast + quantize" kernel saves one launch per call × 1000+ calls — ~10 ms
  total. Low priority vs. the multi-row MMVQ wins above.

## Meta-observations

- **9B / 27B share a dense-hybrid structural profile**: one weight dtype
  dominates (Q4_1 or Q8_0) at 60–90 % of wall on both prefill + decode.
  One kernel optimisation moves the whole number.
- **35B is balanced** (3–4 kernels all 9–37 % each). Any one lever gives
  10–20 %, but compounding them is the path to higher overall.
- **Coder-30B's main prefill bug is missing tile8 MMQ for Q5_K down**
  (parallel to V2.8.b). Same fix pattern.
- **Launch overhead dominates decode**: 27B fires 21 k Q8_0 MMVQ calls for
  64 tokens = 330 calls/token. Multi-row packing is the universal lever.

## Artifacts

- `/tmp/flambeau-profile/{9b,27b,35b,coder}-{pp512,tg64}/threadreaper/*_kernel_stats.csv`
- `crates/models/qwen3-moe/tests/profile_point.rs` — repro harness; select
  model + config via `FLAMBEAU_PROFILE_{GGUF,MESH,L,TG}` env vars.

## Regeneration

```bash
TEST=$(ls -t target/release/deps/profile_point-* | grep -v '\.d$' | head -1)
rocprofv3 --kernel-trace --stats --output-format csv \
  -d /tmp/flambeau-profile/<tag> \
  -- env FLAMBEAU_PROFILE_GGUF=<path> FLAMBEAU_PROFILE_MESH=<n> \
     FLAMBEAU_PROFILE_L=<L> FLAMBEAU_PROFILE_TG=<tg> \
     [FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2] \
     $TEST profile_point --nocapture
```
