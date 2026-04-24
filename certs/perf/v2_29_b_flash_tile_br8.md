# V2.29.b — flash-tile BR=8 at head_dim=256: +24.7% prefill L=4096

First kernel-level win of the V2.29 cycle. Bumped BR (Q rows per
block) from 4 to 8 at head_dim=256 in
`flambeau_attention_prefill_flash_tile_d256_f16`, reducing LDS-tile
redundancy across Q rows by 2×.

## Why this kernel

Per V2.29.a audit: `attention_prefill_flash_tile_d256_f16` was
**22.3 % of total run wall time** (2810 ms / 12600 ms) on 9B Q4_1
Mesh<4> — the #2 kernel after Q4_1 MMQ. All Qwen3-family models
(3.5 + 3.6) use head_dim=256, so this kernel fires on every
attention-prefill layer.

## Change

Added a `BR=8` extern "C" wrapper alongside the existing `BR=4`:

```cpp
extern "C" __global__ __launch_bounds__(512, 1)
void flambeau_attention_prefill_flash_tile_d256_br8_f16(...) {
    flash_attn_prefill_v2_impl</*D=*/256, /*BR=*/8, /*BC=*/16>(...);
}
```

Block: 64 threads × BR rows = 512 threads (up from 256 at BR=4).
Launch-bounds dropped from `(256, 2)` to `(512, 1)` — 1 wave/SIMD
vs the 2 BR=4 allowed. LDS unchanged: `2 · BC · D · 4 = 2 · 16 · 256 · 4 = 32 KiB`.

d=64 and d=128 variants untouched (BR=4 stays optimal for them;
Qwen3's head_dim=256 is the only target shape).

Dispatch in `attention.rs`: head_dim=256 now routes to the new
`_br8_` entry; the `block: (WARP, BR, 1)` uses `br=8` only when
head_dim==256, else keeps BR=4.

## Measured (9B Q4_1 Mesh<4> async ub=128 lanes=2, triple-run avg)

| L | BR=4 (baseline) | **BR=8** | Δ |
|---|---:|---:|---:|
| 1024 | 1828 | **1965** | **+7.5 %** |
| 2048 | 2088 | **2354** | **+12.7 %** |
| 4096 | 2004 | **2498** | **+24.7 %** |

Decode tg=64 unchanged: 54.5 tok/s (decode uses separate
`attention_decode_f16` kernel, not the prefill flash-tile).

## 35B portability check

| L | V2.27.b (BR=4) | BR=8 | Δ |
|---|---:|---:|---:|
| 1024 | 1414 | 1495 | +5.7 % |
| 2048 | 1613 | 1783 | +10.5 % |
| 4096 | 1529 | 1906 | +24.7 % |

Same +24.7 % at L=4096 on Qwen3.6-35B-A3B-UD-Q4_K_S. Win is model-
agnostic within the head_dim=256 family.

## Why BR=8 wins despite halved occupancy

- BR=8 blocks run at 1 wave/SIMD (down from BR=4's 2 waves/SIMD).
  On paper this halves latency-hiding capacity.
- But each block now amortises one K/V tile load across 8 Q rows
  instead of 4 → **2× reduction in redundant LDS loads per unique
  tile**.
- The kernel is LDS-load dominated (BC=16 tile × D=256 floats ×
  2 [K+V] = 8 KiB × 2 reads per iteration = not small). At 50 ms/call
  previously, LDS-load amortisation saves more than the occupancy
  loss costs.
- BR=16 would push LDS tile amortisation further but block size =
  64×16 = 1024 threads = full CU, zero resident blocks per CU left
  → catastrophic. BR=8 at 512 threads is the sweet spot.

## Cumulative impact

Full arc, 9B Q4_1 Mesh<4> prefill L=4096 async path:

| milestone | tok/s | ratio vs prev |
|---|---:|---:|
| V2.26.b async (w/ barrier) | 714 | — |
| V2.26.a (barrier removed) | 1988 | 2.78× |
| **V2.29.b (BR=8)** | **2498** | **1.26×** |
| vs llamacpp-turbo (963) | — | **2.59× ahead** |

## Gate

- Parity bit-exact (last_id matches) on both models × 3 L values:
  9B `220 / 248046 / 62`, 35B `220 / 198 / 248046`.
- Decode unchanged on both models (separate kernel).
- Build clean; kernels-hip rebuilt (HSACO for d256_br8 added to
  catalogue).
- Other head_dim specializations untouched.

## Regeneration

```
cargo clean -p flambeau-kernels-hip
cargo build --release -p flambeau-qwen3-moe --tests --features hip

BIN=$(ls -t target/release/deps/perf_baseline_qwen35_9b-* | grep -v '\.d$' | head -1)
FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  $BIN perf_baseline_qwen35_9b --nocapture
```
