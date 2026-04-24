# V2.29.a — per-kernel perf audit on 9B Q4_1 Mesh<4>

rocprofv3 `--kernel-trace` over the full `perf_baseline_qwen35_9b`
run (prefill_grid = {8, 64, 128, 512, 1024}, decode tg=64). Classifies
kernels by phase (prefill-specific, decode-specific, shared) and
ranks by total wall-time.

## Overall breakdown

| phase | total kernel ms | % of run |
|---|---:|---:|
| Prefill-specific | 9629 | 70 % |
| Decode-specific | 1084 | 8 % |
| Shared (both) | 1887 | 14 % |
| **Total kernel time** | **12600 ms** | 100 % |

Run wall-clock ~13 s — kernel time dominates, driver/host overhead is
sub-10% of the whole run (consistent with V2.26.a / V2.27.a finding
that launch-time is overlapped on this rig).

## Prefill-heavy kernels (top 10)

| total_ms | count | avg_us | % | kernel |
|---:|---:|---:|---:|---|
| **5137.30** | 880 | 5838 | **40.8 %** | flambeau_mmq_q4_1_4warp_lds_q8_1 |
| **2809.78** | 56 | 50175 | **22.3 %** | flambeau_attention_prefill_flash_tile_d256_f16 |
| 1547.48 | 120 | 12896 | 12.3 % | flambeau_mmq_q5_K_wave64_q8_1 |
| 411.16 | 2981 | 138 | 3.3 % | flambeau_quantize_row_f16_q8_1 |
| 131.32 | 240 | 547 | 1.0 % | flambeau_mmq_q8_0_wave64_tile16_q8_1 |
| 108.11 | 936 | 116 | 0.9 % | flambeau_quantize_f16_q8_1_mmq |

**Top prefill levers in descending ROI:**
1. **Q4_1 MMQ 4warp_lds** — 40.8 % of total run. Any % off this
   kernel is big. V2.29.b (flash-tile BR/BC tuning) targets attn,
   not MMQ. Add V2.29.b-alt: Q4_1 MMQ tiling variant sweep.
2. **attn_prefill_flash_tile_d256** — 22.3 %. head_dim=256 is
   Qwen3's attn shape. V2.29.b directly attacks this.
3. **Q5_K MMQ wave64** — 12.3 %. Secondary target; share tuning
   patterns with Q4_K MMQ.

## Decode-specific kernels (what affects tg=64)

| total_ms | count | avg_us | kernel |
|---:|---:|---:|---|
| **766.34** | 18025 | **42.52** | flambeau_mmvq_q4_1_t128_q8_1 |
| 256.16 | 2458 | 104.22 | flambeau_mmvq_q5_k_r2_q8_1 |
| 50.80 | 37 | 1372.93 | flambeau_mmvq_q6_k_dp4a_q8_1 |
| 5.86 | 227 | 25.81 | flambeau_attention_decode_f16 |
| 4.41 | 683 | 6.46 | flambeau_mmvq_q8_0_gate_up_dp4a_q8_1 |

**Decode hot path is ~60 % `flambeau_mmvq_q4_1_t128_q8_1`.**

Math: 64-token decode ≈ 1210 ms wall. Within that, per-token Q4_1
MMVQ = 766 ms / 64 tokens × (decode share of total MMVQ calls) —
estimating decode-only Q4_1 MMVQ = ~580 ms of 64-token decode
→ **~9 ms/token** just for Q4_1 MMVQ, out of **~19 ms/token**
total = **47 %** of decode wall.

Attention decode is ~6 ms total across the whole run (227 calls at
26 µs each). Negligible for 9B short-context decode.

## The 28 % gap to turbo — mapped to kernels

9B tg=64 = 53 tok/s (flambeau) vs 73.5 (turbo). 19.0 ms/token vs
13.6 ms/token = **5.4 ms/token** of per-token work we're doing that
turbo isn't.

Given 47 % of our wall is `mmvq_q4_1_t128`, if turbo's Q4_1 MMVQ is
20-30 % faster (via multi-row packing or different thread shape),
that alone closes ~3 ms/token = **most of the gap**.

## Top-3 recommendations, ranked by ROI

### #1 — V2.29.c: Q4_1 MMVQ multi-row packing (DECODE)
- Current kernel: `flambeau_mmvq_q4_1_t128_q8_1` (128 threads, 1 row/block).
- Target: `mmvq_q4_1_r2_q8_1` or `_r4_q8_1` (2 or 4 output rows per
  block). Cert exists for Q4_K / Q5_K / Q6_K r2/r4 variants; same
  pattern.
- Expected: 15-30 % on the kernel → 7-14 % on decode wall-time →
  53 → 57-60 tok/s. Closes most of the 28 % turbo gap.
- **Already queued as task #175 — execute next.**

### #2 — Q4_1 MMQ 4warp_lds optimization (PREFILL)
- Current kernel: `flambeau_mmq_q4_1_4warp_lds_q8_1` dominates at
  40.8 % of prefill wall.
- Targets (ordered): (a) `tile16` variant like Q8_0 has
  (`flambeau_mmq_q8_0_wave64_tile16` — already 4× faster per-op
  than Q4_1 4warp_lds), (b) `wave64` variant like Q5_K, (c) MMQ_X
  tuning per the V2.3.a cert (memory says it's compute-bound, so
  VGPR tuning is null; structural change needed).
- Expected: 10-20 % on kernel → 4-8 % on prefill wall at L=4096.
- **Not queued — add as V2.29.e.**

### #3 — V2.29.b: attention_prefill_flash_tile BR/BC tuning (PREFILL)
- Current: head_dim=256 variant, BR=4 BC=64. 56 calls in this run
  at 50 ms/call = ~22.3 % of prefill wall.
- Candidates per the original V2.29.b scope: sweep BR ∈ {2,4,8,16}
  × BC ∈ {32,64,128} on head_dim 256.
- Expected: 5-15 % on kernel → 1-3 % on prefill wall. Dwarfed by
  Q4_1 MMQ improvements.
- **Already queued as task #174; deprioritise vs #175 and the new
  Q4_1 MMQ task.**

### #4 — V2.29.d: fused decode kernels (DECODE)
- Small fractions per op (rmsnorm 0.3 %, quantize 0.4 %, cast 0.4 %,
  swiglu 0.4 %). Fusing would save a few % at best.
- Already queued; keep as future tuning, not immediate ROI.

## Per-op PMC hints (from this trace — no counter pass required)

- `mmvq_q4_1_t128`: VGPR=24 per V2.2.b kernel. Sub-threshold
  (2 waves/SIMD ≈ 256 VGPR / 10 waves). At 42 µs/call on 64-thread
  blocks, there's room for r2 (doubling output rows per block
  halves the launch count + amortises weight loads). Same pattern
  as Q4_K r2 (V1.3 multi-row DPP port) which landed 1.8× on Q4_K.
- `attention_prefill_flash_tile_d256`: 50 ms/call at L=1024.
  Per-token cost = 50/1024 = 49 µs. O(L²·d), so at L=4096 we
  project ~800 ms/call. Already measured in V2.26.b: 3.14 s total
  for L=4096 L prefill attention = matches.

## V2.29 queue update

- #173 V2.29.a (this audit) — complete.
- **#175 V2.29.c — promote to next task.** Q4_1 MMVQ r2/r4.
- #174 V2.29.b (attn flash-tile) — still valid but lower ROI.
- **New: V2.29.e — Q4_1 MMQ prefill tile16 variant.** Biggest single
  kernel in the run.
- #176 V2.29.d (fused decode kernels) — stays low priority.

## Regeneration

```
BIN=$(ls -t target/release/deps/perf_baseline_qwen35_9b-* | grep -v '\.d$' | head -1)
mkdir -p /tmp/v29a && cd /tmp/v29a
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.5-9B-Q4_1.gguf \
  /opt/rocm-7.1.1/core-7.13/bin/rocprofv3 \
    --kernel-trace --output-format csv --output-file run \
    -- $BIN perf_baseline_qwen35_9b --nocapture
# Then aggregate via the Python script in cert body.
```
