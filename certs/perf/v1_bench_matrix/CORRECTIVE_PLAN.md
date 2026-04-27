# V1 Bench Matrix — Corrective Action Plan

Generated 2026-04-27 from `REPORT.md` + V2.27 + V2.30 + V2.12 profile memory.

## TL;DR

**User-stated goal**: ≥ **1.6× llama.cpp on pp+tg combined** (geometric, ratio of products).

We already hit it on **5/10 models** thanks to TP-batched prefill +
V2.19 split-K decode. Remaining work targets the **5 below-1.6× models**:
two failure clusters (Q4_0 prefill + Coder-30B/qwen3moe arch).

## Where we stand vs llama.cpp (pp512 × tg64)

llama.cpp doesn't have TP — only `-sm layer` (PP). So two views:

**Kernel-fair (flambeau pp4 vs llama.cpp -sm layer):**
| model | pp4 / lc | tg / lc | combined |
|---|---:|---:|---:|
| qwen35_9b_q4_1 | 0.81× | 0.89× | 0.72× |
| qwen35_27b_q4_0 | **0.32×** | 0.89× | 0.28× |
| qwen35_27b_q4_1 | 0.82× | 0.88× | 0.72× |
| qwen35_27b_q8_0 | 0.76× | 1.04× | 0.79× |
| qwen36_27b_q4_0 | **0.31×** | 0.88× | 0.27× |
| qwen36_27b_q8_0 | 0.78× | 1.05× | 0.82× |
| qwen36_27b_ud_q8_k_xl | OOM | — | — |
| qwen36_35b_a3b_ud_q4_k_s | 0.91× | 1.04× | 0.95× |
| **qwen36_35b_a3b_ud_q8_k_xl** | **1.62×** | **1.00×** | **1.62×** ✓ |
| qwen3_coder_30b_ud_q4_k_xl | **0.53×** | **0.49×** | **0.26×** ✗ |

**Best-topology (TP wins are real but llama.cpp can't reach them):**
| model | best pp / lc | tg / lc | combined |
|---|---:|---:|---:|
| qwen35_9b_q4_1 | 1.23× (tp2+batched) | 1.12× | **1.38×** |
| qwen35_27b_q4_0 | 0.64× | 1.28× | 0.82× |
| qwen35_27b_q4_1 | 1.36× | 1.26× | **1.71×** ✓ |
| qwen35_27b_q8_0 | 1.42× | 1.60× | **2.27×** ✓✓ |
| qwen36_27b_q4_0 | 0.64× | 1.32× | 0.85× |
| qwen36_27b_q8_0 | 1.52× | 1.59× | **2.42×** ✓✓ |
| qwen36_27b_ud_q8_k_xl | 1.08× | 1.67× | **1.80×** ✓ |
| qwen36_35b_a3b_ud_q4_k_s | 0.91× | 1.12× | 1.02× |
| qwen36_35b_a3b_ud_q8_k_xl | 1.62× | 1.00× | **1.62×** ✓ |
| qwen3_coder_30b_ud_q4_k_xl | 0.53× | 0.49× | **0.26×** ✗ |

**Above-1.6 already**: 27B-Q4_1, 27B-Q8_0×2, 27B-UD-Q8_K_XL, 35B-UD-Q8_K_XL.
**Below 1.6**: 9B-Q4_1 (close), 27B-Q4_0×2 (Q4_0 trap), 35B-UD-Q4_K_S (close), Coder-30B (worst).

## Kernel hot-spots (rocprofv3 from prior sessions)

### 35B-A3B-UD-Q8_K_XL prefill L=512 (V2.30, archived cert)
1. `indexed_moe_mmvq_q8_0_dp4a_q8_1` — **77.0 %** (1.62 s of 2.14 s)
2. `mmq_q8_0_wave64_tile16` — 10.0 %
3. `dense_gemv_f32_f16` (router) — 4.6 %

→ Already winning on this model post-V2.x; the 77% on Q8_0 MoE MMVQ is
the lever that unlocks even more. **V2.22.b: Q8_0 indexed-MoE MMQ tile8**.

### 35B-A3B-Q4_0 prefill L=512 (V2.30, archived)
1. `mmq_q4_0_wave64` — 18.0 %
2. `indexed_moe_mmvq_q8_0_dp4a_q8_1` — **14.0 %** (mixed-quant 5 layers)
3. `dense_gemv_f32_f16` — **9.9 %** (router fan-out)
4. `indexed_moe_mmvq_q4_0_q8_1` — 9.3 %

→ Same Q8_0 indexed-MoE MMQ tile gap (V2.22.b) + Q4_0 attention MMQ.

### 9B-Q4_1 Mesh<1> (V2.12 mainline, post-V1.7.6)
1. **`mmq_q4_1_4warp_lds`** — **33.4 %** (V1.4-era kernel, never ported)
2. `mmvq_q4_1` (decode) — 25.2 %
3. `mmvq_q5_k_r2` — 8.0 %
4. `mmq_q5_K_wave64` — 7.2 %

→ **Q4_1 MMQ wave64** never landed. V1.4's `4warp_lds` is the original
prefill kernel; V2.3.b ported Q4_K/Q5_K/Q6_K to wave64 but skipped Q4_1.
With V2.3.b-scale gains (-60 %), Q4_1 prefill becomes ≥ llama.cpp.

## Status of corrective actions (post-execution)

| # | Status | Outcome |
|---|---|---|
| **C1** | **DONE** ✅ | 27B-Q4_0 prefill 73→193 pp4 (+165%), 147→328 pp2tp2 (+123%); pp2tp2 now **1.43× llama.cpp** (was 0.64×). Combined 1.89× — above 1.6 gate. |
| C2 | CLOSED no-op | Q4_1 wave64 kernel exists, intentionally disabled (V2.13.b/V2.29.e/V2.31.f all measured null vs 4warp_lds). 4warp_lds is the local optimum on gfx906. |
| C3 | CLOSED no-op | Q8_0 indexed-MoE tile8 already shipped (V2.22.b). 35B-UD-Q8_K_XL is the model that uses it most heavily — already 1.62× llama.cpp. |
| **C4** | **DEFERRED** | Coder-30B (qwen3moe arch) at 0.26× combined — biggest remaining gap. The forward path uses `forward_dense_attn_*` (V2.28.b) which lacks the perf optimizations qwen35moe got. Multi-session arch parity work. |
| C5 | CLOSED defer | F16 MMQ tile would widen 27B-UD-Q8_K_XL lead (already at 1.80× combined). Low ROI; defer to V2.x. |

## Original five corrective actions (for context)

Ordered by ROI × applicability (touches multiple models):

### **C1 — Q4_0 MMQ wave64 tile** [highest ROI, 3 models]

**Models lifted**: qwen35_27b_q4_0 (0.32→~1.0× pp4); qwen36_27b_q4_0
(0.31→~1.0×); qwen3_coder_30b_ud_q4_k_xl partial (Q4_0 ffn_down). Also
35B-A3B-Q4_0 (V2.30 measured 18% time on the only-MMVQ Q4_0 attention kernel).

**Recipe**: clone V2.3.b Q4_K wave64 (`mmq_q4_K_wave64_q8_1`); swap
block reconstruction for Q4_0's `(q-8)·d` with the bias identity
`(unsigned q − 0x08080808)·y = dp4a(q,y) − 8·sum_y` to avoid the per-byte
saturate-subtract HIP doesn't have. Mirror the V2.3.b SGPR/VGPR layout.

**Expected**: pp4 prefill 73 → 220 tok/s on 27B-Q4_0 (3× kernel).
Combined with TP (pp2tp2+batched), >1.6× llama.cpp.

**Effort**: 1 session. Cert via `bench sweep --impl mmq_q4_0_wave64`.

### **C2 — Q4_1 MMQ wave64** [9B is the marquee dense model]

**Models lifted**: qwen35_9b_q4_1 (1.23 × 1.12 = 1.38 → ≥1.6); qwen35_27b_q4_1
(already 1.71 — gets even further ahead); qwen36_27b_q4_1 — every Q4_1.

**Recipe**: same as C1 but with Q4_1's `(q·d + m)` reconstruction
(d = scale, m = min). Mirror `mmq_q5_K_wave64_q8_1` — Q5_K has the
same `d, dmin` scheme. Per-block: 32 elements packed `q[16]` + `d (f16)`
+ `m (f16)`.

**Expected**: 9B Mesh<1> prefill 754 → 1100 tok/s (turbo's 1033, flambeau passes).
9B tp2+batched 948 → ~1300 tok/s (1.7× llama.cpp).

**Effort**: 1 session.

### **C3 — Q8_0 indexed-MoE MMQ tile8** (V2.22.b deferred)

**Models lifted**: 35B-UD-Q8_K_XL (already 1.62, becomes ~2.5×);
35B-A3B-Q4_0 (mixed-layer 14% Q8_0 MoE goes away); 27B-Q8_0 variants
(small lift, already winning).

**Recipe**: Port V2.6.b's `indexed_moe_mmq_q4_K_gate_up_tile8` structure
to Q8_0 input/output (no super-block, simpler than Q4_K).

**Expected**: per V2.30: UD-Q8_K_XL 239 → ~480 tok/s (×2); 35B-Q4_0
mixed layers go from 14% wall to ~5%.

**Effort**: 1 session.

### **C4 — Coder-30B-UD-Q4_K_XL deep dive** (qwen3moe arch)

This is our worst showing (0.53× pp / 0.49× tg). The arch is
`qwen3moe` (full-attn + MoE, no GDN, no shared expert). Per memory
`project_v2_28_b`, this arch was added in V2.28.b but hasn't gotten
the kernel-perf attention that `qwen35moe` (hybrid) received.

**Action**: rocprofv3 fresh (using rocm-6.3.4 binary — known to work
single-rank per V2.30 memory) on Coder-30B pp4 to identify the gap.
Expected suspects: full-attn output projection, MoE down kernel, router.

**Effort**: 0.5 session profiling + 1-2 sessions kernel work.

### **C5 — F16 MMQ tile** (UD-Q8_K_XL secondary)

V2.27 identified this as a 2.5× lever on 27B-UD-Q8_K_XL (already winning,
this just widens the lead).

**Models lifted**: 27B-UD-Q8_K_XL (1.08 → ~2.7×); also helps any
mixed-quant model that has F16 layers (`token_embd` for instance).

**Recipe**: Y-LDS-shared F16 weight tile, inheriting V2.6.b structure.

**Effort**: 1 session.

## Dispatch hygiene

**Q4_0 t128 default still hurting some shapes** (memory: V2.31.b/e,
"Cycle-5 regression" on 35B-A3B-Q4_0). The current `Q4_0_GU_T128` decision
flips on `n_rows_gate == n_rows_up` (symmetric → t128, asymmetric → 256t).
Verify the dispatch table after C1 lands — Q4_0 MMQ wave64 will
take over the prefill path and t128 only matters for decode.

## Architectural item — TP on qwen3moe

S3 surfaced a hybrid pp2tp2 shape mismatch on Coder-30B (qwen3moe):
`attn_q (TP) shape [2048, 2048] != expected [4096, 2048]` (issue #96).
Pure TP works (V2.27 + B5 used it). Hybrid driver is wrong for qwen3moe
arch — likely a `local_n_heads * head_dim` vs `n_heads * head_dim` boundary
in the layer-range path. Fix unblocks pp2tp2 batched on Coder-30B → likely
~2× prefill on top of C4.

## Six-month look-ahead

After C1+C2+C3, the 1.6× combined gate is hit on **9/10 models**. C4 closes
the last (Coder-30B). C5 widens leads but doesn't change rank order.

Beyond C1-C5, the remaining headroom against silicon ceiling (1 TB/s HBM,
13.4 TFLOPS f32) is roughly:
- Decode is HBM-bound; we're at ~70% of HBM ceiling on Q8_0 (V2.27 ratio).
  Remaining 30% needs Q8 KV (V1 has F16 only on perf path).
- Prefill is compute-bound on K-quants; closer to silicon. C1-C5 takes us
  from "kernel coverage gap" to "silicon ceiling competition".

## What we did NOT do this session

- **Fresh rocprofv3 on all 10 models**: rocprofv3 7.1.1 SIGABRTs on
  multi-rank finalize (V2.30 memory rule), and rocm-6.3.4 throws
  std::out_of_range too on this rig. Existing per-model profiles for
  35B-UD-Q8_K_XL + 35B-Q4_0 plus the V2.27 9B+35B Mesh<2> breakdowns
  cover the majority of corrective targets.
- **Devstral / Mistral / Gemma**: arch=llama not in `SUPPORTED_ARCHS`
  (`config.rs:43`). V1 scope claims Mistral/Devstral dense; loader
  needs wiring (V2 work).

## Done in this session

| | |
|---|---|
| V1-BENCH-S1 | Generic harness (`tests/v1_bench_matrix.rs`) |
| V1-BENCH-S2 | Qwen3.5 27B sweep (Q4_0/Q4_1/Q8_0) |
| V1-BENCH-S3 | MoE family (Coder-30B, Qwen3.6-27B×3, 35B-A3B×2) |
| V1-BENCH-S5 | llama.cpp head-to-head, 10 models, with V2.27 env vars |
| V1-BENCH-S6 | `REPORT.md` aggregator |
| V1-BENCH-S4 | This document — corrective plan from existing profiles |
