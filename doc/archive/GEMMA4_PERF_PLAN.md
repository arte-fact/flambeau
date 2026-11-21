# Gemma4 perf closure plan — chase the 26 % gap vs qwen

Measured 2026-06-01 on MI50 pp2tp2 (`hip:0,2,1,3`), single-request
decode, ~1.5k-token prompt, max_tokens 128, greedy, mixed-batch
default-on:

| Model                | Quant | Decode tps | GB/s/GPU  |
|----------------------|-------|-----------:|----------:|
| Qwen3.6-27B          | Q4_0  |      33.40 |      123  |
| Gemma4-31B           | Q4_0  |      22.59 |       91  |

Gemma4 delivers **0.74×** qwen's per-GPU HBM bandwidth at comparable
model size + quantization + topology. Same qmatmul kernels run both —
the gap is structural (gemma4-specific) overheads on top of the
qmatmul path. This plan slices the gap by lever and orders by ROI
per LOC, validated against the gap budget.

## Gap budget (decode wall, gemma4-31B-Q4_0 pp2tp2, ctx ≈ 1500)

Decode wall per token = 1 / 22.59 tps = **44.3 ms**.
Qwen3.6-27B-Q4_0 at same prompt + topology = 1 / 33.40 tps = **29.9 ms**.
Wall delta: **14.4 ms / tok = 33 % slower**.

Per-token bytes read at decode (weights + KV cache at ctx 1500):

| Model | Weights | KV reads | Total | Achieved BW |
|-------|--------:|---------:|------:|------------:|
| Qwen3.6-27B  (64 layers, 4 KV heads)   | 14.7 GB | 0.4 GB | 15.1 GB | 504 GB/s aggregate |
| Gemma4-31B   (60 layers, 16 KV heads)  | 16.1 GB | 3.4 GB | 19.5 GB | 441 GB/s aggregate |

Two distinct contributions to the 14.4 ms wall delta:

- **Structural: 1.29× more bytes / token.** 19.5 / 15.1 = 1.29. At
  qwen's achieved rate this alone makes gemma4 wall = 29.9 × 1.29 =
  38.6 ms (= +8.7 ms vs qwen, 19 % wall).
- **Implementation: 0.87× per-byte rate.** Gemma4 achieves 441
  GB/s vs qwen's 504; that's a 13 % per-byte deficit. On top of the
  structural floor: 38.6 × (1/0.87) = 44.3 ms (= +5.7 ms more, 13 %
  wall).

So of the 33 % wall gap:

- **~19 % is structural** — gemma4 reads more bytes per token.
- **~14 % is implementation** — gemma4's kernels achieve less per-byte
  bandwidth.

### Structural ~19 % decomposition

| Source                                            | Δ bytes / tok | Driver                  |
|---------------------------------------------------|--------------:|-------------------------|
| KV cache reads (GQA factor 2 vs 6 + hd_global=512)|        +3.0 GB | KV layout / position    |
| Q + O projections (32 heads vs 24, hd_global=512) |        +0.7 GB | Model spec              |
| Hidden + FFN dim modestly larger (5376 vs 5120)   |        +0.7 GB | Model spec              |

KV-cache reads dominate the structural gap and grow linearly with
context (~5 % wall at ctx=1500, ~15-20 % wall at ctx=8k). They are
**not** out of scope — KV quant changes the per-byte cost without
touching the model spec (S7).

### Implementation ~14 % decomposition

| Source                                            |   Δ ms / tok  | Slice    |
|---------------------------------------------------|--------------:|----------|
| head_dim=512 attn kernel undertuning (10 layers)  |    1.5 – 2.0  | S5b-d    |
| F32 AR on attn (post_attn_norm path)              |    0.5 – 1.0  | S1, S3   |
| F32 AR on FFN (post_ffn_norm path)                |    0.5 – 1.0  | S2       |
| V-from-K DtoD memcpy + V-unit-norm + scale_inplace|    0.5 – 0.8  | n/a      |
| PP stage handoffs on Hybrid topology              |    0.3 – 0.8  | S8       |
| Per-layer launch density (4 rmsnorms vs qwen's 2) |    0.3 – 0.5  | (in S1/S2) |
| Unaccounted (measurement floor + launch jitter)   |    0.5 – 1.0  | —        |
| **Total**                                         | **~5.7**      |          |

### Out of scope (model-spec)

- n_q_heads, n_kv_heads — baked into Q/K/V weight shapes (retrain
  only).
- head_dim — same. Per-head attention is dimensionally what it is.
- FFN width — same.

What *is* in scope: the per-byte cost of reading those weights and
that KV cache. Quantizing KV (S7) is the lever that mitigates the
structural KV term without touching the model spec.

## Slices

Ordered by ROI per LOC, not by gap size. Each slice ships
independently with its own cert and parity test.

### S1 — F16 AR split-launch — NULL ON DECODE (shipped 2026-06-01)

**Shipped** the infrastructure (kernel + ops trait + BAR1 wrappers +
runtime helper + topology hook); **reverted the composite branch**
after the bench measured null.

**Measured** gemma4-31B-Q4_0 pp2tp2, 5 reps:

| Metric    | Baseline | S1 split-launch | Δ     |
|-----------|---------:|----------------:|------:|
| Decode tps| 22.59    | 22.63           | +0.04 (noise) |
| TTFT      | 6799 ms  | 6540 ms         | -259 ms (~4 %, not validated) |

**Diagnosis** — launch-count miscount. Old gemma4 path = 2 launches
per layer (`ar_sum_f32` + `rmsnorm_f32_to_f16_add_residual`). S1 path
= 3 (added a `cast_f32_to_f16` launch). Per-layer BAR1 payload saving
(~0.9 µs) was dwarfed by the added cast launch (~10 µs). Net zero.

**What landed** (334 LOC infrastructure across 9 files, all consumed
by S3):
- `flambeau_rmsnorm_f16_to_f16_add_residual` kernel + Ops trait
  method + HipOps launcher + impl.
- `BarP2pAllReduce::sum_tp{2,4}_rank` F16 rank-local wrappers
  (kernels already existed).
- `bar_ar_sum_f16` runtime helper (event / host-sync split mirror of
  the F32 sibling).
- `TopologyHooks::{supports_ar_sum_f16, ar_sum_f16}` + TP / Hybrid
  HipHooks impls.
- `output_proj_safe_for_f16_ar(weights)` predicate.

**What didn't land**: the composite branch in `standard_attn.rs` +
`standard_attn_mixed.rs`. Reverted after measurement showed null.

### S2 — DROPPED (would inherit S3's regression)

Was the FFN mirror of S3. Same `ar_sum_f32 + fused-rmsnorm-add`
2-launch pattern on the post_ffn_norm path. S3's regression
diagnosis (single-block fused kernel sacrifices BAR1-per-CU
parallelism on gfx906) applies symmetrically to the FFN side.
Don't ship until S3's redesign clears that constraint.

### S3 — Fused `bar_ar_postattn_residual_rmsnorm_f32_to_f16` — REGRESSED (shipped 2026-06-01)

**Shipped** the kernel + launcher + runtime helper + topology hook;
**reverted the composite branch** after measuring a regression.

**Measured** gemma4-31B-Q4_0 pp2tp2, 5 reps:

| Metric    | Baseline | S3 fused | Δ        |
|-----------|---------:|---------:|---------:|
| Decode tps| 22.59    | 21.62    | **-4.3 %** |

**Diagnosis** — CU parallelism loss. The fused kernel launches one
block per row; at decode `n_rows = 1` → only 1 block × 1 CU active.
The old `ar_sum_f32` uses `n*hidden / 256 = 21` blocks fanning out
across ~21 CUs, hiding BAR1 read latency through inter-block
parallelism. Replacing 2 launches with 1 was a real launch-count win
(~10 µs / layer) but the per-CU BAR1 read serialisation cost more
than that (~15 µs / layer / kernel × 60 layers).

**What landed** (~280 LOC infrastructure, kept for future tuning):
- `flambeau_p2p_allreduce_postattn_residual_rmsnorm_f32_to_f16_tp{2,4}`
  kernels (1 block per row, register-cached AR sums, hidden ≤ 8192).
- `ArKind::PostAttnResidualRmsNormF32ToF16Tp{2,4}` + launchers in
  `bar_p2p.rs`.
- `bar_ar_postattn_residual_rmsnorm_f32_to_f16` runtime helper.
- `TopologyHooks::{supports_, ar_}postattn_residual_rmsnorm_f32_to_f16`
  trait surface + TP / Hybrid impls.

**What didn't land**: the composite branches in `standard_attn.rs`
+ `standard_attn_mixed.rs`. Reverted after measurement showed -4.3 %.

**Possible future revival**: redesign the kernel with multi-block
AR-sum into LDS + 1-block rmsnorm reduce — essentially recreating
the 2-launch structure inside one launch — OR larger block (1024
threads, 16 wave64 warps to hide BAR1 latency single-CU). Both are
multi-session structural work; neither is a one-session port.

### S3-original — fully-fused (deferred indefinitely)

The original S3 plan (fused F16 kernel) is superseded by the
shipped-then-reverted F32-to-F16 fused kernel above. The lesson is
that **post-attn norm-AR-residual fusion on gemma4's TP path
sacrifices the BAR1-parallelism the unfused `ar_sum_f32` provides
across many CUs**. Don't reattempt without first re-baselining the
multi-block tradeoff with rocprofv3 PMC.

**Shape**: new BAR1 kernel that takes (proj_f32_local, peer_proj_f32,
post_norm_w, residual_in_f16, residual_out_f16, eps) and does in one
pass per row:

1. AR-sum F32: `sum_f32 = proj_f32_local + Σ peer_proj_f32`.
2. rmsnorm over the F32 sum with `post_norm_w` (F16) — reduce + scale
   stays in F32 registers.
3. Add to F16 residual: `residual_out = residual_in + cast<F16>(normed)`
   with F16-saturating clamp (envelope matches existing
   `rmsnorm_f32_to_f16_add_residual`).

**Why F32 over BAR1, not F16**: S1 proved BAR1 payload size isn't the
bottleneck at decode (event-overhead bound). Keeping the F32 read
avoids the upstream cast launch AND avoids the saturation risk on
head_dim=512 — same safety as the existing F32 path. The win is
purely from collapsing 2 → 1 launch (~10 µs/layer × 60 layers = 600
µs / token = ~1.4 % wall).

**Reuses S1 infrastructure** — `output_proj_safe_for_f16_ar`
predicate is irrelevant (F32 over BAR1), but the
`TopologyHooks::ar_*` plumbing pattern transfers directly.

**Files** (~300 LOC):
- `crates/kernels-hip/src/kernels/p2p_allreduce_residual.cu` —
  `flambeau_p2p_postattn_residual_rmsnorm_f32_to_f16_tp{2,4}`. Block
  shape mirrors `flambeau_rmsnorm_f32_to_f16_add_residual` (one
  block / row, 256 threads); adds an inner BAR1 read of peer's F32
  partial for the sum.
- `crates/backend-hip/src/bar_p2p.rs` —
  `postattn_residual_rmsnorm_f32_to_f16_tp{2,4}_rank` rank-local
  launchers. New `ArKind::PostAttnResidualRmsNormF32ToF16Tp{2,4}`.
- `crates/forward/src/runtime/ar.rs` —
  `bar_ar_postattn_residual_rmsnorm_f32_to_f16` runtime helper with
  the event / host-sync split.
- `crates/forward/src/core/hooks.rs` + `engine.rs` —
  `TopologyHooks::{supports, _f32_to_f16_…}` method pair on the
  trait + TP / Hybrid impls.
- `crates/forward/src/core/composites/standard_attn.rs` +
  `standard_attn_mixed.rs` — single branch above the
  `ar_sum_f32` fallthrough that fires when `post_attn_norm.is_some()`
  AND `hooks.supports_ar_postattn_residual_rmsnorm_f32_to_f16()`.

**Tests**:
- BAR1 fused-kernel parity smoke (mirror existing
  `residual_rmsnorm_tp2_rank` test pattern).
- Synth gemma topology-parity vs the existing F32-AR fallthrough.
- End-to-end gemma4-31B-Q4_0 / pp2tp2 / 1.5k-token prompt
  before/after. Expect 22.6 → 22.9-23.0 tps.

### S4 — DROPPED: V-from-K does not cause read amplification

Earlier plan claimed +1.0 tps from a `KvCache<F16ContigKV>`
separate-V layout. Re-reading `standard_attn.rs:282-297` confirms
V is **already** stored in its own cache slab — `attn_v: None`
means the V projection weight is absent; V is built by a DtoD
memcpy from K's scratch buffer at *kv_append time* and written
into the V slab normally. At decode the K-cache read and V-cache
read are independent.

The per-token cost of V-from-K is just one DtoD memcpy per layer
of size `n_kv × head_dim × 2 B` (~580 KB / token total across
60 layers) — negligible at the F16 path. No read amplification
exists to eliminate.

The S4 lever is gone. The "structural KV bandwidth" gap is
addressed by S7 (Q8 KV cache) instead.

### S5 — head_dim=512 attention kernel pass

**Target**: 3-5 % at decode on gemma4-31B (10 / 60 layers are
head_dim=512 global attention). Current `attention_decode_f16`
declares head_dim ∈ {64, 128, 256} supported but accepts 512 via
buffer ceiling — runs at the 4-warp config which is undertuned at
head_dim=512.

**Shape**: gfx906-tuned head_dim=512 path. Likely 8-warp block, LDS
tile re-sized, V load pattern adjusted for the 2× row stride.
Possibly a sibling kernel `attention_decode_f16_hd512`.

Multi-session — split into 4 sub-slices. S5a sets the perf floor
(PMC + sweep-baseline at the current path), S5b-d each ship one
tuned kernel.

#### S5a — baseline characterisation + cert plumbing (~150 LOC + cert)
One session. No new kernels; profile + cert what we have today so
S5b-d wins are measurable, not hand-waved.

- `crates/bench/src/sweep/attn.rs` — extend sweep to cover
  head_dim=512 cells (decode + batched-decode + prefill_flash_tile).
- `cargo run -p bench -- sweep --arch gfx906 --op attention` —
  emits `certs/hip/gfx906/attention_*_hd512.json` baselines.
- rocprofv3 snapshot of gemma4-31B-Q4_0 / pp2tp2 single-request,
  attributing the 10 global layers' attention kernel time. Filed
  in PR body as the gap-to-close target.

#### S5b — `attention_decode_f16` head_dim=512 path (~250 LOC)
One session. Single-token decode is the steady-state case; close
this first because the per-token cost compounds 10× per forward.

- `crates/kernels-hip/src/kernels/attention_decode_f16.cu` — gated
  head_dim=512 path. Likely shape: 8-warp block (block_dim=512),
  LDS tile widened, separate Q-broadcast and V-accumulate phases
  tuned for 8-warp occupancy + gfx906 VGPR ceiling (10 waves/SIMD
  budget, see memory `feedback_v906_vgpr_ceiling` if present).
- Sweep cert green vs S5a baseline (must beat, not tie).
- Parity test on synth head_dim=512 shape.

#### S5c — `attention_decode_f16_batched` head_dim=512 path (~200 LOC)
One session. Required for mixed-batch engagement on gemma4 global
layers (currently bails). Mirror of S5b on the batched kernel.

- `crates/kernels-hip/src/kernels/attention_decode_f16_batched.cu`
  — port the S5b tile shape. Same `window_size` arg as the
  existing batched kernel; same per-slot block-table indirection.
- Drop the `head_dim <= 256` bailout in `standard_attn_mixed.rs`
  for gemma4 global layers if present.
- Parity test + mixed-batch end-to-end on gemma4-31B.

#### S5d — `attention_prefill_f16_flash_tile` head_dim=512 path (~200 LOC)
One session. Prefill is less frequent but each call is fat —
matters for TTFT.

- `crates/kernels-hip/src/kernels/attention_prefill_f16_flash_tile.cu`
  — head_dim=512 path. Online softmax invariants per memory
  `feedback_flash_tile_swa_nan_init` apply unchanged.
- Sweep cert; gemma4-31B-Q4_0 TTFT bench before/after on the
  1.5k-token prompt used in 2026-06-01 baseline.

### S7 — Q8 KV cache on v2 stack — **shipped 2026-06-01, context-dependent win**

Three sub-slices shipped (S7a typestate + scratch + plumbing; S7b
model-ops wrappers + parity tests; S7c composite branch + splitk
wiring). Final bench data:

**qwen3.5-9B-Q4_1 / pp / hip:0 / single device**:

| ctx ≈ | F16 decode tps | Q8 + splitk decode tps | Q8 vs F16 |
|-------|---------------:|-----------------------:|----------:|
| 1500  | 65.34          | 60.28                  | −7.7 %    |
| 3000  | 59.17          | 52.32                  | −11.6 %   |
| ~5000 | 36.56          | 41.18                  | **+12.6 %** |

Crossover sits around ctx 3500–4000 on qwen3.5-9B. KV-cache reads
are ~3 % of per-token bytes at ctx 1500 and grow to ~15 % at ctx
5000; only above the crossover does the ~47 % Q8 BW saving exceed
the kernel-overhead tax (per-block Q→Q8 quantize in LDS + scalar
`K.d × Q.d` dequant per Q8_0 block on the Q8 path).

**Gemma4 stays on F16 by validation.** Per S7c's
`scratch_config_for` check: gemma4-31B has `head_dim=512` global
layers (excluded by the Q8 attention kernel's `head_dim ∈
{64,128,256}` ceiling) AND `window_size=1024` SWA layers (Q8
attention has no SWA mask path). Server bails at boot with
"--kv q8: layer N has head_dim=512" or "window_size=1024 (SWA)".

**Production read**: default `--kv f16` for typical chat
workloads; `--kv q8` for RAG / long-context (ctx ≥ 4k) on
qwen35 / qwen35moe family. The flag is meaningful, not a stub.

**Wasn't a gemma4 lever after all** — the plan's projected +1.2
tps at ctx 1500 was wrong on two counts: (1) bandwidth share at
ctx 1500 too small for Q8's kernel-overhead tax to be amortized;
(2) gemma4 architecturally excluded by the kernel head_dim+SWA
constraints. The gemma4-specific path forward is **S5 head_dim=512
kernel tuning** (which is also independently the biggest single
gemma4 lever in the plan).

#### S7a — Q8Contig typestate + scratch wiring — shipped
- `KvLayout` enum (F16Contig, Q8Contig) + `Q8_0_BLOCK_BYTES` const
  in `forward/src/core/scratch.rs`.
- `KvCache` gains `bytes_per_row` + `layout` fields; `ScratchConfig`
  gains `kv_layout`; `ScratchPool::new` branches alloc on layout.
- Per-arch `Arch::scratch_config` + `scratch_config_for` thread the
  layout end-to-end.
- `KvLayerShape::head_dim_at` + `window_size_at` accessors with
  per-arch impls; `scratch_config_for` bails when `--kv q8`
  requested on a layer the kernel can't handle.
- 4 unit tests on the bytes-per-row math green.

#### S7b — Q8 model-ops + parity tests — shipped
- `kv_append_f16_to_q8` (composes existing `quantize_f16_q8_0`
  launcher with per-row destination offset math).
- `attn_decode_q8_kv` (thin wrapper over `Ops::attention_decode_q8_kv`).
- `attn_prefill_q8_kv` (thin wrapper over `Ops::attention_prefill_q8_kv`).
- `attn_decode_q8_kv_splitk` (thin wrapper over `Ops::attention_decode_q8_kv_splitk`)
  — added in S7c after the no-splitk regression measurement.
- 2 parity tests green on device: kv_append byte/offset parity +
  dequant-vs-F16 envelope (abs ≤ 0.05); F16 reference vs Q8 attn
  decode output on same synth K/V within Q8 noise (abs ≤ 0.05).

#### S7c — composite branch + per-arch wiring + bench — shipped
- `standard_attn.rs` decode + prefill arms branch on `kv.layout ==
  Q8Contig`. Q8 path uses splitk under the same `n_tokens_kv > 256
  && n_chunks > 1` conditions as F16 — restored the inter-block
  parallelism that the regression-causing initial wire lost.
- Server `--kv q8` accepted (was hard-error placeholder).
- 5-rep + 3-rep benches at ctx 1500 / 3000 / ~5000 file the
  context-dependent win above.

### S8 — Hybrid stage handoff: sync-bounce → async DtoD — **shipped 2026-06-01, +2.9 %**

Two sub-slices shipped. The plan's original thesis ("4 separate
per-layer peer-copies coalesced into 1") was wrong: `peer_send` /
`peer_recv` fire exactly **once per forward** at the stage
boundary. The actual lever is replacing `HybStage`'s host-bounce
`DtoH + Stream::synchronize + barrier + HtoD` with `PpStage`'s
existing event-based async DtoD pattern.

**S8a measurements (gemma4-31B-Q4_0 / pp2tp2, FLAMBEAU_TRACE_HANDOFF=1):**
- Per-decode-token `peer_send` DtoH+sync ≈ 8 ms
- Per-decode-token `peer_recv` barrier wait ≈ 15 ms (mostly stage 0's
  GPU compute still landing — not pure overhead)
- Saveable overhead per token: ~1-2 ms (host bounce + CPU
  Stream::synchronize + barrier coordination)

**S8b shipped:**

| Metric (gemma4-31B-Q4_0 / pp2tp2) | Baseline | S8b | Δ |
|---|---:|---:|---:|
| Decode tps (median, 5 reps) | 22.59 | **23.25** | **+0.66 (+2.9 %)** |
| Coherence smoke | "Paris." | "Paris." | ✓ |

**What landed** (~235 LOC across 4 files):
- `crates/forward/src/runtime/orchestrate.rs` — per-transition
  per-rank edges (rank k of stage i ↔ rank k of stage i+1) via
  `new_peer_edge_prealloc(consumer_dev, prefill_ubatch *
  MAX_HIDDEN * 2)`. Replaces single shared host-bounce slot + global
  `Barrier`.
- `crates/forward/src/runtime/ar.rs` — `new_peer_edge_prealloc`
  helper. Eliminates the alloc-on-first-send race when both stage
  workers start in parallel.
- `crates/forward/src/runtime/workers.rs` — `WorkerRole::Hybrid`
  replaces `peer_buffer` + `handoff` with `send_edge` + `recv_edge`
  matching `WorkerRole::Pp` shape.
- `crates/forward/src/engine.rs` — `HybStage` ported to event-based
  async DtoD `memcpy_peer_async` + `HipEvent` record/stream_wait,
  exactly mirroring `PpStage`. Drops the `handoff_barrier` (events
  provide GPU-side ordering). Adds a brief CPU spin-wait in
  `peer_recv` to handle the case where the receiver thread reaches
  the handoff before the producer's CPU-side `peer_send` enqueues
  the event (a real race that didn't exist in the sync-bounce path).
- Constraint: requires equal `tp_size` across adjacent stages; pp2tp2
  satisfies this. Mixed-TP hybrids would need rank-pair re-mapping
  (not in scope).

**Closes** the engineering inconsistency between `PpStage` (already
event-based async DtoD for pp4) and `HybStage` (was sync-bounce).
Code shape now matches. Same win applies to any pp+tp hybrid arch
(qwen35moe, gemma4, future). No regression risk because the path is
literally what PpStage already does.

## Cumulative target

Per-session lifts at ctx ≈ 1500 (matches 2026-06-01 baseline);
numbers are expected, not measured. S7's wins scale with context —
the parenthesised value is the ctx-8k figure. Each session ends with
the measured delta filed against this row.

| Session | Slice                                  | Files | LOC | Δ tps  | Cum. tps | Cum. gap |
|---------|----------------------------------------|------:|----:|-------:|---------:|---------:|
|    —    | Baseline (2026-06-01)                  |     — |   — |   —    |   22.59  |   −33 %  |
|    1    | S1  F16 AR split-launch (**NULL**)     |     9 | 334 | +0.04  |   22.59  |   −33 %  |
|    2    | S3  fused F32→F16 AR+norm+resid (**REGRESSED**) | 5 | 280 | -0.97  |   22.59  |   −33 %  |
|    3    | S7a Q8Contig typestate + scratch       |     8 | 350 |   0    |   22.59  |   −33 %  |
|    4    | S7b Q8 model-ops + parity tests        |     4 | 600 |   0    |   22.59  |   −33 %  |
|    5    | S7c composite + splitk wiring + bench  |     5 | 230 | 0 on gemma4 (excluded by validation; **qwen35 +12.6 % at ctx ≥ 5k**) | 22.59 | −33 %  |
|    6    | S5a baseline cert + rocprofv3          |     1 | 150 |   0    |   22.59  |   −33 %  |
|    7    | S5b decode_f16 hd512 path              |     2 | 250 | +0.7   |   23.29  |   −30 %  |
|    8    | S5c decode_f16_batched hd512 path      |     2 | 200 | +0.4   |   23.69  |   −29 %  |
|    9    | S5d prefill_flash_tile hd512 path      |     2 | 200 | +0.4   |   24.09  |   −28 %  |
|   10    | S8a handoff instrumentation + measure  |     1 | 100 |   0    |   22.59  |   −33 %  |
|   11    | S8b HybStage async DtoD (shipped)      |     4 | 235 | **+0.66 measured** | **23.25** | **−30 %** |

**Projected ceiling** at ctx 1500: **~24.6 tps gemma4-31B-Q4_0
pp2tp2**, vs 33.4 tps qwen3.6-27B-Q4_0. Gap closes from 33 % to
~26 %.

**Projected ceiling at ctx 8k**: still ~24.6 tps for gemma4 —
S7's KV-quant win doesn't engage on gemma4 (excluded by Q8 kernel
constraints). For arches where Q8 does engage (qwen35 / qwen35moe
at head_dim ≤ 256, no SWA), measured gain is **+12.6 % at ctx
5000** on qwen3.5-9B-Q4_1 single-device; crossover sits around
ctx 3500–4000.

**Tracks abandoned / re-scoped**:
- S1 (F16 AR split-launch) and S3 (fused F32→F16 AR+norm+resid):
  both measured non-positive on decode. Single-block fused-norm
  kernel sacrifices the BAR1-per-CU parallelism the multi-block
  `ar_sum_f32` provides on gfx906. Don't reattempt without
  multi-block kernel structure.
- S2 (FFN-side fused mirror): dropped, same regression shape.
- S7 (Q8 KV cache): shipped, context-dependent win on qwen35
  family. Null on gemma4 by Q8 kernel-constraint validation
  (head_dim=512 globals + window=1024 SWA layers excluded).

**Remaining gemma4 levers**: S5 (head_dim=512 kernel tuning — the
direct lever for gemma4's 10 global layers per forward) and S8
(PP handoff coalescing on Hybrid topology).

The residual ~14-19 % is model-spec — larger Q/O projections (head
count + head_dim) + larger FFN. No closure without retrain. Quality
sanity: S7's Q8 KV introduces a small numerical delta; quality cert
(delta-perplexity ≤ 0.02 + chat smoke) must hold per rule 6.

Total: ~2600 LOC across 12 sessions. Each session is independently
shippable (parity test green) and most are independently mergeable.

## Sequencing

Independent tracks that can run in parallel sessions:

- **Track A** (sessions 1-3): S1 → S2 → S3. F16-AR scaffolding
  chain. Ship sessions 1+2 as one PR pair, session 3 follow-up.
- **Track B** (sessions 4-7): S5a → S5b → S5c → S5d.
  head_dim=512 kernel pass. S5a sets the perf floor (PMC + cert);
  S5b is the largest single win.
- **Track C** (sessions 8-10): S7a → S7b → S7c. Q8 KV cache.
  S7a + S7b are no-op-at-runtime landings; S7c ships the actual
  wall-clock delta. **Biggest cumulative win in the plan at long
  context.**
- **Track D** (sessions 11-12): S8a → S8b. Conditional. S8a's
  measurement decides whether S8b ships.

Hard ordering:
- S7b after S7a (kernels need the typestate).
- S7c after S7a + S7b (composite needs both).
- S5b before S5c (decode-batched mirrors decode-single's tile).
- S8 last (handoff refactor is sensitive to per-layer launch
  counts the prior slices change).

Tracks A, B, C can run **in parallel** across sessions — they touch
disjoint files for the most part.

Stop-gates between sessions:
- Each session ships its parity test green before bench claims.
- Each session's wall-clock delta is measured on real
  gemma4-31B-Q4_0 + 26B-A4B-Q8_0 / pp2tp2, not extrapolated.
- Null session = filed honestly + `cfg(unverified)` if structural.
- Diverging > 30 % from the projected Δ tps row triggers re-scope
  before continuing to the next session in the track.

Stop-gates between slices:
- Each slice ships its parity test green before bench claims.
- Each slice's wall-clock delta is measured on real
  gemma4-31B-Q4_0 + 26B-A4B-Q8_0 / pp2tp2, not extrapolated.
- Null slice = filed honestly + `cfg(unverified)` if structural.

## What this plan does NOT chase

- **n_q_heads / n_kv_heads change** — baked into trained Q/K/V
  projection shapes. Each KV head was trained to attend to its
  assigned Q heads; can't fuse/split without retraining. The
  per-byte *cost* of GQA's bigger KV cache is mitigated by S7
  (Q8 KV), but the head-count ratio itself is fixed.
- **head_dim shrink** — same reason. Per-head dimension is fixed
  by training.
- **FFN dim shrink** — same.
- **F32 rmsnorm cascade rollback to F16** — load-bearing for MoE
  numerics per memory `feedback_gemma4_moe_f16_overflow`. Don't
  undo.
- **Qmatmul kernel rewrites for gemma4** — same kernels run both
  archs at the same per-byte efficiency on head_dim ≤ 256; the
  qmatmul gap is in S5 (head_dim=512 kernel tuning), not the
  baseline qmatmul.
- **Separate-V KV layout** — earlier plan revision included this;
  dropped because V is already stored separately (S4 dropped).

## Related memory

- `feedback_gemma4_attn_output_proj_f16_saturate` — F32 widening
  rationale (S1 safety predicate origin).
- `feedback_gemma4_moe_f16_overflow` — F32 rmsnorm cascade
  necessity (S1/S2 don't touch this).
- `feedback_gemma4_v2_dense_31b_incoherent` — BOS prepend fix; not
  perf-related but a reminder gemma4 has had latent correctness
  bugs hiding behind ostensibly-passing smokes.
- `project_gemma4_mixed_batch_partial_2026_05_30` — mixed-batch
  end-to-end on gemma4 (default-on as of 2026-06-01).
- `feedback_kv_q8_not_honored` — `--kv q8` server flag parsed but
  silently ignored on the v2 stack; v1 had Q8Contig typestate and
  attention_decode_q8_kv but they aren't wired into v2. S7 is the
  port.
