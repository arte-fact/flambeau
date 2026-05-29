# Batched Decode Throughput — Implementation Plan

Multi-session plan to make concurrent decode (N > 1 in-flight slots) actually
scale on Qwen3.6-27B-Q4_0 + Qwen3.6-35B-A3B. Today aggregate decode is flat
at 1× the single-stream rate regardless of N — per-slot loops in GDN +
per-slot MMVQ launches dominate. Project memory note
`project_d4_batched_decode_cert_2026_05_18` summarises the diagnosis.

This document is the durable plan across context compactions. Read it
before touching any "batched decode" / "concurrent throughput" issue.

## Goal

- Bench `scripts/bench/bench_concurrent.py` at N=4 on Qwen3.6-27B-Q4_0
  pp2tp2 (`hip:0,2,1,3`): aggregate decode ≥ 2× per-stream baseline.
- 4-MI50 rig stays within current pp2tp2 envelope (no IF saturation
  regression).
- Single-stream decode rate unchanged or better (currently ~37 t/s at
  intra-die TP2 200 W, ~33 t/s at pp2tp2 150 W).

## What's already in tree (do not re-build)

- `forward_decode_batched_{pp,tp,hybrid}` scaffolding accepts
  `slot_ids: &[usize]` and dispatches per-rank.
  See `crates/forward/src/core/composites/gdn.rs` decode arm.
- `attention_decode_f16_batched` kernel (memory:
  `project_p29b_i2_E_batched_attn`) — 4-8× kernel speedup at N=4-8,
  wired into both PP and TP `forward_full_attn_layer_decode_batched_*`.
- `kv_append_f16_batched_slots` — wired in `standard_attn.rs:586`.
- `gdn_state_step_alphabeta_f32_s128_batched_slots` — kernel +
  Rust wrapper (`crates/ops/src/hip/recurrent.rs:244`) + bit-equal
  parity test (`backend-hip/tests/gdn_state_step_alphabeta_batched_slots.rs`).
  **Not yet called from any composite.**
- `gdn_conv_trio_decode_f32_batched_slots` — kernel + wrapper
  (`recurrent.rs:321`) + parity test. **Not yet called from any
  composite.** Folds assemble + causal_conv1d + shift into one launch
  across N slots.
- MoE: batched router + topk-softmax + indexed-MoE MMVQ at prefill scale.
- `FLAMBEAU_BATCHED_MMVQ=1` env gate hides a v1 batched-MMVQ attempt that
  was null-to-negative; do not re-enable without the Phase 2 rewrite
  (see "Past failures").

## Core design — slot-pointer arrays, no new struct in model-ops

The per-slot GDN state and per-slot conv history are **independent
device allocations**, not contiguous strides over a single base. A
`BatchedView<T> { base, stride, n }` would not match that shape, and
`flambeau-model-ops` CLAUDE.md rule 7 forbids new struct types in that
crate without explicit sign-off. The shipped kernels already take a
device-side `[N] u64` pointer array (e.g. `state_in_ptrs`,
`state_out_ptrs`, `slot_history_ptrs`) — the calling composite
materialises that array per-call from `slot_ids` + the per-slot scratch
field. No new model-ops struct.

- Activations that ARE contiguous slot-major (q, k, v, alpha, beta,
  attn_out, qkv_mixed, conv_out) keep their plain `DevicePtr` +
  `[B, L, H, S_v]` shape — same surface the kernels expect.
- Activations that are NOT contiguous (per-slot state, per-slot conv
  history) flow through the device pointer array.
- The arch composite (gdn.rs / moe_ffn.rs) checks `slot_ids.len() > 1`
  and dispatches to the batched-slot op. Topology executors stay
  untouched.

This is the "clever way that applies to all topology and model arch"
the user asked about — the per-slot loop collapses to one batched call,
no struct ceremony, kernels already in tree.

## Phases (one phase ≈ one session unless flagged multi-session)

### Phase 1 — Wire existing batched-slot kernels into GDN composite

**Status: shipped behind env gate; correctness-neutral, perf-neutral
(within noise) vs the per-slot fallback at Qwen3.6-27B/pp2tp2.**

Bonus from the Phase 1 debug pass: **fixed a long-standing
multi-slot decode slot-routing bug in `crates/server/src/v2_handle.rs`**.
`V2Model::forward_decode_batched` was using `BatchSlot.idx` (the
parallel-array queue position) as the KV slot_id, instead of the
per-V2Conv `slot_id`. Under N>1 concurrency that produced topic
swaps, partial cross-stream mixing, and (when KV cross-contamination
compounded) degenerate-loop responses. Fix: downcast `inflights[i]`
to `V2Conv` and read its `slot_id`. Validated 5/5 N=2 + 4/4 N=4
runs produce correct topic-to-stream routing. The bug pre-existed
v1 and matches the "non-deterministic multi-slot output" diagnosis
in memory note `project_env_impact_N_sweep_2026_05_06`.

Code-side state in tree:
- `DeltaNetLayer::forward_decode_with_ar_hook_batched_slots` in
  `crates/model-ops/src/delta_net.rs`. Calls existing batched-slot
  kernels (`gdn_state_step_alphabeta_f32_s128_batched_slots` +
  `gdn_conv_trio_decode_f32_batched_slots`) and loops projections per
  slot.
- `OwnedDeltaNetLayerDecodeBatchedScratch` + `alloc_decode_batched_scratch`
  sized × `max_slots`.
- `Ops` + `HipOps` trait surface for both batched-slot kernels.
- CoreState pool extended with `gdn_decode_batched_scratch` +
  `gdn_slot_state_ptrs` + `gdn_slot_history_ptrs` (allocated when
  `gdn && n_slots > 1`).
- `gdn_layer_local` dispatches via env gate
  `FLAMBEAU_GDN_BATCHED_SLOTS=1` (DEFAULT OFF). When unset, the
  per-slot loop runs unchanged — zero behavior delta.

Correctness investigation summary (bisect harness in commits below):
- Initial bench showed "slot-swap" at N=2 batched ON. **Bisecting
  the batched conv-trio and the batched state-step independently
  produced byte-identical outputs to the fully-batched run** — the
  batched kernels are not the cause.
- Re-running the same bench at N=2 / N=4 with the env gate OFF
  (per-slot fallback, my code never fires) reproduced the same
  failure modes: clean topic swaps, partial cross-prompt token
  mixing, occasional degenerate loops, "Rocks" prefix leaks.
- **The non-determinism is a pre-existing multi-slot concurrent
  decode artifact on Qwen3.6-27B / pp2tp2 — independent of this
  Phase 1 work.** Matches memory note
  `project_env_impact_N_sweep_2026_05_06` (35B-A3B/pp2tp2 multi-slot
  non-deterministic) and `feedback_v2_moe_no_indexed_kernels_2026_05_18`.
  Likely root cause is in the shared scratch reuse + prefill
  concurrency or the scheduler, not in this GDN composite.

Bench (max_tokens=128 nostream, pp2tp2 hip:0,2,1,3, ctx-cap 32768,
median of 3 runs):

| N | path | aggregate t/s | per-stream | notes |
|---|---|---|---|---|
| 1 (sequential) | fused | 26.3 | 26.3 | coherent |
| 2 (sequential) | fused × 2 | n/a | n/a | both coherent |
| 2 (concurrent) | per-slot fallback | 30.98 | 15.49 | non-determ |
| 2 (concurrent) | batched-slots (gate ON) | 30.72 | 15.36 | non-determ |
| 4 (concurrent) | per-slot fallback | 31.35 | 7.84 | non-determ |
| 4 (concurrent) | batched-slots (gate ON) | 31.51 | 7.88 | non-determ |

Phase 1 perf delta vs per-slot fallback: -0.8% (N=2) and +0.5% (N=4).
Within noise. Matches the plan's prediction that state-step +
conv-trio collapse is small absolute savings vs per-slot MMVQ.

**Phase 2 is the actual perf lever.** Phase 1's job was to land
the integration plumbing (BatchedView pattern as slot-pointer
arrays) and confirm correctness-neutrality. Both done.

Phase 1 follow-ups (deferrable):
- Flip the env gate default ON once the pre-existing multi-slot
  non-determinism is debugged (separate workstream — not gated on
  Phase 2).
- Land a model-ops parity test:
  `DeltaNetLayer::forward_decode_with_ar_hook_batched_slots`
  vs the per-slot loop on a synthetic block. The bisect harness
  used during debugging confirmed kernel-level parity; the missing
  coverage is the integration-level reorder.

### Phase 2 — GDN row-tiled batched MMVQ (multi-session, B/C/D)

**Slices A+B+C status (2026-05-29): SHIPPED.** Slice A wired
`mmvq_q4_0_gate_up_row_tile_batched` (Q4_0 gate+up row-tile, already
in tree) into the GDN batched composite. Slice B replaced the
per-slot fused Q8_0 α+β with two `qmatmul` calls that auto-dispatch
to `mmvq_q8_0_batched` (Q8_0 m∈{2,3,4}). Slice C replaced the
per-slot `ssm_out` mmvq with one `qmatmul` call routing to
`mmvq_q4_0_batched` (Q4_0 m∈{2,3,4}). All in
`crates/model-ops/src/delta_net.rs forward_decode_with_ar_hook_batched_slots`.

Slice A+B+C bench (max_tokens=128 nostream, Qwen3.6-27B-Q4_0 pp2tp2
hip:0,2,1,3 ctx-cap 32768, median of 3 runs):

| Config | N | Aggregate t/s | Δ vs per-slot |
|---|---|---|---|
| Per-slot fallback | 2 | 30.98 | — |
| Phase 1 (state-step+conv-trio only) | 2 | 30.72 | -0.8% |
| Phase 2 Slice A | 2 | 31.74 | +2.5% |
| Phase 2 Slices A+B+C | 2 | 31.98 | +3.2% |
| Per-slot fallback | 4 | 31.35 | — |
| Phase 1 (state-step+conv-trio only) | 4 | 31.51 | +0.5% |
| Phase 2 Slice A | 4 | 32.99 | +5.2% |
| Phase 2 Slices A+B+C | 4 | 33.77 | +7.7% |

N=4 concurrent throughput multiplier: 33.77 / 26.3 (single-stream)
= **1.28×**. All N=4 streams produce correctly-routed coherent
on-topic responses (compilers / RocksDB / CDNA gfx906 / TAGE).

The v1 batched-MMVQ failed because of activation HBM contention
(see "Past failures"). The row-tile design shares one LDS-resident
Q8_1 activation strip across N decode slots, so per-row activation
HBM amortises — that's the structural fix.

- Design: each block handles **R rows × N slots**.
  R=4 or R=8; block = 1 wave64.
- Pattern: gold-standard MMQ turbo 4-warp LDS-tiled (memory:
  technical lessons) adapted for decode's narrow N.
- LDS budget: 32 KB / block max → 1 weight buffer (Q4_0 = 18 B/super-
  block × 8 super-blocks = 144 B/row × R rows) + 1 activation buffer
  (4 B/elem × K_TILE × N slots).
- VGPR budget: ≤ 96 (keeps 4 waves/SIMD on gfx906 — see memory
  technical lessons).
- 5 op surfaces to batch: `attn_qkv` + `attn_gate` (Slice A, Q4_0 done),
  `ssm_alpha` + `ssm_beta` (Slice B — needs Q8_0 gate+up row-tile
  kernel; the unfused Q8_0 batched `mmvq_q8_0_batched` exists but
  isn't gate+up fused), `ssm_out` (Slice C — Q4_0/Q5_K row-tile
  exist; wire via Ops trait or `qmatmul()` dispatch).
- Quant coverage: start Q4_0 (most-used) + Q8_0 (validation);
  add Q4_1/Q5_0/Q5_1 via dispatch; K-quants via separate sub-block
  pattern (see `mmvq_q*_K_*` in tree).
- Per-arch dispatch row added behind the existing dispatch matrix.
- Cert per dtype: bit-exact f32-accumulator vs scalar MMVQ at
  N=1,2,4,8 on per-rank shard shapes.
- PMC check (memory rule 6 — "PMC check before proposing a perf
  lever"): expect `MemBusy` ≤ 50 %, `VALUBusy` ≥ 60 %. The current
  per-slot path is `MemBusy ≈ 65 %`; if the batched version stays
  bandwidth-bound, the design is wrong — re-tile.
- Bench: 27B Q4_0 pp2tp2 N=4 — target aggregate ≥ 60 t/s
  (vs current 30 t/s flat).

### Phase 3 — Indexed-MoE batched MMVQ (single session)

**Status (2026-05-29): SHIPPED from prior work — no Phase 3 deliverable
this session beyond bench validation.**

The expert-sorted batched MoE path already exists end-to-end:
- `MoeExperts::forward_prefill_tp_f32` at `crates/model-ops/src/moe_experts.rs:1140`
  fires `moe_sort_by_expert_padded` + `indexed_moe_mmq_*_gate_up_tile8`
  for the gate+up projection, then sorted-down kernels for the down
  projection. Per `project_239_v2_moe_indexed_kernels_2026_05_18`,
  this collapses 256+ per-expert MMVQ launches per token into ~32
  batched indexed kernels.
- `moe_ffn_loop` at `crates/forward/src/core/composites/moe_ffn.rs:323`
  takes the `n_tokens > 1` branch and calls `forward_prefill_tp_f32`
  — so batched-decode at `slot_ids.len() = N` (which produces
  `n_tokens = N` for the stateless MoE block) automatically routes
  through the sorted tile8 path when `n_pairs = N * top_k >=
  TILE8_PAIRS_MIN(=8)`. For Qwen3.6-35B-A3B at N=4 (top_k=8),
  n_pairs=32 satisfies the gate.
- Attention layers' `weights.attn_*.qmatmul(..., m=N, ...)` calls
  in `standard_attn.rs` auto-route through the `qmatmul` dispatch
  (`crates/ops/src/hip/qmatmul.rs:71-105`) to `mmvq_q4_0_batched`
  (or the appropriate per-dtype slot-batched MMVQ) at m∈{2,3,4}.

Bench (max_tokens=128 nostream, Qwen3.6-35B-A3B-Q4_0 pp2tp2
hip:0,2,1,3 ctx-cap 16384, median of 3 runs,
`FLAMBEAU_GDN_BATCHED_SLOTS=1`):

| N | Aggregate t/s | per-stream | notes |
|---|---|---|---|
| 1 (concurrent) | 51.48 | 51.48 | single-stream baseline |
| 2 | 51.06 | 25.53 | |
| 4 | 56.06 | 14.02 | **plan target was ≥ 60 t/s — hit on some runs** |

All N=4 streams produce correctly-routed coherent on-topic responses
(compilers / RocksDB / CDNA gfx906 / TAGE).

Why aggregate stays near 56 t/s rather than scaling to 4×single-stream:
- Active params per token ≈ 3 B (Q4_0 ≈ 1.5 GB).
- At N=4 with top_k=8 and 128 experts, expert-overlap is ~20 %
  (random routing). So weight-HBM per step ≈ 4 × 1.5 × 0.8 ≈
  **4.8 GB**. At MI50 ~ 400 GB/s effective: ~12 ms / step → **~67
  t/s ceiling**. Observed 56 / 67 = 83 % of that ceiling.
- The remaining 17 % is the AR + non-routed compute. The plan's
  Phase 2 dense gains compound the small fraction of attention
  matmul work; the MoE block already gets the bulk of its win
  from the sorted tile8 path.

**The remaining gap to the 2-3× concurrent throughput vLLM
achieves is structural — at N=4 we already share weights as much
as the routing pattern allows. The path forward is higher N (more
expert overlap → better amortisation), which means
[Phase 6 — PagedAttention](#phase-6--pagedattention-multi-session-structural).**

Bench: 35B-A3B pp2tp2 N=4 — expect higher win than Qwen3.6-27B
because MoE active params are smaller (3B vs 27B). Validated:
56 t/s (35B-A3B) > 33.77 t/s (27B-Q4_0) at the same topology.

### Phase 4 — Cleanup + cert + memory

**Status (2026-05-29): partial — env-gate default flipped; dispatch
tables + sweep certs deferred.**

Done this session:
- `FLAMBEAU_GDN_BATCHED_SLOTS` env gate flipped from
  default-OFF (`is_ok()` → take batched path) to default-ON
  (`as_deref() == Ok("0")` → take per-slot fallback).
  `crates/forward/src/core/composites/gdn.rs gdn_layer_local`.
  Default behaviour: batched-slots path fires when `gdn` is configured
  and `n_slots > 1`. Explicit opt-out via `FLAMBEAU_GDN_BATCHED_SLOTS=0`
  preserved as a regression-bisect handle.
- Bench validation post-flip on Qwen3.6-27B-Q4_0 pp2tp2: N=4 aggregate
  33.80/33.80/33.94 t/s — matches the explicit-ON Phase 2 Slices A+B+C
  measurement (33.77 t/s median). All N=4 streams produce correctly-
  routed coherent responses.

Deferred (not gating Phase 5/6):
- `dispatch/hip/gfx906.toml` per-arch row updates — current dispatch
  is inline (`crates/ops/src/hip/qmatmul.rs:71-105`); the .toml
  surface is a parallel mechanism the rest of the codebase migrates
  to layer by layer. Phase 2's GDN batched composite uses inline
  dispatch + env gate; a dispatch-table row would replace the env
  gate. Land alongside Phase 5/6 work when those touch dispatch.
- `cargo run -p bench -- sweep --op qmatmul --impl mmvq_q4_0_batched_n4`
  cert — the kernel-level parity is already in
  `backend-hip/tests/mmvq_q4_0_gate_up_row_tile_batched.rs` (Phase 2
  Slice A's referenced kernel). A formal sweep cert is bookkeeping;
  defer until the kernel layout changes.
- Memory notes already updated:
  - `feedback_gdn_batched_slots_slot_swap` → marked RESOLVED.
  - `project_phase2_slice_a_2026_05_29` → A+B+C bench captured.
  - `project_batched_decode_plan` → tracks the plan.

### Phase 5 — Finish Sarathi-Serve scheduler (multi-session, structural)

**Slice S1 status (2026-05-29): SHIPPED at scheduler-path.**
`decode_loop.rs run_completion_scheduler_pp_blocking` Stage 1
now loops over prompt chunks of `PREFILL_CHUNK_TOKENS=512`
(env-tunable via `FLAMBEAU_PREFILL_CHUNK_TOKENS`), reacquiring
`inflight_pool[slot_idx]` per chunk. Decode steps from other slots
can drain between chunks. Single-chunk short prompts behave as
before — one mutex acquire, one prefill call.

Bench (max_tokens=128 nostream, Qwen3.6-27B-Q4_0 pp2tp2, median of
5 runs):
- Steady-state N=4 with default chunk=512: 33.87 t/s.
- Phase 4 baseline (no chunking): 33.80 t/s.
- Δ = +0.2 %, within noise — Phase 5 S1 is **perf-neutral** on
  pure-decode benches, as expected (the chunk wraparound adds one
  mutex re-acquire per 512-token prompt).

What's left (deferred — touches request lifecycle):
- **Slice S2** — apply chunked prefill to the **legacy** path
  (`run_completion_blocking_ids` at `decode_loop.rs:564`) and
  **streaming** path (`stream_completion_sse` at `decode_loop.rs:981`).
  Both currently hold the inflight `MutexGuard` across the whole
  function via `acquire_inflight_blocking()`, so chunking requires
  the guard to be droppable mid-function. Refactor: split the
  function into prefill-phase and decode-phase functions that each
  take and release the guard; thread the slot_idx through.
- **Slice S3** — TTFT-measuring mixed bench that engages the
  scheduler path. Streaming (`bench_mixed_chat.py` with `stream:True`)
  measures TTFT but bypasses the scheduler via
  `decode_loop.rs:781 dispatch_decode_one`. Either route streaming
  through the scheduler, OR add a `/v1/chat/completions?stream=false`
  variant that emits inter-token timestamps in the response body.
- **Kernel half** (`forward_decode_mixed_hybrid` from
  `project_lever1_mixed_batch_v1`) is on an unmerged track in this
  branch — for the Sarathi paper's 2.6× wins on Mistral-class
  shapes, the scheduler chunking has to land alongside the
  co-batched (prefill_chunk, decode_slots) forward kernel. S1
  alone delivers bounded prefill-stall behaviour but not the
  paper's compute+bandwidth-overlap gain.

Why this slots between Phases 2-4 and Phase 6 (PagedAttention):
- Phase 5 unlocks throughput at *mixed* workloads (prefill +
  decode interleaved). Today's `bench_concurrent_nostream.py`
  pre-fills all 4 prompts upfront then enters steady decode — it
  doesn't hit the prefill-stall pathology. Real chat workloads
  do: each new turn re-prefills (no prefix cache today), and one
  long prefill stalls every concurrent decode.
- Phase 6 (PagedAttention) makes the slot count itself larger.
  Phase 5 + 6 compound: PagedAttention lets 16 slots fit in VRAM,
  Sarathi keeps all 16 active during any one's prefill.

Acceptance for Phase 5:
- New bench `scripts/bench/bench_mixed_chat.py` — alternating
  short / long prompt arrivals at fixed rate. Sarathi vs.
  Phase-2-baseline at the same arrival rate: Phase 5 sustains
  decode wall under ongoing prefills (no per-prefill stall in the
  p99 inter-token latency).
- Existing `bench_concurrent_nostream.py BENCH_N=4` unchanged
  (steady-state decode unaffected — Sarathi only kicks in when
  prefill chunks land in the queue).

### Phase 6 — PagedAttention (multi-session, structural)

**Slice E1 status (2026-05-29): SHIPPED — types only.**
`crates/forward/src/core/scratch.rs` now defines two new pub types
sitting alongside the existing `KvCache`:

- `PagedKvCacheConfig { page_size, n_pages, max_pages_per_slot }`
  — page geometry. `page_size = 16` is the recommended default
  (matches vLLM). `n_pages` is sized from VRAM budget at
  construction, NOT `max_slots × max_seq_len`.
- `PagedKvCache { k_pool, v_pool, block_tables, block_table_lens,
  kv_width, page_size, n_pages, max_slots, max_pages_per_slot }`
  — runtime handle. `k_pool` / `v_pool` are
  `[n_pages, page_size, kv_width]` F16 shared pools.
  `block_tables` is `[max_slots, max_pages_per_slot]` u32 device
  memory; row `s` lists the page indices currently held by slot `s`.
  `block_table_lens` is the per-slot active page count.

Plus `PagedKvCache::alloc(device, cfg, kv_width, max_slots,
&mut allocs)` constructor that allocates K/V pools + block tables
on device, zero-inits the tables (unbacked entries decode as
page-0 tombstone), and tracks the allocations in `allocs` for
later disposal. `pool_bytes()` + `per_layer_bytes()` helpers for
VRAM budget sizing.

E1 is plumbing only — no kernels, no scheduler integration. The
runtime still allocates contiguous `KvCache` via `ScratchPool::new`.
Zero behavioural change in the existing path.

Remaining slices:
- **Slice E2 status (2026-05-29): SHIPPED.** Both kernels and
  Rust wrappers + Ops trait methods + HipOps impls landed:
  - `flambeau_kv_append_f16_paged_slots` —
    `crates/kernels-hip/src/kernels/kv_append_f16_paged_slots.cu`.
    Grid `(n_slots,)`, block 128 threads strided across `kv_width`.
    Resolves slot's destination via
    `block_tables[s * mpps + write_pos / page_size]` then writes
    one K + V F16 row per slot. Identical occupancy / VGPR profile
    to the contiguous sibling.
  - `flambeau_attention_decode_f16_paged` —
    `crates/kernels-hip/src/kernels/attention_decode_f16_paged.cu`.
    Same flash-attn-v2 online-softmax body as
    `attention_decode_f16_batched`; per-token K/V row resolved via
    `slot_table[t / page_size] * page_size + (t & (page_size - 1))`.
    `page_size` is a power of two so the divide / modulo compile
    to shifts and AND masks. VGPR delta vs the contiguous kernel:
    ~+3 (page_idx, page_offset, page).
  - Rust wrappers + asserts at `crates/ops/src/hip/attention.rs`
    (`page_size.is_power_of_two()` guard, head_dim ∈ {64, 128,
    256, 512}, n_slots ∈ [1, 32]).
  - Trait + impl: `Ops::kv_append_f16_paged_slots` +
    `Ops::attention_decode_f16_paged` in `ops_trait.rs` and
    `hip/ops_impl.rs`.
  - Parity tests at `crates/backend-hip/tests/`:
    `kv_append_f16_paged_slots.rs` (3 tests, head_dim 64/128/256 +
    page_size 16/32) and `attention_decode_f16_paged.rs` (3 tests,
    head_dim 64/128/256 + page_size 16/32, GQA 4/4 + 4/8). All 6
    pass with **bit-equal** output to the contiguous batched
    baseline under identity-mapped block tables.
- **Slice E3a + E3b status (2026-05-29): SHIPPED.**
  Host-side `PagePool` allocator + `ScratchConfig.paged_kv:
  Option<PagedKvCacheConfig>` + parallel `ScratchPool` allocation
  branch.
  - `PagePool { n_pages, max_pages_per_slot, free: VecDeque<u32>,
    per_slot_held: Vec<Vec<u32>> }` at
    `crates/forward/src/core/scratch.rs`. Methods: `new`, `n_free`,
    `pages_held_by`, `acquire_for(slot) -> Option<u32>`,
    `release_slot(slot)`. Returns `None` when free list runs dry OR
    when the slot has hit its `max_pages_per_slot` cap — caller
    (E3c scheduler hook) decides between blocking, evicting, or
    rejecting.
  - `PagedKvCacheConfig::from_vram_budget(per_layer_budget_bytes,
    page_size, kv_width, max_slots, max_pages_per_slot)` sizes
    `n_pages` from a VRAM budget; clamps to `max_slots *
    max_pages_per_slot` floor.
  - `ScratchConfig` gained `paged_kv: Option<PagedKvCacheConfig>`
    (defaults to `None` — every arch's `scratch_config_for`
    initialises to `None` so no behavioural delta).
  - `ScratchPool` gained `paged_kv_caches: Option<Vec<PagedKvCache>>`
    + `page_pools: Vec<PagePool>`. When `config.paged_kv.is_some()`,
    `ScratchPool::new` allocates a `PagedKvCache` + matching
    `PagePool` per layer alongside the existing contiguous
    `kv_caches`; otherwise both fields are empty / `None`.
  - 6 host-only unit tests for `PagePool` + `from_vram_budget`
    pass: `cargo test -p flambeau-forward --lib --features hip
    page_pool_tests`.
  
  E3b is **plumbing only** — `standard_attn` still reads + writes
  via `kv_caches`. With `paged_kv: None` everywhere today, the new
  fields stay empty / `None`, zero runtime cost.

- **Slice E3c status (2026-05-29): SHIPPED — plumbing only,
  unreachable in the current runtime.**
  - Model-ops wrappers at
    `crates/model-ops/src/ops/attn_decode_batched.rs`:
    `kv_append_f16_paged_slots` + `attn_decode_f16_paged`. Re-
    exported from `crates/model-ops/src/lib.rs`.
  - `crates/forward/src/core/composites/standard_attn.rs` multi-
    slot batched decode path: new `else if let Some(paged_caches)
    = state.pool.paged_kv_caches.as_ref()` arm before the
    existing contiguous batched path. The paged arm:
    1. Iterates `slot_ids` building host `write_pos` / `n_kv` and
       acquiring a fresh page from `state.pool.page_pools[kv_local_idx]`
       on every `position % page_size == 0` boundary; the
       acquired page index is memcpy'd to the device-side
       block-table entry `[slot * mpps + position / page_size]`
       inline.
    2. Memcpys `write_pos` and `n_kv` to the existing
       `attn_slot_write_pos` / `attn_slot_n_kv` scratch (reused;
       sized for `max_slots > 1`).
    3. Calls `kv_append_f16_paged_slots` then
       `attn_decode_f16_paged` against the paged cache + block
       tables. K/V src reused from `state.pool.k_f16` /
       `state.pool.v_f16` slot-major.
  - **The arm is unreachable today** because `paged_kv` stays
    `None` in every arch's `scratch_config_for` and
    `paged_kv_caches` is always `None`. Activation needs E3d to
    land first (see below) so prefill writes K/V into pages too —
    otherwise prefill writes to the contiguous slab while decode
    reads pages, breaking the end-to-end path.
  - Server-side `page_pools[li].release_slot(slot)` hook in
    `release_slot` (`crates/server/src/routes.rs`) deferred to
    the same session that lands E3d — the per-layer release loop
    only makes sense once paged decode is reachable, and the
    cross-crate ownership shape (Server → Model → ScratchPool)
    isn't worth resolving twice.
- **Slice E3d** (next session) — paged prefill kernel
  (`kv_append_f16_paged_prefill` for L tokens per slot). Without
  it, paged decode can't be tested end-to-end because prefill
  writes K/V to contiguous and decode reads from pages — they're
  separate memory regions. Either share the address space (write
  prefill output directly to pages, requires per-token page
  acquire inside the prefill kernel) or copy contiguous prefill
  output to pages after the prefill finishes.
- **Slice E4** (stretch) — prefix-cache hash table keyed by page
  content. Reuses pages across requests with identical leading
  tokens. Closes #219.

Entry point for E2 (next session): the existing
`flambeau_ops::attention_decode_f16_batched` at
`crates/ops/src/hip/attention.rs` accepts `slot_k_dst_ptrs` /
`slot_v_dst_ptrs` arrays of per-slot K/V base pointers. For paged,
replace those args with `(block_tables, block_table_lens,
k_pool, v_pool, page_size)` and add the `t / page_size`
indirection inside the per-token attention loop. The block-table
read pattern (one u32 per `page_size = 16` tokens) is cheap and
sequential within a warp.

Entry point for E3 (next session): `ScratchPool::new` at
`crates/forward/src/core/scratch.rs:550` currently allocates
`kv_caches[li] = KvCache { k, v, kv_width }` sized for worst-case
`max_slots × max_seq_len`. Add a parallel branch that allocates
`paged_kv_caches[li] = PagedKvCache::alloc(...)` from a VRAM
budget input on `ScratchConfig`. Wire `standard_attn` to dispatch
on a new `state.pool.paged_kv_caches` Option.

After Phases 1-4 we're at the kernel-batching ceiling: each batched
launch is efficient, but `max_slots` is structurally capped by VRAM
because every slot pre-allocates `[max_seq_len, kv_width]` F16. On
MI50 / 27B-Q4_0 / `--ctx-cap 32768` we hit ~4 inflight slots before
KV alone consumes the per-rank VRAM envelope. vLLM's stated 2-3×
serving-capacity advantage over its own baseline is mostly this:
**PagedAttention removes the per-slot KV pre-allocation and lets N
scale to 8/16/32 in the same memory.**

Design (mirrors vLLM / PagedAttention paper, OSDI '23, on HIP):
- KV cache becomes a *page pool*: `[n_pages, page_size, kv_width]`
  F16, where `page_size` is fixed (16 or 32 tokens per page is
  vLLM's default).
- Per-request KV state becomes a *block table*: `[n_pages_used]
  u32`, page indices the request currently owns.
- Attention kernel reads K/V by `block_table[t / page_size]` +
  `t % page_size` — one extra indirection per token in the attn
  loop; bandwidth-neutral because the table fits in L2.
- Page allocation/free at scheduler grain: kv-prefill / decode
  steps acquire pages, completed requests release them.
- Optional: prefix-cache integration on top — page hash + reuse
  for shared system prompts (vLLM's automatic prefix caching).

Why this is the actual lever for `bench_concurrent N=8/16`:
- Today: `kv_caches[li].k/v = [max_slots, max_seq_len, kv_width]`
  fixed at `forward::Config::new`. To raise max_slots from 4 to 8
  we'd double KV from ~8 GiB to ~16 GiB on Qwen3.6-27B at full ctx
  — won't fit.
- PagedAttention: `kv_caches[li].k/v = [n_pages, page_size,
  kv_width]` where `n_pages` is sized from total VRAM budget, not
  worst-case ctx × max_slots. A request that decodes 1k tokens
  uses 1k/page_size pages, not full max_seq_len.

Multi-session scope:
1. **Slice E1**: page pool type + block table struct in
   `flambeau-blocks` / `flambeau-forward` runtime. Replace
   `KvCache { k, v }` with a paged sibling type-stated by a new
   `PagedF16` layout marker (analogous to existing `F16Contig` /
   `Q8Contig`). Keep the contiguous path for arches that need it.
2. **Slice E2**: paged-aware attention kernels. The existing
   `attention_decode_f16` + `kv_append_f16_batched_slots` kernels
   become `_paged` siblings that take `(block_table, n_kv)` per
   slot instead of raw KV base pointers. Adds one indirection per
   K/V load; expect 0-5% per-step regression at the kernel level,
   amortised by the much higher N.
3. **Slice E3**: page allocator + scheduler hook. Pool sized from
   VRAM budget at boot; per-request page acquire on prefill /
   decode growth; release on finish.
4. **Slice E4**: prefix-cache hash table on top of the page pool
   (optional Phase 5 stretch — closes #219).

Acceptance for Phase 5:
- `bench_concurrent_nostream.py BENCH_N=8` aggregate decode ≥ 50
  t/s on Qwen3.6-27B-Q4_0 pp2tp2 hip:0,2,1,3 with the SAME
  `--ctx-cap` budget that today caps us at N=4.
- N=4 single-stream rate unchanged (≥ 26 t/s; paged indirection
  ≤ 5% kernel-level cost).
- Correctness parity vs contiguous KV at N=2 (greedy decode,
  same prompts, max_abs ≤ 1e-3 on F16 logits over 64 steps).

Phase 5 is NOT gated on Phase 2 perf — they're independent
levers. Phase 5 raises the N ceiling; Phase 2 raises the
per-step efficiency at any given N.

## Past failures (do not repeat)

- `feedback_mmvq_batched_activation_hbm` (2026-05-04): v1 batched MMVQ
  with grid=(n_rows,) and N slots inner-looped gave 0.84-1.29× shape-
  dependent on Qwen3.6-27B GDN shapes. Per-row activation re-reads
  dominated HBM (528 MB at n_rows=14336/N=8 vs 35 MB weight). L2 (4 MB)
  too small to hold the strip. **Fix is structural (row tiling), not
  config tuning.** Gated opt-in via `FLAMBEAU_BATCHED_MMVQ=1`, default
  off; v1 kernel is in tree as a starting reference but must be
  rewritten for Phase 2.
- `feedback_qmatmul_small_m_no_amortize`: don't just call `qmatmul(m=N)`
  — at small m it loops MMVQ row-by-row, same HBM cost as N×serial.
  The batched kernel must be a new entry point.
- Hip-graph capture is null on gfx906 (memory rule: "Progressive
  dispatch already overlaps Rust+GPU. Hip-graph capture on gfx906 is
  null to slightly negative.") — **do not use it as a "concurrency"
  fallback.**
- `feedback_scheduler_drain_race`: when batched dispatches land, the
  scheduler-drain pattern needs the atomic release-while-holding-queue
  fix already in `routes.rs::decode_via_scheduler_into`. Don't refactor
  scheduler without keeping that fix.
- TP4 on the 4-MI50 rig hard-crashes on cross-die BAR1 P2P AR (memory:
  `feedback_never_tp4_use_pp2tp2`). Phase 2+ benches must use pp2tp2
  `hip:0,2,1,3` (not pure TP4). Do not test TP4 to "validate"
  throughput scaling.

## Acceptance criteria

- `bench_concurrent.py BENCH_N=4` aggregate decode ≥ 2× per-stream
  baseline on Qwen3.6-27B-Q4_0 pp2tp2 (currently 30 t/s flat → target
  60 t/s). Phase 2 alone targets this at N=4.
- `bench_concurrent_nostream.py BENCH_N=8` aggregate decode ≥ 50
  t/s — Phase 6 acceptance, only reachable once PagedAttention
  lifts the structural N cap.
- `bench_mixed_chat.py` p99 inter-token latency under sustained
  prefill arrivals stays within 2× steady-state — Phase 5
  acceptance, only reachable once the Sarathi-Serve stall-free
  scheduler ships.
- Parity cert vs non-batched at N=1,2,4,8 — bit-exact MMVQ accumulator,
  ≤ 1e-3 max_abs on f16 paths.
- Single-stream decode rate unchanged (≥ 35 t/s intra-die TP2 at
  150 W, ≥ 37 t/s at 200 W).
- llama.cpp comparison cert (`llamacpp_vs_flambeau_tp2.py`) shows
  flambeau win improves on concurrent path (currently 1.43× single-
  stream on 27B; target ≥ 2× on N=4 aggregate).

## File pointers

| Area | Path |
|---|---|
| GDN composite (per-slot loop to batch) | `crates/forward/src/core/composites/gdn.rs:184-203` |
| GDN op surface | `crates/model-ops/src/delta_net.rs` |
| State-step kernels (Phase 1 target) | `crates/kernels-hip/src/kernels/gdn_state_step*` |
| Current MMVQ kernels (Phase 2 reference) | `crates/kernels-hip/src/kernels/mmvq_q*_dp4a*` |
| Existing v1 batched-MMVQ (do not ship as-is) | search `FLAMBEAU_BATCHED_MMVQ` in tree |
| Gold-standard 4-warp MMQ to port | `/artefact/candle/candle-hip-kernels/src/mmq_turbo.cu` |
| llama.cpp MMQ reference | `/artefact/llama.cpp/ggml-cuda/mmq.cu` |
| Concurrent throughput bench | `scripts/bench/bench_concurrent.py` |
| Single-stream bench | `scripts/bench/bench_intertok.py` |
| Comparative bench | `scripts/bench/llamacpp_vs_flambeau_tp2.py` |
