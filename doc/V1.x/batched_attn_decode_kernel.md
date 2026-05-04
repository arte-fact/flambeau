# Batched-attention decode kernel — design (P2.9b-i2-E / #266)

**Goal:** replace the per-slot attention loop in
`forward_full_attn_layer_decode_batched_tp` (and the PP twin) with a single
kernel launch that handles N (Q row, per-slot KV) pairs in one grid. The
per-slot loop is the structural ceiling that pinned #267 hybrid throughput at
0.98×; lifting it is the precondition for the ≥ 3× target.

## Status quo (post-#275)

`crates/models/qwen3-moe/src/forward/attn_tp.rs` lines 1007-1026:

```rust
for s in 0..n_tokens {
    let q_row   = scratch.q_f16.offset_bytes(s * q_per_token_bytes);
    let out_row = scratch.attn_out_f16.offset_bytes(s * q_per_token_bytes);
    let kv      = &slot_kv_caches[s];
    let n_k_tokens = kv.current_tokens();
    attention_decode_f16_slots(
        ops, stream,
        q_row, kv.k_buffer(), kv.v_buffer(), out_row,
        local_n_heads, local_n_kv_heads, head_dim,
        n_k_tokens, scale, None,
    )?;
}
```

`attention_decode_f16` (the underlying kernel) is launched as
`grid=(n_heads_q,)` `block=(head_dim,)`. For Qwen3.6-27B / pp2tp2:
`local_n_heads = 32 / 2 = 16`, `head_dim = 128`. Each launch uses
**16 blocks × 2 wave64 warps = 32 wavefronts** per slot. MI50 has 60 CUs ×
~4 waves/CU resident ≈ 240 wave slots. So a single slot already underfills the
device (~13 % occupancy ceiling); N slots launched serially get N times the
launch overhead with no parallelism gain — exactly the observed 1.0× scaling.

## Design

### Kernel signature

```c
extern "C" __global__ void flambeau_attention_decode_f16_batched(
    const half* __restrict__ q_batched,        // [N, n_heads_q, head_dim]
    const uint64_t* __restrict__ k_cache_ptrs, // [N] device-side pointers, one per slot
    const uint64_t* __restrict__ v_cache_ptrs, // [N] device-side pointers, one per slot
    half* __restrict__ out_batched,            // [N, n_heads_q, head_dim]
    const int* __restrict__ n_tokens_kv,       // [N] per-slot KV-tail (post-append)
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int n_slots,                               // == N
    float scale
);
```

Per-slot KV pointer/length tables are uploaded once per `forward_*_batched`
call: each `slot_kv_caches[s]` already has its own contiguous K/V allocation
on the relevant device. Shipping the pointer table HtoD once costs `N × 16 B`
(two u64s per slot) — negligible vs the saved per-slot launch overhead
(~ 30-50 µs per kernel launch on ROCm).

### Tile / block layout (gfx906)

- `gridDim  = (n_heads_q, n_slots)`
- `blockDim = (head_dim,)`
- One block owns `(slot, q_head)`. Each block is exactly the same shape as
  `flambeau_attention_decode_f16` today — no change to LDS / VGPR budget.

For Qwen3.6-27B / pp2tp2 / N=4 / `local_n_heads = 16` / `head_dim = 128`:
- single-slot today: 16 blocks × 32 waves resident
- batched N=4: **64 blocks** × 32 waves resident → ~53 % of MI50's 240 wave
  slots. Crosses the SIMD-occupancy threshold where memory latency stops
  dominating. Projected speedup vs N×serial: **2.5–3×** at N=4 (literature
  bound; see vLLM PagedAttention paper §4.3).

For N≥8 we saturate the device; further N gains return through the next
ceiling (HBM bandwidth) rather than launch overhead.

### Split-K (flash-decoding) interaction

The existing `attention_decode_f16_splitk` partitions context across `grid.y`
to attack the single-slot-occupancy starvation at long-context decode (the
27 B head_dim=128 case is borderline; 35 B-A3B head_dim=256 is the strong
case). When N>1 covers the same occupancy gap, split-K is no longer a
strict win — the two strategies compete for the same `grid.y` slot.

**Decision:** use a `gridDim = (n_heads_q, n_slots)` baseline kernel for
N≥2. When `N × n_heads_q < 2 × CU_count`, fall back to
`gridDim = (n_heads_q, n_slots * n_chunks)` with `n_chunks` chosen to fill
the device — i.e., split-K on top of batching. Dispatch decision lives in
the ops-layer wrapper, not the kernel, mirroring the existing
`attention_decode_f16_splitk` chunk-size selector.

### F16 KV only (V1)

V1 implements F16-KV only. Q8-KV (`attention_decode_q8_kv`) gets a sibling
implementation in V1.x once F16 lands and certs green. F16 covers the
35 B-A3B-Q4_0 + 27 B-Q4_1 prod targets.

### VGPR / LDS budget

Reusing the `attention_decode_f16` kernel body verbatim per (slot, q_head)
block:
- VGPR per thread: ~32 (matches existing kernel — same ALU budget)
- LDS per block: 2064 B (same as existing)
- Resident waves: 4 waves/SIMD (same as existing single-slot kernel)

No new VGPR pressure. The hsaco grows by ~10 % from the extra slot-index
arithmetic; not a constraint.

## Cert plan

Correctness gate (`crates/kernels-hip/tests/attention_decode_f16_batched.rs`,
new test):
- N ∈ {1, 2, 4, 8}
- seq_len_kv ∈ {128, 1024, 4096}
- (n_heads_q, n_heads_kv) ∈ {(32, 4), (16, 2)}  // Qwen3.5 + Qwen3.6
- head_dim ∈ {128, 256}
- max element-wise delta vs serial-per-slot baseline < 1e-3 (flash-attn-v2
  reordering tolerance — same bar as the split-K cert)
- N=1 path bit-identical to single-slot baseline (regression guard)

Per-impl cert artefact at `certs/gfx906/attention_decode_f16_batched.json`
once green; entry behind `cfg(unverified)` until cert exists.

## Dispatch table row

```toml
# dispatch/hip/gfx906.toml
[[op.attention_decode_f16]]
impl_id = "attention_decode_f16_batched_v1"
predicate = { n_slots = ">= 2" }   # falls through to single-slot for N=1

[[op.attention_decode_f16]]
impl_id = "attention_decode_f16"
predicate = { n_slots = "1" }
```

The N=1 row keeps the existing single-slot kernel as the fast path (no
pointer-table HtoD, no slot dimension). The new row engages only when
`forward_*_batched` hands N≥2 to the dispatcher.

## Subtask split (#266b / #266c / #266d)

This doc closes #266a. Implementation tasks remain:
- **#266b** Implement kernel + correctness sweep (cfg(unverified) until cert).
- **#266c** Wire into `forward_full_attn_layer_decode_batched_tp` and the PP
  twin; replace the per-slot loop. Verify N=1 bit-identity + N=2/N=4 parity
  vs the per-slot version.
- **#266d** Re-run #267 throughput cert; if still under 3×, profile the next
  ceiling (likely batched GDN state-step) and file a follow-up.

## Open questions for implementation

1. **Per-slot pointer table source-of-truth.** The simplest path is to
   build it once per `forward_*_batched` call and HtoD via pinned host
   buffer. Alternative: keep a long-lived device-resident pointer table on
   each rank's `RankForwardPrefillScratchTp` and update only when a slot's
   KV pointer changes (rare — at session reset). Decision: start with
   per-call HtoD; revisit if profiling shows it's > 1 % of decode wall.
2. **N upper bound.** With pinned-pointer-table, the kernel scales linearly
   in resident wave slots until hitting the 240-wave ceiling on MI50.
   `local_n_heads = 16` and `n_slots = 16` already saturates. Cap the
   kernel's contract at `N ≤ 32` for V1 and bail out otherwise; the
   inflight-pool ceiling is below 32 anyway.
3. **Q8 KV path interaction with #275 sub_cluster-stream finding.** The
   batched kernel itself doesn't change stream semantics, but the wiring
   in `forward_full_attn_layer_decode_batched_tp` already runs on
   `device.default_stream()` of the rank's sub_cluster. Same draining
   discipline applies; no additional sync needed.
