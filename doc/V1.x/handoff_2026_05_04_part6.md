# Handoff 2026-05-04 part 6 — Lever 2 closed (null), Lever 1 (#304) layer-body queued

## State

- Branch: `main` (24 commits ahead of origin/main).
- Last commits:
  - `cdf2627` #304 scaffold: `MixedPrefillChunk` + `forward_decode_mixed_hybrid` stub
  - `b882778` #302: Sarathi mixed-batch design
  - `4602807` #298/#301: 1F1B-decode shipped + null at PP=4/N=4

## What landed this session

1. **Lever 2 (#298) — true 1F1B PP-pipelined decode** shipped behind
   `FLAMBEAU_DECODE_1F1B=1`. Correct (md5-bit-identical). Wall ≈26 s
   at PP=4/N=4 — same as slot-major pipelined and batched-decode.
   **Null lever for decode** because decode at n_tokens=1 is HBM-bound
   and batched-decode at N=4 already amortizes that HBM in one call.
   Cert: `certs/perf/p29b_i2_F_throughput/qwen36_27b_pp4_1f1b_2026_05_04.md`.

2. **Lever 1 (#302) — Sarathi mixed-batch design** committed.
   Key finding from surface scan: 7 of 10 layer subsystems already
   accept arbitrary `n_tokens` (RMSNorm, QKV proj, MoE router, MoE
   experts, FFN, embed, output head). Only full-attn and GDN need a
   K|N split. **No new device kernels needed for v1** (v2 varlen-attn
   kernel deferred). Doc: `doc/V1.x/sarathi_mixed_batch_design.md`.

3. **#304 scaffold landed.** API surface in
   `crates/models/qwen3-moe/src/forward/batched.rs` and
   `crates/models/qwen3-moe/src/forward/hybrid.rs`:
   ```rust
   pub struct MixedPrefillChunk {
       pub idx: usize,            // session index
       pub tokens: Vec<u32>,
       pub chunk_start: usize,    // global pos of tokens[0]
       pub is_final_chunk: bool,
   }

   pub fn forward_decode_mixed_hybrid(
       model, sessions, global_cluster, stage_ars, scratch,
       prefill_chunk: Option<&MixedPrefillChunk>,
       slots: &[BatchSlot],
       decode_logits_out: &mut [&mut Vec<f32>],
       prefill_final_logits_out: Option<&mut Vec<f32>>,
   ) -> Result<()>
   ```
   When `prefill_chunk = None`: delegates to existing
   `forward_decode_batched_hybrid` (zero regression). When `Some`:
   currently bails — **the layer-body K|N split is the next work item.**

## Next step (resume here)

**Implement the layer-body K|N split inside
`forward_decode_mixed_hybrid`.** Strategy: clone
`forward_decode_batched_hybrid` to a new private impl
`forward_decode_batched_hybrid_inner(prefill_chunk: Option<...>, ...)`.
Make both public functions thin wrappers around the inner.

Inside the inner, with `T = K + N` where `K = chunk.map_or(0,
|c|c.len())`:

1. **Embed (stage 0)**:
   - Rows `[0..K]` from `chunk.tokens[0..K]`.
   - Rows `[K..T]` from `slots[i].token_id` (current code).

2. **Per-stage layer loop** (lines 933-1685): each layer body
   currently dispatches with `n_tokens = N`. Change to `n_tokens = T`
   for these subsystems (no other change needed):
   - RMSNorm + add residual
   - Q/K/V projection
   - Output projection
   - MoE router + experts + shared expert
   - Dense FFN (qwen35 dense layers)

3. **RoPE positions**: existing `slot_positions: Vec<usize>` becomes
   `token_positions: Vec<usize>`:
   ```rust
   let mut token_positions = Vec::with_capacity(T);
   if let Some(chunk) = prefill_chunk {
       for i in 0..chunk.len() {
           token_positions.push(chunk.chunk_start + i);
       }
   }
   for s in slots {
       token_positions.push(s.position);
   }
   ```
   The existing `forward_full_attn_layer_decode_batched_tp` accepts
   `slot_positions` already; pass `token_positions[K..T]` for the
   decode slice.

4. **Full-attn layer (lines 994-1047)**: SPLIT the kernel call.
   - Prefill rows `[0..K]`: call
     `super::attn_tp::forward_full_attn_prefill_tp_layer` (or
     equivalent — check the name in `attn_tp.rs`) on a Q/K/V slice
     view of rows `[0..K]`, with the chunk's KV cache from
     `sessions[chunk.idx].stages[stage_idx].caches[r][il_local]`.
     Use `chunk.chunk_start` as `start_position`.
   - Decode rows `[K..T]`: existing
     `forward_full_attn_layer_decode_batched_tp` call but pass only
     the N-slot subset (slot KVs unchanged from current code).

   Note: full-attn QKV proj is already done in step 2 over T rows.
   The split happens after RoPE — slice the Q/K/V tensors for the
   two attn calls. Currently RoPE is fused into
   `forward_full_attn_layer_decode_batched_tp`; for v1 may need to
   split RoPE call too. Check the function body before deciding.

5. **GDN layer (lines 1048-1685)**: SPLIT.
   - Prefill rows `[0..K]`: call
     `super::gdn_tp::forward_gdn_prefill_tp_layer` (or equivalent)
     with `n_tokens = K` on the chunk request's GdnLayerState
     (`sessions[chunk.idx].stages[stage_idx].caches[r][il_local]`).
   - Decode rows `[K..T]`: existing
     `forward_gdn_decode_batched_tp` (or per-slot loop) over the N
     slot states — current code already handles this.

6. **Stage boundary peer_copy**: T rows transferred instead of N.
   `chunk_bytes = T * row_bytes`. The existing peer_copy logic
   should handle the increased size with no other change.

7. **Output head**:
   - If `chunk.is_final_chunk`: call `forward_output_head_decode` on
     row `K - 1` of the final hidden activation, write to
     `prefill_final_logits_out`.
   - For each decode slot `s` in `0..N`: call output_head on row
     `K + s`, write to `decode_logits_out[s]`.

8. **Sessions parallel array**: `chunk.idx` and `slots[i].idx` index
   into the same `sessions` array; the chunk request's session
   provides KV/GDN state for the prefill rows, distinct from any
   decode slots' sessions. The borrow-disjoint `sessions_ptr.add(s)`
   pattern at lines 1009-1024 needs to extend to also borrow
   `sessions[chunk.idx]` if `chunk` is set.

## Validation steps after #304 lands

- **#307**: parity test — run a 100-token prompt as
  (a) chunked-prefill-then-decode (existing path), output 32 tokens;
  (b) the same prompt+decode sequence via mixed dispatch, with K
  chunking that matches (a) exactly, and 0 co-batched decodes during
  prefill (legacy mode). Output token IDs should match bit-exact.
- **#308**: throughput cert — pp2tp2 / Qwen3.6-27B / mixed traffic
  (one prefill-512 + 4 decodes per iteration, run 20 iterations)
  vs baseline (sequential prefill then 4 decodes). 3× wall ratio is
  the gate; v2 cert recorded 0.92× pp2tp2, so target is 2.7× lift.

## Open scheduler/server work (#305/#306)

Independent of the driver. Can start once driver is correct:
- `#305`: `build_mixed_iteration` builder picks one pending prefill,
  fills with decodes up to budget. State machine for chunked-in-
  progress prefills.
- `#306`: routes.rs replaces separate prefill+decode dispatch with
  mixed dispatch when `FLAMBEAU_MIXED_BATCH=1`.

## Untouched

- `1F1B (#298)` stays as opt-in for prefill / future use cases. Don't
  delete.
- `MTP refactor side-task` (mentioned in earlier session notes —
  dedup forward_mtp_step variants, scratch struct, dead F16 buffers)
  not started; orthogonal to Lever 1.

## Don't repeat

- 1F1B at decode is null — don't re-propose more PP-pipelining levers
  for decode.
- Varlen-attention kernel (#303) is queued for v2 only after v1
  shows two-launch overhead is the next bottleneck.
- `sub_cluster` vs `global_cluster` default streams have separate
  HipStream handles for the same physical device. Sync both at
  sub↔global boundaries (#275 lesson, applies to mixed-batch too).
