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

4. **#304 layer-body K|N split landed (commit `ddc0e9b`).** Full
   driver implementation:
   - When `prefill_chunk = None`: delegates to
     `forward_decode_batched_hybrid` (zero regression).
   - When `prefill_chunk = Some(chunk)`:
     - Stage 0 embed K prefill rows + N decode rows.
     - Per layer: full-attn split → `forward_full_attn_prefill_tp` on
       `[0..K]` rows + chunk's KV cache, then
       `forward_full_attn_layer_decode_batched_tp` on `[K..T]` rows +
       N slot KVs. GDN split → `forward_gdn_prefill_tp` on `[0..K]` +
       chunk's GDN state, then `forward_gdn_decode_batched_tp` on
       `[K..T]` + N slot states.
     - AR over T rows; ffn_norm / router / shared / MoE / FFN at
       `n_tokens=T` (the co-batching point — one weight read for K+N
       rows).
     - Stage handoff: peer_copy `chunk_bytes = T * row_bytes`.
     - Output head: optional `K-1` row → `prefill_final_logits_out`,
       N rows → `decode_logits_out`.
   - Compiles clean; runtime validation pending.

## Next step (resume here)

**Validation + wiring**, in order:

### A. Parity test (#307) — run first

Write `crates/models/qwen3-moe/tests/mixed_batch_parity.rs` modeled
on `chunked_prefill_kv_parity_hybrid.rs`. Two parallel runs, compare
logits:

```
// Reference: two independent sessions
sess_A_ref: prefill prompt_A, decode 1 step → logits_A_dec_ref
sess_B_ref: prefill prompt_B → logits_B_pre_ref (last-row from
                              forward_prefill_hybrid_logits)

// Mixed: same end-state via one mixed call
sess_A_mix: prefill prompt_A normally
sess_B_mix: (fresh)
forward_decode_mixed_hybrid(
    chunk = Some({tokens: prompt_B, chunk_start: 0,
                  is_final_chunk: true, idx: B-index}),
    slots = [BatchSlot{idx: A-index, token_id: A.last,
                       position: A.len}],
    decode_logits_out, prefill_final_logits_out,
)?;

assert!(close(logits_B_pre_ref,  prefill_final_logits_out));
assert!(close(logits_A_dec_ref,  decode_logits_out[A-index]));
```

F16 rounding tolerance ~1e-3 relative. Top-1 token-id parity is the
stricter secondary check.

Initial config: Qwen3.5-9B-Q4_1 / pp2tp2 / [0,2,1,3]. K=32, N=1.

Likely failure modes (where to debug if it fails):
- **FullAttnPrefillScratch reuse between prefill K-call and
  decode N-call on same stream**: the prefill writes scratch tensors
  the decode then overwrites. They run sequentially on the same
  rank's default_stream so this should be safe — but if logits are
  garbled, this is suspect #1. Check via two sequential calls each
  storing intermediate via `FLAMBEAU_AR_DUMP=1` style env.
- **Scratch sizing**: `scratch.max_tokens >= T` is checked at entry.
  If unit test sizes via `ShardedForwardPrefillScratchHybrid::new(model, T)`
  before calling, it's fine.
- **shared_delta_f16 buffer sizing**: it's allocated based on
  `LayerPrefillScratch::new`'s n_tokens; sizing is downstream of
  scratch.max_tokens.
- **kv_replicated / kq_replicated mismatch**: chunk session and
  decode sessions share the same model so flags should agree;
  print and verify if logits diverge.

### B. Scheduler + server wiring (#305 / #306)

Once parity passes, wire it through:
- `#305`: `build_mixed_iteration` builder picks one pending prefill,
  fills with decodes up to budget. State machine for chunked-in-
  progress prefills (request can have multiple chunks before
  is_final_chunk fires).
- `#306`: routes.rs replaces separate prefill+decode dispatch with
  mixed dispatch when `FLAMBEAU_MIXED_BATCH=1`. **Important**:
  scratch sizing — `ShardedForwardPrefillScratchHybrid::new(model,
  max_slots)` must use `max_tokens >= chunk_budget + max_slots`,
  not just `max_slots` like today's batched-decode does. Current
  routes.rs:585 passes `max_slots = inflight_pool.len().max(n)`;
  needs an env-driven chunk budget added.

### C. Throughput cert (#308)

- pp2tp2 / Qwen3.6-27B / mixed traffic harness: one prefill-512 +
  4 decodes per iteration × 20 iterations vs baseline (sequential
  prefill-then-decode). The relevant metric is **aggregate
  throughput** (prefill_tokens + decode_tokens) / wall_time, not
  decode-wall-only.
- Pure-decode wall may slightly regress under mixed batching (we
  add prefill cost to each step). The win shows up on **mixed
  traffic** where prefill amortises over decode steps that would
  otherwise have been blocked.

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
