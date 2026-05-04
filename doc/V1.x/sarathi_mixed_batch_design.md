# Sarathi-style mixed prefill+decode batching — design (#302)

**Goal:** Fill GPU-idle that pure-decode batches leave behind by
co-batching one prefill chunk with N decode slots in the same forward
pass. Prefill chunk drives tensor cores (compute-bound); decodes drive
HBM bandwidth (bandwidth-bound). Both subsystems saturate together.

**Lever projected to lift v2 cert from current 1.34× (PP=4) / 0.92×
(pp2tp2) toward 3.0×+ across all topologies.** Lever 2 (1F1B) was
shipped null at decode (HBM-bound at n_tokens=1); this is the
remaining lever.

## Pattern (Sarathi-Serve, OSDI'24)

Per scheduler iteration, pack one mixed batch:
- Pick one pending prefill request, slice off K tokens (chunk).
- Fill remaining (token_budget - K) with decodes from already-prefilled
  slots (each contributes 1 token).
- Total tokens in the forward pass: T = K + N (chunk + decodes).

A single forward call processes T tokens through the model. Each layer
sees T-token activation; attention sees per-query KV history that
varies by source (prefill chunk → growing KV of one request; decode
token → that slot's KV).

## What we already have, what's missing

Per surface scan (#302 sub-survey):

| Subsystem | Prefill | Decode-batched | Mixed-T capable today? |
|-----------|---------|----------------|------------------------|
| Embed gather | loop N IDs | 1 ID | trivial — gather T IDs once |
| RMSNorm | n_tokens=K | n_tokens=N | yes — call with n_tokens=T |
| QKV proj | n_tokens=K (MMQ) | n_tokens=N (MMQ batched) | yes — call with n_tokens=T |
| RoPE | n_tokens=K, contig pos | n_tokens=N, per-slot pos | **needs per-token pos** |
| **Full-attn** | varlen Q vs single contig KV | Q=1×N vs per-slot KV ptrs | **NO — two separate kernels** |
| KV append | append K to one cache | append 1 to each of N caches | yes — split per source |
| **GDN** | 1 state × K tokens | N states × 1 token (loop) | **NO — fundamentally per-state** |
| FFN dense | n_tokens=K | n_tokens=N | yes — n_tokens=T |
| MoE router | dense_gemv batched(K) | dense_gemv batched(N) | yes — n_tokens=T |
| MoE experts | indexed batched(K) | indexed batched(N) | yes — n_tokens=T |
| Output head | argmax T rows? | argmax N rows | yes — argmax T, take last N |

So 7/10 layer-internal blocks accept arbitrary T already. Two require
real work: full-attn and GDN.

## Variant choice for v1: two-attn-calls + GDN sub-pass

For full-attn:
- **v1 (this design)**: dispatch the K prefill-chunk tokens via the
  existing `forward_full_attn_prefill` against the prefill request's
  KV cache. Dispatch the N decode tokens via the existing
  `forward_full_attn_layer_decode_batched` against the N slot KVs.
  Two kernel calls per attn layer. **No new kernel.**
- **v2 (future)**: write `attention_mixed_varlen_batched` — single
  kernel, Q pool of T queries, per-query slot table indexes K/V
  pointers + causal mask. One kernel call. Harder; defer until v1
  proves the lever.

For GDN:
- **v1 (this design)**: dispatch the K prefill-chunk tokens via
  `forward_gdn_prefill` against the prefill request's GDN state.
  Loop over N decode slots, each via `forward_gdn_layer_decode`.
  Same kernel count as today's batched-decode path; we just also do
  the prefill-chunk pass. **No new kernel.**
- **v2 (future)**: batched per-slot decode kernel (already separately
  scoped as #284-#287). Independent of Sarathi.

Net for v1: **zero new device kernels.** All work is host-side
scheduler + driver + dispatch glue. The lever still applies because
the wins come from co-saturating compute (prefill side) and HBM
(decode side) in the same step — not from kernel fusion.

## Token-pool layout

Per-step inputs:
- `prefill_req: Option<PrefillReq>` — request id, KV cache handle,
  base position, chunk start offset, chunk len K.
- `decode_slots: &[BatchSlot]` — N slots (existing type).
- `token_budget: usize` — typical 512 (configurable).

Pack:
1. K = min(prefill_req.remaining(), token_budget).
2. N = min(decode_slots.len(), token_budget - K).
3. Embed K prompt tokens into rows [0..K] of stage-0 hidden.
4. Embed N decode tokens into rows [K..K+N] of stage-0 hidden.
5. Build per-token position table (size T): rows [0..K] get
   `prefill_base + chunk_start..chunk_start+K`; rows [K..K+N] get the
   decode slots' `position`.

The K prefill rows and N decode rows live in one `[T, hidden]`
activation tensor — no reshuffling between layers.

## Layer body (mixed-batch version)

```rust
fn forward_layer_mixed(
    layer_idx: usize,
    ctx: &mut MixedCtx,
    x: TensorMut<[T, hidden]>,
) -> Result<()> {
    // Pre-attn rmsnorm + Q|K|V proj at n_tokens = T (existing kernels).
    rmsnorm(x, ...);
    qkv_proj(x, ...);  // produces Q[T], K[T], V[T]

    // Split Q/K/V into prefill region [0..K] and decode region [K..T].
    let q_pre = q.slice(0..K);
    let q_dec = q.slice(K..T);
    let k_pre = k.slice(0..K);
    let k_dec = k.slice(K..T);
    let v_pre = v.slice(0..K);
    let v_dec = v.slice(K..T);

    if layer is full_attn {
        // v1: TWO kernel calls.
        // Append k_pre/v_pre into prefill_req's KV cache, run prefill attn.
        prefill_req.kv.append(k_pre, v_pre);
        attention_prefill_f16(q_pre, prefill_req.kv.k, prefill_req.kv.v,
                              n_q_tokens=K, q_offset=prefill_req.base);
        // For each decode slot i in 0..N: append k_dec[i]/v_dec[i] into
        // its slot KV cache (existing batched op).
        kv_append_batched(slot_caches, k_dec, v_dec, slot_positions);
        attention_decode_f16_batched(q_dec, slot_kv_ptrs[N],
                                     slot_kv_lengths[N]);
    } else {
        // GDN layer.
        // Run prefill GDN over K rows on prefill_req.gdn_state.
        gdn_prefill(prefill_req.gdn_state, x.slice(0..K), n=K);
        // Loop over decode slots (existing per-slot path).
        for i in 0..N { gdn_decode(slot_gdn[i], x.slice(K+i..K+i+1)); }
    }

    // Output proj: n_tokens = T (existing kernel handles).
    o_proj(...);

    // Post-attn rmsnorm + MoE/FFN: n_tokens = T (existing).
    rmsnorm + moe_router + moe_experts + shared_expert (n_tokens=T).

    Ok(())
}
```

For PP/TP topologies, the same pattern applies on each rank — the
`[T, hidden]` activation flows through all stages exactly like a
T-token prefill, with the attn split happening locally per stage.

## Output & sampling

After the final layer:
- **Prefill chunk**: only the LAST chunk of a request needs logits
  (for the first generated token). Non-final chunks are KV-only —
  skip output head for those K rows.
- **Decode slots**: each contributes 1 logit row → output head over
  N rows → sample N tokens.

Total output-head work: 0–1 (final-chunk) + N rows. Same as today's
batched-decode in the steady state where prefill chunk is non-final.

## Scheduler / budget

```rust
fn build_mixed_iteration(scheduler: &mut Scheduler) -> MixedBatch {
    let token_budget = std::env::var("FLAMBEAU_MIXED_BUDGET")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(512);

    let mut batch = MixedBatch::default();

    // 1. Try to attach one pending prefill chunk.
    if let Some(req) = scheduler.pop_pending_prefill() {
        let k = (req.remaining()).min(token_budget);
        batch.prefill = Some(PrefillChunk { req, len: k });
    }

    // 2. Fill remaining budget with ready decodes.
    let used = batch.prefill.as_ref().map_or(0, |p| p.len);
    let cap = token_budget - used;
    for slot in scheduler.ready_decode_slots().take(cap) {
        batch.decodes.push(slot);
    }

    batch
}
```

Knobs:
- `FLAMBEAU_MIXED_BATCH=1` — engage mixed-batch dispatch (v1 default
  off; once stable, becomes default).
- `FLAMBEAU_MIXED_BUDGET=K+N` — per-iteration token budget. Default
  512 per Sarathi sweet spot.

Tradeoffs:
- Larger budget → fewer iterations per prefill (lower TTFT) but each
  step delays the decodes more (higher TPOT during co-batch).
- Smaller budget → smoother decode latency but more iterations per
  prefill.

## Implementation steps (revised task list)

| # | task | scope |
|---|------|-------|
| #303 | varlen attention kernel (v2) | DEFERRED — v1 uses two existing kernels |
| #304 | mixed-batch forward driver | hybrid layer body that splits Q/K/V at K|N boundary, dispatches per-region; reuses existing attn + GDN + MoE kernels |
| #305 | mixed-batch scheduler | `build_mixed_iteration` + state machine for in-progress chunked prefills |
| #306 | server wiring | replace separate prefill+decode dispatch with mixed-batch dispatch when `FLAMBEAU_MIXED_BATCH=1` |
| #307 | correctness validation | mixed-batch result bit-identical to (chunked-prefill-then-decode) for the same prompt+slot sequence |
| #308 | throughput cert | Qwen3.6-27B / pp2tp2 / 4-decode + 1-prefill mixed traffic; 3× gate target |

#303 is deferred — v1's two-attn-call path delivers the lever
without varlen kernel work. We retain the task but unblock it from
the critical path. v2 work re-opens it once measurements show the
two-launch overhead is the next bottleneck.

## Expected win

Sarathi paper + vLLM blog: 3–5× over naive static batching on H100.
Our v2 cert shows GPU% mean 25–46 % at 4-concurrent decode-only —
the 50–75 % idle silicon is what mixed batching fills.

For our 27B / pp2tp2 / 4-decode + 1-prefill-chunk:
- Prefill chunk K=384 occupies tensor cores (~3.5 ms compute-bound).
- 4 decode tokens occupy HBM (~6 ms HBM-bound).
- Same forward pass; max(compute, HBM) wall ≈ 6 ms — the decode
  portion of the work, with the prefill chunk effectively free
  (piggybacks on HBM idle).

Projected wall: same ~6 ms/step that pure decode-only-N=4 takes.
Throughput: (4 decodes + 384 prefill tokens) / 6 ms = ~64.7 tok/ms
**system throughput**, vs ~0.67 tok/ms for pure-decode-N=4 — a
~9× *aggregate* throughput win on mixed traffic, of which:
- ~1.0× for any individual decode (no degradation)
- ~10× for the prefill request (chunks now share decode-step time
  rather than blocking)
- TTFT for new requests no longer spikes during ongoing decodes

Closing the v2 cert gap: mixed batching's effect on the 4-concurrent-
decode-only bench is ~1.0× (no prefill to co-batch). The 3× gate
needs the **realistic-traffic cert**: mixed prompts arriving while
decodes are in-flight, which is precisely the workload Sarathi
optimizes. New cert harness required (#308).

## References

- `doc/V1.x/gpu_idle_fill_levers.md` — research synthesis
- Agrawal et al., Sarathi-Serve OSDI'24 — https://www.usenix.org/system/files/osdi24-agrawal.pdf
- vLLM chunked-prefill default-on — https://docs.vllm.ai/en/v0.8.5/performance/optimization.html
- v2 cert — `certs/perf/topology_compare/qwen36_27b_topo_x_mode_pp1024_2026_05_04.md`
- 1F1B null cert — `certs/perf/p29b_i2_F_throughput/qwen36_27b_pp4_1f1b_2026_05_04.md`
