# V2.27.a-i1 — decode-path audit for graph capture

Scope of this audit: enumerate everything in `forward_one_token_pp`
that would need slot tagging or host-memory stabilisation to make
the per-token decode chain captureable + replayable.

## Call tree (Qwen3.5-9B-Q4_1 Mesh<4>, hybrid dense arch, per token)

```
forward_one_token_pp(token_id, position)
├─ Rank 0: forward_embed_decode_host(token_id)        // HOST roundtrip + 2 syncs
├─ For rank in 0..N:
│    ├─ (rank > 0) cluster.peer_copy_via_host(hidden_prev → hidden_curr)
│    └─ For layer in shard.layers:
│         ├─ [full-attn] forward_layer_decode → forward_full_attn_decode
│         │    ├─ rmsnorm + quantize_q8_1 (fused)
│         │    ├─ qmatmul(attn_q) × mmvq r2
│         │    ├─ cast_f32_f16
│         │    ├─ split_q_gate
│         │    ├─ qmatmul(attn_k) × mmvq r2
│         │    ├─ cast_f32_f16
│         │    ├─ qmatmul(attn_v) × mmvq r2
│         │    ├─ cast_f32_f16
│         │    ├─ rmsnorm(q_norm) + rmsnorm(k_norm)
│         │    ├─ upload_position(position as i32)    // HOST roundtrip + 1 sync
│         │    ├─ rope_neox_partial_f16(Q)
│         │    ├─ rope_neox_partial_f16(K)
│         │    ├─ kv_cache.append(K, V)               // 2 memcpys, pos-dependent dst
│         │    ├─ attention_decode_f16                // reads n_k_tokens + q_offset
│         │    ├─ sigmoid_mul_f16
│         │    ├─ quantize_q8_1 × 2
│         │    ├─ qmatmul(attn_out) + cast + add_f16 residual
│         │    └─ ... ~20 kernel launches + 1 sync
│         └─ [GDN]    ~25 kernel launches, state advances in place
└─ (rank == N-1) forward_output_head_decode
   ├─ rmsnorm_quant_q8_1
   ├─ mmvq (LM head → logits F32)
   └─ argmax_token_host                                // DtoH + sync + Rust scan
```

## Slot inventory per replay (Mesh<4>, 9 layers/rank, 25% full-attn)

Roughly 2 full-attn + 7 GDN layers per rank.

| slot type | per-layer | per-rank (full-attn only) | per-token total (Mesh<4>) |
|---|---:|---:|---:|
| pos `ScalarSlot` (kv append + attn) | 2 (n_k, q_off) | 4 | 16 |
| KV-append K dst `MemcpySlot` | 1 | 2 | 8 |
| KV-append V dst `MemcpySlot` | 1 | 2 | 8 |
| embed token-id (rank 0 only) | — | 1 | 1 |
| output-head pos (rank N-1 only) | — | 0 | 0 |
| **Total** | | | **33 slots** |

All of these already have infra in place from V2.26.a. No new slot
primitives needed.

## Host-memory hazards — which ones need persistent backing

| site | hazard | fix |
|---|---|---|
| `upload_position` (common.rs) | transient `[i32; 1]` on stack + sync | persistent 1-slot buffer per layer or per rank; drop the sync under capture (Relaxed mode swallows it) |
| `forward_embed_decode_host` (io.rs) | transient Vec for row + 2 syncs | persistent `DecodeEmbedHostScratch`; replace with gather-on-device kernel OR keep host-roundtrip with slot-tagged memcpys |
| `argmax_token_host` | transient Vec<f32>[vocab] + sync | persistent host buffer sized to vocab; the sync is needed (argmax is host-side) — keep uncaptured OR write GPU argmax kernel (~10 lines, trivially captureable) |
| `cluster.peer_copy_via_host` | pinned bounce is already persistent | no change; memcpy is stable src/dst already |

The capture can cover everything EXCEPT `argmax_token_host` (host
compute after a sync). argmax runs post-replay per token; trivial to
keep out of the graph.

## Per-token driver-overhead budget

Decode at 53 tok/s → 18.9 ms/token wall. Estimated rank-level work:

- Per layer, full-attn: ~20 kernel launches × ~3 µs = 60 µs driver
- Per layer, GDN: ~25 launches × ~3 µs = 75 µs driver
- Per rank (9 layers, 2 full-attn + 7 GDN): 60·2 + 75·7 = **645 µs driver** on the Rust dispatcher thread
- Plus 9 × `upload_position` syncs at ~50 µs each = **450 µs sync** per rank
- Plus peer-copy: ~3 × 2 memcpys = small
- Across 4 ranks in serial decode: 4 × (645 + 450) = **4.38 ms/token driver + sync overhead**

Measured 18.9 ms/token wall × 23 % = **~4.3 ms plausibly recoverable** by graph capture.

## Expected wins from V2.27.a graph capture

Two sources:

1. **Launch overhead collapse**: 4 × 645 µs driver = 2.58 ms/token → 4 × ~5 µs replay = 20 µs. Saves **~2.56 ms/token = 13.5 %**.

2. **Sync drops under Relaxed capture**: the 9 per-layer
   `upload_position` syncs per rank don't record into the graph at
   all (HIP swallows `hipStreamSynchronize` during Relaxed capture).
   Replay skips them entirely. Saves **4 × 450 µs = 1.8 ms/token
   = ~9.5 %**.

**Combined: ~23 % decode speedup** if the capture lands cleanly.
53 tok/s → **~65 tok/s**. Closes 60 % of the gap to turbo's 73.5.

Remaining 40 % of the gap lives at the per-kernel level — covered by
V2.29.a audit + V2.29.b/c/d kernel tuning.

## What V2.27.a-i2 / i3 must deliver

**i2 (token-id slot):**
- Pick host-roundtrip-with-slot for now — simplest integration.
- Add `DecodeEmbedHostScratch { row_raw: Vec<u8>, row_f16: Vec<half::f16> }` to decode scratch.
- New `forward_embed_decode_host_slot(...)`: issues the HtoD with a
  caller-provided MemcpySlot tagging the F16 row upload.
- Before each replay, caller updates the host row_raw + row_f16
  (dequant from device weights OR pre-dequanted persistent cache
  for hot tokens). Persistent host storage means no syncs needed.
- A future V2.27.a-i2b could add the gather-kernel for full
  on-device embed; marginal win if i2 delivers ~23%.

**i3 (decode capture):**
- Per-rank `HipGraphExec` on `ShardedForwardOneTokenScratch` — one
  exec per rank, captured on first token.
- Slot layout per the inventory above.
- Replay dance per token:
  1. Update embed host row (rank 0 only).
  2. For each rank: update pos slots, KV-append dst slots, replay
     the exec, bump KV tail per layer.
  3. Rank N-1: launch argmax_token_host (uncaptured tail).
- Env gate: FLAMBEAU_DECODE_GRAPH=1 opt-in.

**i4 (measurement):**
- tg=64 / 128 / 512 on 9B + 35B with/without DECODE_GRAPH.
- Parity: 8-token decode greedy sequence must match between paths.
- Head-to-head vs turbo 9B: expect 0.72× → **0.88×** if +23% lands
  (or 0.95× if attention kernel has room too, per V2.29.a).

## Gate

Research-only; no code changes in this iteration. Unblocks i2.
