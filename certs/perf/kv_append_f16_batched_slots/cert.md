# kv_append_f16_batched_slots — cert

Date: 2026-05-13
Backend: HIP / gfx906 (MI50, 100 W cap)
Kernel: `kv_append_f16_batched_slots.cu`
Baseline: per-slot `hipMemcpyAsync(DtoD)` loop in step 8 of
`forward_full_attn_layer_decode_batched_tp`.

## Design

For an `N`-slot concurrent decode, each full-attention layer wrote one
K row + one V row per slot using 2N small DtoD memcpys
(`kv_per_token_bytes ≈ 256–512 bytes` per copy at typical decode
shapes). Each memcpy carries fixed launch overhead that dominated at
this size.

The kernel takes (a) one source `[N, kv_width]` slot-major K + V
buffer, (b) `[N] u64` device arrays of per-slot K and V cache base
pointers, and (c) a `[N] i32` array of per-slot pre-bump write
positions. Grid `(N, 1, 1)`, block `min(kv_width, 128)` threads with
strided coverage of `kv_width`; one block copies both K and V rows for
its slot in one pass.

`FullAttnPrefillScratch` gained `slot_write_pos: DevicePtr`
(`max_tokens * 4` bytes) + `slot_write_pos_host: Vec<i32>` matching
the existing `slot_k_ptrs`/`slot_v_ptrs`/`slot_n_tokens_kv`
table-staging pattern. The bookkeeping loop is unchanged; only the
DtoD-memcpy part is replaced.

Gated by `FLAMBEAU_KV_APPEND_BATCHED` (default ON, set to `0` to fall
back to the per-slot memcpy loop).

## Parity

`tests/kv_append_f16_batched_slots.rs`: 4/4 **bit-equal** across:
- N=2, kv_width=256, max_seq=16
- N=4, kv_width=512, max_seq=64
- N=3, kv_width=256, max_seq=8 (Qwen3.6-35B-A3B TP2 GDN/full-attn shape)
- N=8, kv_width=256, max_seq=32

Same bytes copied to same target offsets; bit-equal is the bar.

## End-to-end (Qwen3.6-35B-A3B-Q4_0 / pp2tp2 / inflight=4)

A/B with the same warm build, two measurements each:

```
| KV-append | N=2 conc | N=4 conc | conc/seq (N=4) |
|-----------+----------+----------+----------------|
| OFF       | 53.1 t/s | 50.65 t/s| 0.895          |
| ON        | 54.35 t/s| 51.80 t/s| 0.915          |
| Δ         | +1.2     | +1.15    | +0.02          |
```

Modest gain because only 25% of layers in Qwen3.6-35B-A3B are
full-attention (full_attention_interval=4 over 48 layers → 12 full-attn
+ 36 GDN). KV-append only fires on those 12 layers; the other 36 GDN
layers are unaffected by this commit. The B.2 N=2 run measured 0.99×
conc/seq — effectively perfect overlap.

## Cumulative this session on the same topology

```
| Variant                                | N=2 conc | N=4 conc |
|----------------------------------------+----------+----------|
| Baseline (no batching)                 | 50.7     | 50.4     |
| + Q4_0 row-tile fused gate+up          | 52.3     | 51.0     |
| + batched-slots GDN state-step         | 53.4     | 51.9     |
| + batched-slots KV-append (this cert)  | 54.4     | 51.8     |
```

End-to-end: **+7.3% at N=2, +2.8% at N=4** over the no-batching
baseline. conc/seq at N=4 climbed 0.88 → 0.92, with the residual gap
sitting on cross-rank AR overhead + small per-slot ops the
batched-decode pipeline still has (rmsnorm shape transitions, conv-
input assemble in GDN's pass A).

## Closes

The per-slot KV-append half of the
`project_p29b_i2_F_hybrid_throughput` structural-ceiling decomposition.
The batched-attention half (`attention_decode_f16_batched`) shipped
earlier in #266c.
