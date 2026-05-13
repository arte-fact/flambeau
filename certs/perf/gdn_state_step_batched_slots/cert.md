# gdn_state_step_alphabeta_f32_s128_batched_slots — cert

Date: 2026-05-13
Backend: HIP / gfx906 (MI50, 100 W cap)
Kernel: `gdn_state_step_alphabeta_f32_batched_slots.cu`
Baseline: per-slot loop of `flambeau_gdn_state_step_alphabeta_f32_s128`
called N times with B=1 inside `forward_gdn_decode_batched_tp`.

## Design

Same compute body as `gdn_state_step_alphabeta_f32_s128`. The existing
kernel assumes all B batches' states live in one contiguous buffer at
`state_in + b * H * S_v * S_v`; in the batched-decode driver each
slot's `GdnLayerState::state` is an independent device allocation. The
new variant takes a `[B] u64` device pointer array and dereferences
`state_in_ptrs[b_idx]` per block instead. Per-call: a small
`max_tokens * 8`-byte HtoD memcpy_async (e.g. 32 bytes at N=4) on the
same stream as the kernel — ordering free, no host sync.

## Parity

`tests/gdn_state_step_alphabeta_batched_slots.rs`: 4/4 **bit-equal**
across:
- B=2, H=32, n_rep=1
- B=3, H=16, n_rep=1 (odd batch)
- B=4, H=32, n_rep=4, rep_outer (qwen35moe)
- B=4, H=32, n_rep=4, rep_inner (qwen3next)

Same FP32 accumulation order, same lane/warp reduce signatures, only
the state base pointer fetch changes — guarantees bit-equal at FP32.

## End-to-end (Qwen3.6-35B-A3B-Q4_0 / pp2tp2 / inflight=4)

Cumulative perf vs the original two-separate-launch / per-slot
state-step path (sequential vs concurrent decode tok/s):

```
| Variant                                  | N=2 conc | N=4 conc | conc/seq (N=4) |
|------------------------------------------+----------+----------+----------------|
| Baseline (no batching)                   | 50.7     | 50.4     | 0.88×          |
| + Q4_0 row-tile fused gate+up            | 52.3     | 51.0     | 0.90×          |
| + batched-slots state-step (this cert)   | 53.4     | 51.9     | 0.94×          |
```

Cumulative end-to-end: **+5.3% at N=2, +3.0% at N=4** vs the original
baseline. The conc/seq ratio climbed from 0.88 → 0.94 at N=4, closing
about half of the gap to a perfect-overlap topology, with the rest
sitting on the per-slot full-attention path (memory note `#266`).

## Wiring

`forward_gdn_decode_batched_tp` now splits its per-slot work loop into
two passes: pass A runs steps 6–10 (conv/silu/split/l2_norm/scale) for
each slot, then a single batched-slots state-step launch covers all N
slots, then pass B runs per-slot ssm_norm. Gated by
`FLAMBEAU_GDN_STATE_STEP_BATCHED` (default ON, set to `0` to disable
and fall back to per-slot state-step).

`GdnPrefillScratch` gained `slot_state_ptrs: DevicePtr` sized
`max_tokens * 8` bytes for the per-call pointer array upload.

## Notes

- The first cut included a stream.synchronize() after the HtoD copy;
  that host stall was a measurable regression. Removed: memcpy_async +
  kernel on the same stream is ordering-safe by HIP's stream-queue
  semantics.
- ssm_norm is still per-slot. Batching it is the next-smaller lever
  (sizeable but small absolute time at this shape); deferred until a
  measurement justifies the plumbing.
- Per-slot full-attention KV-append + flash-attn in
  `forward_decode_batched_hybrid` is the remaining structural ceiling
  for the 25% of hybrid layers that are full-attention. That's the
  next big lever (#266 in memory notes).

## Closes

The batched-GDN-state-step half of the
`project_p29b_i2_F_hybrid_throughput` structural-ceiling decomposition.
