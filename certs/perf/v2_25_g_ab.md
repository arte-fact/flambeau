# V2.25.g — async PP refinement iter 1: per-lane bounces + non-blocking streams

Two refinements targeting the V2.25.f hypothesis that async peer-copies
on the same source rank serialise via shared pinned memory.

## Changes

1. **Per-lane pinned bounces** (`cluster.rs`)
   - New `HipCluster.lane_bounces: Vec<Mutex<Vec<RankBounce>>>` indexed
     by `(rank, lane)`.
   - New `reserve_lane_bounces(n_lanes, bytes_per_rank)` grows each
     lane's slab at session init.
   - Split `ensure_bounce` into a `(slot, rank, need)` helper reusable
     for both shared and per-lane bounces.
   - `peer_copy_via_host_async` now forwards to
     `peer_copy_via_host_async_laned(..., Some(lane))` when the caller
     passes a lane.
   - Dispose path frees both shared and per-lane pinned allocations.

2. **Non-blocking aux streams** (`device.rs`, `cluster.rs`)
   - New `HipStream::new_non_blocking` via `hipStreamCreateWithFlags`
     with `hipStreamNonBlocking=1`.
   - `reserve_aux_streams` now uses non-blocking streams so lanes on
     the same device don't serialise through the null stream.

3. **`forward_prefill_pp_async` wiring**
   - Calls `reserve_lane_bounces(u_lanes, ubatch_size × hidden × 2)`
     at entry.
   - Swaps `peer_copy_via_host_async` → `peer_copy_via_host_async_laned`
     with `Some(lane)`.

## Results (9B Q4_1 Mesh<4>, L=1024)

| | tok/s | Δ vs sync |
|---|---:|---:|
| Sync reference | 766.94 | — |
| V2.25.d async (shared bounce) | 696.15 | −9.2 % |
| V2.25.g async (per-lane bounces) | 696–700 | −9 % |
| V2.25.g + non-blocking streams | 678.36 | **−11.6 %** |

**Both refinements are noop-to-regression.** Per-lane bounces don't
help because the bottleneck wasn't bounce contention. Non-blocking
streams marginally hurt (driver path difference on gfx906).

Parity preserved: `last_id=220` on every config — async arithmetic
remains bit-exact vs sync.

## Root cause — wrong one hypothesised in V2.25.f

V2.25.f speculated that the perf regression came from shared-bounce
serialisation. V2.25.g disproves that: per-lane bounces (and non-
blocking streams) give no benefit.

The ACTUAL bottleneck is the **control flow shape** in
`forward_prefill_pp_async`:

```rust
for ub_idx in 0..n_ubatches {
    for rank_idx in 0..n_ranks {
        dispatch(ub_idx, rank_idx)
    }
}
```

This is **serial across ubatches**. Ubatch 1 only begins issuing on
rank 0 AFTER ubatch 0 has been fully issued across all N ranks. There
is no "pipeline fill" — at any moment, exactly one ubatch is being
dispatched; the others are either queued on aux_streams (waiting for
upstream peer-copy events) or already done.

Driver-side concurrency across lanes is possible but unused: ubatch 1
on lane 1 is issued AFTER ubatch 0 has been peer-copied to rank 3,
which means by the time rank 0 stream 1 has work queued, rank 0's
layer work is already idle — no overlap with rank 1+ work that's
still running.

## Fix planned for V2.25.h — interleaved 1F1B dispatch

Correct dispatch for an N-rank × K-ubatch pipeline:

```
for t in 0..N+K-1:             # N+K-1 time-step "issues"
    for r in 0..N:             # each rank gets at most one issue per t
        if 0 <= t - r < K:
            dispatch(ubatch=t-r, rank=r)
```

At steady state (t ≥ N-1), all N ranks are issuing DIFFERENT ubatches
concurrently. Real 1F1B pipeline fill.

Per-lane bounces + non-blocking streams (this commit) become
prerequisite infrastructure for V2.25.h to land: the shared bounce
would be raced by concurrent issues, and blocking streams would
serialise through the null stream.

Net of V2.25.g: **no perf change but infrastructure ready for V2.25.h**.
cert-check hip/gfx906 unchanged. UD-Q4_K_S 8-tok parity preserved.

## Regeneration

```
# compare sync vs async at L=1024 M<4>
FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=...Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L=1024"

FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=512 FLAMBEAU_U_LANES=2 \
  FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=...Qwen3.5-9B-Q4_1.gguf \
  ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
  --nocapture 2>&1 | grep "prefill L=1024"
```
