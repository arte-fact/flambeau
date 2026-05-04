# Pipelined batched-decode — design

**Goal**: at PP ≥ 2 with N concurrent slots, overlap stage 0's work for slot
k+1 with stage 1's work for slot k. The two stages live on disjoint
sub_clusters → disjoint physical GPUs → run in parallel by hardware.
Today's `forward_decode_batched_hybrid` runs all N slots through stage 0
THEN all through stage 1; one stage's GPUs are idle 50 % of the time.

## Speedup ceiling (PP=2, balanced stages)

| topology     | serial (today) | pipelined  | speedup |
|--------------|---------------:|-----------:|--------:|
| PP=2 / N=2   | 2N=4 units     | N+1=3      | 1.33×   |
| PP=2 / N=4   | 8 units        | 5          | 1.60×   |
| PP=2 / N=8   | 16 units       | 9          | 1.78×   |
| PP=4 / N=4   | 16 units       | 7          | 2.29×   |
| PP=4 / N=8   | 32 units       | 11         | 2.91×   |

`unit` = stage-cost (the time one slot takes through one stage). Serial
uses 2 units per slot per step × N slots = 2N. Pipelined: stage_0 can
issue slot k+1 while stage_1 finishes slot k → wall = (N + PP - 1) units.
Combined with #266c attention batching (~1.05×) and a future #288
batched-MMVQ (~1.5× projected), the 3× cert gate is reachable on
hybrid pp2tp2 and pure PP=4.

## The pipeline pattern

For PP=n_stages, sub_cluster_streams[s] are independent (different
physical GPUs). Each stage has its own `ShardedForwardPrefillScratchTp`
(the per-stage scratch already in `ShardedForwardPrefillScratchHybrid`).
**Within** a stage, slot processing is serial (we reuse the per-stage
scratch for each slot). **Across** stages, two slots can be in flight
simultaneously.

```
  step t for slots [0..N), n_stages=2:

  stage_0_streams: |slot0_fwd→DtoH|slot1_fwd→DtoH|slot2_fwd→DtoH|slot3_fwd→DtoH|
  stage_1_streams:               |bridge0|HtoD0,slot0_s1,head0|bridge1|HtoD1,slot1_s1,head1|...
                                  ^                            ^
                                  wait peer_copy slot0          wait peer_copy slot1
```

Each peer_copy from stage_0 to stage_1 uses
`peer_copy_via_host_async(bridge_event=Bk, done_event=Dk)`:
- DtoH queued on stage_0's src stream → records `Bk` when DtoH done.
- HtoD queued on stage_1's dst stream after `streamWaitEvent(Bk)` → records
  `Dk` (we don't actually need `Dk`; the next op on stage_1 stream is the
  layer forward, which is already serial after the HtoD).

The fan-out (one src rank → all tp_size dst ranks of next stage) is
already the pattern in current `forward_decode_batched_hybrid`. We
preserve it; just swap blocking `peer_copy_via_host` for the async
variant and use bridge events.

## Scratch reuse (no new allocations)

- `scratch.per_stage[s]` exists per stage.
- Within stage s, slot k uses scratch.per_stage[s] for its layer forward;
  slot k+1 reuses the same scratch after slot k completes its DtoH.
  `stage_0_stream` orders this naturally: ops on the same stream are
  serial.
- Across stages, slot k uses scratch.per_stage[1] while slot k+1 uses
  scratch.per_stage[0]. **Different physical buffers on different
  devices — no aliasing**.

So pipelining requires **zero new scratch allocations**. The only new
state is one `HipEvent` per slot per pipeline boundary (4 events at
PP=2 / N=4) + done_event slots. Pre-allocate per-rank in
`ShardedForwardPrefillScratchTp::new`.

## Bounce-buffer pacing at TP fan-out

`HipCluster::peer_copy_via_host_async` uses one pinned bounce buffer
per src_rank. The fan-out loop in current `forward_decode_batched_hybrid`
issues `tp_size` HtoDs sequentially from the same src_rank. With the
async variant, all `tp_size` copies of slot k *plus* the eventual peer
copy of slot k+1 share the same bounce buffer. They MUST be paced.

Pacing comes for free when DtoH and HtoD all live on `src_dev`'s
**default stream** (DtoH) and the dst_devs' default streams (HtoDs):
- Slot k DtoH on stage_0 src stream.
- Slot k HtoD_for_dst_local=0, HtoD_for_dst_local=1 on respective dst
  streams. Each WaitEvents the same bridge event.
- Slot k+1 DtoH on the SAME stage_0 src stream after slot k's full
  forward. Sequential. Bounce reused safely.

For PP=2 / N=4 the same bounce slot turns over 4 times per token step.
Per-lane bounces (`peer_copy_via_host_async_laned`) are NOT needed for
the first-cut implementation — the tp-fan-out pattern is already
serialised through one src stream.

## #275-style stream-handle pitfalls

The #275 fix established that sub_cluster default streams ≠
global_cluster default streams. The pipelined design only uses
sub_cluster streams (the layer kernels, the AR-residual collectives,
the output head). The cross-stage hand-off goes through
`HipCluster::peer_copy_via_host_async` on the **global_cluster** —
which uses src_dev/dst_dev's default streams. So the function
boundary needs explicit pass-through: callers pass
`stage_0_sub_cluster.device(0).default_stream()` AND
`stage_1_sub_cluster.device(dst_local).default_stream()` as the
src/dst streams to the async peer_copy.

The #275 entry-time stream-drain stays at the function entry, ensuring
no in-flight kernels from the previous step interfere with this step's
peer_copies.

## Topology applicability

- **PP=1 / TP-only**: degenerate to current batched dispatch. Pipelining
  has nothing to interleave (one stage). Caller flag falls through.
- **PP=2 / TP=2 (hybrid pp2tp2 — prod target)**: full pipeline. 1.6× at
  N=4.
- **PP=4 / TP=1 (pure PP)**: full pipeline. 2.3× at N=4, ~3× at N=8.
- **PP=2 / TP=1 (rare)**: same code path as pp2tp2 with tp_size=1.

Single function `forward_decode_pipelined_hybrid` covers all PP≥2
cases; the n_stages==1 case bails to the old batched function.

## Per-slot logits delivery

Each slot's logits are written to a per-slot device buffer at the
output-head stage. The scheduler's `logits_out: &mut [&mut Vec<f32>]` is
indexed by slot; we just need to produce all entries before returning
to the scheduler. With pipelining, slot 0 completes earlier than slot
N-1, but the function only returns once all slots have finished
output-head + DtoH-of-logits to host. We sync the stage-1 streams at
the tail and download all N logits batches in one host loop.

## Cert plan

- N=2/4 on Qwen3.6-27B / pp2tp2: bit-identical-within-batch
  vs `forward_decode_batched_hybrid` (correctness regression guard).
  Coherent output. Wall ≤ today's batched-decode.
- 4-concurrent vs 4-sequential cert (#287's recipe): expect 1.6×.
- A/B vs `FLAMBEAU_DECODE_PIPELINE=0`: confirms the pipelining is what
  delivers the win.
