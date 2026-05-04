# 1F1B PP-pipelined decode — design (#297)

## Pattern

Adapt the V2.25.h 1F1B prefill loop (`pp.rs:1480-1700`) to decode:
- "ubatch" = one slot at n_tokens = 1
- "rank" = one PP stage (each stage may itself be TP-replicated; the
  scheduling unit is the stage)
- n_timesteps = n_slots + n_stages - 1

```
At tick t, for each stage s in 0..n_stages:
    slot k = t - s
    if k not in [0, n_slots): continue
    if s == 0:
        embed slot[k].token_id → stage 0 hidden_a row 0 (per rank in stage 0)
    run all stage_s layers for slot k at n_tokens = 1
    if s + 1 < n_stages:
        peer_copy_async fan-out stage s → stage s+1 with bridge[s, k]
    if s == n_stages - 1:
        output_head + async logits DtoH for slot k
tail-sync head_rank stream + stage_0 streams
```

## How it differs from `forward_decode_pipelined_hybrid` (#290/#294)

| | current (slot-major) | 1F1B (stage-major) |
|---|---|---|
| outer | `for slot in 0..N` | `for t in 0..N + n_stages - 1` |
| inner | `for stage in 0..n_stages` | `for stage in 0..n_stages` |
| pairing | one slot through all stages, then next | at each tick, every stage advances by one slot |
| host enqueueing | sequential per slot | all stages queued in parallel per tick |

Concrete at PP=4, N=4:

| tick | stage 0 | stage 1 | stage 2 | stage 3 |
|-----:|:-------:|:-------:|:-------:|:-------:|
|   0  | slot 0  |    -    |    -    |    -    |
|   1  | slot 1  | slot 0  |    -    |    -    |
|   2  | slot 2  | slot 1  | slot 0  |    -    |
| **3**|**slot 3**|**slot 2**|**slot 1**|**slot 0**| ← steady state, all 4 ranks busy
|   4  |    -    | slot 3  | slot 2  | slot 1  |
|   5  |    -    |    -    | slot 3  | slot 2  |
|   6  |    -    |    -    |    -    | slot 3  |

7 ticks total. Steady-state ticks (3) keep all 4 ranks busy; ramp-up
(0..2) and ramp-down (4..6) are the unavoidable pipeline bubble.

Wall ratio vs. serial:
- 1F1B: 7 stage-time-units
- Serial-per-slot: 4 slots × 4 stages = 16 stage-time-units
- **Speedup ceiling = 16/7 = 2.29×** at PP=4/N=4 (matches design table).

## Why this fills GPU% gap that current pipelined doesn't

Current `forward_decode_pipelined_hybrid`:
```
for slot 0: stage 0 work (rank 0 busy, others idle)
            peer_copy 0→1
            stage 1 work (rank 1 busy, others idle)
            peer_copy 1→2
            stage 2 work (rank 2 busy, others idle)
            peer_copy 2→3
            stage 3 work (rank 3 busy, others idle)
            head + DtoH
for slot 1: ... (same serial walk through stages)
```
Only one rank works at a time per slot. Pipelining benefit at most:
when slot k+1 stage 0 starts before slot k stage 1 finishes — but
host serializes slot iteration so this is limited.

1F1B: **at every tick host enqueues work for ALL ranks** (the ones
that have a valid slot). All 4 GPUs have queued work at tick 3+,
running concurrently on separate sub_cluster streams.

## Scratch + bridge events: unchanged

- Each stage's per-rank scratch holds `hidden_a` row 0 reused across
  slots within that stage (sequential on the stage's sub_cluster
  stream → no clobber).
- `pipeline_bridge_events`: one per (src_stage, slot) — already lazily
  allocated by current pipelined function for non-final stages' rank 0.
- Lane bounces: tp_size lanes pre-reserved on global_cluster — no
  change.

## Stream-ordering correctness check

At PP=4/N=4, tick t=3 (steady state) host enqueues:
- (stage 0, slot 3): embed + layers + peer_copy(0→1, bridge[0,3])
- (stage 1, slot 2): layers + peer_copy(1→2, bridge[1,2])
- (stage 2, slot 1): layers + peer_copy(2→3, bridge[2,1])
- (stage 3, slot 0): layers + output_head + logits DtoH

These 4 host enqueueings target 4 different sub_cluster streams (one
per stage). The work runs concurrently on 4 GPUs.

On stage 0's sub_cluster stream specifically, the queue looks like:
- (tick 0) slot 0 layers, slot 0 DtoH (peer_copy 0→1)
- (tick 1) slot 1 layers, slot 1 DtoH
- (tick 2) slot 2 layers, slot 2 DtoH
- (tick 3) slot 3 layers, slot 3 DtoH

Sequential on the same stream — `hidden_a` row 0 reused safely (each
slot's layers complete + DtoH consumes the row before next slot's
embed overwrites). Same pattern as current pipelined; only the host
issue order changes.

## Implementation plan (#298)

Refactor `forward_decode_pipelined_hybrid` to use the timestep loop.
Most code reuses (lazy bridge init, lane bounces, peer_copy fan-out,
`pipelined_run_slot_through_stage` helper, output_head). Only the
outer two loops change.

```rust
let n_timesteps = n + n_stages - 1;
for t in 0..n_timesteps {
    for src_stage in 0..n_stages {
        let slot_signed = t as isize - src_stage as isize;
        if slot_signed < 0 || (slot_signed as usize) >= n { continue; }
        let slot_idx = slot_signed as usize;

        if src_stage == 0 { embed slot[slot_idx]; }
        pipelined_run_slot_through_stage(.., src_stage, slot_idx, ..);
        if src_stage + 1 < n_stages {
            peer_copy_async_fan_out(src_stage, slot_idx);
        }
        if src_stage == n_stages - 1 {
            output_head + async_logits_dtoh(slot_idx);
        }
    }
}
tail_sync;
```

The stage-major-with-overlap iteration is the lever. Everything else
stays.
