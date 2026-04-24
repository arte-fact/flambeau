# V2.25.f — async PP sweep (ubatch × u_lanes) baseline

Pre-refinement sweep of the V2.25.d async ubatch path on Qwen3.5-9B-Q4_1
Mesh<4>, prefill L=1024. Identifies the optimal (ubatch, u_lanes) pair
and characterises the bottleneck that V2.25.g needs to fix.

## Results (tok/s, prefill L=1024)

| ubatch | u_lanes=2 | u_lanes=3 | u_lanes=4 |
|---:|---:|---:|---:|
| 128 | 517.57 | 516.02 | 516.99 |
| 256 | 632.28 | 633.71 | 633.31 |
| **512** | **696.15** | 624.00 | 668.91 |

Sync reference (same model + mesh): **765.75 tok/s**.
Turbo reference: 1569 tok/s (non-apples — uses `n_batch > n_ubatch`
pipelining via ggml_backend_sched, a superset of what we're building).

**All async configs regress vs sync.** Best config (ubatch=512,
u_lanes=2) is −9.1 %.

Parity: `last_id=220` on every config — bit-exact vs sync. Correctness
of the async path is confirmed; only perf is TBD.

## Two-axis read

**ubatch size** (finer = more chunks → worse):
- 128 (8 chunks): ~517 tok/s regardless of u_lanes — saturated in
  per-chunk overhead.
- 256 (4 chunks): ~633 — still saturated; u_lanes doesn't help.
- 512 (2 chunks): 624-696 — some u_lanes variation, 2 is best.

**u_lanes count** (at ubatch=512 only — the others are saturated):
- 2: 696 — best; minimum lanes for 2-ubatch overlap.
- 3: 624 — worse; extra lane adds alloc overhead but no benefit.
- 4: 669 — similar to 3; allocated but unused.

## Root cause (hypothesis, pending rocprofv3 trace for V2.25.g)

`HipCluster.bounces` is a **single pinned-host slab per rank**, not
per-lane. When rank 0 issues async DtoH for ubatch 0 on lane 0's aux
stream, the pinned bounce on rank 0 is in use. When rank 0 later issues
async DtoH for ubatch 2 on lane 0 (same rank, same lane), the driver
serialises on the bounce memory — because it's shared, the DAG edge
from the earlier HtoD (on rank 1) must complete before the new DtoH
can start writing.

This effectively makes the "async" pipeline behave as serial across
ubatches on the same rank, which is most of the work. Larger ubatches
are less regressive because there are fewer cross-ubatch bounce
conflicts per unit of prefill work.

The u_lanes count doesn't matter because each lane still serialises on
the shared bounce — more lanes = more aux streams kicked into flight,
all queued behind the same bounce dependency.

## V2.25.g target — per-lane bounces

Change `HipCluster.bounces: Vec<RankBounce>` →
`Vec<Vec<RankBounce>>` indexed by `(rank, lane)`. Each lane gets its
own pinned slab so concurrent DtoH across lanes truly overlap.

Memory cost: `rank × lane × chunk_bytes`. For 9B hidden=5120 at
ubatch=512: 5120 × 2 × 512 = 5 MiB/lane. At 4 ranks × 2 lanes = 8 slabs
= 40 MiB pinned host. Acceptable.

Expected gain (post-refinement): L=1024 async 696 → ≥ 1100 tok/s (close
to turbo's single-request scaling ceiling). Per TD-Pipe paper §2.3 and
llama.cpp PR #6017 methodology, 1F1B-style microbatch PP should scale
linearly with pipeline depth minus bubble cost.

## Other observations

- Runs at u_lanes > 2 allocate more scratch (+ubatch_size × hidden × 2
  bytes per extra lane) but current driver path gets zero additional
  overlap. Keep u_lanes=2 as the default until V2.25.g lifts the
  bounce bottleneck.
- UD-Q4_K_S 8-token parity remains bit-exact (async path doesn't
  trigger at L=1).
- Sync path (no FLAMBEAU_ASYNC_UBATCH) unchanged — 765 tok/s matches
  pre-V2.25 baseline.

## Regeneration

```
for u in 128 256 512; do for lanes in 2 3 4; do
  FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=$u FLAMBEAU_U_LANES=$lanes \
    FLAMBEAU_MESH_RANKS=4 FLAMBEAU_QWEN35_GGUF=...Qwen3.5-9B-Q4_1.gguf \
    ./target/release/deps/perf_baseline_qwen35_9b-* perf_baseline_qwen35_9b \
    --nocapture 2>&1 | grep "prefill L=1024"
done; done
```

Closes task 146. Next: V2.25.g adds per-lane bounces + rocprofv3
profile confirms the bottleneck shifts.
