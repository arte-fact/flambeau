# V2.30.a — event-ordered GDN state eliminates 3 cross-lane race guards

## What was wrong

Three guards accumulated over V2.27.d, V2.28.a-i1, and V2.28.c.1 all
bailed out of `forward_prefill_pp_async` for configurations that
produced non-deterministic `last_id` across runs on hybrid (GDN) models:

| cert       | condition                                 | symptom                     |
|------------|-------------------------------------------|-----------------------------|
| V2.27.d    | `u_lanes > 1 && ubatch_size < 128`        | race at small ubatch        |
| V2.27.d    | `u_lanes > 1 && (L % ubatch_size) < 128`  | race on tail ubatch         |
| V2.28.a-i1 | `u_lanes > 1 && gdn_per_rank > 10`        | race at any K on Qwen3.6-27B|
| V2.28.c.1  | `u_lanes > 1 && K > 64` (K = L/ubatch)    | race at long context        |

**Root cause (common):** per-rank `session.caches[layer]::Gdn(state)`
is a single device buffer shared across aux-stream lanes. Concurrent
ubatches on lane 0 and lane 1 both call `gdn_state_step_f32_s128`
which does an in-place read-modify-write on that state. There is no
stream ordering between the two lanes' `state_step` calls, so the
output depends on driver-side scheduling.

## The fix

Event-ordered serialisation of `state_step` across lanes:

1. `ShardedForwardPrefillScratch.gdn_state_events: Vec<Vec<Option<HipEvent>>>`
   (outer = rank, inner = local layer). Allocated in `new_with_lanes`;
   `Some(event)` for recurrent layers, `None` for full-attn layers.
2. `forward_gdn_prefill` takes `state_event: Option<&HipEvent>`:
   - `ev.stream_wait(stream)` before `gdn_state_step_f32_s128`
   - `ev.record(stream)` after
3. `forward_layer_prefill` threads `gdn_state_event` through.
4. Async callers (rank 0 embed+layers, rank r>0 captured, rank r>0
   uncaptured) pull `&mut scratch.gdn_state_events[rank_idx]` out of
   the split-borrow and pass `rank_events[local_idx].as_ref()`.
5. Sync / decode callers pass `None`.

Only `state_step` needs ordering. The rest of the GDN chain
reads/writes lane-local ubatch-sized buffers (qkv_f32, conv_f32,
alpha_beta, etc.) on per-lane `LayerPrefillScratch.gdn`. Event cost
is driver-side — no host sync.

## Qwen3.6-27B Mesh<4> — the headline win (100W/GPU)

V2.28.a-i1 predicted "~3× async prefill win would unlock if GDN state
race were fixed." **We get 3.45–3.76×.**

Triple-run determinism at u_lanes=2 ubatch=128 (previously
hard-bailed — "no known ubatch/K combo is safe"):

| L    | run 1 last_id | run 2 last_id | run 3 last_id | det |
|------|--------------:|--------------:|--------------:|:---:|
|   8  |           328 |           328 |           328 |  ✓  |
|  64  |           279 |           279 |           279 |  ✓  |
| 128  |            79 |            79 |            79 |  ✓  |
| 512  |        248046 |        248046 |        248046 |  ✓  |
| 1024 |        248046 |        248046 |        248046 |  ✓  |
| 2048 |        248046 |        248046 |        248046 |  ✓  |
| 4096 |            62 |            62 |            62 |  ✓  |
| 8192 |        248046 |        248046 |        248046 |  ✓  |

Perf vs u_lanes=1 baseline at 100W/GPU:

| L    | u_lanes=1 | u_lanes=2 | speedup  |
|------|----------:|----------:|---------:|
|  128 |     99.15 |     98.41 |    -1 %  |
|  512 |     98.52 |    160.01 |   +62 %  |
| 1024 |     97.40 |    229.66 | **+136 %** |
| 2048 |     95.28 |    283.81 | **+198 %** |
| 4096 |     91.44 |    315.29 | **+245 %** |
| 8192 |     84.73 |    318.21 | **+276 %** |
| decode |   18.67 |     18.61 |     0 %  |

Decode is flat (expected — prefill-only path). Prefill breaks
100 tok/s at L≥512 for the first time on 27B.

## Qwen3.6-35B Mesh<4> — V2.28.c.1 K>64 fix (100W/GPU)

35B has gdn_per_rank=7 (safe from V2.28.a-i1) but previously blocked
at K>64 by V2.28.c.1. Triple-run at u_lanes=2 ubatch=64 L=8192 →
K=128:

| L    | run 1 last_id | run 2 last_id | run 3 last_id | det |
|------|--------------:|--------------:|--------------:|:---:|
|   8  |           220 |           220 |           220 |  ✓  |
|  64  |           289 |           289 |           289 |  ✓  |
| 128  |            64 |            64 |            64 |  ✓  |
| 512  |           263 |           263 |           263 |  ✓  |
| 1024 |           220 |           220 |           220 |  ✓  |
| 2048 |           198 |           198 |           198 |  ✓  |
| 4096 |        248046 |        248046 |        248046 |  ✓  |
| 8192 |        248046 |        248046 |        248046 |  ✓  |

35B prefill at 100W: 322 tok/s at L=8192 (K=128 was blocked before).

## Qwen3.5-9B Mesh<1> correctness at ubatch=64 (V2.27.d + V2.28.c.1)

9B Mesh<1> u_lanes=2 ubatch=64 triple-run alternating GPU 1/2 — this
config triggers BOTH ubatch<128 (V2.27.d) AND K=128 at L=8192
(V2.28.c.1) simultaneously:

| L    | run 1 (dev=1) | run 2 (dev=2) | run 3 (dev=1) | det |
|------|--------------:|--------------:|--------------:|:---:|
|   8  |            73 |            73 |            73 |  ✓  |
|  64  |         95761 |         95761 |         95761 |  ✓  |
| 128  |         95759 |         95759 |         95759 |  ✓  |
| 512  |         24178 |         24178 |         24178 |  ✓  |
| 1024 |         96519 |         96519 |         96519 |  ✓  |
| 2048 |            82 |            82 |            82 |  ✓  |
| 4096 |         96128 |         96128 |         96128 |  ✓  |
| 8192 |            11 |            11 |            11 |  ✓  |

Note: Mesh<1> u_lanes=2 is a *correctness* cert only — Mesh<1>
doesn't benefit from u_lanes=2 (one GPU, no cross-rank overlap), and
at ubatch=64 the kernel launch overhead overwhelms any concurrency
benefit. The Mesh<1> 9B perf table is not the headline case; 27B
Mesh<4> above is.

## Guards removed

`forward_prefill_pp_async` in `crates/models/qwen3-moe/src/forward/pp.rs`
had ~60 LOC of `bail!` code covering the three guards. All removed;
replaced with a single 8-line comment citing V2.27.d / V2.28.a-i1 /
V2.28.c.1 as the resolved issues.

## Ship status

- Build: green (`cargo build --release -p flambeau-qwen3-moe --features hip`).
- 27B Mesh<4> u_lanes=2 ubatch=128: **deterministic + 3.76× prefill win**
  at L=8192 (84.7 → 318.2 tok/s, 100 W/GPU).
- 9B Mesh<1> u_lanes=2 ubatch=64: deterministic at two previously
  forbidden configs (ubatch<128 AND K=128 simultaneously).
- 35B Mesh<4> u_lanes=2 ubatch=64 L=8192 (K=128, V2.28.c.1 trigger):
  **bit-exact across 3 runs** at every L. Prefill 322 tok/s at L=8192.
- All three guards removed from code.

## Regeneration

```bash
# 27B triple-run (the headline):
TEST_BIN=$(ls -t target/release/deps/perf_baseline_qwen35_9b-* | grep -v '\.d$' | head -1)
for i in 1 2 3; do
  FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=128 FLAMBEAU_U_LANES=2 FLAMBEAU_MESH_RANKS=4 \
    FLAMBEAU_QWEN35_GGUF=/artefact/models/Qwen3.6-27B-Q8_0.gguf \
    $TEST_BIN perf_baseline_qwen35_9b --nocapture
done

# 35B K=128 triple-run (V2.28.c.1 cleared):
TEST_BIN_35B=$(ls -t target/release/deps/perf_baseline_qwen3_moe-* | grep -v '\.d$' | head -1)
for i in 1 2 3; do
  FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=64 FLAMBEAU_U_LANES=2 FLAMBEAU_MESH_RANKS=4 \
    FLAMBEAU_QWEN3_GGUF=/artefact/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
    $TEST_BIN_35B perf_baseline_qwen3_moe --nocapture
done

# 9B Mesh<1> ubatch=64 alternating GPUs:
for i in 1 2 3; do
  dev=$((((i-1)%2)+1))
  HIP_VISIBLE_DEVICES=$dev FLAMBEAU_ASYNC_UBATCH=1 FLAMBEAU_UBATCH=64 \
    FLAMBEAU_U_LANES=2 FLAMBEAU_MESH_RANKS=1 \
    $TEST_BIN perf_baseline_qwen35_9b --nocapture
done
```
