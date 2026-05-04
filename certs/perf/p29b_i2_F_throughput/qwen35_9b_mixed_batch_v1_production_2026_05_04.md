# Sarathi mixed-batch v1 — production wiring null at 4×MI50/pp2tp2

**Date:** 2026-05-04
**Rig:** 4× MI50 (gfx906) PCIe 3.0 x16, ROCm 7.1.1
**Model:** Qwen3.5-9B-Q4_1 (qwen35 — hybrid GDN+full-attn+MoE)
**Topology:** pp2tp2 over devices [0,2,1,3]
**Task #:** #306 + #308 production cert
**Commits:** `73c9e06` (server wiring), this branch

## TL;DR

The v1 mixed-batch dispatch (#306 server wiring) is **correct
end-to-end** — both legacy and mixed return coherent text on every
workload tested — but **does not deliver a throughput win** in
production scheduler-driven traffic on this rig.

| workload | legacy wall (ms) | mixed wall (ms) | mixed vs legacy |
|----------|-----------------:|----------------:|----------------:|
| 2 concurrent, K=487/N=2, simultaneous arrival | 10763 | 10808 | 1.00× (noise) |
| 2 concurrent, staggered (r2 prefill mid-r1-decode) | 7085 | 8104 | **0.87× (slower)** |
| 4 concurrent (3 decoders + 1 long-prefill arrival) | 16903 | 22876 | **0.74× (slower)** |
| 4 concurrent, mixed_chunk=64 instead of 256 | 16903 | 26608 | **0.64× (worse)** |

All wall numbers are total time for the bench's heaviest request to
finish. Total tokens identical across both paths.

## Why the lever doesn't activate

The microbench at K=512/N=4 (single-call comparison) showed mixed at
**1.063×** and at K=128/N=16 **1.174×**. Production wiring underperforms
by 14–36% — the gap is the **per-iteration scheduler overhead**.

Per mixed iteration (this implementation):
1. Lock `mixed_scheduler`, run `next_iteration()` to build plan.
2. Lock all referenced inflight slot mutexes (sorted, deterministic).
3. Lock the mixed scratch.
4. Build `sessions: Vec<&mut HybridSession>`, `chunk`, `slots` structs;
   compute pool→vec slot index remap.
5. Call `forward_decode_mixed_hybrid(...)`.
6. Send results to per-request mpsc channels; remove channels.
7. Drop guards.

Compared to legacy `dispatch_batched_pending`:
- Legacy prefill: ONE `forward_prefill_hybrid_logits` call upfront,
  no per-step overhead, kernel internally batches across the prompt.
- Legacy decode: `dispatch_batched_pending` coalesces N decodes into
  one `forward_decode_batched_hybrid` call. Per-step overhead is
  amortized over many tokens.

Mixed adds the chunk-prefill cost to **every decode iteration** that
co-runs with a chunk. For chunk=256 and decode N=3:
- Legacy decode step: ~50ms
- Mixed iteration with chunk: ~280ms (256-token compute + 3 decodes)

Net effect: r4's prefill chunks slow down all 3 decoders by ~5×
during the chunk-arrival window, eating ~560ms × 3 = 1680ms of
decoder TPOT. Mixed-batch only "saves" the prefill→decode handoff
latency (~50ms) — net 1.6 sec worse.

## Per-iteration overhead breakdown (estimated)

Each iteration has ~5–10ms of host-side overhead (channel setup,
mutex acquisitions, scratch mutex, sessions vec building, leader
election). With chunk=64 / N=3, r4 has 487/64 = 8 chunks =
8 mixed iterations + many decode-only iterations. Total iterations
~80–100. 80 × 7ms = 560ms of pure overhead beyond the GPU work.

With chunk=256 / N=3: 487/256 = 2 chunks = 2 mixed iterations.
Smaller chunk = more iterations = more host overhead. That's why
chunk=64 was 16% slower than chunk=256.

## What would unlock the lever

1. **Varlen attention kernel (#303, deferred)**: cuts per-layer
   kernel calls in half. Saves ~25% of the chunk-iteration GPU time
   and reduces TPOT impact on co-batched decoders.
2. **Higher N (≥8 decoders)**: more decodes amortize the prefill
   chunk's compute. The microbench K=128/N=16 showed 1.174×; that
   ratio scales with N. Production needs ≥8 concurrent users for the
   amortization to outweigh per-iteration overhead.
3. **Larger model (35B-A3B)**: per-decode HBM-bound work is bigger
   relative to scheduler overhead. The 9B per-layer decode is too
   fast for the host-side scheduling to disappear into.
4. **Async leader pattern**: vLLM v1 RFC #11945 spawns the leader as
   a separate task that polls the scheduler and dispatches
   continuously, decoupling the dispatcher loop from per-request
   handlers' latency. Our v1 has handlers act as ad-hoc leaders,
   adding handler-context overhead per iteration.

## Recommendation

Mixed-batch v1 ships behind opt-in flags (`FLAMBEAU_MIXED_BATCH=1`)
and stays available for future experimentation. It is **not the
default** — production traffic on this rig should continue using the
legacy `FLAMBEAU_BATCHED_DECODE=1` path.

The 3× cert gate is **not closeable** by mixed-batch v1 on this rig
at current Qwen3.5-9B/pp2tp2 scale. The v1 cert remains the v2
topology bench (1.34× at PP=4, 0.92× at pp2tp2) until either a
varlen kernel or a higher-N workload reshapes the equation.

## Code state

The wiring is correct + tested:
- Driver (#304): `forward_decode_mixed_hybrid` parity bit-exact at
  small K, top-1 match at K=128/N=4.
- Scheduler (#305): unit-tested + integration-tested.
- Server wiring (#306): live-validated coherent output on every
  workload.

The implementation is ready to be reactivated when the prerequisites
land:
- #303 varlen-attn kernel
- New dense large-context model that benefits from chunked-prefill

For the v2 milestone, the path forward is **NOT more mixed-batch
optimization** — it's investigating other levers documented in
`doc/V1.x/gpu_idle_fill_levers.md`:
- Lever 3 (out of scope): prefill-decode disaggregation requires
  xGMI/NVLink rig.
- Other directions: dispatch matrix tuning, fused decode-stage
  kernels, KV-cache layout experiments.

## References

- `doc/V1.x/sarathi_mixed_batch_design.md` — design doc
- `certs/perf/p29b_i2_F_throughput/qwen35_9b_mixed_batch_v1_2026_05_04.md` — microbench cert
- `crates/server/src/mixed_scheduler.rs` — scheduler primitive
- `crates/server/src/routes.rs::MixedBatchCtx` + helper methods
- vLLM v1 RFC #11945 — async two-stage engine loop reference
