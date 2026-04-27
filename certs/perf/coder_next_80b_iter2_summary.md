# CN-80B-5 (perf iter 2) — HipEvent section profiler + Coder-Next pp4 breakdown

## Infrastructure shipped

- `flambeau_backend_hip::HipEvent::new_timing(device_id)` — timing-
  enabled event constructor (the existing `new()` uses
  `hipEventDisableTiming` for cheap ordering).
- `HipEvent::synchronize()` and `HipEvent::elapsed_ms_since(start)` —
  wraps `hipEventSynchronize` + `hipEventElapsedTime`.
- `flambeau_backend_hip::profile` module — thread-local section timer:
  - `profile::enable()` arms recording on this thread
  - `profile::mark(name, device, stream)` records a section boundary
    on the supplied stream (no-op when disabled, ~ns)
  - `profile::flush()` syncs every recorded event, computes pairwise
    `hipEventElapsedTime` deltas (skipping cross-device pairs), and
    aggregates by section name. Sorted by total ms desc.
- `forward_one_token_pp` instrumented with markers at: `step_start`,
  `embed_done`, `stage_start`, `stage_end`, `output_head_start`. F16
  parity preserved (35B-A3B prefill L=1→11, L=2→271 still bit-exact
  against llama.cpp). The same instrumentation is in
  `forward_one_token_pp_inner` (used by `forward_one_token_pp_logits`).

## Profile — Coder-Next-Q4_0 pp4 decode (32 steps, post-warmup)

Wall: **773.17 ms / 32 tokens = 24.16 ms/tok = 41.39 tok/s**.
(Matches the CN-80B-3 baseline of 41.3 tok/s.)

| section            | total ms | count | mean ms/call | % wall |
|--------------------|---------:|------:|-------------:|-------:|
| stage_end          |   719.89 |   128 |        5.624 |  93.1% |
| embed_done         |     5.47 |    32 |        0.171 |   0.7% |
| stage_start        |     0.22 |    32 |        0.007 |   0.0% |
| output_head_start  |     0.15 |    32 |        0.005 |   0.0% |
| **attributed**     |   725.73 |       |              |  93.9% |

Each of 4 ranks owns 12 of 48 layers. `stage_end` aggregates the per-
stage layer-chain wall captured at each rank's `stage_start →
stage_end` event pair: **5.62 ms per stage, 0.47 ms per layer per
token**.

## Findings

- **Layer compute owns 93 % of decode wall.** Embed (0.7 %), peer-
  copies + bind (≤ 0.05 %), and the output-head dispatch (≤ 0.05 %)
  are all negligible.
- **Communication is not the bottleneck.** The PP host-bounce peer-
  copy was suspected to add latency on this 80 B model — the profile
  rules it out. Each stage's wait+sync at `stage_start` is < 10 µs,
  meaning the previous rank's output lands well before the current
  rank's first kernel needs it.
- **All headroom is inside the layer body.** Per-layer 0.47 ms ÷
  ~7-10 kernel launches/layer ≈ 50-70 µs per kernel. Common decode
  kernels at this size: indexed-MoE MMVQ (Q4_0/Q4_1) for the top-8
  active experts, sigmoid-gated full-attn or GDN, attention decode,
  router GEMV, topk_f32 over 512 experts, RMSNorm+quant fusions,
  cast/output projections.

## No iter-2 lever shipped

Section-level granularity confirms the layer body is the right place
to dig but doesn't pick a kernel. Iter-3 (CN-80B-6) extends the
markers INSIDE the per-layer call chain (split full-attn vs GDN,
attn vs FFN, MoE vs shared expert) so the next iteration has a
specific kernel to optimize.

The infrastructure is now in place — iter-3 just needs more `mark()`
call sites at the layer-internal section boundaries.

## Closes

- CN-80B-5 #131 — infrastructure + first profile shipped, no perf
  delta. Iter-3 picks the kernel-level lever once finer instrumentation
  lands.
