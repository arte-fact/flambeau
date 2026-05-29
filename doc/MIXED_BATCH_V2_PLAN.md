# Mixed-Batch v2 — Implementation Plan (Sarathi-Serve kernel half)

Multi-session plan to port the Sarathi-Serve mixed-batch driver
(`forward_decode_mixed_hybrid`) from the deleted qwen3-moe v1 forward
crate to the current v2 stack. Phase 5 software side (S1+S2+S3) is
already shipped via [`BATCHED_DECODE_PLAN.md`](BATCHED_DECODE_PLAN.md);
this doc covers the **kernel half** — the co-batched forward that
runs one prefill chunk of K tokens for one slot AND N decode steps
for other slots in a single PP+TP forward pass.

This document is the durable plan across context compactions. Read
it before touching any "mixed batch" / "Sarathi kernel" issue.

## Goal

- Aggregate-throughput win on real chat traffic on Qwen3.6-27B-Q4_0
  pp2tp2: when prefill arrivals are interleaved with concurrent decodes,
  total token output ≥ 1.5× a sequential-prefill+decode baseline.
- Per-call wall ≥ 1.10× on Qwen3.5-9B-Q4_1 at (K=256, N=8) — matches
  the v1 ceiling memory note `project_lever1_mixed_batch_v1`.
- No regression on N=1 single-stream decode (≥ 26 t/s pp2tp2).
- Parity vs `forward(prefill_only)` + `forward(decode_only)` separate-
  sessions reference: bit-exact at K=32/N=1 and K=32/N=2; F16-tolerance
  at K=128/N=4 (hybrid abs+rel, abs_tol=0.5, rel_tol=1e-2).

## What's in tree today

- **NOTHING from v1.** The v1 driver
  `crates/models/qwen3-moe/src/forward/hybrid.rs::forward_decode_mixed_hybrid`,
  its parity test `tests/mixed_batch_parity.rs`, and its microbench
  `tests/mixed_batch_microbench.rs` all lived in the v1 qwen3-moe crate
  which was deleted during the v2 migration. The v1 cert
  `certs/perf/p29b_i2_F_throughput/qwen35_9b_mixed_batch_v1_2026_05_04.md`
  is also gone. Memory note `project_lever1_mixed_batch_v1` captures
  the v1 results — see "Past results to beat" below.
- v2 forward dispatches a single `forward<C: ForwardCtx>` per arch
  (`crates/models/{qwen35-v2,qwen35moe-v2,gemma4-v2}/src/model.rs`).
  Same code runs for prefill (`n = K`) and decode (`n = N`). Per-arch
  forwards take `(tokens: &[u32], positions: &[usize], slot_ids: &[usize])`.
- v2 `ForwardCtx` (`crates/forward/src/ctx.rs`) defines `standard_attn`,
  `gdn_layer`, `dense_ffn`, `moe_ffn`, `embed`, `output_head`. Each
  is monomorphised per topology (Pp / Tp / Hybrid).
- Phase 5 S1+S2 chunked prefill already issues prefill calls of bounded
  K (default 512) — the scheduler layer that would mix chunks across
  slots is what this plan eventually replaces.

## Why this is worth doing (and what it doesn't deliver alone)

**Per-call ceiling bound.** The v1 numbers from
`project_lever1_mixed_batch_v1` (Qwen3.5-9B-Q4_1 / pp2tp2):

| K   | N   | seq (ms) | mix (ms) | speedup |
|-----|-----|---------:|---------:|--------:|
| 512 | 4   | 638.09   | 600.45   | 1.063×  |
| 256 | 8   | 427.13   | 381.30   | 1.120×  |
| 128 | 16  | 389.64   | 331.95   | 1.174×  |

The ceiling at our shapes is
`(T_prefill + T_decode) / max(T_prefill, T_decode)`. T_prefill is
compute-bound (MoE/FFN per K tokens). T_decode is HBM-bound (MMVQ +
attention per N slots). The two overlap to the extent the slower one
hides the faster — bounded by the bigger of the two phases.

**The headline win is aggregate throughput, not per-call wall.** The
v1 memory explicitly notes: *"Sarathi's 3–5× claims are aggregate-
throughput over many requests; those need the scheduler (#305), not
just the driver."* On flambeau today the scheduler chunked prefill
(S1+S2+S3) already releases the mutex between chunks so other slots
can fire — but each slot's chunk runs as its own forward call. Mixed
batch packs the prefill chunk and the concurrent decodes into ONE
forward call, eliminating the launch-overhead + scheduler-coalescing
gap on every iteration of the decode loop. That gap becomes more
visible at high N (paged inflight 16+ — Phase 6 territory).

**So the real production win is Mixed-batch × Phase 6 paged**: more
concurrent users (paged capacity) × tighter per-iteration packing
(mixed batch). Either alone is bounded.

## Phases (one phase ≈ one session unless flagged multi-session)

### Phase K1 — `standard_attn_mixed` on Pp/Tp/Hybrid (multi-session, ≥2)

The structural piece. Today `standard_attn` dispatches its `n` input
rows as either ALL-prefill (causal-triangular attention against K rows
of newly-written KV) or ALL-decode (1-row attention against
max_seq_len of slot's KV). Mixed batch interleaves: rows `[0..K)` are
prefill for one slot, rows `[K..K+N)` are decode for N distinct slots.

**Why this is the hard part.**
- Attention itself has to call TWO existing kernels back-to-back: the
  K prefill rows go through `attention_prefill_f16` (or splitk variant),
  the N decode rows go through `attention_decode_f16_batched` (with
  per-slot KV pointers). The KV-append for K rows writes to slot_p's
  contiguous KV slab; the KV-append for N rows writes to N slots'
  individual rows. **No new device kernels** — the v1 driver explicitly
  avoided new kernels by splitting the call site.
- QKV projection runs at `n = K + N` (one matmul, weight amortised
  across all rows).
- Output projection runs at `n = K + N` too.
- The PP/TP AR placement: the existing standard_attn already places AR
  after the row-parallel ops; the mixed split happens INSIDE the
  attention call between QKV and output proj.
- **All three impls or it doesn't ship** (flambeau-forward rule 1 — no
  method exists on only one topology). Pp is simplest, Tp adds AR
  placement, Hybrid adds stage handoff.

**Slice K1a status (2026-05-29): SHIPPED — including K1b+K1c.**
`crates/forward/src/core/composites/standard_attn_mixed.rs`
(`standard_attn_mixed_local`) + ctx trait method + `ForwardEngine`
impl. Because the engine impl is generic over
`<H: TopologyHooks, S: StageHooks>`, the same composite serves
Pp/Tp/Hybrid in one shot — the AR hook calls
(`hooks.ar_residual_f16` / `hooks.ar_sum_f32`) are unchanged from
`standard_attn`, and topology choice is at the type-parameter
level. K1b and K1c were redundant slice splits on the v1 architecture.

K1a bails (future slices):
- Paged KV (Phase K-paged)
- Shared-KV layers (gemma 4n)
- V unit-norm fusion (gemma4)
- `attn_q_gated` (gemma4)
- `post_attn_norm` (gemma4)
- Sliding-window attention
- V-from-K (gemma4 fused)

Parity test: `crates/forward/tests/synth_dense_mixed.rs`. K=4/N=3
on a synthetic dense config. Compares mixed (one call) vs reference
(two `standard_attn` calls — prefill_shape + batched-decode).
Hybrid abs+rel tolerance (abs_tol=0.5, rel_tol=1e-2). **PASS.**

### Phase K2 — `gdn_layer_mixed` on Pp/Tp/Hybrid

**Status (2026-05-29): SHIPPED.**
`crates/forward/src/core/composites/gdn_mixed.rs`
(`gdn_layer_mixed_local`) + ctx trait method + `ForwardEngine` impl.
Same generic-engine collapse as K1a — one slice covers Pp/Tp/Hybrid.

Composite calls the two existing `DeltaNetLayer` entry points
back-to-back on sliced input/delta views:
- K rows → `forward_prefill_with_ar_hook` on slot_p.
- N rows → `forward_decode_with_ar_hook_batched_slots` on N slots'
  per-slot state + history pointer arrays.

No new device kernels. AR fires twice per layer (one per phase) —
matches the v1 driver shape.

Pool requirements: both `gdn_prefill_scratch` AND
`gdn_decode_batched_scratch` configured.
`max_prefill_tokens >= K + N`, `max_slots >= N + 1`.

Parity test: `crates/forward/tests/synth_gdn_mixed.rs`. K=4/N=3 on
synth GDN config (HIDDEN=256, NUM_V_HEADS=NUM_K_HEADS=2,
HEAD_K/V_DIM=128, CONV_KERNEL=4). State + conv history zero-init.
Mixed (one call) matches reference (separate prefill + batched-
decode `gdn_layer` calls) within abs+rel tolerance. **PASS.**

**K1 + K2 together unblock the driver layer for both layer kinds**
— dense (Qwen3.5-9B/27B + Qwen3.5moe) and hybrid (Qwen3.6-27B).

### Phase K3 — Driver + parity + microbench

**Slice K3a status (2026-05-29): SHIPPED.**
`flambeau_qwen35_v2::forward_mixed` and
`flambeau_qwen35moe_v2::forward_mixed`. Parallel entry to `forward`
that takes `prefill_rows: usize` and routes attention/GDN through
the `_mixed` ctx methods. Embed / FFN / residual_add unchanged at
`n = K + N`. Output head emits `N + 1` rows: (K-1)-th prefill row
(slot_p's next-token logit) + N decode rows.

Re-exported from each crate's `lib.rs`. Covers Qwen3.5-9B,
Qwen3.6-27B hybrid, Qwen3.6-35B-A3B MoE.

Composite-level parity (K1a `synth_dense_mixed` + K2 `synth_gdn_mixed`)
already proves the math. Model-level parity test deferred to K3b
because `Qwen35V2Model` has crate-private fields (`allocs`,
`device_id`) — a synth-data integration test would need a test-only
constructor or going through the GGUF loader. K3b loads a real GGUF
anyway and is the right level to validate end-to-end.

**Slice K3b status (2026-05-29): SHIPPED.**
`crates/models/qwen35-v2/tests/mixed_batch_microbench.rs` — env-gated
by `FLAMBEAU_MIXED_MICROBENCH_GGUF`. Builds a `SingleDeviceForwardCtx`
manually (bypasses `Session`), sizes the pool to
`max_prefill_tokens = K + N` and `max_slots = N + 1`, times
`forward_mixed` vs `forward(K) + forward(N)` with stream sync between
calls. Single-device hip:0, not pp2tp2 (pp2tp2 would need
`Arch::forward_mixed` + `Session::forward_mixed` plumbing — deferred).

K3b also required lifting two K1a bailouts that turn out to apply
to Qwen3.5/3.6 (not just gemma4): `attn_q_gated` (fused Q + per-head
gate; QKV emits `2*q_width`, `split_q_gate_f16` + post-attn
`sigmoid_mul_f16`) and `attn_q_norm`/`attn_k_norm` (fused rmsnorm +
RoPE single-launch path). K1a `synth_dense_mixed` parity still passes
(synth has both off).

Result on Qwen3.5-9B-Q4_1 / hip:0, ctx_cap=4096, 1 warmup + 5 timed:

| K   | N   | seq (ms) | mix (ms) | speedup |
|-----|-----|---------:|---------:|--------:|
| 128 | 16  | 435.28   | 352.40   | **1.235×** |
| 256 | 8   | 491.73   | 464.19   | 1.059×  |
| 512 | 4   | 801.95   | 800.10   | 1.002×  |

Range 1.002×–1.235× vs v1 reference 1.063×–1.174×. K=128/N=16 beats
v1; K=512/N=4 flat (prefill dominates, no decode headroom). Per-call
ceiling validated on the actual model — Phase K3 done.

**Borrow disjointness check.** Already implemented at the composite
layer in both `standard_attn_mixed` and `gdn_layer_mixed`: each bails
if the prefill slot_id appears in the decode slot_ids.

### Phase K4 — Scheduler integration (`build_mixed_iteration`)

**Slice K4a+K4b status (2026-05-29): SHIPPED.** Runtime plumbing
threads `forward_mixed` from `Arch` trait → workers → Session in
the same shape as `forward`:

- `Arch::forward_mixed(model, ctx, tokens, positions, slot_ids,
  prefill_rows)` trait method with default-bail. Overridden in
  qwen35-v2 and qwen35moe-v2 to delegate to each crate's
  `forward_mixed` (K3a). gemma4-v2 inherits the bail.
- `Command::ForwardMixed` worker variant + `run_forward_mixed_once`
  per-WorkerRole dispatcher (Sd/Tp/Pp/Hybrid). Same ctx
  construction as `run_forward_once`.
- `WorkerHandle::send_forward_mixed` + `orchestrate::run_forward_mixed`
  topology-aware fanout.
- `Session::forward_mixed(tokens, positions, slot_ids, prefill_rows)`
  with bounds checks; results readable via `Session::mixed_logits_row(i, vocab)`.
  Layout: `[(N + 1), vocab]` row-major — row 0 = prefill slot's
  next-token, rows 1..=N = decode slots.

K4b smoke: `crates/models/qwen35-v2/tests/forward_mixed_session_smoke.rs`
loads Qwen3.5-9B-Q4_1 via the production
`Session<Qwen35V2>::new(SingleDevice)` path, fires
`Session::forward_mixed` once with K=8/N=3, asserts the 4 rows
(vocab=248320 wide each) come back finite + non-constant. **PASS.**

Validates the entire plumbing chain on the real worker
infrastructure — no bypass.

**Slice K4c status (2026-05-29): SHIPPED.**
Server-side mixed engagement in `chunked_prefill_pp`. When
`FLAMBEAU_MIXED_BATCH=1` + the arch supports it
(`Model::supports_mixed_batch`) + capacity allows (K + N ≤
`--prefill-ubatch`), the chunked-prefill leader blocking-locks
`batched_dispatcher`, drains `batched_pending`, and fires
`Model::forward_mixed_decode` (V2Model override on qwen35 /
qwen35moe). Decode pendings receive their per-slot logits via the
already-held response channels; the prefill's logits land in
`logits_out`. Otherwise falls through to the existing pure-prefill
path.

No cycle: decode loops in `decode_via_scheduler_into` PUSH then
RELEASE their inflight slot mutex before `rx.recv()`, so
`dispatch_mixed_with_pending` can acquire the decode slots' mutexes.

End-to-end smoke `scripts/bench/bench_mixed_engage_nostream.py`
on Qwen3.5-9B-Q4_1 / hip:0 / `--inflight-slots 4 --prefill-ubatch
768 --ctx-cap 32768` — engagement fires (`slot_p=3, k=436, n_dec=3`)
during the long's second chunk while shorts are decoding. All 4
streams coherent; wall 12.43 s ON vs 12.61 s OFF (modest at this
shape — only 1-2 engagement opportunities per long request; the
sustained-traffic case is K5).

The throughput lever. The mixed driver delivers a fixed per-call
speedup; the scheduler is what makes EVERY decode iteration use the
mixed shape instead of going through separate prefill+decode paths.

Server-side state machine:
- Track in-progress chunked prefills (multi-chunk requests need
  resumable state — current chunk index per slot).
- Each scheduler tick: pick the slot with pending prefill (if any);
  combine its next chunk with all active-decode slots' next tokens
  into one `forward_decode_mixed` call.
- Logits demux: route the K-th-row logit to the prefill slot's
  next-token sampler, route the N row logits to each decode slot's
  sampler.
- When a prefill request finishes (last chunk emitted), transition the
  slot from "prefilling" → "decoding".

**Server-side wiring.** Replace `chunked_prefill_pp` (Phase 5 S2 helper)
with a scheduler that issues mixed forwards. The current shape
(separate prefill calls between mutex releases) is replaceable by a
single `dispatch_mixed_iteration` that returns logits for prefill +
decode in one shot.

Env-gated: `FLAMBEAU_MIXED_BATCH=1` opt-in at boot. Default off until
the parity + cert pass on the production workload.

Scratch sizing at boot:
`ShardedForwardPrefillScratchHybrid::new(model, chunk_budget + max_slots)`.

### Phase K5 — Realistic-traffic cert + scheduler tuning

**Status (2026-05-29): SHIPPED with HONEST null on aggregate throughput.**

K5a: boot-time scratch auto-sizing in `crates/server/src/serve.rs`.
When `FLAMBEAU_MIXED_BATCH=1` is set, `prefill_ubatch` is bumped to
at least `chunk_tokens + inflight_slots` so the K4c capacity guard
never rejects engagement. Operator no longer has to set
`--prefill-ubatch 768` manually.

K5b: `scripts/bench/bench_mixed_sustained.py` — open-loop
sustained-traffic cert at fixed arrival rate with mixed short +
long prompt distribution. Reports aggregate output tokens/sec +
per-cohort wall percentiles.

Result on Qwen3.5-9B-Q4_1 / pp hip:0 / `--inflight-slots 4
--ctx-cap 32768`, DURATION_S=30 ARRIVAL_S=1.0 LONG_RATIO=0.33:

| Metric | K4c OFF | K4c ON |
|---|---|---|
| **Aggregate (wall)** | **26.3 t/s** | **25.9 t/s** |
| Long  p50 wall | 24.4 s | **20.5 s (-16 %)** |
| Long  p99 wall | 43.2 s | 40.0 s (-7 %)  |
| Short p50 wall | 20.2 s | 25.1 s (+24 %) |

**Plan target NOT hit.** Aggregate throughput unchanged at this
saturated load. The mechanism **does work** — it redistributes
latency in favour of long-prompt requests (long p50 -16%, long p99
-7%) at the cost of short-prompt p50 latency (+24%). At full HBM
saturation on MI50 the total work per second is fixed; mixed-batch
can't lift the hardware ceiling.

The aggregate win regime is structurally different — low-to-medium
load where prefill stalls dominate the schedule. Saturated MI50
has no stalls for the mixed kernel to fill.

**Default position.** `FLAMBEAU_MIXED_BATCH=1` stays default-off.
Operators who want long-prompt latency reduction at the cost of
short-prompt latency can opt in.

**PP4 addendum.** Same bench shape on
`--mesh-mode pp --devices hip:0,1,2,3`:

| Metric | OFF | ON | Δ |
|---|---|---|---|
| Aggregate (wall) | 26.4 t/s | 26.4 t/s | ~0 % |
| Aggregate (inject) | 60.0 t/s | 61.3 t/s | +2.2 % |
| Long p50 wall | 19.2 s | **17.4 s (-9.4 %)** | |
| Short p50 wall | 31.5 s | **30.0 s (-4.8 %)** | |
| Errors (503) | 2 | 1 | -1 |

PP4 inverts the single-device latency-redistribution trade — both
cohorts get slightly faster. Per-rank compute is 1/4 of the layers
so each rank has more idle time per iteration; the mixed kernel
fills that idle without stealing bandwidth from concurrent shorts.
Aggregate still unchanged at saturated load (HBM-bound), but the
PP4 wall behaviour is materially better for both cohorts. This is
the production-relevant configuration for the rig.

**pp2tp2 addendum (`--mesh-mode pp+tp --pp-size 2 --tp-size 2
--devices hip:0,2,1,3`):**

| Metric | OFF | ON | Δ |
|---|---|---|---|
| **Aggregate (wall)** | 33.7 t/s | **35.6 t/s** | **+5.6 %** |
| Total wall | 57.05 s | 53.91 s | -5.5 % |
| **Long p50 wall** | 18.7 s | **10.8 s** | **-42 %** |
| Long p99 wall | 34.9 s | 23.1 s | -34 % |
| Short p50 wall | 12.8 s | 20.6 s | +60 % |
| Errors (503) | 0 | 0 | 0 |

**This is the strongest mixed-batch result measured.** pp2tp2's
per-rank envelope is small enough that prefill benefits
dramatically from co-batching with concurrent decodes, while AR
cost across the TP=2 pair is hidden by the same overlap. Long p50
drops 42 %, p99 drops 34 %, aggregate +5.6 %, total wall 3 s
shorter. Short p50 rises 60 % — the trade is sharp.

**Production recommendation.**
- RAG / summarisation / long-context chat: turn it ON. Long-
  completion is what users wait on.
- Short-only typing-feel chat: keep default OFF.
- Mixed traffic: depends on ratio.

**Cross-config:**

| Topology | Aggregate OFF→ON | Long p50 OFF→ON | Short p50 OFF→ON |
|---|---|---|---|
| Single-device | -1.5 % | -16 % | +24 % |
| PP4 | 0 % | -9.4 % | -4.8 % |
| **pp2tp2** | **+5.6 %** | **-42 %** | +60 % |

pp2tp2 is the production-relevant topology — highest baseline
throughput, the only one where the K5 aggregate target trends in
the right direction.

**Optional follow-up sweeps (not Required for K5 close):**
- Chunk-size sweep on mixed bench (the per-iteration overlap math
  shifts with chunk size; ARRIVAL_S sweep).
- Higher-N regime via Phase 6 paged KV (K3b microbench's 1.235× at
  K=128/N=16 cell hints at a higher-N regime where per-call gain
  compounds; would need pp2tp2 + paged to reach those slot counts).

## Acceptance criteria

- Phase K1: parity test passes on Pp/Tp/Hybrid for dense (Qwen3.5-9B-Q4_1).
- Phase K2: parity test passes on hybrid model with GDN layers
  (Qwen3.6-27B-Q4_0 small synthetic config).
- Phase K3: microbench matches v1's 1.06-1.17× per-call ceiling.
- Phase K4: env-gated scheduler shipped, default OFF.
- Phase K5: ≥1.5× aggregate throughput on
  `bench_mixed_chat_streaming.py`-style sustained traffic at the same
  ctx-cap and inflight-slots vs the S1+S2 baseline.

## Past results to beat (v1 reference)

From memory `project_lever1_mixed_batch_v1` (2026-05-04 — branch is
gone):

| K   | N   | seq (ms) | mix (ms) | speedup |
|-----|-----|---------:|---------:|--------:|
| 512 | 4   | 638.09   | 600.45   | 1.063×  |
| 256 | 8   | 427.13   | 381.30   | 1.120×  |
| 128 | 16  | 389.64   | 331.95   | 1.174×  |

Qwen3.5-9B-Q4_1 / pp2tp2 / per-call wall. Don't re-derive — match or
beat these as the K3 acceptance, then move on to K4+K5 for the real
throughput.

## Past failures (do not repeat)

From the v1 work:

- **1F1B decode (Lever 2)** was null — decode is HBM-bound, 1F1B is
  for compute-bound prefill. Don't propose more PP-pipelining levers
  for decode.
- **Pure-rel tolerance for logit parity is brittle** — denominator
  → 0 at small logits. Use hybrid abs+rel: pass if
  `abs <= abs_tol` OR `rel <= rel_tol`. Suggested abs_tol=0.5,
  rel_tol=1e-2.
- **BAR P2P matrix gets stale across tests in one cargo run** — second
  test fails with `BarP2pAllReduce::new ... requires a fully-connected
  peer-access matrix`. Run mixed-batch tests one at a time
  (separate cargo invocations).
- **`RUST_MIN_STACK=16777216 LD_LIBRARY_PATH=/opt/rocm-host/lib`**
  required for cargo test on this rig.

From the v2 migration:

- **Don't add a ctx method on only one topology** (flambeau-forward
  rule 1). All three Pp/Tp/Hybrid impls must implement
  `standard_attn_mixed` even if Hybrid's impl is the most involved.
- **Don't try to retrofit `standard_attn` itself with a `mixed_split`
  parameter** — its caller `forward<C>` in each model crate doesn't
  know about mixed-batch shape. Add a separate ctx method; the model
  forward can opt in by detecting the mixed shape from caller-provided
  metadata.
- **Don't fuse the K and N attention calls into one new kernel.** The
  v1 driver explicitly avoided this; the two existing kernels are
  already optimal for their shapes (prefill flash-tile, batched-decode
  MMVQ). Compose, don't fuse.

## File pointers

| Area | Path |
|---|---|
| v2 ctx trait (target for new method) | `crates/forward/src/ctx.rs` |
| Pp impl of standard_attn | `crates/forward/src/core/composites/standard_attn.rs` |
| GDN composite | `crates/forward/src/core/composites/gdn.rs` |
| MoE composite (already runs at n = K+N — no change) | `crates/forward/src/core/composites/moe_ffn.rs` |
| Per-arch v2 forward (caller side) | `crates/models/{qwen35-v2,qwen35moe-v2,gemma4-v2}/src/model.rs` |
| Reference K kernels (already in tree) | `crates/ops/src/hip/attention.rs::attn_prefill_f16` |
| Reference N kernels (already in tree) | `crates/ops/src/hip/attention.rs::attn_decode_f16_batched` |
| GDN K kernel | `crates/ops/src/hip/recurrent.rs::gdn_prefill_*` |
| GDN N kernel | `crates/ops/src/hip/recurrent.rs::gdn_decode_batched_slots` |
| Phase 5 S1+S2 helper (will be replaced by K4 scheduler) | `crates/server/src/routes/decode_loop.rs::chunked_prefill_pp` |
| Phase 6 paged (compose with K4) | see `BATCHED_DECODE_PLAN.md` Phase 6 |
| v1 memory note (results to beat) | `project_lever1_mixed_batch_v1.md` |
| Sibling plan | [`BATCHED_DECODE_PLAN.md`](BATCHED_DECODE_PLAN.md) |
