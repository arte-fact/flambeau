# Lever A — pooled TP prefill scratch — 2026-05-05

`#324`. Shared `ShardedForwardPrefillScratchTp` lazy-initialised on
`ServerState`, reused across every TP prefill in the server's
lifetime. Replaces the per-call alloc/dispose inside
`forward_prefill_tp_batched_logits` that motivated the `#321`
serialiser.

## Design

A per-slot pre-allocation (one scratch per Inflight) was tried first
and **failed at boot** with VRAM OOM on Qwen3.6-27B/tp2/slots=8 — 8 ×
~35 MB scratches × 2 ranks pushes past the ~7.5 GB free per device
after weights+KV. The shipped design uses **one shared scratch** on
`ServerState`, gated by the existing `prefill_serialiser` mutex
(prefills already serialise on the GPU stream regardless of host
orchestration, so the shared scratch design loses no parallelism).

VRAM cost: ~35 MB × n_ranks per server (lazy-init, zero cost when
not used). PP-only models don't pay this. Hybrid is a V2 follow-up.

## Measurements (Qwen3.6-27B-Q4_1 / TP2 / 4×MI50, prompt=203 tokens, max_tokens=64)

Same N-concurrent test as `#321` validation:

| N   | #321 (serialiser only) | #324 (pooled scratch) | speedup |
|-----|------------------------|------------------------|---------|
| 4   | 27.7s                  | **14.3s**              | 1.94×   |
| 8   | 55.7s                  | **29.3s**              | 1.90×   |

8/8 streams succeed at both N values; all return `finish_reason=length`
with 64 tokens.

The ~2× wall-clock improvement at concurrency is bigger than expected
from pure host-side alloc cost (~80 ms × N). Likely contributors:
- HIP allocator: per-call alloc + dispose of multiple device buffers
  across 2 ranks isn't free (~50-100 ms each).
- Repeated alloc fragments the device-side free list.
- The serialiser's hold time was longer because it covered alloc +
  prefill + dispose; pooled scratch shortens it to just prefill.

## Code changes

- `crates/models/qwen3-moe/src/forward/tp.rs`:
  - `forward_prefill_tp_batched_logits` accepts
    `pooled: Option<&mut ShardedForwardPrefillScratchTp>`. `None`
    keeps the old alloc-per-call semantics for tests/back-compat.
  - New `pub fn forward_prefill_tp_logits_pooled(...)` — required
    `&mut pool_prefill` arg, threads through to the batched path.
- `crates/server/src/routes.rs`:
  - `ServerState.tp_prefill_scratch: Mutex<Option<...>>`.
  - `ServerState::lock_tp_prefill_scratch()` lazy-inits and locks.
  - The 3 callers of `prefill_logits` lock the scratch under the
    existing `prefill_serialiser` and pass `Some(&mut pool)` for TP.
- `crates/server/src/model.rs`:
  - `prefill_logits(..., tp_pool_prefill: Option<&mut ...>)` — when
    `Some` and topology is TP, dispatches via `_pooled` variant.

## What's NOT done

- **Hybrid (pp+tp)** still uses per-call alloc. Hybrid prefill scratch
  is per-stage; needs Vec-of-scratch in `Inflight::Hybrid` or
  `ServerState`. V2 follow-up — pp2tp2 wasn't OOMing in the matrix
  bench so it's not blocking.
- The `prefill_serialiser` mutex stays as belt-and-braces. The
  shared scratch fundamentally requires it (one-at-a-time prefill);
  removing it would need per-slot scratch which already proved
  infeasible at slots=8.

## Reproducer

```sh
RUST_LOG=info,server.prefill=info FLAMBEAU_INFLIGHT_SLOTS=8 \
FLAMBEAU_GPU_SAMPLER=1 FLAMBEAU_CTX_CAP=4096 FLAMBEAU_PREFILL_UBATCH=512 \
target/release/flambeau serve \
  --model /artefact/models/Qwen3.6-27B-Q4_1.gguf \
  --devices hip:0,1 --mesh-mode tp --tp-size 2 --port 18181

# Server log on first prefill:
# server.prefill: lazy-init shared TP prefill scratch (#324) prefill_ubatch=512

python3 /tmp/dbg_n.py 8   # all 8 streams succeed in ~29s
```
