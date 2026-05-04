# Handoff 2026-05-04 part 7 — Lever 1 v1 driver+scheduler done, server wiring (#306) is the remaining piece

## State

- Branch: `main`. ~30 commits ahead of `origin/main`.
- Lever 1 v1: **driver shipped**, **scheduler shipped**,
  **parity bit-exact**, **microbench 1.06×–1.17×**, **scheduler+driver
  integrated end-to-end on multi-chunk prefill**.
- Only #306 (server wiring) + #308 (production cert) remain.

## What landed across last 2 sessions

| #    | what                                                | result                           |
|------|-----------------------------------------------------|----------------------------------|
| 302  | Sarathi mixed-batch design                          | docs in `doc/V1.x/sarathi_*`     |
| 304  | `forward_decode_mixed_hybrid` driver                | shipped, ~700 LOC, K\|N split    |
| 307  | parity test (3 configs)                             | bit-exact at small K, top-1 at K=128/N=4 |
| 308  | microbench (3 shapes)                               | 1.063× → 1.174× wall, scales with N/K |
| 305  | `MixedScheduler` primitive + 9 unit tests           | shipped                          |
| 305b | scheduler+driver integration test                   | passes — multi-chunk + cross-request transitions |

Cert: `certs/perf/p29b_i2_F_throughput/qwen35_9b_mixed_batch_v1_2026_05_04.md`.

## Where the work lives

- Driver: `crates/models/qwen3-moe/src/forward/hybrid.rs::forward_decode_mixed_hybrid`
- Driver scaffold + types: `crates/models/qwen3-moe/src/forward/batched.rs::MixedPrefillChunk`
- Scheduler: `crates/server/src/mixed_scheduler.rs`
- Parity test: `crates/models/qwen3-moe/tests/mixed_batch_parity.rs`
- Microbench: `crates/models/qwen3-moe/tests/mixed_batch_microbench.rs`
- Integration test: `crates/server/tests/mixed_scheduler_integration.rs`

## Next step: #306 server wiring

Plumb the scheduler into the chat handler so production traffic
benefits from mixed dispatch. The integration is at the
`run_completion_scheduler_pp_blocking` + `decode_via_scheduler_into`
seam.

### Architectural change

Today's per-request flow (prefill-then-decode):
```
acquire inflight slot mutex
  prefill_logits(...)         # synchronous, blocks decode
  sample first token
release mutex
loop:
  decode_via_scheduler_into(slot_idx, last_tok, pos, &mut logits)
                              # leader-coalesced batched-decode
  sample next token
```

Mixed-batch flow (prefill-via-scheduler):
```
acquire inflight slot mutex (briefly: just to set position=0 + zero KV)
release mutex
prefill_via_mixed_scheduler(slot_idx, prompt_ids) -> first_logits
                              # registers in MixedScheduler queue,
                              # waits on per-request channel for
                              # final-chunk logits delivery
sample first token
loop:
  decode_via_mixed_scheduler(slot_idx, last_tok, pos, &mut logits)
                              # same leader pattern, but iteration
                              # may also include a pending prefill chunk
```

### Concrete steps

1. **State**: add to `ServerState`:
   ```rust
   mixed_scheduler: Mutex<MixedScheduler>,
   prefill_done_channels: HashMap<MixedRequestId, oneshot::Sender<Vec<f32>>>,
   ```
   Init `MixedScheduler::new(token_budget=512, max_chunk_size=256)` at server boot. Token budget reads from env: `FLAMBEAU_MIXED_BUDGET`.

2. **Scratch sizing**: change routes.rs:585 from
   ```rust
   let max_slots = self.inflight_pool.len().max(n);
   ShardedForwardPrefillScratchHybrid::new(model, max_slots)
   ```
   to
   ```rust
   let chunk_budget = std::env::var("FLAMBEAU_MIXED_CHUNK")
       .and_then(...).unwrap_or(256);
   let max_t = self.inflight_pool.len().max(n) + chunk_budget;
   ShardedForwardPrefillScratchHybrid::new(model, max_t)
   ```
   when `FLAMBEAU_MIXED_BATCH=1`.

3. **Prefill entry**: new method
   `state.prefill_via_mixed_scheduler(slot_idx, prompt_ids) -> Result<Vec<f32>>`:
   - allocate request_id via `scheduler.allocate_request_id()`
   - build oneshot channel (tx, rx); insert tx into `prefill_done_channels`
   - `scheduler.submit_prefill(PendingPrefillReq { request_id, slot_idx, tokens, chunk_start: 0 })`
   - call `state.run_mixed_dispatch_step()` in a loop or wake leader
   - block on `rx.blocking_recv()` for final-chunk logits

4. **Leader logic**: extend `decode_via_scheduler_into` (or split into
   `decode_via_mixed_dispatch_into`) to:
   - lock `mixed_scheduler`
   - call `next_iteration()` — gets plan
   - acquire inflight mutexes for chunk.slot_idx + each decode.slot_idx
   - call `forward_decode_mixed_hybrid(...)` with the plan
   - if chunk.is_final_chunk: pop the request's channel from
     `prefill_done_channels`, send the final logits
   - for each decode: send logits to its waiter (existing path)

5. **Gating**: only engage mixed dispatch for the qualifying topology
   (`LoadedModel::Hybrid` + cfg.arch ∈ {qwen35moe, qwen36moe}).
   Default off via env: `FLAMBEAU_MIXED_BATCH=1`.

6. **Tests**: add a 2-process or async-test that drives
   `/v1/chat/completions` × 2 concurrent with one long-prompt + one
   short-prompt; verify both return coherent text and that aggregate
   throughput improves.

### Estimated size

~300–500 LOC in routes.rs + ~50 LOC of state changes. Most of the
risk is in the leader's iteration loop: deadlock potential between
`mixed_scheduler` lock + per-slot inflight mutexes. The existing
batched-decode leader already manages multi-mutex acquisition via
the borrow-disjoint `sessions_ptr.add(s)` pattern; mixed extends
that to also include `chunk.slot_idx`.

## Subsequent: #308 production cert

Once #306 lands:
- Use `flambeau_dispatch_ab` MCP tool or write a bench harness
  that drives Qwen3.6-27B / pp2tp2 with **mixed traffic**:
  - 1 long-prompt request arriving every N decode steps
  - 4 ongoing decode-state requests
- Compare aggregate `(prefill_tokens + decode_tokens) / wall_time`
  with vs without `FLAMBEAU_MIXED_BATCH=1`.
- Gate: 3× wall ratio over baseline (v2 cert recorded 0.92× pp2tp2).

## Don't repeat

- Per-rel tolerance for logit parity is brittle for small logits;
  use hybrid abs+rel (see #307).
- 1F1B at decode is null; don't propose more PP-pipelining levers
  for decode.
- Multi-cluster setup races: `BarP2pAllReduce::new` fails on
  second test in same process; run parity/microbench tests one at
  a time (separate cargo invocations).
- Don't run multiple parity tests in one cargo invocation.

## Run commands cheat-sheet

```bash
# Build (release+hip)
RUST_MIN_STACK=16777216 cargo build --release -p flambeau-server -p flambeau-qwen3-moe --features hip

# Scheduler unit tests (no GPU needed)
RUST_MIN_STACK=16777216 cargo test --release --lib -p flambeau-server --no-default-features mixed_scheduler

# Driver parity tests (GPU; one at a time)
RUST_MIN_STACK=16777216 LD_LIBRARY_PATH=/opt/rocm-host/lib \
  cargo test --release -p flambeau-qwen3-moe --features hip \
  --test mixed_batch_parity mixed_batch_parity_pp2tp2_k32_n1 -- --nocapture

# Driver microbench
RUST_MIN_STACK=16777216 LD_LIBRARY_PATH=/opt/rocm-host/lib \
  FLAMBEAU_BENCH_K=512 FLAMBEAU_BENCH_N=4 \
  cargo test --release -p flambeau-qwen3-moe --features hip \
  --test mixed_batch_microbench -- --nocapture

# Scheduler+driver integration
RUST_MIN_STACK=16777216 LD_LIBRARY_PATH=/opt/rocm-host/lib \
  cargo test --release -p flambeau-server --features hip \
  --test mixed_scheduler_integration -- --nocapture
```
