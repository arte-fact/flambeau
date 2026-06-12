# Prefix-Cache v2 Reimplementation Plan (#219)

Scope doc for resurrecting prompt prefix caching on the v2 stack.
Server-side + a thin per-rank KV snapshot/restore op; no new attention
kernels. Multi-session arc.

## Current state (verified on `feature/tool-calling-fixes` @ f950a21)

The flag is parsed and the LRU index is built, but the device-side
snapshot/restore is stubbed — `--prefix-cache` is a **complete no-op**.

Verified empirically: same 661-token prompt fired 3× under
`--prefix-cache --prefix-cache-max-gb 8` on Qwen3.6-27B-Q8 pp2tp2
returned prefill+1tok of 2935 / 2934 / 2935 ms — zero speedup.

What exists and works:
- `crates/server/src/prefix_cache.rs` (613 LOC) — the index:
  - `ChunkKey` rolling hash + `PrefixKeys::from_prompt(prompt,
    chunk_tokens)` chunk-chain builder.
  - `PrefixCache { RwLock<by_terminal: HashMap<ChunkKey,
    Vec<CacheEntry>>, lru_order, used_bytes>, vram_budget_bytes,
    enabled }`.
  - `longest_match(&keys, topo, touch)` → `Option<CacheHit>` — walks
    chunk keys longest→shortest, verifies the **full chain** (not just
    terminal key) to defend against partial-chain collisions.
  - `insert_with_kv(PrefixCacheInsert{topology, chunk_tokens,
    n_tokens, bytes}, chain, kv, last_logits)`, `snapshot_for`,
    `logits_for`. LRU eviction under `vram_budget_bytes`.
  - 12 unit-test references exercise the index in isolation — it's
    sound.
- `CacheEntry { topology: TopologyTag, chunk_tokens, chain:
  Vec<ChunkKey>, n_tokens, kv: Option<Arc<KvSnapshot>>, last_logits:
  Option<Arc<Vec<f32>>> }`.
- CLI → `ServeConfig` → `build_prefix_cache` → `ServerState{
  prefix_cache: Arc<PrefixCache>, prefix_cache_chunk_tokens }`.

What's stubbed (the dead half):
- `RankSnapshot = Vec<u8>` — placeholder type.
- `KvSnapshot = Vec<RankSnapshot>` — one per rank.
- `ServerState::prefix_cache_try_restore(...)` → always
  `Ok(PrefixCacheRestore::Miss)`. **Not called anywhere.**
- `prefix_cache_try_capture_full(...)` / `prefix_cache_insert_
  intermediate(...)` → empty bodies. **Not called anywhere.**
- `longest_match` / `insert_with_kv` — never called from the request
  path (only their own unit tests reference them).

Root cause: the legacy qwen3-moe snapshot/restore was deleted with the
v1 model stack in #221; the v2 reimplementation (#219) never landed.
Code comment (`routes.rs:298`): *"After #221 deleted the legacy
qwen3-moe snapshot/restore path, prefix cache is a no-op regardless of
`FLAMBEAU_PREFIX_CACHE`."*

## Goal

A chat request whose prompt shares a prefix with a recently-served
prompt restores the cached prefix KV into a fresh slot and prefills
only the **new tail tokens**, cutting TTFT proportional to the shared
prefix. Standard automatic prefix caching (vLLM APC / SGLang
RadixAttention equivalent), bounded by `--prefix-cache-max-gb` of host
RAM.

Non-goals (this plan):
- New attention kernels. Snapshot/restore is pure DtoH/HtoD memcpy of
  existing KV slabs + GDN state.
- Cross-process / persistent cache. Process-local only.
- Streaming-path changes beyond TTFT.
- Speculative or partial-token KV reuse — chunk-aligned only.

## What has to be snapshotted

Per slot, the post-prefill device state is three families. The
snapshot is **per rank** (each rank holds only its shard), and is only
restorable under the **same topology** (the `TopologyTag` check
already in `longest_match`).

| State | Holder | Per-slot size | Topology shard |
|---|---|---|---|
| Standard-attn KV | `ScratchPool.kv_caches[li].{k,v}` slab `[n_slots, max_seq_len, bytes_per_row]` | `n_tokens × bytes_per_row` per layer × 2 (K+V) | PP: layer subset · TP: head shard · Hyb: both |
| GDN recurrent state | `gdn_state[li].state` + `.conv_history` | fixed `num_v_heads·head_k·head_v·4` + `(conv_k−1)·conv_ch·4` per slot | same as attn |
| Paged KV | `paged_kv_caches[li]` page pool + block table | mapped pages for the slot | same |

Layout + arch wrinkles that gate the slices:
- **F16Contig vs Q8Contig** (`KvCache.layout`): `bytes_per_row`
  already encodes it; snapshot is layout-agnostic byte copy, but the
  size accounting differs.
- **SWA layers (gemma4)**: slab is sized at full `max_seq_len` but only
  the last `window_size` tokens are valid. Snapshot must copy the
  valid window, not the whole slab, and restore must place it at the
  right offset (interacts with the SWA cache-pointer-offset lever).
- **GDN state (qwen3.6 hybrid, qwen3-coder-next)**: the recurrent
  state is the *exact* post-prefix state — there's no per-token
  history to truncate; you snapshot the fixed-size state buffer. The
  hard part is that GDN state can only be captured at a position the
  model actually stopped at, so intermediate (chunk-boundary) capture
  needs GDN-at-position snapshotting (the V1 plan flagged this — it's
  why V1 only did full-prompt entries).

## Architecture

**Capture point.** After a fresh prefill finishes for a slot, the
slot's KV slab + GDN state hold the full-prompt state and the caller
already has the last-position logits row (it's about to sample the
first decode token). `prefix_cache_try_capture_full` does a DtoH copy
of the valid ranges across all layers/ranks into a `KvSnapshot`, and
clones the logits row into `last_logits`. Opportunistic — failures
log and swallow.

**Restore point.** Before prefill, the chat/completions handler builds
`PrefixKeys::from_prompt(prompt_ids, chunk_tokens)` and calls
`longest_match`. On a hit with a real `kv` snapshot + matching
`TopologyTag`:
1. Claim a fresh slot.
2. `Session::restore_slot(slot_id, &snapshot, n_matched_tokens)` —
   HtoD copy into the slot's slab/state at the right offsets, per rank.
3. Prefill only `prompt_ids[n_matched..]` via
   `forward_prefill_logits_slot` starting at position `n_matched`.
4. If `n_matched == prompt.len()` (full-prompt hit) and `last_logits`
   is present, skip prefill entirely and sample the first token from
   the cached logits.

**API surface to add.**
- `Session::snapshot_slot(&self, slot_id, n_tokens) ->
  Result<KvSnapshot>` — fans out a `Command::SnapshotKv` to every
  rank worker; each DtoH-copies its local shard and returns
  `RankSnapshot` bytes; orchestrate collects them in rank order.
- `Session::restore_slot(&mut self, slot_id, &KvSnapshot, n_tokens)
  -> Result<()>` — fans out `Command::RestoreKv`; each rank HtoD-copies
  its shard.
- Worker `Command::SnapshotKv { slot_id, n_tokens, reply }` +
  `Command::RestoreKv { slot_id, snapshot, n_tokens, reply }` in
  `workers.rs`, dispatched in the rank loop alongside `Forward` /
  `ResetKvSlot`.
- A model-ops-free helper on the rank's `ScratchPool` that, given
  `slot_id` + `n_tokens`, produces/consumes the per-layer byte ranges
  (it already knows `kv_caches[li].{k,v,bytes_per_row}`,
  `max_seq_len`, `gdn_state[li].{state,conv_history}`,
  `per_layer_kv_layouts`, `window_size_at`).

`RankSnapshot` becomes a real struct (still `Vec<u8>`-backed for the
index's byte accounting) carrying a small header: per-layer offsets +
the layout/window metadata needed to restore without re-deriving it.

## Slices

Each slice ends green on a parity test (snapshot→restore round-trip
produces bit-identical decode to a no-cache run) + a TTFT measurement
where it applies. pp2tp2 / hip:0,2,1,3 is the live target.

### P0 — Honest no-op guard (ship immediately, independent)
Startup warning when `--prefix-cache` is set but the path is a no-op,
so the flag stops implying a TTFT win it can't deliver. One-liner in
`build_prefix_cache`. (Already offered separately; land it first so
the dead flag isn't silently misleading while P1–P9 cook.)

### P1 — Snapshot/restore op, dense F16, PP/TP
- `Command::SnapshotKv` / `RestoreKv` + orchestrate fanout +
  `Session::{snapshot_slot, restore_slot}`.
- `ScratchPool` byte-range helper for `kv_caches` only (F16Contig),
  full-context layers (no SWA, no GDN, no paged).
- Parity test: prefill a synthetic prompt to slot A, snapshot, reset,
  restore into slot B, decode N tokens from both → bit-identical.
- No request-path wiring yet.

### P2 — Request-path wiring (dense F16, PP/TP)
- Implement `prefix_cache_try_restore` for real: `longest_match` →
  `restore_slot` → return `PrefixCacheRestore::Hit{ n_matched,
  last_logits }`.
- Implement `prefix_cache_try_capture_full`: `snapshot_slot` →
  `insert_with_kv`.
- Wire both into `routes/chat.rs` + `routes/completions` around the
  prefill call; partial-tail prefill from `position = n_matched`.
- **Gate**: live TTFT — repeat a 600-tok shared-prefix prompt, expect
  run-2 prefill ≈ (tail/total)× run-1. On Qwen3.6-27B this should turn
  the 2935 ms measured above into a few hundred ms for a 1-chunk tail.

### P3 — Q8Contig layout
- Byte-range helper handles `Q8Contig` `bytes_per_row`.
- Parity test under `--kv q8`. Size accounting in `PrefixCacheInsert.
  bytes` reflects the Q8 footprint.

### P4 — SWA windowing (gemma4)
- Snapshot only the valid `window_size` window for SWA layers;
  restore at the offset the SWA cache-pointer-offset lever expects.
- Full-attention (global) layers stay full-range.
- Parity test on gemma4-26B-A4B across the SWA boundary (prompt >
  window).

### P5 — GDN recurrent state (qwen3.6 hybrid, qwen3-coder-next)
- Snapshot `gdn_state[li].{state, conv_history}` per slot for GDN
  layers; restore in place.
- Full-prompt entries only (V1 parity) — intermediate chunk-boundary
  GDN capture is a follow-up (needs GDN-at-position snapshotting).
- Parity test on Qwen3.6-27B hybrid: GDN layers must decode identically
  post-restore.

### P6 — Hybrid topology
- Per-stage-per-rank snapshot under pp+tp. `TopologyTag` already
  rejects cross-topology restores; this slice makes the within-Hybrid
  fanout correct.
- Parity on pp2tp2.

### P7 — Paged KV
- Snapshot the slot's mapped pages + block table; restore allocates
  fresh pages and re-points the block table. Interacts with
  `release_paged_slot`.
- Defer if paged-KV adoption is low; gate on a real consumer.

### P8 — Eviction, budget, concurrency
- `vram_budget_bytes` enforcement on insert (LRU evict to fit); the
  index already has the hooks. Confirm host-RAM accounting matches
  real `RankSnapshot` sizes.
- Concurrency: capture happens at request end while other requests
  read; confirm the `RwLock` discipline holds under the sustained
  bench.

### P9 — Cert + sustained TTFT bench
- A `scripts/` harness firing a realistic chat trace (shared system
  prompt + growing history) and reporting TTFT with/without the cache,
  hit rate, and host-RAM high-water.
- Cert under `certs/perf/`.

## Correctness risks

- **Stale slot reuse.** A restored slot must have its decode position
  set to `n_matched`, its non-KV scratch zeroed, and any
  `release_paged_slot` bookkeeping consistent — a half-restored slot
  decoding from position 0 is a silent corruption. Parity tests catch
  it only if they decode enough tokens; assert ≥ 8 decode steps.
- **Chain-collision.** `longest_match` already verifies the full chain,
  not just the terminal hash — keep that invariant; a terminal-only
  match would restore the wrong KV.
- **Topology / layout drift.** A snapshot captured under pp2tp2/F16 is
  invalid under pp2tp2/Q8 or pp4/F16. The `TopologyTag` check covers
  topology; add a layout tag (`KvLayout`) to the entry and reject on
  mismatch (cheap, closes a silent-corruption hole).
- **GDN position aliasing.** GDN state is only valid at the exact
  position it was captured. A chunk-boundary GDN snapshot restored for
  a prompt that diverges one token earlier is wrong — full-prompt-only
  (P5) sidesteps this; intermediate GDN capture must verify the chain
  to the exact token, which the chunk-key chain already does.

## Measurement (per project norms)

- No "faster TTFT" claim without a measured before/after on the same
  GGUF + topology + ctx. The P2 gate above is the canonical shape:
  identical shared-prefix prompt, run-1 (cold) vs run-2 (warm), report
  prefill ms and the tail/total ratio.
- Null result is first-class: if restore DtoH/HtoD cost approaches the
  prefill it saves (small prefixes, fast prefill, slow PCIe), file the
  crossover prompt length and gate capture on a minimum prefix.

## Architectural-rule audit (project root CLAUDE.md)

- **Rule 11** (server stays a JSON HTTP API): unaffected — this is
  internal KV reuse, no new endpoints, no MCP, no multi-turn loop.
- **Rule 13** (arch glue in the model crate, not shared handlers): the
  snapshot byte-range logic is per-arch (SWA windows, GDN state) — it
  belongs behind a generic `Session::snapshot_slot` that the shared
  `routes.rs` calls polymorphically, with the per-arch byte ranges
  computed inside the worker/ScratchPool, NOT via
  `gguf.architecture()` branches in `routes.rs`.
- **Rule 9** (`{op}_{dtype}_{backend}_{shape}_{variant}` naming): the
  DtoH/HtoD copy is a model-ops free fn (`kv_snapshot_copy` /
  `kv_restore_copy`), co-located with a parity test, not a method on a
  shared trait with a `None` default.
- **Rule 12** (no `Hip*`/`<Arch>*` on shared server types):
  `KvSnapshot` / `RankSnapshot` stay backend-neutral byte carriers;
  no `Gemma4Snapshot` variant — SWA/GDN specifics live in the
  per-rank `ScratchPool` helper, addressed by `slot_id` + metadata.

## Lineage

- #221 — deleted the legacy qwen3-moe stack, taking its snapshot/
  restore path with it.
- #228 — added the optional `kv` snapshot field on `CacheEntry`.
- #229 — V1 design: full-prompt capture + last-position logits; GDN-
  at-chunk-boundary flagged as V2.
- #219 — **this plan**: the v2 device-side snapshot/restore that makes
  the index do work again.

---

## Revised priority — agentic-first (2026-06-09)

Triggered by the `perf/long-context-speed` investigation: the user's pain is
**single-stream agentic context growth** (each turn re-sends the whole
conversation, `--inflight-slots 1`). Two scoping findings change the priority:

### Finding 1 — full-prompt-only capture (P5 "V1") will not hit a growing chat
Turn N's prompt = turn N-1's prompt + assistant reply + new user message. The
P1 cache entry is keyed by turn N-1's *partial-tail* chunk; turn N re-chunks
across that boundary so the tail chunk key never reappears in turn N's chain →
`longest_match` MISSES. Agentic hits require **chunk-boundary (intermediate)
capture** (entries keyed at each `chunk_tokens` boundary). This is NOT the "V2
hard" GDN-at-arbitrary-position problem the original plan feared: the
chunked-prefill loop already stops at clean chunk boundaries (it releases the
inflight mutex between chunks), and `gdn_state[li].state` holds the post-chunk
state there — capture = a DtoH at each boundary. So intermediate capture is
tractable and is the part that delivers the host-cache agentic win; it should
be pulled forward, not deferred.

### Finding 2 — in-place same-slot reuse is a cheaper lever for single-stream
With `--inflight-slots 1` and a growing conversation, consecutive requests land
on the SAME slot whose KV + GDN state from the prior turn is still ON-DEVICE.
If the server (a) skips the per-request slot reset and (b) tracks the slot's
current token sequence, then a new request computes
`lcp = longest_common_prefix(prompt, slot_sequence)` and prefills only
`prompt[lcp..]` from `position = lcp`. **Zero DtoH/HtoD, zero GDN host
snapshot, no worker commands** — it sidesteps the entire hard part of the
host-snapshot design. Trade-off: single-stream only (breaks under
concurrency / slot eviction); the host-snapshot cache (P1–P9) remains the
general multi-slot solution.

### Revised slice order
- **P0** — honest no-op startup warning (unchanged, ship first).
- **Strategy A — in-place same-slot continuation** (NEW, highest ROI for the
  reported scenario): per-slot token-sequence tracking + skip-reset-on-extend +
  tail-only prefill from `position = lcp`. Parity (≥8 decode steps
  bit-identical vs cold) + before/after TTFT gate on Qwen3.6-27B pp2tp2.
  Independent of the host-snapshot machinery. ~1 session.
  - A1: per-slot `Vec<u32>` token history + `valid_len` on the inflight/slot
    state; populated after each prefill+decode.
  - A2: request path computes `lcp` vs the claimed slot's history; gate the
    `reset_kv_slot`/`reset_gdn_state_slot` calls on `lcp == 0`; prefill
    `prompt[lcp..]` at `start_position = lcp`. Decode appends generated tokens
    to the history.
  - A3: correctness guards — only reuse when the SAME slot is re-claimed (slot
    affinity for the stream), invalidate history on any divergence, cap
    history to `ctx_cap`. Parity + TTFT cert.
- **B1–B3** — the host-snapshot cache (existing P1–P9), with **intermediate
  chunk-boundary capture promoted from "follow-up" to in-scope for the GDN
  slice** (Finding 1). General multi-slot/concurrent solution.

### Correctness deltas specific to Strategy A
- **Slot affinity**: in-place reuse is only valid if the new request re-claims
  the slot whose history we matched against. Under `inflight-slots 1` this is
  automatic; with >1 slot, only reuse when the scheduler hands back the same
  slot, else `lcp = 0` (cold). Never match against a different slot's history.
- **Divergence invalidation**: if `lcp < slot.valid_len` (the new prompt
  diverges mid-history — e.g. an edited/regenerated turn), the KV/state past
  `lcp` is stale; set `valid_len = lcp` and prefill the tail (the slab past
  `lcp` is simply overwritten — no explicit clear needed for full-attention;
  GDN state is overwritten in-place from `lcp` forward).
- **Position counter**: the slot's decode position must be set to `lcp + tail`,
  not 0 — same half-restore risk as the host path.

---

## Strategy A tried + reverted — null result (2026-06-09)

Strategy A (in-place same-slot reuse) was implemented end-to-end and **reverted**:
empirically defeated for the chat API on the GDN hybrid. Diagnostic on
Qwen3.6-27B-Q8_0 `/v1/chat/completions`: turn 2 `lcp = 258 < valid_len = 290`
— chat-template re-tokenization is not prefix-stable at turn boundaries (the
assistant response is detokenized→retokenized + generation-prompt/think-primer
framing shifts token boundaries near the assistant header). A GDN hybrid can
only reuse at `lcp == valid_len` (recurrent state exists only at the final
position), so reuse never engaged (0 hits, TTFT unchanged). Full detail in
memory `feedback_longctx_decode_splitk_cliff_2026_06_09.md`.

**Consequence:** the host-snapshot cache below (B / P1–P9) is now the PRIMARY
lever. It is robust to the same re-tokenization because it matches at **chunk
granularity** — in a real (long) conversation the stable early chunks (system
prompt + early turns; identical text → identical token ids) still hit even when
a late chunk diverges. Strategy A's failure (whole prompt < 1 chunk, diverged
at 258) is the worst case that chunked matching sidesteps. The GDN-state slice
(P5) must therefore land **chunk-boundary (intermediate) capture**, not just
full-prompt — that is the part that delivers the agentic win. Next concrete
step: P1 (device snapshot/restore op + ScratchPool byte-range helper + parity).

---

## Review findings → revised slice order (2026-06-09, executing)

1. **Hybrid moves into P1.** The user's production topology is pp2tp2
   (`--mesh-mode pp+tp`); as ordered, P1–P5 deliver nothing there until P6.
   Unnecessary deferral: `reset_kv_slot`'s fanout (`runtime/mod.rs:323`)
   already iterates all rank handles topology-agnostically and runs on
   pp2tp2 today. Snapshot/restore mirrors it → hybrid-correct by
   construction; `TopologyTag` already blocks cross-topology restores.
   Same logic applies to Q8Contig: the user runs `--kv q8`, so the byte-range
   helper covers F16Contig + Q8Contig from the start (P3 folded into P1).
2. **Intermediate capture must not be one-full-snapshot-per-boundary** —
   that is O(n²) host RAM (a 32k prompt ≈ 36 GB). Fix: KV rows are
   positional and contiguous per layer, so ONE full-length KV buffer per
   request serves every boundary (entry at boundary b restores tokens
   [0..b) of each layer's range; entries share the buffer via the existing
   `Arc<KvSnapshot>`). Only the GDN state is per-boundary (fixed size);
   capture the **last K=2 full-chunk boundaries** per request — the next
   turn's divergence sits near the end of the previous prompt (Strategy A
   diagnostic: lcp 258/290), so last-2 covers the realistic hit set.
   Follow-ons: `used_bytes` must not double-count Arc-shared buffers;
   `vram_budget_bytes` is host RAM (misnomer).
3. **Chunk-boundary GDN capture is well-defined, not "V2 hard".**
   `chunked_prefill_pp` runs the full layer stack per chunk, so every GDN
   layer's state is committed at each boundary (the next chunk's forward
   depends on it). The snapshot point is exactly between
   `forward_prefill_logits` calls — where the loop already pauses and
   releases the mutex. Supersedes the "GDN-at-position" caveat above.
4. **The P2 exact-retry gate does not validate the agentic case** (a
   full-prompt terminal entry never matches the next turn — Strategy A null
   result). The agentic gate lands with the intermediate-capture slice: a
   real multi-turn growing conversation showing turn-N prefill ≈ tail-only.

Execution order: **P1′** (snapshot/restore op, all-rank fanout, F16+Q8 KV +
GDN state byte copy, pp2tp2 parity) → **P2′** (request wiring + full-prompt
capture, exact-retry gate) → **P5′** (last-K chunk-boundary capture +
shared-KV-buffer entries, agentic gate) → **P8/P9** (Arc-aware accounting,
eviction under load, cert). P4 (SWA) and P7 (paged) stay deferred.

---

## P8 finding — host budget must be clamped to RAM headroom (2026-06-12, fixed `4caf12f`)

P5′ (gemma4 SWA chunk-boundary capture, shipped `e6df3fd`) was validated by a
live growing-context A/B, which surfaced a budget bug, not a snapshot bug.

**Symptom.** gemma-4-31B-it-Q8_0 pp2tp2 `--kv q8 --prefix-cache
--prefix-cache-max-gb 12` died at ~3k ctx with
`HybStage::peer_recv: timed out waiting for producer event (10s)`, then wedged
(GPUs idle). The 10 s pipeline-handoff timeout is a **symptom**, not the cause.

**Root cause — host-RAM oversubscription.** The snapshot cache (anonymous
`Vec<u8>`, LRU-capped at `vram_budget_bytes`, the host-RAM misnomer in §P8)
competes with the resident model weights. The rig has **31 GB RAM**; the gemma
GGUF is **32.6 GB** — the model alone exceeds RAM. Under TP/hybrid the loader
cannot `advise_drop` the mmap after upload (every rank reads the shared region;
see memory `feedback_tp_no_advise_drop`), so 32.6 GB stays mapped. A 12 GB live
cache on top tips the box onto the direct-reclaim cliff; the large per-capture
snapshots (gemma is full-attention every layer, SWA+global → ~800 MB KV) stall,
the process slows, and the PP `peer_recv` spin trips its 10 s budget.

**Decisive probe (single variable = the cap).** Same workload, 7-turn growing
chat to 3478 tok:
- `--prefix-cache-max-gb 2` → CLEAN, TTFT flat ~7 s, restores cross the SWA
  window (P5′ correctness holds).
- `--prefix-cache-max-gb 12` → deterministic timeout at ~3k.

So it is the **total live cache bytes**, not per-capture size. The first
hot-stream-contention hypothesis was wrong (snapshot and forward share
`device.default_stream()` sequentially under one slot lock, synced between) —
disproven by reading `workers.rs`/`engine.rs` before coding.

**Fix (`4caf12f`).** `clamp_host_cache_budget` in `prefix_cache.rs`, wired in
`serve_common::build_prefix_cache`: bound the requested budget to
`(MemTotal − resident_weights − 2 GiB reserve)`, floored at a confirmed-safe
**2 GiB**, and `warn!` when it clamps. `resident_weights` = GGUF size for
TP/hybrid, 0 for PP (drops its mmap). Pure helper, 4 unit tests. The
`mesh_kind == "pp"` branch is a topology property (mmap residency), not a
`gguf.architecture()` arch branch — rule 13 holds.

**Validation (both archs, cap 12 → clamp 2 GiB).**
- gemma-31B-Q8_0: was a crash; now **CLEAN** to 3478 tok, TTFT flat ~7 s
  (cache-off climbs to ~39 s).
- qwen-27B-Q8_0: off 103.4 s → on 31.2 s = **3.31×**, transcript parity PASS —
  no regression (better than the prior 2.97× at unclamped 12 GB; lighter evict
  churn). qwen's GDN snapshots are small so its working set fits well under
  2 GiB.

**Operational note.** On this 31 GB rig a 32 GB-class model can only host a
~2 GiB prefix cache; the clamp now enforces that automatically — do not
hand-tune `--prefix-cache-max-gb`. Boxes with real headroom (weights ≪ RAM)
honor the requested cap unchanged.

**Open follow-up.** The clamp removes the crash but the floor of 2 GiB is the
*confirmed-safe* point, not a measured optimum; the true cliff for gemma sits
between 2 and 12 GB. A pressure-aware dynamic budget (evict harder as
`MemAvailable` drops, rather than a static ceiling) would reclaim the
in-between headroom on boxes where the model doesn't fully fill RAM. Also
unaddressed: the per-capture snapshot still does an `alloc_zeros` of the host
buffer (rule 8) before the DtoH overwrites it — a cheap hot-path cleanup.
