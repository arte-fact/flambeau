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
