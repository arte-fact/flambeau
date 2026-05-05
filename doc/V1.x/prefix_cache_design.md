# Prompt prefix caching — design (P2.10) — 2026-05-05

V1.x feature: skip re-prefilling tokens that were already prefilled by
a prior request whose prompt is a *prefix* of the current one. Standard
on vLLM, llama.cpp, sglang. The win is huge for chat — every multi-turn
turn re-prefills `system + user1 + asst1 + ... + userN-1` even though
the KV state for that prefix is identical to what we computed last
turn.

Tracked across three task IDs:
- **#227 P2.10a** (this doc) — design + cache-key types + lookup
  scaffold. No KV ops yet.
- **#228 P2.10b** — lookup + KV restore (read path). Hot path.
- **#229 P2.10c** — write path: KV snapshot at request end, eviction.

## Goals

- **Cut TTFT to 0** (or near-zero) when the entire prompt is a prefix
  of a cached entry.
- **Cut TTFT to (new tokens / chunk_size) × per-chunk-prefill** when
  the prompt extends a cached entry.
- **No correctness change**: KV from cache must produce bit-equivalent
  next-token logits to a fresh prefill of the same token sequence.
- **No regression** when caching disabled (`FLAMBEAU_PREFIX_CACHE=0`)
  or unsupported (e.g. logprobs path that diverges per-step).

## Non-goals (V1)

- Cross-request *during-decode* radix caching (vLLM's full radix tree
  with shared subtrees). V1 only caches at request boundaries —
  on cache write we snapshot the KV at the prompt's prefill end, no
  earlier.
- Disk persistence. Cache lives in process memory only; flushed on
  server restart.
- Model-aware swapping (e.g. CPU↔GPU). Device-side only in V1.
- Spec-decode interaction. The cache is bypassed when MTP spec-decode
  is engaged for this request.

## Wire format

OpenAI doesn't expose the cache in its API — it's transparent. The
client sends `messages[]`, the server tokenises into `prompt_ids[]`,
and the prefix cache sits between tokenization and `prefill_logits`.

Server log:
```
server.prefix_cache: hit prefix=2048/3313 chunks=4 ttft_saved_ms=4521
server.prefix_cache: miss
```

Optional response header `x-flambeau-prefix-cache-hit: 2048/3313` for
debugging.

## Data structures

### Granularity: chunks of `prefill_ubatch`

The chunked-prefill path already operates on `FLAMBEAU_PREFILL_UBATCH`
(default 512) token chunks. Cache entries are aligned to those same
chunk boundaries — a 1300-token prompt has potential cache entries
at boundaries 512, 1024 (no entry at 1300, that's mid-chunk). The
cache stores KV state at chunk boundaries; partial-chunk prefixes
fall back to fresh prefill from the previous boundary.

This matches the existing parity guarantee: KV at chunk boundaries
is bit-identical to single-shot prefill of the same prefix (Phase
A2 + A2-TP certs).

### Cache key: chained hash over chunks

```
chunk_key[0] = H(seed,         tokens[0..512])
chunk_key[1] = H(chunk_key[0], tokens[512..1024])
chunk_key[2] = H(chunk_key[1], tokens[1024..1536])
...
```

`H` = `std::hash::DefaultHasher` (SipHash-1-3, 64-bit). Collision
probability on adversarial input is non-trivial for cryptographic
threat models, but irrelevant here — clients aren't adversarial,
and a collision would only produce wrong tokens (not memory unsafety).
Detected at request boundary if next-token distributions diverge
catastrophically.

Lookup: walk the prompt's chunks, computing `chunk_key[i]` and probing
the cache. The longest contiguous prefix found wins.

### Cache value: per-rank KV snapshot

For each rank in the active cluster:
- Per-layer KV state at `position = chunk_end`. For full-attn layers
  this is the K and V tensors filled to position `chunk_end`. For
  GDN layers this is the recurrent state (`[B, H, head_v, head_v]`
  matrix and similar shapes).

The snapshot lives in a separately-allocated device buffer per layer
per rank. On hit, the buffers are `hipMemcpyDtoD`'d into the inflight
session's KV cache layers. On miss/eviction, the buffer is freed.

VRAM cost per snapshot at chunk_end:
- Qwen3.6-27B / TP2 / ctx=512:
  - K + V per layer = 2 × 8 × 128 × 512 × 2 = 2 MB per rank per layer
  - 32 layers per rank → 64 MB per rank → 128 MB across 2 ranks
- Qwen3.6-27B / TP2 / ctx=4096 (full prompt):
  - 16 MB × 32 layers × 2 ranks = 1 GB per snapshot

### Eviction: LRU with VRAM cap

`FLAMBEAU_PREFIX_CACHE_MAX_GB` (default 2 GB). Tracks total VRAM
across all snapshots. On insert, evict LRU until under cap.

Touched on every hit (move to MRU).

## Lookup → restore flow

```text
prompt_ids[]  →  chunk into [c0, c1, c2, ...]
              →  compute chunk_keys [k0, k1, k2, ...]
              →  cache.longest_matching_prefix(keys) → (n_matched_chunks, cached_state)
              →  if n_matched_chunks > 0:
                     restore cached_state into Inflight.session.caches  (1 dtod per layer)
                     prefill_logits(prompt_ids[n_matched*chunk_size..])  (only the tail)
                     start_position = n_matched * chunk_size
                 else:
                     prefill_logits(prompt_ids[0..])  (full path; today's behaviour)
```

The KV cache layers' `current_tokens` field gets set to
`n_matched * chunk_size` after restore so subsequent prefill / decode
appends correctly.

## Write flow (cache populate)

After `prefill_logits` completes, walk the chunks in the prompt:
- For each chunk boundary that wasn't already in cache, snapshot the
  KV at that boundary and insert.
- Skip chunks that were in cache (already there — bump LRU).

This means the FIRST request with a given prompt populates the entire
chain `[k0, k0→k1, k0→k1→k2, ...]`. Subsequent prompts that start
with that prefix get full hits.

## Topology constraints

### PP

Each rank holds a contiguous layer range. Snapshot is per-rank-local:
each rank stages its own layers' KV into its own snapshot buffer.
On restore, each rank dtod's its own layers. No cross-rank
coordination needed.

### TP

Each rank holds the FULL layer set, but per-layer K/V tensors are
sharded along `n_kv_heads`. Snapshot is per-rank-per-layer (each
rank stages its shard of every layer).

### Hybrid (PP+TP)

Per-stage: stage owns layers, each TP rank within the stage owns
shard. Snapshot = stage_idx × tp_rank × per-layer.

The chunk_key is **topology-independent** — it's a function of the
token IDs alone. But the cache value's layout is topology-specific.
A snapshot taken on TP=2 cannot be restored on PP=4 even for the
same prompt. We tag each cache entry with the topology + cluster
identity and reject mismatches at lookup time.

## Bypass conditions (V1)

Skip the cache (force fresh prefill) when:
1. `FLAMBEAU_PREFIX_CACHE` env unset (default OFF in V1; flip to
   default-on after the feature certs).
2. Request has `logprobs` set (cached entries don't have per-token
   logprobs).
3. MTP spec-decode is engaged (`FLAMBEAU_SPEC_MTP=path` AND topology
   matches).
4. Q8 KV layouts (cached snapshot would need Q8 quantize on restore;
   V2 work).

## Failure modes + recovery

- **Hash collision**: caller computes wrong KV but proceeds. Detected
  only by output divergence. V1 doesn't do collision detection
  (would require storing the chunk's tokens alongside the key, which
  is a 2 KB redundancy per cached chunk). Document the exact failure
  mode in the cert; consider adding per-chunk token-list comparison
  in V2 if collision turns out to bite.
- **VRAM exhaustion mid-restore**: snapshot dtod's bail with the same
  error path as a fresh prefill OOM. Cache entry stays valid (it's
  still on device), the request errors. Operator can lower
  `MAX_GB` or `INFLIGHT_SLOTS`.
- **Topology change**: `flambeau serve` boots fresh per session; cache
  is process-local. No cross-session restore so no topology
  mismatch unless someone tampers in-process. The check is
  defensive.

## Tasks breakdown

### #227 (this) — design + key + lookup scaffold

- Module `crates/server/src/prefix_cache.rs`
- Types:
  - `ChunkKey(u64)` — hashed chunk identifier
  - `PrefixKeys` — chained chunk keys for a prompt
  - `PrefixCache` — the index (DashMap or RwLock<HashMap>)
  - `CacheEntry` (stub for now — KV ptrs filled in #228)
- `PrefixCache::longest_match(&PrefixKeys) -> Option<(usize, &CacheEntry)>`
- No KV ops; no allocation; just the index + lookup logic.
- Wire as `Arc<PrefixCache>` on `ServerState`.
- Unit test: insert sequence of keys, query, validate longest match.

### #228 — KV restore (hot path)

- Per-topology restore: copy snapshot device buffers into the
  active inflight session's KV cache layers. Set
  `LayerCache::current_tokens = n_matched * chunk_size`.
- Engage from `prefill_logits` before the chunked-prefill loop.
- For matched-but-not-final tail, slice the prompt: skip the
  first `n_matched * chunk_size` tokens, prefill the rest.
- Live cert: chat with N turns, verify TTFT drops >50% on turn 2+.

### #229 — KV snapshot at request end + eviction

- After `prefill_logits` returns successfully, snapshot any newly-
  created chunk boundaries.
- LRU eviction when total VRAM > `MAX_GB`.
- Concurrent-safe: writes serialise on a per-key mutex; reads can
  race with writes (latest writer wins).

## Open questions deferred to implementation

- **Do we cache the GDN recurrent state or just the attn KV?** GDN
  state is small (~per-layer-rank, fixed-size matrices) so probably
  yes, but parity at chunk boundary needs verification (we have
  `chunked_prefill_kv_parity_hybrid.rs` but not a GDN-snapshot test
  yet).
- **What's the right `MAX_GB` default?** Probably 2 GB / device for
  the V1 rig (16 GB MI50 with 8.5 GB weights + KV). Operator can
  override via env.
- **Scope of "topology mismatch"**: if pp_size or tp_size changes
  between requests on the same server, that's impossible (server
  immutable post-boot). But if the cluster's `peer_access` changes
  mid-session (link flake), the cached BAR pointers would still
  be valid since they're per-rank. No issue.
