//! **#227 P2.10a** — prompt prefix caching: keys, lookup, scaffold.
//! Skips re-prefilling tokens already prefilled by a prior request whose
//! prompt is a prefix of the current one. Wins big for multi-turn chat,
//! where every turn N's prompt is `system + user1 + asst1 + ... + userN-1`,
//! identical KV state to turn N-1's prefill.
//! Granularity: chunks of `FLAMBEAU_PREFILL_UBATCH` tokens (default 512).
//! KV at chunk boundaries is bit-identical to single-shot prefill of the
//! same prefix (Phase A2 + A2-TP parity certs).
//! V1 scope is design + key types + lookup; KV restore lands in
//! [`crate::prefix_cache_restore`] (#228) and write/eviction in #229. This
//! module is the index — read-only on the hot path, write-protected on
//! request end.
//! Design doc: `doc/V1.x/prefix_cache_design.md`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// Default chunk size when `FLAMBEAU_PREFILL_UBATCH` is unset.
/// Matches the chunked-prefill default in `model.rs`. Cache entries are
/// only valid when produced under the same chunk size; mismatches are
/// rejected at lookup.
pub const DEFAULT_CHUNK_TOKENS: usize = 512;

/// Chained 64-bit hash of one chunk's tokens.
/// Computed as `H(prev_chunk_key, tokens_in_chunk)` so that the i-th
/// chunk key uniquely identifies the *full prefix* `tokens[0..i*chunk]`.
/// `prev_chunk_key` is `0` for the first chunk (the seed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkKey(pub u64);

impl ChunkKey {
    /// Seed for the first chunk in a prompt.
    pub const SEED: ChunkKey = ChunkKey(0);

    /// Hash one chunk's tokens against a previous chunk key.
    pub fn extend(self, tokens: &[u32]) -> ChunkKey {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut h);
        tokens.len().hash(&mut h);
        for &t in tokens {
            t.hash(&mut h);
        }
        ChunkKey(h.finish())
    }
}

/// Chained chunk keys for a full prompt.
/// Built by walking the prompt's tokens in `chunk_tokens`-sized strides
/// and chaining each chunk's hash against the prior key. The last
/// element is the key for the full prompt's complete chunks (a partial
/// final chunk is *not* keyed — only chunk boundaries are cacheable).
/// Examples
/// ```ignore
/// // 1300-token prompt at chunk=512:
/// // -> 2 complete chunks (0..512, 512..1024)
/// // -> chunk_keys.len() == 2
/// // -> tail tokens 1024..1300 are not cacheable, must be prefilled
/// ```
#[derive(Debug, Clone)]
pub struct PrefixKeys {
    /// Per-chunk hash chain. Length = `prompt_len / chunk_tokens`
    /// (integer division — partial final chunks excluded).
    pub chunk_keys: Vec<ChunkKey>,
    /// Chunk size used when computing the chain (`FLAMBEAU_PREFILL_UBATCH`
    /// at the time of the request). Cache entries store this and reject
    /// lookups with a different chunk size.
    pub chunk_tokens: usize,
    /// Total prompt length (tokens). Convenience for the lookup return.
    pub prompt_tokens: usize,
}

impl PrefixKeys {
    /// Compute the chunk-key chain for a prompt at the given chunk size.
    /// **#229 V1**: the chain includes a final partial-tail chunk when
    /// `prompt.len() % chunk_tokens != 0`, so the last `chunk_keys` entry
    /// always uniquely identifies the *full* prompt. Prefix-only matches
    /// at chunk boundaries are still expressible (callers can probe the
    /// shorter-chain entries) but are not used in V1 because GDN
    /// recurrent state isn't snapshotted at chunk boundaries — we only
    /// hit on full-prompt match.
    pub fn from_prompt(prompt: &[u32], chunk_tokens: usize) -> PrefixKeys {
        assert!(chunk_tokens > 0, "chunk_tokens must be > 0");
        let n_full = prompt.len() / chunk_tokens;
        let has_tail = prompt.len() % chunk_tokens != 0;
        let n_chunks = n_full + (has_tail as usize);
        let mut chunk_keys = Vec::with_capacity(n_chunks);
        let mut prev = ChunkKey::SEED;
        for i in 0..n_full {
            let chunk = &prompt[i * chunk_tokens..(i + 1) * chunk_tokens];
            prev = prev.extend(chunk);
            chunk_keys.push(prev);
        }
        if has_tail {
            let tail = &prompt[n_full * chunk_tokens..];
            prev = prev.extend(tail);
            chunk_keys.push(prev);
        }
        PrefixKeys {
            chunk_keys,
            chunk_tokens,
            prompt_tokens: prompt.len(),
        }
    }

    /// **#229** — token count covered by the i-th chunk key (1-based).
    /// Mostly returns `(i+1) * chunk_tokens` but the last chunk in a
    /// prompt with a partial tail covers `prompt_tokens` total.
    pub fn tokens_for_chunk_index(&self, idx: usize) -> usize {
        let candidate = (idx + 1) * self.chunk_tokens;
        candidate.min(self.prompt_tokens)
    }

    /// Number of complete chunks in this prompt (= `chunk_keys.len()`).
    pub fn n_chunks(&self) -> usize {
        self.chunk_keys.len()
    }
}

/// Topology fingerprint stored on each cache entry. Mismatches cause
/// lookup to skip the entry (defensive — a fresh `serve` starts a fresh
/// cache, so cross-topology pollution shouldn't happen normally, but
/// the check is cheap and preempts catastrophic restores if it ever
/// did).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyTag {
    /// "pp" / "tp" / "pp+tp"
    pub mesh_kind: &'static str,
    /// Total ranks in the cluster.
    pub ranks: u32,
    /// PP stages (1 for non-PP). For pp+tp: the outer pp_size.
    pub pp_size: u32,
    /// TP world / per-stage size (1 for non-TP).
    pub tp_size: u32,
}

/// Host-side KV snapshot covering one rank's full layer set at a given
/// prefix length.
/// V1 stores the snapshot in host RAM rather than device VRAM — the
/// per-snapshot footprint is too big to keep many on-device, but host
/// RAM is plentiful (64+ GB). The DtoH→HtoD round-trip on hit is
/// ~700 ms at 6 GB/s for a 4 GB snapshot — still a net win vs the
/// 2-3 s prefill it replaces.
/// Hidden behind a feature gate at the type level: when `hip` is off
/// (CPU-only build of the server crate) this module compiles, but
/// the snapshot type is feature-gated since it carries
/// `LayerCacheSnapshot` from the qwen3-moe crate which itself is
/// `#[cfg(feature = "hip")]`.
/// Opaque per-rank snapshot bytes. Layout is arch-specific; the prefix
/// cache treats it as bytes for size accounting + storage. After the
/// legacy qwen3-moe stack was removed (#221) the snapshot/restore path
/// is on hold pending #219 (v2 prefix-cache reimplementation); the type
/// stays in the public surface as a `Vec<u8>` placeholder.
#[cfg(feature = "hip")]
pub type RankSnapshot = Vec<u8>;

/// Full multi-rank KV snapshot — one [`RankSnapshot`] per rank in the
/// captured topology.
#[cfg(feature = "hip")]
pub type KvSnapshot = Vec<RankSnapshot>;

/// One cache entry. **#228** adds the optional `kv` snapshot field;
/// `kv = None` means the entry exists in the index for lookup-only
/// purposes (e.g. tests, future write-deferred state).
/// `n_chunks` is the number of chunks this entry covers (so the matched
/// prefix is `n_chunks * chunk_tokens` tokens). The entry's chain is
/// stored so lookup can verify the full prefix matches, not just the
/// final chunk's hash (defends against partial-chain collisions).
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Topology this entry was captured under. Lookup rejects on mismatch.
    pub topology: TopologyTag,
    /// Chunk size when this entry was captured. Lookup rejects on mismatch.
    pub chunk_tokens: usize,
    /// Full chain up to and including this entry's terminal key. Used
    /// for the lookup integrity check (a probe matches only when the
    /// full chain agrees, not just the terminal key).
    pub chain: Vec<ChunkKey>,
    /// Total prompt-token count this entry covers — equals the original
    /// prompt length (chain may include a partial-tail chunk).
    pub n_tokens: usize,
    /// **#228** — host-side KV snapshot covering every rank's layers at
    /// `n_tokens`. `None` = index-only entry (lookup hits, restore
    /// no-ops).
    #[cfg(feature = "hip")]
    pub kv: Option<std::sync::Arc<KvSnapshot>>,
    /// **#229 V1** — host-copy of the LAST-position logits row from the
    /// originating prefill (one F32 per vocab entry, ~600 KB on
    /// Qwen3.6). Returned to the caller on full-prompt hit so it can
    /// sample the first decode token without re-running prefill. None
    /// means index-only / pre-#229 entry; a full hit with `None`
    /// degrades to "restore + re-prefill last token" which doesn't
    /// work on GDN topologies, so callers treat None as a miss in V1.
    pub last_logits: Option<std::sync::Arc<Vec<f32>>>,
}

impl CacheEntry {
    /// Number of chunks this entry covers.
    pub fn n_chunks(&self) -> usize {
        self.chain.len()
    }
}

/// Outcome of [`PrefixCache::longest_match`].
#[derive(Debug, Clone)]
pub struct CacheHit<'a> {
    /// Number of chunks matched (`>= 1` on a real hit).
    pub n_chunks: usize,
    /// Tokens in the matched prefix (`n_chunks * chunk_tokens`).
    pub n_tokens: usize,
    /// Reference to the cache entry covering the longest match. The
    /// caller's lifetime constraint is the read-lock guard returned by
    /// `longest_match`'s API; callers should `clone()` device pointers
    /// they need to outlive the guard.
    pub entry: &'a CacheEntry,
}

/// Process-local prefix-cache index.
/// Multi-reader, single-writer via `RwLock<HashMap>`. The hot path is
/// read-heavy (every prefill consults it once); writes happen at request
/// end and on eviction. A more elaborate `DashMap` is justified later
/// if profiling shows lock contention; for V1 a single RwLock is fine
/// (the chat handler already serialises on bigger primitives).
/// Indexed by **terminal chunk key** — `chain[chain.len() - 1]`. The
/// lookup function walks the prompt's chunk_keys longest-to-shortest,
/// returning the first match whose stored chain prefix-matches the
/// caller's keys exactly.
pub struct PrefixCache {
    inner: RwLock<PrefixCacheInner>,
    /// VRAM budget across all entries (bytes). Sized at construction
    /// from `cfg.prefix_cache_max_gb`. **#229** uses this for eviction;
    /// #227 just stores it.
    pub vram_budget_bytes: usize,
    /// True iff this server has the prefix cache enabled. Stored at
    /// construction (from `cfg.prefix_cache`); read by every cache-
    /// consulting hot-path site.
    enabled: bool,
}

#[derive(Default)]
struct PrefixCacheInner {
    /// terminal chunk key → entry. Multiple entries with the same
    /// terminal key but different chains can exist (rare; resolved by
    /// the chain-equality check at lookup).
    by_terminal: HashMap<ChunkKey, Vec<CacheEntry>>,
    /// LRU order: front = most recently touched. Populated by
    /// `insert_with_kv` (#229).
    lru_order: std::collections::VecDeque<ChunkKey>,
    /// Total VRAM used by all entries.
    used_bytes: usize,
}

impl PrefixCache {
    /// Construct an empty cache with the given VRAM budget and enabled flag.
    pub fn new(vram_budget_bytes: usize, enabled: bool) -> PrefixCache {
        PrefixCache {
            inner: RwLock::new(PrefixCacheInner::default()),
            vram_budget_bytes,
            enabled,
        }
    }

    /// Convert a GB budget to bytes (rounded toward zero).
    pub fn gb_to_bytes(gb: f64) -> usize {
        (gb * 1024.0 * 1024.0 * 1024.0) as usize
    }

    /// True iff prefix-cache is enabled for this server.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Find the longest prefix of `keys` whose chain matches a stored
    /// entry under the given topology + chunk size.
    /// Walks the prompt's chunk_keys from longest to shortest. The first
    /// terminal that hits the index AND whose stored chain agrees byte-
    /// for-byte with `keys.chunk_keys[..n]` wins.
    /// Returns `None` when:
    /// - the prompt has no complete chunks (`keys.chunk_keys.is_empty()`),
    /// - no terminal key matches under the given topology + chunk size,
    /// - or every terminal that matched had a divergent chain (collision).
    pub fn longest_match<F>(
        &self,
        keys: &PrefixKeys,
        topology: TopologyTag,
        mut on_hit_touch_lru: F,
    ) -> Option<MatchInfo>
    where
        F: FnMut(ChunkKey),
    {
        if keys.chunk_keys.is_empty() {
            return None;
        }
        let inner = self.inner.read().unwrap();
        // Walk from longest to shortest.
        for n in (1..=keys.chunk_keys.len()).rev() {
            let terminal = keys.chunk_keys[n - 1];
            let Some(candidates) = inner.by_terminal.get(&terminal) else {
                continue;
            };
            for entry in candidates {
                if entry.topology != topology
                    || entry.chunk_tokens != keys.chunk_tokens
                    || entry.chain.len() != n
                {
                    continue;
                }
                if entry.chain[..] != keys.chunk_keys[..n] {
                    continue;
                }
                // Hit — caller updates LRU outside the read lock.
                on_hit_touch_lru(terminal);
                // n_tokens uses the entry's stored count, not a
                // chunk-aligned multiple, so partial-tail prompts
                // (where the last chunk covers < chunk_tokens) round-
                // trip correctly.
                return Some(MatchInfo {
                    n_chunks: n,
                    n_tokens: entry.n_tokens,
                    terminal,
                });
            }
        }
        None
    }

    /// **#229 stub** — insert an index-only entry (no KV, no logits).
    /// Used by tests and as a fallback path when capture is disabled.
    /// The hit path no-ops on entries without a `kv` snapshot.
    /// `n_tokens` is the prompt-token count this entry covers; for
    /// chunk-aligned prompts that's `chain.len() * chunk_tokens`, but
    /// the caller must pass it explicitly because partial-tail prompts
    /// have a final chunk covering < `chunk_tokens` tokens.
    pub fn insert_skeleton(
        &self,
        chain: Vec<ChunkKey>,
        topology: TopologyTag,
        chunk_tokens: usize,
        n_tokens: usize,
    ) {
        let Some(&terminal) = chain.last() else {
            return;
        };
        let entry = CacheEntry {
            topology,
            chunk_tokens,
            n_tokens,
            chain,
            #[cfg(feature = "hip")]
            kv: None,
            last_logits: None,
        };
        let mut inner = self.inner.write().unwrap();
        inner
            .by_terminal
            .entry(terminal)
            .or_default()
            .push(entry);
    }

    /// **#228** — clone out the `Arc<KvSnapshot>` for a hit terminal so
    /// the caller can drop the read lock before doing the (slow)
    /// host→device restore. Returns `None` when the entry has no
    /// snapshot attached (index-only).
    #[cfg(feature = "hip")]
    pub fn snapshot_for(&self, terminal: ChunkKey) -> Option<std::sync::Arc<KvSnapshot>> {
        let inner = self.inner.read().unwrap();
        let entries = inner.by_terminal.get(&terminal)?;
        entries
            .iter()
            .find_map(|e| e.kv.as_ref().map(std::sync::Arc::clone))
    }

    /// **#229 V1** — clone out the cached last-position logits for a
    /// hit terminal. Pair with `snapshot_for`; on full-prompt match the
    /// caller restores KV/GDN, then samples first decode token from
    /// these logits without re-running prefill.
    pub fn logits_for(&self, terminal: ChunkKey) -> Option<std::sync::Arc<Vec<f32>>> {
        let inner = self.inner.read().unwrap();
        let entries = inner.by_terminal.get(&terminal)?;
        entries
            .iter()
            .find_map(|e| e.last_logits.as_ref().map(std::sync::Arc::clone))
    }

    /// **#229 P2.10c** — insert an entry with its KV snapshot, account
    /// the bytes against the budget, evict LRU until under cap.
    /// Caller passes `bytes` (size of the snapshot in host RAM).
    /// Touches the entry's terminal as MRU. Idempotent: re-inserting
    /// the same chain replaces the prior entry's `kv` field.
    #[cfg(feature = "hip")]
    pub fn insert_with_kv(
        &self,
        chain: Vec<ChunkKey>,
        topology: TopologyTag,
        chunk_tokens: usize,
        n_tokens: usize,
        kv: std::sync::Arc<KvSnapshot>,
        last_logits: Option<std::sync::Arc<Vec<f32>>>,
        bytes: usize,
    ) {
        let Some(&terminal) = chain.last() else {
            return;
        };
        let mut inner = self.inner.write().unwrap();
        // Replace if a same-chain entry already exists; else push new.
        let existing = inner.by_terminal.entry(terminal).or_default();
        let mut replaced_bytes = 0usize;
        if let Some(idx) = existing.iter().position(|e| {
            e.topology == topology && e.chunk_tokens == chunk_tokens && e.chain == chain
        }) {
            // Replace — release prior accounting.
            if let Some(prior) = existing[idx].kv.as_ref() {
                replaced_bytes = snapshot_bytes_arc(prior);
            }
            if let Some(prior_lp) = existing[idx].last_logits.as_ref() {
                replaced_bytes += prior_lp.len() * std::mem::size_of::<f32>();
            }
            existing[idx].kv = Some(std::sync::Arc::clone(&kv));
            existing[idx].last_logits = last_logits.as_ref().map(std::sync::Arc::clone);
        } else {
            existing.push(CacheEntry {
                topology,
                chunk_tokens,
                n_tokens,
                chain,
                kv: Some(std::sync::Arc::clone(&kv)),
                last_logits: last_logits.as_ref().map(std::sync::Arc::clone),
            });
        }
        inner.used_bytes = inner.used_bytes.saturating_sub(replaced_bytes) + bytes;
        // Move to MRU.
        inner.lru_order.retain(|k| *k != terminal);
        inner.lru_order.push_front(terminal);
        // Evict LRU until under budget.
        while inner.used_bytes > self.vram_budget_bytes {
            let Some(victim) = inner.lru_order.pop_back() else {
                break;
            };
            if let Some(victim_entries) = inner.by_terminal.remove(&victim) {
                for e in victim_entries {
                    if let Some(arc) = e.kv {
                        inner.used_bytes = inner
                            .used_bytes
                            .saturating_sub(snapshot_bytes_arc(&arc));
                    }
                    if let Some(lp) = e.last_logits {
                        inner.used_bytes = inner.used_bytes.saturating_sub(
                            lp.len() * std::mem::size_of::<f32>(),
                        );
                    }
                }
            }
        }
    }
}

/// **#229** — sum bytes of one snapshot held by `Arc<KvSnapshot>` for
/// LRU accounting. Mirrors `model::snapshot_bytes` but takes the Arc
/// view (avoids importing the model crate's helper into this module).
#[cfg(feature = "hip")]
fn snapshot_bytes_arc(snap: &KvSnapshot) -> usize {
    snap.iter().map(|rank| rank.len()).sum()
}

/// Standalone helpers don't need to live in `impl PrefixCache` — keeping
/// them here avoids forcing callers to construct an instance just to
/// hash a prompt.
impl PrefixCache {

    /// Number of stored entries (sum across all terminal keys).
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .unwrap()
            .by_terminal
            .values()
            .map(|v| v.len())
            .sum()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().by_terminal.is_empty()
    }

    /// Used VRAM (bytes) — populated by #229.
    pub fn used_bytes(&self) -> usize {
        self.inner.read().unwrap().used_bytes
    }
}

/// Public match descriptor. The `terminal` key is exposed so callers
/// can re-fetch the entry under whatever lock discipline they prefer
/// (read for KV-restore in #228; write for LRU touch).
#[derive(Debug, Clone, Copy)]
pub struct MatchInfo {
    pub n_chunks: usize,
    pub n_tokens: usize,
    pub terminal: ChunkKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topo() -> TopologyTag {
        TopologyTag {
            mesh_kind: "tp",
            ranks: 2,
            pp_size: 1,
            tp_size: 2,
        }
    }

    #[test]
    fn chunk_chain_is_deterministic_and_distinct() {
        let prompt = (0u32..1100).collect::<Vec<_>>();
        let a = PrefixKeys::from_prompt(&prompt, 512);
        let b = PrefixKeys::from_prompt(&prompt, 512);
        assert_eq!(a.chunk_keys, b.chunk_keys);
        // 1100 / 512 = 2 full chunks + 1 partial tail (76 tokens)
        assert_eq!(a.n_chunks(), 3);
        assert_ne!(a.chunk_keys[0], a.chunk_keys[1]);
        assert_ne!(a.chunk_keys[1], a.chunk_keys[2]);
        assert_eq!(a.tokens_for_chunk_index(0), 512);
        assert_eq!(a.tokens_for_chunk_index(1), 1024);
        assert_eq!(a.tokens_for_chunk_index(2), 1100);
    }

    #[test]
    fn diverging_chunks_diverge_chains() {
        let p1: Vec<u32> = (0..1024).collect();
        let mut p2 = p1.clone();
        p2[600] = 9999; // differs in second chunk
        let a = PrefixKeys::from_prompt(&p1, 512);
        let b = PrefixKeys::from_prompt(&p2, 512);
        // First chunk identical → chain[0] equal.
        assert_eq!(a.chunk_keys[0], b.chunk_keys[0]);
        // Second chunk differs → chain[1] differs.
        assert_ne!(a.chunk_keys[1], b.chunk_keys[1]);
    }

    #[test]
    fn longest_match_walks_longest_first() {
        let cache = PrefixCache::new(0, true);
        // 1536-token prompt at chunk=512 = 3 full chunks, no tail.
        let prompt: Vec<u32> = (0..1536).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        assert_eq!(keys.n_chunks(), 3);
        cache.insert_skeleton(keys.chunk_keys[..2].to_vec(), topo(), 512, 1024);
        cache.insert_skeleton(keys.chunk_keys[..3].to_vec(), topo(), 512, 1536);

        let m = cache
            .longest_match(&keys, topo(), |_| {})
            .expect("expected hit");
        assert_eq!(m.n_chunks, 3);
        assert_eq!(m.n_tokens, 1536);
    }

    #[test]
    fn longest_match_returns_partial_when_full_not_cached() {
        let cache = PrefixCache::new(0, true);
        let prompt: Vec<u32> = (0..1536).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        // Cache only the first 2 chunks.
        cache.insert_skeleton(keys.chunk_keys[..2].to_vec(), topo(), 512, 1024);

        let m = cache
            .longest_match(&keys, topo(), |_| {})
            .expect("expected partial hit");
        assert_eq!(m.n_chunks, 2);
        assert_eq!(m.n_tokens, 1024);
    }

    #[test]
    fn topology_mismatch_rejects() {
        let cache = PrefixCache::new(0, true);
        let prompt: Vec<u32> = (0..1024).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        let other = TopologyTag {
            mesh_kind: "pp",
            ranks: 4,
            pp_size: 4,
            tp_size: 1,
        };
        cache.insert_skeleton(keys.chunk_keys.clone(), other, 512, 1024);

        let hit = cache.longest_match(&keys, topo(), |_| {});
        assert!(hit.is_none(), "topology mismatch should reject");
    }

    #[test]
    fn empty_prompt_no_match() {
        let cache = PrefixCache::new(0, true);
        let keys = PrefixKeys::from_prompt(&[], 512);
        assert!(cache.longest_match(&keys, topo(), |_| {}).is_none());
    }

    #[test]
    fn partial_final_chunk_included_in_chain() {
        // **#229 V1**: 700-token prompt at chunk=512 → 1 full chunk +
        // 1 partial-tail chunk = 2 entries, last covering 700 tokens.
        let prompt: Vec<u32> = (0..700).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        assert_eq!(keys.n_chunks(), 2);
        assert_eq!(keys.prompt_tokens, 700);
        assert_eq!(keys.tokens_for_chunk_index(0), 512);
        assert_eq!(keys.tokens_for_chunk_index(1), 700);
    }
}
