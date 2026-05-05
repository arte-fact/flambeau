//! **#227 P2.10a** — prompt prefix caching: keys, lookup, scaffold.
//!
//! Skips re-prefilling tokens already prefilled by a prior request whose
//! prompt is a prefix of the current one. Wins big for multi-turn chat,
//! where every turn N's prompt is `system + user1 + asst1 + ... + userN-1`,
//! identical KV state to turn N-1's prefill.
//!
//! Granularity: chunks of `FLAMBEAU_PREFILL_UBATCH` tokens (default 512).
//! KV at chunk boundaries is bit-identical to single-shot prefill of the
//! same prefix (Phase A2 + A2-TP parity certs).
//!
//! V1 scope is design + key types + lookup; KV restore lands in
//! [`crate::prefix_cache_restore`] (#228) and write/eviction in #229. This
//! module is the index — read-only on the hot path, write-protected on
//! request end.
//!
//! Design doc: `doc/V1.x/prefix_cache_design.md`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// Default chunk size when `FLAMBEAU_PREFILL_UBATCH` is unset.
///
/// Matches the chunked-prefill default in `model.rs`. Cache entries are
/// only valid when produced under the same chunk size; mismatches are
/// rejected at lookup.
pub const DEFAULT_CHUNK_TOKENS: usize = 512;

/// Chained 64-bit hash of one chunk's tokens.
///
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
///
/// Built by walking the prompt's tokens in `chunk_tokens`-sized strides
/// and chaining each chunk's hash against the prior key. The last
/// element is the key for the full prompt's complete chunks (a partial
/// final chunk is *not* keyed — only chunk boundaries are cacheable).
///
/// Examples
///
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
    pub fn from_prompt(prompt: &[u32], chunk_tokens: usize) -> PrefixKeys {
        assert!(chunk_tokens > 0, "chunk_tokens must be > 0");
        let n_full = prompt.len() / chunk_tokens;
        let mut chunk_keys = Vec::with_capacity(n_full);
        let mut prev = ChunkKey::SEED;
        for i in 0..n_full {
            let chunk = &prompt[i * chunk_tokens..(i + 1) * chunk_tokens];
            prev = prev.extend(chunk);
            chunk_keys.push(prev);
        }
        PrefixKeys {
            chunk_keys,
            chunk_tokens,
            prompt_tokens: prompt.len(),
        }
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

/// One cache entry. **Stub for #227** — the actual KV-snapshot device
/// pointers land in #228. Today this just records what would be cached
/// so the lookup path can be wired and unit-tested.
///
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
    /// Prompt-token-position the entry covers (= `chain.len() * chunk_tokens`).
    /// Convenience for callers that want to slice the input prompt.
    pub n_tokens: usize,
    // **#228** — fields landed in the next task:
    //   pub kv_per_rank: Vec<Vec<DevicePtr>>,  // [rank][layer] K+V buffers
    //   pub gdn_state_per_rank: Vec<Vec<...>>, // for hybrid models
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
///
/// Multi-reader, single-writer via `RwLock<HashMap>`. The hot path is
/// read-heavy (every prefill consults it once); writes happen at request
/// end and on eviction. A more elaborate `DashMap` is justified later
/// if profiling shows lock contention; for V1 a single RwLock is fine
/// (the chat handler already serialises on bigger primitives).
///
/// Indexed by **terminal chunk key** — `chain[chain.len() - 1]`. The
/// lookup function walks the prompt's chunk_keys longest-to-shortest,
/// returning the first match whose stored chain prefix-matches the
/// caller's keys exactly.
pub struct PrefixCache {
    inner: RwLock<PrefixCacheInner>,
    /// VRAM budget across all entries (bytes). Sized at construction
    /// from `FLAMBEAU_PREFIX_CACHE_MAX_GB`. **#229** uses this for
    /// eviction; #227 just stores it.
    pub vram_budget_bytes: usize,
}

#[derive(Default)]
struct PrefixCacheInner {
    /// terminal chunk key → entry. Multiple entries with the same
    /// terminal key but different chains can exist (rare; resolved by
    /// the chain-equality check at lookup).
    by_terminal: HashMap<ChunkKey, Vec<CacheEntry>>,
    /// LRU order: front = most recently touched. **#229** populates
    /// this; #227 leaves it empty.
    #[expect(dead_code, reason = "wired in #229 (write path + eviction)")]
    lru_order: std::collections::VecDeque<ChunkKey>,
    /// Total VRAM used by all entries.
    used_bytes: usize,
}

impl PrefixCache {
    /// Construct an empty cache with the given VRAM budget.
    pub fn new(vram_budget_bytes: usize) -> PrefixCache {
        PrefixCache {
            inner: RwLock::new(PrefixCacheInner::default()),
            vram_budget_bytes,
        }
    }

    /// Read the `FLAMBEAU_PREFIX_CACHE_MAX_GB` env (default 2 GB).
    pub fn budget_from_env() -> usize {
        let gb: f64 = std::env::var("FLAMBEAU_PREFIX_CACHE_MAX_GB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        (gb * 1024.0 * 1024.0 * 1024.0) as usize
    }

    /// True iff prefix-cache is enabled for this server.
    /// Default OFF in V1 (set `FLAMBEAU_PREFIX_CACHE=1` to engage).
    pub fn enabled() -> bool {
        std::env::var("FLAMBEAU_PREFIX_CACHE")
            .map(|v| !matches!(v.as_str(), "0" | ""))
            .unwrap_or(false)
    }

    /// Find the longest prefix of `keys` whose chain matches a stored
    /// entry under the given topology + chunk size.
    ///
    /// Walks the prompt's chunk_keys from longest to shortest. The first
    /// terminal that hits the index AND whose stored chain agrees byte-
    /// for-byte with `keys.chunk_keys[..n]` wins.
    ///
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
                return Some(MatchInfo {
                    n_chunks: n,
                    n_tokens: n * keys.chunk_tokens,
                    terminal,
                });
            }
        }
        None
    }

    /// **#229 stub** — insert an entry. `kv` field will be added to
    /// `CacheEntry` in #228; for #227 we accept the (chain, topology,
    /// chunk_tokens) skeleton so the index can be exercised in tests.
    pub fn insert_skeleton(
        &self,
        chain: Vec<ChunkKey>,
        topology: TopologyTag,
        chunk_tokens: usize,
    ) {
        let Some(&terminal) = chain.last() else {
            return;
        };
        let entry = CacheEntry {
            topology,
            chunk_tokens,
            n_tokens: chain.len() * chunk_tokens,
            chain,
        };
        let mut inner = self.inner.write().unwrap();
        inner
            .by_terminal
            .entry(terminal)
            .or_default()
            .push(entry);
    }

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
        assert_eq!(a.n_chunks(), 2); // 1100 / 512 = 2 (floor)
        assert_ne!(a.chunk_keys[0], a.chunk_keys[1]);
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
        let cache = PrefixCache::new(0);
        // Prompt of 3 chunks. Insert two cached entries: 2 chunks, 3 chunks.
        let prompt: Vec<u32> = (0..1536).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        cache.insert_skeleton(keys.chunk_keys[..2].to_vec(), topo(), 512);
        cache.insert_skeleton(keys.chunk_keys[..3].to_vec(), topo(), 512);

        let m = cache
            .longest_match(&keys, topo(), |_| {})
            .expect("expected hit");
        assert_eq!(m.n_chunks, 3);
        assert_eq!(m.n_tokens, 1536);
    }

    #[test]
    fn longest_match_returns_partial_when_full_not_cached() {
        let cache = PrefixCache::new(0);
        let prompt: Vec<u32> = (0..1536).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        // Cache only the first 2 chunks.
        cache.insert_skeleton(keys.chunk_keys[..2].to_vec(), topo(), 512);

        let m = cache
            .longest_match(&keys, topo(), |_| {})
            .expect("expected partial hit");
        assert_eq!(m.n_chunks, 2);
        assert_eq!(m.n_tokens, 1024);
    }

    #[test]
    fn topology_mismatch_rejects() {
        let cache = PrefixCache::new(0);
        let prompt: Vec<u32> = (0..1024).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        let other = TopologyTag {
            mesh_kind: "pp",
            ranks: 4,
            pp_size: 4,
            tp_size: 1,
        };
        cache.insert_skeleton(keys.chunk_keys.clone(), other, 512);

        let hit = cache.longest_match(&keys, topo(), |_| {});
        assert!(hit.is_none(), "topology mismatch should reject");
    }

    #[test]
    fn empty_prompt_no_match() {
        let cache = PrefixCache::new(0);
        let keys = PrefixKeys::from_prompt(&[], 512);
        assert!(cache.longest_match(&keys, topo(), |_| {}).is_none());
    }

    #[test]
    fn partial_final_chunk_excluded_from_chain() {
        // 700-token prompt at chunk=512 → only 1 complete chunk.
        let prompt: Vec<u32> = (0..700).collect();
        let keys = PrefixKeys::from_prompt(&prompt, 512);
        assert_eq!(keys.n_chunks(), 1);
        assert_eq!(keys.prompt_tokens, 700);
    }
}
