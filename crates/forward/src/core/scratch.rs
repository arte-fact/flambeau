//! Pre-allocated device scratch + per-layer KV cache.

use anyhow::{Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};

use crate::ctx::GdnDims;
use crate::loader::ShardMode;

/// `q_width` / `kv_width` are PER-RANK under TP (caller divides by
/// tp_size) AND act as upper bounds across layers — they size the
/// shared (ephemeral) Q / K / V / projection scratch slots, which
/// every layer reuses. For uniform arches they equal the per-layer
/// values; for gemma4-style SWA/global alternation they are the
/// per-layer max.
///
/// `num_layers` is the count of owned KV slots — PP rank owning a
/// layer slice passes the slice length, not the global total.
///
/// `per_layer_kv_widths`: when `Some`, length must equal `num_layers`
/// and each entry sets that slot's KV-cache stride (used by gemma4
/// SWA layers, which need half the cache of global-attention layers).
/// When `None`, every slot is sized at `kv_width`.
///
/// `max_experts` sizes the MoE router logits slot; 0 for dense-only.
/// `gdn` is Some for hybrid arches; sizes the per-layer state +
/// conv-history slots (allocated once per owned layer).
#[derive(Clone, Debug)]
pub struct ScratchConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub q_width: usize,
    pub kv_width: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub max_experts: usize,
    /// Top-k value (e.g. 8 for Qwen3.6 MoE, 4 for gemma4-MoE). Sizes
    /// the per-token `expert_ids` / `expert_weights` slots and the
    /// per-slot indexed-MoE scratch (`gate_out` / `up_out` /
    /// `activated` / `down`). `0` when no MoE.
    pub max_experts_per_tok: usize,
    pub gdn: Option<GdnDims>,
    pub per_layer_kv_widths: Option<Vec<usize>>,
    /// `true` when the arch has gated full-attention (qwen3.5 /
    /// qwen3.6 / qwen3-Next). Drives allocation of the extra
    /// `q_fused_f16` (2·q_width F16) + `gate_f16` (q_width F16)
    /// scratch slots the split-then-sigmoid-gate path needs.
    pub attn_q_gated: bool,
    /// Per-layer shared-expert FFN intermediate size. `0` when the
    /// MoE arch has no shared expert; positive when one is present
    /// (Qwen3.6-35B-A3B = 512, qwen3next = ...). Drives `shared_x_norm_f32`
    /// scratch sizing.
    pub shared_intermediate: usize,
    /// Upper bound on tokens-per-forward. 1 for decode-only; >1 for
    /// chunked prefill. Every per-token scratch slot (resid, norm,
    /// q/k/v, gate, attn_out, gate_f32/up_f32/gated, down, moe_accum,
    /// shared_x_norm, router_logits, position_i32) is sized at
    /// `max_prefill_tokens * <per-token width>`.
    pub max_prefill_tokens: usize,
    /// Number of independent inflight slots this pool reserves KV/GDN
    /// state for. 1 = single-request decode + chunked prefill. >1 =
    /// batched-decode across N concurrent slots. Each layer's
    /// `kv_caches[li].k/v` is sized `[max_slots, max_seq_len, kv_width]`;
    /// GDN per-layer state + conv_history multiply by `max_slots`.
    pub max_slots: usize,
    /// Per-layer side-channel embedding width (gemma 4n / E2B / E4B,
    /// 256 on E4B). `0` when the arch has no per-layer side-channel;
    /// drives the 6 small F32 / F16 scratch slots for the apply block.
    pub per_layer_embd: usize,
    /// PagedAttention geometry, when configured. `Some(cfg)` allocates a
    /// parallel paged KV cache alongside the existing contiguous
    /// `kv_caches`. The two coexist until E3c rewires `standard_attn` to
    /// dispatch on the paged path — at that point the contiguous slabs
    /// can be skipped when paged is on. Derive via
    /// [`PagedKvCacheConfig::from_vram_budget`].
    pub paged_kv: Option<PagedKvCacheConfig>,
}

impl Default for ScratchConfig {
    fn default() -> Self {
        Self {
            hidden: 0,
            intermediate: 0,
            q_width: 0,
            kv_width: 0,
            vocab: 0,
            max_seq_len: 0,
            num_layers: 0,
            max_experts: 0,
            max_experts_per_tok: 0,
            gdn: None,
            per_layer_kv_widths: None,
            attn_q_gated: false,
            shared_intermediate: 0,
            max_prefill_tokens: 1,
            max_slots: 1,
            per_layer_embd: 0,
            paged_kv: None,
        }
    }
}

/// Maximum split-K chunk count. Sized so that `n_chunks` per
/// `splitk_chunk_size` stays ≤ 32 for any `n_tokens_kv` ≤ 16 384;
/// callers must keep ctx within that bound.
pub const MAX_SPLITK_CHUNKS: usize = 32;

/// Per-layer KV-cache geometry. Arches implement this on their
/// `Config` so `per_layer_kv_widths` can size the ScratchPool's
/// per-layer K/V slabs uniformly. Layers with no KV cache
/// (SSM / GDN / recurrent) MUST return 0 — the alloc loop in
/// `ScratchPool::new` skips zero-width allocations.
///
/// Indexing is GLOBAL: `kv_width_at(li, _)` is called for every
/// `li` in `0..num_layers()`, then the runtime slices the result
/// to the PP layer range in `runtime::workers::build_pool`.
///
/// `n_ranks` is the per-stage TP world size (1 for SD / PP-only).
pub trait KvLayerShape {
    fn num_layers(&self) -> usize;
    fn kv_width_at(&self, li: usize, n_ranks: usize) -> usize;
}

/// Build the per-layer KV width vector for `ScratchConfig`. Length
/// equals `shape.num_layers()`. Pass the result directly into
/// `ScratchConfig::per_layer_kv_widths`. The runtime slices it to the
/// owned PP range; arches should NOT slice themselves.
pub fn per_layer_kv_widths<S: KvLayerShape + ?Sized>(shape: &S, n_ranks: usize) -> Vec<usize> {
    (0..shape.num_layers())
        .map(|li| shape.kv_width_at(li, n_ranks))
        .collect()
}

/// MoE-specific scratch dims, factored out so dense arches return
/// `None` and MoE arches return one value. `shared_intermediate_per_rank`
/// is the row-parallel shared-MLP intermediate (Qwen3.6-35B-A3B = 512,
/// gemma4-26B-A4B = dense `intermediate / n_ranks`); 0 when the MoE
/// arch has no shared expert.
#[derive(Debug, Clone, Copy)]
pub struct MoeShape {
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub shared_intermediate_per_rank: usize,
}

/// Full scratch-pool geometry. Arches implement this on their `Config`
/// and the runtime builds `ScratchConfig` via `scratch_config_for`.
///
/// Width/intermediate accessors take `n_ranks` so each arch's MoE-vs-
/// dense / per-layer-max-vs-uniform choice stays encapsulated rather
/// than leaking into the `Arch::scratch_config` impl. `q_width_per_rank`
/// should return the per-rank max across layers when the arch has
/// per-layer attention shape (gemma4 SWA-vs-global); uniform arches
/// return the single value.
pub trait ScratchShape: KvLayerShape {
    fn hidden(&self) -> usize;
    fn vocab(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn intermediate_per_rank(&self, n_ranks: usize) -> usize;
    fn q_width_per_rank(&self, n_ranks: usize) -> usize;
    fn moe_per_rank(&self, n_ranks: usize) -> Option<MoeShape>;
    fn gdn_per_rank(&self, n_ranks: usize) -> Option<GdnDims>;
    fn attn_q_gated(&self) -> bool;
    fn per_layer_embd(&self) -> usize {
        0
    }
}

/// Build a `ScratchConfig` from a shape spec. Each arch's
/// `Arch::scratch_config` collapses to one call into this helper. The
/// builder derives `kv_width` as `max(per_layer_kv_widths)` so the
/// arch never has to (it's a 0-uniform-or-per-layer-max math identical
/// across qwen and gemma4). `per_layer_kv_widths` is always `Some`;
/// the runtime alloc loop interprets width=0 as "no KV slab for this
/// layer" (GDN / recurrent / shared-KV).
pub fn scratch_config_for<S: ScratchShape + ?Sized>(
    shape: &S,
    shard: ShardMode,
    prefill_ubatch: usize,
    max_slots: usize,
) -> ScratchConfig {
    let n_ranks = shard.n_ranks();
    let per_layer_kv = per_layer_kv_widths(shape, n_ranks);
    let kv_width = per_layer_kv.iter().copied().max().unwrap_or(0);
    let moe = shape.moe_per_rank(n_ranks);
    // Env-gated PagedAttention activation. `FLAMBEAU_PAGED_KV=1`
    // turns on the paged path; n_pages is clamped to `max_slots *
    // max_pages_per_slot` (the contiguous-equivalent floor) so the
    // paged path activates with no VRAM win — sized for correctness
    // validation, not the structural PagedAttention saving. A
    // future flag (or per-arch budget) can size n_pages larger.
    let paged_kv = if std::env::var("FLAMBEAU_PAGED_KV").as_deref() == Ok("1") {
        let page_size = 16;
        let max_seq_len = shape.max_seq_len();
        let mpps = max_seq_len.div_ceil(page_size);
        Some(PagedKvCacheConfig::from_vram_budget(
            0,
            page_size,
            kv_width,
            max_slots,
            mpps,
        ))
    } else {
        None
    };
    ScratchConfig {
        hidden: shape.hidden(),
        intermediate: shape.intermediate_per_rank(n_ranks),
        q_width: shape.q_width_per_rank(n_ranks),
        kv_width,
        vocab: shape.vocab(),
        max_seq_len: shape.max_seq_len(),
        num_layers: shape.num_layers(),
        max_experts: moe.map_or(0, |m| m.num_experts),
        max_experts_per_tok: moe.map_or(0, |m| m.experts_per_tok),
        gdn: shape.gdn_per_rank(n_ranks),
        per_layer_kv_widths: Some(per_layer_kv),
        attn_q_gated: shape.attn_q_gated(),
        shared_intermediate: moe.map_or(0, |m| m.shared_intermediate_per_rank),
        max_prefill_tokens: prefill_ubatch,
        max_slots,
        per_layer_embd: shape.per_layer_embd(),
        paged_kv,
    }
}

#[derive(Clone, Copy)]
pub struct KvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub kv_width: usize,
}

/// Geometry for a paged KV cache.
///
/// Today's `KvCache` allocates `[max_slots, max_seq_len, kv_width]`
/// F16 contiguously — every slot reserves worst-case context, so VRAM
/// caps `max_slots` at ~4 on MI50 for Qwen3.6-27B-Q4_0 at ctx 32k.
/// `PagedKvCache` replaces the per-slot slab with a shared
/// `[n_pages, page_size, kv_width]` pool plus a per-slot block table
/// of page indices. A slot only consumes pages for its actually-
/// written tokens, so a sparse N can share the same VRAM as a dense 4.
///
/// `page_size = 16` matches vLLM's default — small enough to keep
/// per-token waste bounded at the tail, large enough that the block-
/// table-indirected attention kernel hits L2 for the table read.
/// 16 token pages on Qwen3.6 head_dim=128 GQA-4 = 2 KiB / page,
/// fitting comfortably in a single block's L1 tile.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvCacheConfig {
    /// Tokens per page. Recommended 16; powers of two for cheap
    /// `t / page_size = t >> log2(page_size)` in the kernel.
    pub page_size: usize,
    /// Total number of pages in the per-layer pool. Sized from
    /// VRAM budget at construction, NOT from `max_slots × max_seq_len`.
    pub n_pages: usize,
    /// Maximum pages a single slot can hold. Bounded by `ceil(max_seq_len / page_size)`.
    pub max_pages_per_slot: usize,
}

/// Paged KV cache layout. Sibling of the contiguous [`KvCache`].
///
/// `k_pool` and `v_pool` are the shared `[n_pages, page_size, kv_width]`
/// F16 page pools. `block_tables` is `[max_slots, max_pages_per_slot]`
/// `u32` device memory: row `s` of length `<= max_pages_per_slot` lists
/// the page indices currently held by slot `s`. `block_table_lens`
/// holds the active page count per slot (current `ceil((position + 1)
/// / page_size)`).
///
/// Append: write the slot's next token to
/// `block_tables[s][position / page_size] * page_size + position % page_size`,
/// allocating a new page when crossing a page boundary.
///
/// Read (attention): for each token `t` in `[0, n_kv)`, the kernel
/// fetches `page_idx = block_tables[s][t / page_size]` then loads
/// from `k_pool + (page_idx * page_size + t % page_size) * kv_width`.
#[derive(Clone, Copy)]
pub struct PagedKvCache {
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_tables: DevicePtr,
    pub block_table_lens: DevicePtr,
    pub kv_width: usize,
    pub page_size: usize,
    pub n_pages: usize,
    pub max_slots: usize,
    pub max_pages_per_slot: usize,
}

impl PagedKvCacheConfig {
    /// Pick `n_pages` from a per-layer VRAM budget. Reserves bytes for
    /// `max_slots` block-table rows plus the block-table-length array,
    /// then divides the remaining budget by `2 * page_size * kv_width
    /// * 2` (K and V pools, F16). The result is clamped to at least
    /// `max_slots * max_pages_per_slot` so every slot can fill its
    /// table even when the budget is just enough; a larger budget
    /// leaves spare pages for cross-slot sharing (E4 prefix cache).
    pub fn from_vram_budget(
        per_layer_budget_bytes: usize,
        page_size: usize,
        kv_width: usize,
        max_slots: usize,
        max_pages_per_slot: usize,
    ) -> Self {
        assert!(
            page_size > 0 && page_size.is_power_of_two(),
            "PagedKvCacheConfig::from_vram_budget: page_size {page_size} must be a positive power of two"
        );
        let table_bytes = max_slots * max_pages_per_slot * std::mem::size_of::<u32>();
        let lens_bytes = max_slots * std::mem::size_of::<u32>();
        let overhead = table_bytes + lens_bytes;
        let usable = per_layer_budget_bytes.saturating_sub(overhead);
        let bytes_per_page_pair = 2 * page_size * kv_width * std::mem::size_of::<u16>();
        let from_budget = if bytes_per_page_pair == 0 {
            0
        } else {
            usable / bytes_per_page_pair
        };
        let min_pages = max_slots * max_pages_per_slot;
        let n_pages = from_budget.max(min_pages);
        Self {
            page_size,
            n_pages,
            max_pages_per_slot,
        }
    }
}

impl PagedKvCache {
    /// Byte size of the per-layer page pool for K (= for V, doubled
    /// for the per-layer pair). Excludes the block-table allocation
    /// (which is per-pool and small: `max_slots * max_pages_per_slot
    /// * 4` bytes).
    pub fn pool_bytes(&self) -> usize {
        self.n_pages * self.page_size * self.kv_width * std::mem::size_of::<u16>()
    }

    /// Total per-layer allocation including K + V pools + block table
    /// + block-table-lens. Used to size the page pool from a VRAM budget.
    pub fn per_layer_bytes(&self) -> usize {
        2 * self.pool_bytes()
            + self.max_slots * self.max_pages_per_slot * std::mem::size_of::<u32>()
            + self.max_slots * std::mem::size_of::<u32>()
    }

    /// Allocate K + V page pools + block table + block-table-lens for a
    /// single layer on `device`. The block table is zero-initialised on
    /// the host buffer so unbacked entries decode as page 0 (a tombstone);
    /// `block_table_lens` is zero-initialised to mean "no pages yet
    /// assigned". Caller is responsible for tracking the returned
    /// allocations and freeing them on dispose.
    pub fn alloc(
        device: &HipDevice,
        cfg: PagedKvCacheConfig,
        kv_width: usize,
        max_slots: usize,
        allocs: &mut Vec<(DevicePtr, usize)>,
    ) -> Result<Self> {
        let f16_bytes = std::mem::size_of::<u16>();
        let pool_bytes = cfg.n_pages * cfg.page_size * kv_width * f16_bytes;
        let table_bytes = max_slots * cfg.max_pages_per_slot * std::mem::size_of::<u32>();
        let lens_bytes = max_slots * std::mem::size_of::<u32>();
        let k_pool = device.alloc(pool_bytes).context("PagedKvCache::alloc k_pool")?;
        allocs.push((k_pool, pool_bytes));
        let v_pool = device.alloc(pool_bytes).context("PagedKvCache::alloc v_pool")?;
        allocs.push((v_pool, pool_bytes));
        let block_tables = device
            .alloc(table_bytes)
            .context("PagedKvCache::alloc block_tables")?;
        allocs.push((block_tables, table_bytes));
        let block_table_lens = device
            .alloc(lens_bytes)
            .context("PagedKvCache::alloc block_table_lens")?;
        allocs.push((block_table_lens, lens_bytes));
        // Zero the block table + lens. Pool stays uninit — the append
        // kernel writes every element of each newly-acquired page
        // before the attention kernel reads it.
        let zero = vec![0u8; table_bytes.max(lens_bytes)];
        let stream = device.default_stream();
        // SAFETY: block_tables owns table_bytes; block_table_lens owns
        // lens_bytes; zero has at least max(table_bytes, lens_bytes).
        unsafe {
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    block_tables,
                    DevicePtr(zero.as_ptr() as usize),
                    table_bytes,
                )
                .context("PagedKvCache::alloc zero block_tables")?;
            device
                .memcpy_async(
                    stream,
                    CopyDirection::HostToDevice,
                    block_table_lens,
                    DevicePtr(zero.as_ptr() as usize),
                    lens_bytes,
                )
                .context("PagedKvCache::alloc zero block_table_lens")?;
        }
        flambeau_core::Stream::synchronize(stream).context("PagedKvCache::alloc sync zero")?;
        Ok(Self {
            k_pool,
            v_pool,
            block_tables,
            block_table_lens,
            kv_width,
            page_size: cfg.page_size,
            n_pages: cfg.n_pages,
            max_slots,
            max_pages_per_slot: cfg.max_pages_per_slot,
        })
    }
}

/// Host-side free-list allocator over the `n_pages` pages of a single
/// layer's [`PagedKvCache`]. Each [`PagePool`] owns its layer's free
/// list and tracks which pages each slot currently holds so a slot
/// release (request finish) returns all its pages atomically.
///
/// Per-step acquisition fires when a slot crosses a page boundary
/// (the host computes `position % page_size == 0`). The acquired page
/// index is then written into the slot's block-table row before the
/// next `kv_append_f16_paged_slots` launch reads it.
///
/// The pool is intentionally generic over the layer's `PagedKvCache`
/// geometry — `acquire_for` returns `None` once the free list runs
/// dry. Callers (E3c scheduler hook) handle eviction.
#[derive(Debug)]
pub struct PagePool {
    /// Total pages in this layer's pool. Matches the linked
    /// `PagedKvCache.n_pages`. Stored for assertion sanity-checks.
    pub n_pages: usize,
    /// Maximum pages a single slot can ever hold. Matches
    /// `max_pages_per_slot` on the linked `PagedKvCache`.
    pub max_pages_per_slot: usize,
    free: std::collections::VecDeque<u32>,
    per_slot_held: Vec<Vec<u32>>,
}

impl PagePool {
    /// Build a pool with all `n_pages` pages free. `per_slot_held` is
    /// pre-sized to `max_slots` empty vectors so `acquire_for(slot)`
    /// never reallocates the outer vec.
    pub fn new(n_pages: usize, max_slots: usize, max_pages_per_slot: usize) -> Self {
        let free: std::collections::VecDeque<u32> = (0..n_pages as u32).collect();
        let per_slot_held = (0..max_slots).map(|_| Vec::new()).collect();
        Self {
            n_pages,
            max_pages_per_slot,
            free,
            per_slot_held,
        }
    }

    /// Number of pages currently free. Used by E3c to decide whether
    /// to evict before serving the next prefill or to admit a new
    /// request.
    pub fn n_free(&self) -> usize {
        self.free.len()
    }

    /// Page indices currently held by `slot`. Caller must guard
    /// `slot < per_slot_held.len()`.
    pub fn pages_held_by(&self, slot: usize) -> &[u32] {
        &self.per_slot_held[slot]
    }

    /// Acquire one free page for `slot`. Returns `None` when the free
    /// list is empty — caller (E3c scheduler) decides between
    /// blocking, evicting, or rejecting the request.
    pub fn acquire_for(&mut self, slot: usize) -> Option<u32> {
        assert!(slot < self.per_slot_held.len(), "PagePool::acquire_for: slot {slot} out of bounds");
        if self.per_slot_held[slot].len() >= self.max_pages_per_slot {
            return None;
        }
        let page = self.free.pop_front()?;
        self.per_slot_held[slot].push(page);
        Some(page)
    }

    /// Release every page held by `slot` back to the free list.
    /// Called at request finish (slot lifecycle in
    /// [`crate::routes::ServerState::release_slot`] sibling).
    pub fn release_slot(&mut self, slot: usize) {
        assert!(slot < self.per_slot_held.len(), "PagePool::release_slot: slot {slot} out of bounds");
        let held = std::mem::take(&mut self.per_slot_held[slot]);
        for page in held {
            self.free.push_back(page);
        }
    }

    /// Ensure the slot owns at least `target_page_count` pages —
    /// acquiring fresh ones from the free list until it does. Returns
    /// the indices of any newly-acquired pages in slot-internal page-
    /// index order (i.e. `[old_count, target_page_count)`). Used by
    /// prefill paths that span multiple page boundaries in a single
    /// kernel call and need to pre-populate the slot's block-table
    /// row before launching.
    ///
    /// Returns `Err` (with the count of pages successfully acquired
    /// before exhaustion) when the free list runs dry OR when the
    /// slot would exceed `max_pages_per_slot`. The pages it managed
    /// to acquire are RETAINED — caller can release the slot to
    /// recycle them.
    pub fn ensure_pages_up_to(
        &mut self,
        slot: usize,
        target_page_count: usize,
    ) -> std::result::Result<Vec<(usize, u32)>, usize> {
        assert!(slot < self.per_slot_held.len(), "PagePool::ensure_pages_up_to: slot {slot} out of bounds");
        if target_page_count > self.max_pages_per_slot {
            return Err(self.per_slot_held[slot].len());
        }
        let already = self.per_slot_held[slot].len();
        if already >= target_page_count {
            return Ok(Vec::new());
        }
        let mut new_pages: Vec<(usize, u32)> = Vec::with_capacity(target_page_count - already);
        for page_idx_in_slot in already..target_page_count {
            match self.free.pop_front() {
                Some(p) => {
                    self.per_slot_held[slot].push(p);
                    new_pages.push((page_idx_in_slot, p));
                }
                None => return Err(self.per_slot_held[slot].len()),
            }
        }
        Ok(new_pages)
    }
}

#[cfg(test)]
mod page_pool_tests {
    use super::*;

    #[test]
    fn new_pool_has_all_pages_free() {
        let pool = PagePool::new(8, 4, 4);
        assert_eq!(pool.n_free(), 8);
        for slot in 0..4 {
            assert!(pool.pages_held_by(slot).is_empty());
        }
    }

    #[test]
    fn acquire_returns_distinct_pages() {
        let mut pool = PagePool::new(4, 2, 4);
        let p0 = pool.acquire_for(0).unwrap();
        let p1 = pool.acquire_for(0).unwrap();
        let p2 = pool.acquire_for(1).unwrap();
        assert_ne!(p0, p1);
        assert_ne!(p0, p2);
        assert_ne!(p1, p2);
        assert_eq!(pool.n_free(), 1);
        assert_eq!(pool.pages_held_by(0).len(), 2);
        assert_eq!(pool.pages_held_by(1).len(), 1);
    }

    #[test]
    fn acquire_returns_none_when_pool_empty() {
        let mut pool = PagePool::new(2, 2, 4);
        assert!(pool.acquire_for(0).is_some());
        assert!(pool.acquire_for(0).is_some());
        assert!(pool.acquire_for(1).is_none());
    }

    #[test]
    fn acquire_returns_none_when_slot_full() {
        let mut pool = PagePool::new(8, 2, 2);
        assert!(pool.acquire_for(0).is_some());
        assert!(pool.acquire_for(0).is_some());
        // Slot 0 has hit its per-slot cap even though 6 pages are
        // still free — caller must release-slot or reject.
        assert!(pool.acquire_for(0).is_none());
        assert_eq!(pool.n_free(), 6);
    }

    #[test]
    fn release_slot_returns_pages_to_free_list() {
        let mut pool = PagePool::new(4, 2, 4);
        let p0 = pool.acquire_for(0).unwrap();
        let p1 = pool.acquire_for(0).unwrap();
        assert_eq!(pool.n_free(), 2);
        pool.release_slot(0);
        assert_eq!(pool.n_free(), 4);
        assert!(pool.pages_held_by(0).is_empty());
        // Released pages are reusable in any order.
        let p2 = pool.acquire_for(1).unwrap();
        assert!(p2 == p0 || p2 == p1 || p2 == 2 || p2 == 3);
    }

    #[test]
    fn ensure_pages_up_to_acquires_missing() {
        let mut pool = PagePool::new(8, 2, 4);
        // Slot starts with 0 pages. Request 3 → acquires 3.
        let new = pool.ensure_pages_up_to(0, 3).unwrap();
        assert_eq!(new.len(), 3);
        assert_eq!(new[0].0, 0);
        assert_eq!(new[1].0, 1);
        assert_eq!(new[2].0, 2);
        assert_eq!(pool.pages_held_by(0).len(), 3);
        // Request 3 again → no-op.
        let new = pool.ensure_pages_up_to(0, 3).unwrap();
        assert!(new.is_empty());
        // Request 4 → acquires 1 more (page_idx 3).
        let new = pool.ensure_pages_up_to(0, 4).unwrap();
        assert_eq!(new, vec![(3, new[0].1)]);
    }

    #[test]
    fn ensure_pages_up_to_returns_err_when_exhausted() {
        let mut pool = PagePool::new(2, 2, 4);
        // Slot 0 takes all 2 pages.
        assert!(pool.ensure_pages_up_to(0, 2).is_ok());
        // Slot 1 wants 1 page but pool is empty.
        let err = pool.ensure_pages_up_to(1, 1).unwrap_err();
        assert_eq!(err, 0); // acquired 0 before exhaustion
    }

    #[test]
    fn ensure_pages_up_to_returns_err_when_over_cap() {
        let mut pool = PagePool::new(8, 1, 2);
        // Cap is 2. Asking for 3 → Err with already-acquired count.
        let err = pool.ensure_pages_up_to(0, 3).unwrap_err();
        assert_eq!(err, 0);
        assert!(pool.pages_held_by(0).is_empty());
    }

    #[test]
    fn from_vram_budget_picks_max_of_budget_and_min() {
        // Tight budget: only enough for the min_pages = max_slots *
        // max_pages_per_slot. Allocator must still return at least
        // that minimum.
        let cfg = PagedKvCacheConfig::from_vram_budget(0, 16, 64, 4, 8);
        assert_eq!(cfg.n_pages, 4 * 8); // min_pages floor
        // Generous budget: 1 MiB per layer should comfortably exceed
        // the floor.
        let cfg = PagedKvCacheConfig::from_vram_budget(1 << 20, 16, 64, 4, 8);
        assert!(cfg.n_pages > 4 * 8);
        assert_eq!(cfg.page_size, 16);
        assert_eq!(cfg.max_pages_per_slot, 8);
    }
}

/// Caller must invoke `dispose(device)` before drop to release HBM.
pub struct ScratchPool {
    pub config: ScratchConfig,

    pub resid_a: DevicePtr,
    pub resid_b: DevicePtr,
    pub norm: DevicePtr,
    pub delta: DevicePtr,

    pub norm_q8_1: DevicePtr,
    pub norm_q8_1_mmq: DevicePtr,
    pub q_f16: DevicePtr,
    pub k_f16: DevicePtr,
    pub v_f16: DevicePtr,
    /// `[2 * q_width]` F16 — fused `[Q | gate]` projection output for
    /// gated full-attention arches. `DevicePtr::NULL` otherwise.
    pub q_fused_f16: DevicePtr,
    /// `[q_width]` F16 — per-head sigmoid gate for gated full-attn.
    /// `DevicePtr::NULL` otherwise.
    pub gate_f16: DevicePtr,
    pub attn_out_f16: DevicePtr,
    pub attn_out_q8_1: DevicePtr,
    pub attn_out_q8_1_mmq: DevicePtr,
    pub attn_proj_f32: DevicePtr,

    /// `[n_heads_q * MAX_SPLITK_CHUNKS]` F32 — split-K online-softmax
    /// per-chunk running max. NULL until `q_width > 0`.
    pub splitk_partials_m: DevicePtr,
    /// `[n_heads_q * MAX_SPLITK_CHUNKS]` F32 — split-K per-chunk
    /// running denom.
    pub splitk_partials_s: DevicePtr,
    /// `[n_heads_q * MAX_SPLITK_CHUNKS * head_dim]` F32 — split-K
    /// per-chunk numerator outputs. Sized as `q_width * MAX_SPLITK_CHUNKS`
    /// (= n_heads_q * head_dim * MAX_SPLITK_CHUNKS).
    pub splitk_partials_o: DevicePtr,

    pub gate_f32: DevicePtr,
    pub up_f32: DevicePtr,
    pub gated_f16: DevicePtr,
    pub gated_q8_1: DevicePtr,
    pub gated_q8_1_mmq: DevicePtr,
    pub down_f32: DevicePtr,

    pub logits_f32_dev: DevicePtr,
    pub position_i32: DevicePtr,

    /// `[max_experts]` F32. NULL when `config.max_experts == 0`.
    pub router_logits_f32: DevicePtr,
    /// `[hidden]` F16 MoE per-expert accumulator. NULL when `max_experts == 0`.
    pub moe_accum_f16: DevicePtr,
    /// `[hidden]` F32 — F32 cast of x_norm for the shared expert's
    /// per-token gate scale step. NULL when no shared expert.
    pub shared_x_norm_f32: DevicePtr,

    /// `[max_slots]` U64 — per-slot K-cache base pointers, filled
    /// host-side per forward call and consumed by
    /// `attn_decode_f16_batched`. NULL when `max_slots == 1`.
    pub attn_slot_k_dst_ptrs: DevicePtr,
    pub attn_slot_v_dst_ptrs: DevicePtr,
    /// `[max_slots]` I32 — per-slot write position (= positions[i]).
    pub attn_slot_write_pos: DevicePtr,
    /// `[max_slots]` I32 — per-slot KV length (= positions[i] + 1).
    pub attn_slot_n_kv: DevicePtr,

    /// Shared MoE prefill scratch (`flambeau_model_ops` owned, sized
    /// `max_prefill_tokens × top_k × intermediate`). `None` when no MoE
    /// or `max_prefill_tokens <= 1`.
    pub moe_prefill_scratch: Option<flambeau_model_ops::OwnedMoeExpertsPrefillScratch>,

    /// `[max_experts_per_tok]` I32 — top-k expert indices. NULL when no MoE.
    pub moe_expert_ids: DevicePtr,
    /// `[max_experts_per_tok]` F32 — top-k normalised expert weights. NULL when no MoE.
    pub moe_expert_weights: DevicePtr,
    /// `[max_experts_per_tok * intermediate]` F32 — indexed gate output.
    pub moe_gate_out_f32: DevicePtr,
    pub moe_up_out_f32: DevicePtr,
    pub moe_activated_f16: DevicePtr,
    pub moe_activated_q8_1: DevicePtr,
    /// `[max_experts_per_tok * hidden]` F32 — indexed down output before combine.
    pub moe_down_f32: DevicePtr,
    pub moe_down_f16: DevicePtr,

    /// Per-layer side-channel embedding scratch (E2B / E4B).
    /// NULL when `config.per_layer_embd == 0`.
    pub ple_gate_out_f32: DevicePtr,
    pub ple_activated_f32: DevicePtr,
    pub ple_activated_f16: DevicePtr,
    pub ple_proj_out_f32: DevicePtr,
    pub ple_proj_out_f16: DevicePtr,
    pub ple_normed_f16: DevicePtr,

    pub kv_caches: Vec<KvCache>,

    /// Per-layer paged KV cache, when `config.paged_kv` is `Some`.
    /// Coexists with `kv_caches` until E3c rewires `standard_attn`
    /// to dispatch on the paged path; today's runtime still reads
    /// and writes via `kv_caches` regardless of this Option.
    pub paged_kv_caches: Option<Vec<PagedKvCache>>,
    /// Per-layer host-side free-list allocator. Empty when
    /// `config.paged_kv` is `None`. Each `PagePool` owns the
    /// corresponding `paged_kv_caches[li].n_pages` pages and tracks
    /// per-slot ownership for the page-acquire / release lifecycle.
    pub page_pools: Vec<PagePool>,

    /// Per-owned-layer recurrent state + conv history. Empty when
    /// `config.gdn` is None.
    pub gdn_state: Vec<GdnLayerState>,
    /// Shared GDN per-decode scratch wrapping blocks's owned scratch.
    /// `None` when `config.gdn` is None.
    pub gdn_decode_scratch: Option<flambeau_model_ops::OwnedDeltaNetLayerDecodeScratch>,
    /// Shared GDN batched-decode scratch sized for `max_slots`. `None`
    /// when GDN is off or `max_slots <= 1`. Used by the GDN composite
    /// when `slot_ids.len() > 1` (multi-slot decode path).
    pub gdn_decode_batched_scratch: Option<flambeau_model_ops::OwnedDeltaNetLayerDecodeBatchedScratch>,
    /// `[max_slots] u64` device arrays of per-slot GDN state and conv
    /// history base pointers. Filled host-side per forward call and
    /// consumed by `gdn_state_step_alphabeta_f32_s128_batched_slots`
    /// and `gdn_conv_trio_decode_f32_batched_slots`. NULL when GDN is
    /// off or `max_slots <= 1`.
    pub gdn_slot_state_ptrs: DevicePtr,
    pub gdn_slot_history_ptrs: DevicePtr,
    /// Shared GDN prefill scratch sized for `max_prefill_tokens`. Used
    /// by the composite when the call is single-slot, contiguous, and
    /// `n_tokens > 1` (the prefill-shape path). `None` when GDN is off
    /// or `max_prefill_tokens <= 1`.
    pub gdn_prefill_scratch: Option<flambeau_model_ops::OwnedDeltaNetLayerPrefillScratch>,

    pub current_residual_is_a: bool,

    /// When true, `pool.norm` holds an already-rmsnormed F16 buffer
    /// written by the previous composite's BAR1 fused
    /// `residual_rmsnorm_tp2` call. The next composite must consume it
    /// (skip its initial rmsnorm) and clear the flag, OR clear the
    /// flag and ignore the stale value if its rmsnorm uses a different
    /// weight than the one folded in upstream.
    pub input_pre_normed: bool,

    /// When true, the previous composite fused its post-norm+residual
    /// add into the next residual slot directly (gemma4 paths with
    /// post_attn_norm / post_ffn_norm). The matching `residual_add`
    /// must skip its own advance + add and return the incoming `b`
    /// (which IS the new residual) as-is. Set by `standard_attn_local`
    /// / `dense_ffn_local` when they take the fused path; cleared by
    /// `residual_add_local`.
    pub fused_residual_already_done: bool,

    allocs: Vec<(DevicePtr, usize)>,
}

#[derive(Clone, Copy)]
pub struct GdnLayerState {
    /// `[num_v_heads, head_k_dim, head_v_dim]` F32 recurrent state.
    pub state: DevicePtr,
    /// `[conv_kernel - 1, conv_channels]` F32 conv1d history.
    pub conv_history: DevicePtr,
}

impl ScratchPool {
    pub fn new(device: &HipDevice, config: ScratchConfig) -> Result<Self> {
        let mut allocs: Vec<(DevicePtr, usize)> = Vec::new();
        let mut alloc_bytes = |bytes: usize| -> Result<DevicePtr> {
            let p = device.alloc(bytes).context("alloc")?;
            allocs.push((p, bytes));
            Ok(p)
        };

        let f16 = 2;
        let f32 = 4;
        let i32_b = 4;
        let q8_1 = |n: usize| n.div_ceil(32) * 36;
        // MMQ block: 144 B per 128 elements, row-major over (ncols/128, total_b).
        let q8_1_mmq = |cols: usize, rows: usize| cols.div_ceil(128) * rows * 144;

        let h = config.hidden;
        let m = config.intermediate;
        let qw = config.q_width;
        let kvw = config.kv_width;
        let n = config.max_prefill_tokens.max(1);

        let resid_a = alloc_bytes(n * h * f16)?;
        let resid_b = alloc_bytes(n * h * f16)?;
        let norm = alloc_bytes(n * h * f16)?;
        let delta = alloc_bytes(n * h * f16)?;

        let norm_q8_1 = alloc_bytes(q8_1(n * h))?;
        let norm_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(h, n))?
        } else {
            DevicePtr::NULL
        };
        let q_f16 = alloc_bytes(n * qw * f16)?;
        let k_f16 = alloc_bytes(n * kvw * f16)?;
        let v_f16 = alloc_bytes(n * kvw * f16)?;
        let (q_fused_f16, gate_f16) = if config.attn_q_gated {
            (alloc_bytes(n * 2 * qw * f16)?, alloc_bytes(n * qw * f16)?)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let attn_out_f16 = alloc_bytes(n * qw * f16)?;
        let attn_out_q8_1 = alloc_bytes(q8_1(n * qw))?;
        let attn_out_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(qw, n))?
        } else {
            DevicePtr::NULL
        };
        let q_or_fused = if config.attn_q_gated { 2 * qw } else { qw };
        let attn_proj_f32 = alloc_bytes(n * q_or_fused.max(kvw).max(h) * f32)?;

        // Split-K decode-attn partials. Over-allocate `partials_m/s`
        // at q_width elems (= n_heads_q * head_dim) rather than the
        // exact n_heads_q — saves storing head_dim in ScratchConfig
        // and the waste is sub-MB.
        let (splitk_partials_m, splitk_partials_s, splitk_partials_o) = if qw > 0 {
            let m = alloc_bytes(qw * MAX_SPLITK_CHUNKS * f32)?;
            let s = alloc_bytes(qw * MAX_SPLITK_CHUNKS * f32)?;
            let o = alloc_bytes(qw * MAX_SPLITK_CHUNKS * f32)?;
            (m, s, o)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL)
        };

        // FFN gate/up/activated buffers are shared between the dense
        // FFN path (intermediate=ffn_inter), the qwen-shared-expert
        // path (intermediate=shared_expert_inter, usually =
        // expert_inter), and the gemma4 MoE shared MLP path
        // (shared_intermediate > routed expert_intermediate). Size for
        // the max so all callers fit.
        let m_buf = m.max(config.shared_intermediate);
        let gate_f32 = alloc_bytes(n * m_buf * f32)?;
        let up_f32 = alloc_bytes(n * m_buf * f32)?;
        let gated_f16 = alloc_bytes(n * m_buf * f16)?;
        let gated_q8_1 = alloc_bytes(q8_1(n * m_buf))?;
        let gated_q8_1_mmq = if n > 1 {
            alloc_bytes(q8_1_mmq(m_buf, n))?
        } else {
            DevicePtr::NULL
        };
        let down_f32 = alloc_bytes(n * h * f32)?;

        let logits_f32_dev = alloc_bytes(config.max_slots.max(1) * config.vocab * f32)?;
        let position_i32 = alloc_bytes(n * i32_b)?;

        let (router_logits_f32, moe_accum_f16) = if config.max_experts > 0 {
            let r = alloc_bytes(n * config.max_experts * f32)?;
            let a = alloc_bytes(n * h * f16)?;
            (r, a)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let shared_x_norm_f32 = if config.shared_intermediate > 0 {
            alloc_bytes(n * h * f32)?
        } else {
            DevicePtr::NULL
        };

        // Per-layer side-channel embedding scratch (gemma 4n / E2B / E4B).
        // Six small per-token buffers; pe is typically 256 so total is
        // ~10 KB. Skip alloc when the arch has no per-layer side channel.
        let pe = config.per_layer_embd;
        let (
            ple_gate_out_f32,
            ple_activated_f32,
            ple_activated_f16,
            ple_proj_out_f32,
            ple_proj_out_f16,
            ple_normed_f16,
        ) = if pe > 0 {
            (
                alloc_bytes(n * pe * f32)?,
                alloc_bytes(n * pe * f32)?,
                alloc_bytes(n * pe * f16)?,
                alloc_bytes(n * h * f32)?,
                alloc_bytes(n * h * f16)?,
                alloc_bytes(n * h * f16)?,
            )
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };

        // Batched-decode attention scratch. Per-slot pointer/scalar
        // arrays consumed by `attn_decode_f16_batched` and
        // `kv_append_f16_batched_slots`. Allocated only when N > 1
        // (single-slot decode goes through the unbatched kernel).
        let n_slots = config.max_slots.max(1);
        let (attn_slot_k_dst_ptrs, attn_slot_v_dst_ptrs, attn_slot_write_pos, attn_slot_n_kv) =
            if n_slots > 1 {
                let kd = alloc_bytes(n_slots * 8)?;
                let vd = alloc_bytes(n_slots * 8)?;
                let wp = alloc_bytes(n_slots * i32_b)?;
                let nk = alloc_bytes(n_slots * i32_b)?;
                (kd, vd, wp, nk)
            } else {
                (
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                )
            };

        // Per-slot pointer arrays for GDN batched-slot kernels. Allocated
        // only when GDN is enabled AND multi-slot. Filled host-side per
        // forward call by `gdn_layer_local`; consumed by
        // `gdn_state_step_alphabeta_f32_s128_batched_slots` and
        // `gdn_conv_trio_decode_f32_batched_slots`.
        let (gdn_slot_state_ptrs, gdn_slot_history_ptrs) =
            if config.gdn.is_some() && n_slots > 1 {
                let s = alloc_bytes(n_slots * 8)?;
                let h = alloc_bytes(n_slots * 8)?;
                (s, h)
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };

        let topk = config.max_experts_per_tok;
        let (
            moe_expert_ids,
            moe_expert_weights,
            moe_gate_out_f32,
            moe_up_out_f32,
            moe_activated_f16,
            moe_activated_q8_1,
            moe_down_f32,
            moe_down_f16,
        ) = if topk > 0 && config.max_experts > 0 {
            let ids = alloc_bytes(topk * i32_b)?;
            let weights = alloc_bytes(topk * f32)?;
            let gate = alloc_bytes(topk * m * f32)?;
            let up = alloc_bytes(topk * m * f32)?;
            let act_f16 = alloc_bytes(topk * m * f16)?;
            let act_q8 = alloc_bytes(q8_1(topk * m))?;
            let dn_f32 = alloc_bytes(topk * h * f32)?;
            let dn_f16 = alloc_bytes(topk * h * f16)?;
            (ids, weights, gate, up, act_f16, act_q8, dn_f32, dn_f16)
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };

        if let Some(per) = config.per_layer_kv_widths.as_ref() {
            if per.len() != config.num_layers {
                anyhow::bail!(
                    "per_layer_kv_widths.len() {} != num_layers {}",
                    per.len(),
                    config.num_layers
                );
            }
            for (li, &w) in per.iter().enumerate() {
                if w > kvw {
                    anyhow::bail!(
                        "per_layer_kv_widths[{li}] = {w} > kv_width {kvw} \
                         (kv_width must be >= max per-layer kv_width — \
                         it sizes the shared K/V scratch)",
                    );
                }
            }
        }
        let n_slots = config.max_slots.max(1);
        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for li in 0..config.num_layers {
            let slot_kvw = config
                .per_layer_kv_widths
                .as_ref()
                .map(|p| p[li])
                .unwrap_or(kvw);
            let k = alloc_bytes(n_slots * config.max_seq_len * slot_kvw * f16)?;
            let v = alloc_bytes(n_slots * config.max_seq_len * slot_kvw * f16)?;
            kv_caches.push(KvCache {
                k,
                v,
                kv_width: slot_kvw,
            });
        }

        // Paged-KV allocation runs alongside the contiguous cache when
        // `config.paged_kv` is `Some`. Today's `standard_attn` still
        // reads + writes via `kv_caches`; the paged structures sit
        // ready for E3c's dispatch rewire. Uses a local alloc list
        // (drained into `allocs` after the `alloc_bytes` closure
        // releases its mutable borrow) — same pattern as the GDN +
        // MoE prefill scratch paths below.
        let mut paged_local_allocs: Vec<(DevicePtr, usize)> = Vec::new();
        let (paged_kv_caches, page_pools) = if let Some(paged_cfg) = config.paged_kv.as_ref() {
            let mut caches = Vec::with_capacity(config.num_layers);
            let mut pools: Vec<PagePool> = Vec::with_capacity(config.num_layers);
            for li in 0..config.num_layers {
                let slot_kvw = config
                    .per_layer_kv_widths
                    .as_ref()
                    .map(|p| p[li])
                    .unwrap_or(kvw);
                let cache = PagedKvCache::alloc(
                    device,
                    *paged_cfg,
                    slot_kvw,
                    n_slots,
                    &mut paged_local_allocs,
                )
                .context("PagedKvCache::alloc")?;
                let pool =
                    PagePool::new(cache.n_pages, cache.max_slots, cache.max_pages_per_slot);
                caches.push(cache);
                pools.push(pool);
            }
            (Some(caches), pools)
        } else {
            (None, Vec::new())
        };

        let (gdn_state, gdn_decode_scratch, gdn_decode_batched_scratch, gdn_prefill_scratch) = if let Some(g) = config.gdn {
            let mut state_vec = Vec::with_capacity(config.num_layers);
            let state_bytes = n_slots * g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
            let hist_bytes = n_slots * (g.conv_kernel - 1) * g.conv_channels * f32;
            // Recurrent state + conv history must start at zero —
            // the step kernel reads them every call, including
            // position=0. `device.alloc` is uninitialised.
            let zero_buf = vec![0u8; state_bytes.max(hist_bytes)];
            let stream = device.default_stream();
            for _ in 0..config.num_layers {
                let state = alloc_bytes(state_bytes)?;
                let conv_history = alloc_bytes(hist_bytes)?;
                // SAFETY: state owns state_bytes, conv_history owns
                // hist_bytes, zero_buf has >= max(state_bytes, hist_bytes).
                unsafe {
                    device
                        .memcpy_async(
                            stream,
                            CopyDirection::HostToDevice,
                            state,
                            DevicePtr(zero_buf.as_ptr() as usize),
                            state_bytes,
                        )
                        .context("zero gdn state")?;
                    device
                        .memcpy_async(
                            stream,
                            CopyDirection::HostToDevice,
                            conv_history,
                            DevicePtr(zero_buf.as_ptr() as usize),
                            hist_bytes,
                        )
                        .context("zero gdn conv_history")?;
                }
                state_vec.push(GdnLayerState {
                    state,
                    conv_history,
                });
            }
            flambeau_core::Stream::synchronize(stream).context("sync gdn zero")?;
            // Drain the blocks-owned RawAllocTracker into our list
            // so a single dispose walks every alloc.
            let mut tracker = flambeau_model_ops::RawAllocTracker::new();
            let dims = flambeau_model_ops::DeltaNetScratchDims {
                hidden: h,
                d_inner: g.d_inner,
                num_v_heads: g.num_v_heads,
                num_k_heads: g.num_k_heads,
                head_k_dim: g.head_k_dim,
                head_v_dim: g.head_v_dim,
                conv_channels: g.conv_channels,
                conv_kernel: g.conv_kernel,
            };
            let owned = flambeau_model_ops::DeltaNetLayer::alloc_decode_scratch(
                device,
                &mut tracker,
                dims,
            )?;
            allocs.extend(std::mem::take(&mut tracker.allocs));
            let batched_owned = if n_slots > 1 {
                let mut b_tracker = flambeau_model_ops::RawAllocTracker::new();
                let b = flambeau_model_ops::DeltaNetLayer::alloc_decode_batched_scratch(
                    device,
                    &mut b_tracker,
                    dims,
                    n_slots,
                )?;
                allocs.extend(std::mem::take(&mut b_tracker.allocs));
                Some(b)
            } else {
                None
            };
            let prefill_owned = if config.max_prefill_tokens > 1 {
                let mut p_tracker = flambeau_model_ops::RawAllocTracker::new();
                let p = flambeau_model_ops::DeltaNetLayer::alloc_prefill_scratch(
                    device,
                    &mut p_tracker,
                    dims,
                    config.max_prefill_tokens,
                )?;
                allocs.extend(std::mem::take(&mut p_tracker.allocs));
                Some(p)
            } else {
                None
            };
            (state_vec, Some(owned), batched_owned, prefill_owned)
        } else {
            (Vec::new(), None, None, None)
        };

        // MoE prefill scratch — must happen AFTER the `alloc_bytes`
        // closure's last use (allocs is captured-mutably; extending it
        // here is the only safe spot once the closure has been
        // dropped from the borrow checker's perspective).
        let moe_prefill_scratch =
            if topk > 0 && config.max_experts > 0 && config.max_prefill_tokens > 1 {
                let mut p_tracker = flambeau_model_ops::RawAllocTracker::new();
                let dims = flambeau_model_ops::MoeExpertsScratchDims {
                    hidden: h,
                    intermediate: m,
                    n_experts: config.max_experts,
                    top_k: topk,
                };
                let owned = flambeau_model_ops::MoeExperts::alloc_prefill_scratch(
                    device,
                    &mut p_tracker,
                    dims,
                    config.max_prefill_tokens,
                )?;
                allocs.extend(std::mem::take(&mut p_tracker.allocs));
                Some(owned)
            } else {
                None
            };

        // Drain the paged-KV local allocations into the unified
        // `allocs` list so `dispose` walks every allocation in one pass.
        allocs.append(&mut paged_local_allocs);

        Ok(Self {
            config,
            resid_a,
            resid_b,
            norm,
            delta,
            norm_q8_1,
            norm_q8_1_mmq,
            q_f16,
            k_f16,
            v_f16,
            q_fused_f16,
            gate_f16,
            attn_out_f16,
            attn_out_q8_1,
            attn_out_q8_1_mmq,
            attn_proj_f32,
            splitk_partials_m,
            splitk_partials_s,
            splitk_partials_o,
            gate_f32,
            up_f32,
            gated_f16,
            gated_q8_1,
            gated_q8_1_mmq,
            down_f32,
            logits_f32_dev,
            position_i32,
            router_logits_f32,
            moe_accum_f16,
            shared_x_norm_f32,
            attn_slot_k_dst_ptrs,
            attn_slot_v_dst_ptrs,
            attn_slot_write_pos,
            attn_slot_n_kv,
            moe_prefill_scratch,
            moe_expert_ids,
            moe_expert_weights,
            moe_gate_out_f32,
            moe_up_out_f32,
            moe_activated_f16,
            moe_activated_q8_1,
            moe_down_f32,
            moe_down_f16,
            ple_gate_out_f32,
            ple_activated_f32,
            ple_activated_f16,
            ple_proj_out_f32,
            ple_proj_out_f16,
            ple_normed_f16,
            kv_caches,
            paged_kv_caches,
            page_pools,
            gdn_state,
            gdn_decode_scratch,
            gdn_decode_batched_scratch,
            gdn_slot_state_ptrs,
            gdn_slot_history_ptrs,
            gdn_prefill_scratch,
            current_residual_is_a: true,
            input_pre_normed: false,
            fused_residual_already_done: false,
            allocs,
        })
    }

    /// Idempotent.
    pub fn dispose(&mut self, device: &HipDevice) -> Result<()> {
        for (ptr, bytes) in self.allocs.drain(..) {
            // SAFETY: `ptr` came from `device.alloc(bytes)`; caller
            // contract: no further forward calls past dispose.
            unsafe { device.dealloc(ptr, bytes) }.context("dealloc")?;
        }
        Ok(())
    }

    /// Flip residual ping-pong and return the live slot. Called by
    /// `embed` and every `residual_add`.
    pub fn next_residual_slot(&mut self) -> DevicePtr {
        self.current_residual_is_a = !self.current_residual_is_a;
        if self.current_residual_is_a {
            self.resid_a
        } else {
            self.resid_b
        }
    }

    /// Zero per-layer GDN recurrent state + conv-history slabs so the
    /// next request starts fresh. No-op when the arch is non-GDN.
    /// KV-cache positions are caller-supplied (no counter on the
    /// pool), so this is the only stateful slot that needs reset.
    pub fn reset_gdn_state(&self, device: &HipDevice) -> Result<()> {
        let Some(g) = self.config.gdn else {
            return Ok(());
        };
        if self.gdn_state.is_empty() {
            return Ok(());
        }
        let f32 = 4;
        let n_slots = self.config.max_slots.max(1);
        let state_bytes = n_slots * g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
        let hist_bytes = n_slots * (g.conv_kernel - 1) * g.conv_channels * f32;
        let zero_buf = vec![0u8; state_bytes.max(hist_bytes)];
        let stream = device.default_stream();
        for ls in &self.gdn_state {
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ls.state,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        state_bytes,
                    )
                    .context("reset gdn state")?;
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        ls.conv_history,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        hist_bytes,
                    )
                    .context("reset gdn conv_history")?;
            }
        }
        flambeau_core::Stream::synchronize(stream).context("sync gdn reset")?;
        Ok(())
    }

    /// Zero one slot's GDN state + conv-history across every layer.
    /// Used by the server's shared-Session pool to reset a single
    /// conversation slot without touching others. No-op for non-GDN
    /// archs or when the slot index is out of range.
    pub fn reset_gdn_state_slot(&self, slot_id: usize, device: &HipDevice) -> Result<()> {
        let Some(g) = self.config.gdn else {
            return Ok(());
        };
        if self.gdn_state.is_empty() {
            return Ok(());
        }
        let n_slots = self.config.max_slots.max(1);
        if slot_id >= n_slots {
            anyhow::bail!("reset_gdn_state_slot: slot_id {slot_id} >= max_slots {n_slots}");
        }
        let f32 = 4;
        let state_bytes_per_slot = g.num_v_heads * g.head_k_dim * g.head_v_dim * f32;
        let hist_bytes_per_slot = (g.conv_kernel - 1) * g.conv_channels * f32;
        let zero_buf = vec![0u8; state_bytes_per_slot.max(hist_bytes_per_slot)];
        let stream = device.default_stream();
        for ls in &self.gdn_state {
            let state_off = ls.state.offset_bytes(slot_id * state_bytes_per_slot);
            let hist_off = ls.conv_history.offset_bytes(slot_id * hist_bytes_per_slot);
            unsafe {
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        state_off,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        state_bytes_per_slot,
                    )
                    .context("reset gdn state slot")?;
                device
                    .memcpy_async(
                        stream,
                        CopyDirection::HostToDevice,
                        hist_off,
                        DevicePtr(zero_buf.as_ptr() as usize),
                        hist_bytes_per_slot,
                    )
                    .context("reset gdn conv_history slot")?;
            }
        }
        flambeau_core::Stream::synchronize(stream).context("sync gdn reset slot")?;
        Ok(())
    }
}

