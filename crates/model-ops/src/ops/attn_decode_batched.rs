//! Batched-slot flash-attention decode + KV append. Single-launch
//! variants that process N decode slots in one kernel call. Replaces
//! the per-slot loop pattern (`for slot in 0..N { kv_append + attn_decode }`)
//! that v2 standard_attn used to take when slot_ids are non-uniform.
//!
//! The pointer arrays + scalar arrays are caller-owned device buffers
//! (`scratch.attn_slot_k_dst_ptrs` etc. in v2's ScratchPool). They are
//! filled host-side and uploaded once per forward call.

use anyhow::bail;
use flambeau_ops::Ops;

use crate::error::Result;

/// One-shot batched KV append across `n_slots` slots. Each slot writes
/// one new (K, V) row at its own `write_pos` into its own slot KV
/// slab. `k_src`/`v_src` are slot-major `[n_slots, kv_width]` F16
/// buffers (each slot's freshly-projected row). `slot_k_dst_ptrs` /
/// `slot_v_dst_ptrs` hold `[n_slots] u64` pointer-arrays — each
/// pointer is the slot's KV slab base. `slot_write_pos` is a device
/// `[n_slots] i32` of pre-bump tail indices (each slot's position).
pub fn kv_append_f16_batched_slots(
    buffers: flambeau_ops::KvAppendBatchedSlotsBuffers,
    shape: flambeau_ops::KvAppendBatchedSlotsShape,
    ops: &impl Ops,
) -> Result<()> {
    if shape.n_slots == 0 {
        bail!("kv_append_f16_batched_slots: n_slots must be > 0");
    }
    ops.kv_append_f16_batched_slots(buffers, shape)
}

/// Single-launch GQA decode attention over N slots. Each slot owns
/// its own K/V cache base; per-slot KV length comes from the
/// `n_tokens_kv` device array. `q_batched` is `[n_slots, n_heads_q,
/// head_dim]` slot-major F16; `out_batched` matches. `k_cache_ptrs` /
/// `v_cache_ptrs` are `[n_slots] u64` device pointer arrays.
pub fn attn_decode_f16_batched(
    buffers: flambeau_ops::AttnBatchedBuffers,
    shape: flambeau_ops::AttnDecodeBatchedShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &impl Ops,
) -> Result<()> {
    if !matches!(shape.head_dim, 64 | 128 | 256 | 512) {
        bail!(
            "attn_decode_f16_batched: head_dim {} not in {{64, 128, 256, 512}}",
            shape.head_dim
        );
    }
    if shape.n_slots == 0 || shape.n_slots > 32 {
        bail!(
            "attn_decode_f16_batched: n_slots {} out of range [1, 32]",
            shape.n_slots
        );
    }
    if shape.n_heads_q == 0 || shape.n_heads_kv == 0 || shape.n_heads_q % shape.n_heads_kv != 0 {
        bail!(
            "attn_decode_f16_batched: head counts invalid (q={}, kv={})",
            shape.n_heads_q,
            shape.n_heads_kv
        );
    }
    ops.attention_decode_f16_batched(buffers, shape, knobs)
}

/// PagedAttention prefill attention. Same flash-attn-v2 body as
/// `attn_prefill_f16`; per-`t` K/V row resolved via
/// `block_table[t / page_size] * page_size + (t & (page_size - 1))`.
/// `page_size` must be a power of two.
pub fn attn_prefill_f16_paged(
    buffers: flambeau_ops::AttnPagedPrefillBuffers,
    shape: flambeau_ops::AttnPrefillPagedShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &impl Ops,
) -> Result<()> {
    if !matches!(shape.head_dim, 64 | 128 | 256 | 512) {
        bail!(
            "attn_prefill_f16_paged: head_dim {} not in {{64, 128, 256, 512}}",
            shape.head_dim
        );
    }
    if !shape.page_size.is_power_of_two() {
        bail!(
            "attn_prefill_f16_paged: page_size {} must be a power of two",
            shape.page_size
        );
    }
    if shape.n_heads_q == 0 || shape.n_heads_kv == 0 || shape.n_heads_q % shape.n_heads_kv != 0 {
        bail!(
            "attn_prefill_f16_paged: head counts invalid (q={}, kv={})",
            shape.n_heads_q,
            shape.n_heads_kv
        );
    }
    ops.attention_prefill_f16_paged(buffers, shape, knobs)
}

/// PagedAttention prefill K + V append. Writes L K + V rows for a
/// single slot's prefill into the slot's paged KV cache, walking the
/// slot's row of the block table per token.
///
/// `block_table` is a pointer at the slot's row of the global
/// `[max_slots, max_pages_per_slot]` block table. The host must
/// pre-populate it for the position range `[start_pos, start_pos +
/// n_tokens)` before this call — typically by calling
/// `PagePool::acquire_for` for each new page boundary and memcpying
/// the acquired page indices to the device-side block-table region.
/// `page_size` MUST be a power of two.
pub fn kv_append_f16_paged_prefill(
    buffers: flambeau_ops::KvAppendPagedPrefillBuffers,
    shape: flambeau_ops::KvAppendPagedPrefillShape,
    ops: &impl Ops,
) -> Result<()> {
    if shape.n_tokens == 0 {
        return Ok(());
    }
    if !shape.page_size.is_power_of_two() {
        bail!(
            "kv_append_f16_paged_prefill: page_size {} must be a power of two",
            shape.page_size
        );
    }
    ops.kv_append_f16_paged_prefill(buffers, shape)
}

/// PagedAttention sibling of [`kv_append_f16_batched_slots`]. Writes
/// the per-slot K + V row into the page that the slot's block table
/// currently maps to.
///
/// `block_tables` is a `[n_slots, max_pages_per_slot]` `u32` device
/// region. The host populates
/// `block_tables[slot_ids[i] * max_pages_per_slot + write_pos[i] /
/// page_size]` with a valid page index BEFORE calling — the kernel
/// never allocates pages. `page_size` MUST be a power of two.
pub fn kv_append_f16_paged_slots(
    buffers: flambeau_ops::KvAppendPagedSlotsBuffers,
    shape: flambeau_ops::KvAppendPagedSlotsShape,
    ops: &impl Ops,
) -> Result<()> {
    if shape.n_slots == 0 {
        bail!("kv_append_f16_paged_slots: n_slots must be > 0");
    }
    if !shape.page_size.is_power_of_two() {
        bail!(
            "kv_append_f16_paged_slots: page_size {} must be a power of two",
            shape.page_size
        );
    }
    if shape.max_pages_per_slot == 0 {
        bail!("kv_append_f16_paged_slots: max_pages_per_slot must be > 0");
    }
    ops.kv_append_f16_paged_slots(buffers, shape)
}

/// PagedAttention sibling of [`attn_decode_f16_batched`]. Reads K/V
/// per-token rows through the slot's block-table indirection.
/// `page_size` MUST be a power of two.
pub fn attn_decode_f16_paged(
    buffers: flambeau_ops::AttnPagedDecodeBuffers,
    shape: flambeau_ops::AttnDecodePagedShape,
    scale: f32,
    ops: &impl Ops,
) -> Result<()> {
    if !matches!(shape.head_dim, 64 | 128 | 256 | 512) {
        bail!(
            "attn_decode_f16_paged: head_dim {} not in {{64, 128, 256, 512}}",
            shape.head_dim
        );
    }
    if shape.n_slots == 0 || shape.n_slots > 32 {
        bail!(
            "attn_decode_f16_paged: n_slots {} out of range [1, 32]",
            shape.n_slots
        );
    }
    if shape.n_heads_q == 0 || shape.n_heads_kv == 0 || shape.n_heads_q % shape.n_heads_kv != 0 {
        bail!(
            "attn_decode_f16_paged: head counts invalid (q={}, kv={})",
            shape.n_heads_q,
            shape.n_heads_kv
        );
    }
    if !shape.page_size.is_power_of_two() {
        bail!(
            "attn_decode_f16_paged: page_size {} must be a power of two",
            shape.page_size
        );
    }
    if shape.max_pages_per_slot == 0 {
        bail!("attn_decode_f16_paged: max_pages_per_slot must be > 0");
    }
    ops.attention_decode_f16_paged(buffers, shape, scale)
}
