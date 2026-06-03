//! Batched-slot flash-attention decode + KV append. Single-launch
//! variants that process N decode slots in one kernel call. Replaces
//! the per-slot loop pattern (`for slot in 0..N { kv_append + attn_decode }`)
//! that v2 standard_attn used to take when slot_ids are non-uniform.
//!
//! The pointer arrays + scalar arrays are caller-owned device buffers
//! (`scratch.attn_slot_k_dst_ptrs` etc. in v2's ScratchPool). They are
//! filled host-side and uploaded once per forward call.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// One-shot batched KV append across `n_slots` slots. Each slot writes
/// one new (K, V) row at its own `write_pos` into its own slot KV
/// slab. `k_src`/`v_src` are slot-major `[n_slots, kv_width]` F16
/// buffers (each slot's freshly-projected row). `slot_k_dst_ptrs` /
/// `slot_v_dst_ptrs` hold `[n_slots] u64` pointer-arrays — each
/// pointer is the slot's KV slab base. `slot_write_pos` is a device
/// `[n_slots] i32` of pre-bump tail indices (each slot's position).
pub fn kv_append_f16_batched_slots(
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    slot_k_dst_ptrs: flambeau_core::DevicePtr,
    slot_v_dst_ptrs: flambeau_core::DevicePtr,
    slot_write_pos: flambeau_core::DevicePtr,
    n_slots: usize,
    kv_width: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if n_slots == 0 {
        bail!("kv_append_f16_batched_slots: n_slots must be > 0");
    }
    let need = n_slots * kv_width;
    if k_src.n_elems < need || v_src.n_elems < need {
        bail!(
            "kv_append_f16_batched_slots: k_src/v_src must have >= {need} F16 elems \
             (got k={}, v={})",
            k_src.n_elems,
            v_src.n_elems,
        );
    }
    ops.kv_append_f16_batched_slots(
        flambeau_ops::KvAppendBatchedSlotsBuffers {
            k_src: k_src.ptr,
            v_src: v_src.ptr,
            slot_k_dst_ptrs,
            slot_v_dst_ptrs,
            slot_write_pos,
        },
        flambeau_ops::KvAppendBatchedSlotsShape { n_slots, kv_width },
    )
}

/// Single-launch GQA decode attention over N slots. Each slot owns
/// its own K/V cache base; per-slot KV length comes from the
/// `n_tokens_kv` device array. `q_batched` is `[n_slots, n_heads_q,
/// head_dim]` slot-major F16; `out_batched` matches. `k_cache_ptrs` /
/// `v_cache_ptrs` are `[n_slots] u64` device pointer arrays.
pub fn attn_decode_f16_batched(
    q_batched: &Tensor<F16>,
    k_cache_ptrs: flambeau_core::DevicePtr,
    v_cache_ptrs: flambeau_core::DevicePtr,
    out_batched: &mut Tensor<F16>,
    n_tokens_kv: flambeau_core::DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_slots: usize,
    scale: f32,
    window_size: i32,
    ops: &HipOps<'_>,
) -> Result<()> {
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16_batched: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_slots == 0 || n_slots > 32 {
        bail!("attn_decode_f16_batched: n_slots {n_slots} out of range [1, 32]");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || n_heads_q % n_heads_kv != 0 {
        bail!("attn_decode_f16_batched: head counts invalid (q={n_heads_q}, kv={n_heads_kv})");
    }
    let need = n_slots * n_heads_q * head_dim;
    if q_batched.n_elems < need || out_batched.n_elems < need {
        bail!(
            "attn_decode_f16_batched: q/out must have >= {need} F16 elems \
             (got q={}, out={})",
            q_batched.n_elems,
            out_batched.n_elems,
        );
    }
    ops.attention_decode_f16_batched(
        flambeau_ops::AttnBatchedBuffers {
            q_batched: q_batched.ptr,
            k_cache_ptrs,
            v_cache_ptrs,
            out_batched: out_batched.ptr,
            n_tokens_kv_ptrs: n_tokens_kv,
        },
        flambeau_ops::AttnDecodeBatchedShape {
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_slots,
        },
        flambeau_ops::AttnKnobs { scale, window_size },
    )
}

/// PagedAttention prefill attention. Same flash-attn-v2 body as
/// `attn_prefill_f16`; per-`t` K/V row resolved via
/// `block_table[t / page_size] * page_size + (t & (page_size - 1))`.
/// `page_size` must be a power of two.
pub fn attn_prefill_f16_paged(
    q: &Tensor<F16>,
    k_pool: flambeau_core::DevicePtr,
    v_pool: flambeau_core::DevicePtr,
    block_table: flambeau_core::DevicePtr,
    out: &mut Tensor<F16>,
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    page_size: usize,
    scale: f32,
    window_size: i32,
    ops: &HipOps<'_>,
) -> Result<()> {
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_prefill_f16_paged: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if !page_size.is_power_of_two() {
        bail!("attn_prefill_f16_paged: page_size {page_size} must be a power of two");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || n_heads_q % n_heads_kv != 0 {
        bail!("attn_prefill_f16_paged: head counts invalid (q={n_heads_q}, kv={n_heads_kv})");
    }
    let need = n_q_tokens * n_heads_q * head_dim;
    if q.n_elems < need || out.n_elems < need {
        bail!(
            "attn_prefill_f16_paged: q/out must have >= {need} F16 elems \
             (got q={}, out={})",
            q.n_elems,
            out.n_elems,
        );
    }
    ops.attention_prefill_f16_paged(
        flambeau_ops::AttnPagedPrefillBuffers {
            q: q.ptr,
            k_pool,
            v_pool,
            block_table,
            out: out.ptr,
        },
        flambeau_ops::AttnPrefillPagedShape {
            n_q_tokens,
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_k_tokens,
            q_offset,
            page_size,
        },
        flambeau_ops::AttnKnobs { scale, window_size },
    )
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
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_pool: flambeau_core::DevicePtr,
    v_pool: flambeau_core::DevicePtr,
    block_table: flambeau_core::DevicePtr,
    n_tokens: usize,
    kv_width: usize,
    start_pos: usize,
    page_size: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }
    if !page_size.is_power_of_two() {
        bail!("kv_append_f16_paged_prefill: page_size {page_size} must be a power of two");
    }
    let need = n_tokens * kv_width;
    if k_src.n_elems < need || v_src.n_elems < need {
        bail!(
            "kv_append_f16_paged_prefill: k_src/v_src must have >= {need} F16 elems \
             (got k={}, v={})",
            k_src.n_elems,
            v_src.n_elems,
        );
    }
    ops.kv_append_f16_paged_prefill(
        flambeau_ops::KvAppendPagedPrefillBuffers {
            k_src: k_src.ptr,
            v_src: v_src.ptr,
            k_pool,
            v_pool,
            block_table,
        },
        flambeau_ops::KvAppendPagedPrefillShape {
            n_tokens,
            kv_width,
            start_pos,
            page_size,
        },
    )
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
    k_src: &Tensor<F16>,
    v_src: &Tensor<F16>,
    k_pool: flambeau_core::DevicePtr,
    v_pool: flambeau_core::DevicePtr,
    block_tables: flambeau_core::DevicePtr,
    slot_write_pos: flambeau_core::DevicePtr,
    n_slots: usize,
    kv_width: usize,
    page_size: usize,
    max_pages_per_slot: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if n_slots == 0 {
        bail!("kv_append_f16_paged_slots: n_slots must be > 0");
    }
    if !page_size.is_power_of_two() {
        bail!("kv_append_f16_paged_slots: page_size {page_size} must be a power of two");
    }
    if max_pages_per_slot == 0 {
        bail!("kv_append_f16_paged_slots: max_pages_per_slot must be > 0");
    }
    let need = n_slots * kv_width;
    if k_src.n_elems < need || v_src.n_elems < need {
        bail!(
            "kv_append_f16_paged_slots: k_src/v_src must have >= {need} F16 elems \
             (got k={}, v={})",
            k_src.n_elems,
            v_src.n_elems,
        );
    }
    ops.kv_append_f16_paged_slots(
        flambeau_ops::KvAppendPagedSlotsBuffers {
            k_src: k_src.ptr,
            v_src: v_src.ptr,
            k_pool,
            v_pool,
            block_tables,
            slot_write_pos,
        },
        flambeau_ops::KvAppendPagedSlotsShape {
            n_slots,
            kv_width,
            page_size,
            max_pages_per_slot,
        },
    )
}

/// PagedAttention sibling of [`attn_decode_f16_batched`]. Reads K/V
/// per-token rows through the slot's block-table indirection.
/// `page_size` MUST be a power of two.
pub fn attn_decode_f16_paged(
    q_batched: &Tensor<F16>,
    k_pool: flambeau_core::DevicePtr,
    v_pool: flambeau_core::DevicePtr,
    block_tables: flambeau_core::DevicePtr,
    out_batched: &mut Tensor<F16>,
    n_tokens_kv: flambeau_core::DevicePtr,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_slots: usize,
    page_size: usize,
    max_pages_per_slot: usize,
    scale: f32,
    ops: &HipOps<'_>,
) -> Result<()> {
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16_paged: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_slots == 0 || n_slots > 32 {
        bail!("attn_decode_f16_paged: n_slots {n_slots} out of range [1, 32]");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || n_heads_q % n_heads_kv != 0 {
        bail!("attn_decode_f16_paged: head counts invalid (q={n_heads_q}, kv={n_heads_kv})");
    }
    if !page_size.is_power_of_two() {
        bail!("attn_decode_f16_paged: page_size {page_size} must be a power of two");
    }
    if max_pages_per_slot == 0 {
        bail!("attn_decode_f16_paged: max_pages_per_slot must be > 0");
    }
    let need = n_slots * n_heads_q * head_dim;
    if q_batched.n_elems < need || out_batched.n_elems < need {
        bail!(
            "attn_decode_f16_paged: q/out must have >= {need} F16 elems \
             (got q={}, out={})",
            q_batched.n_elems,
            out_batched.n_elems,
        );
    }
    ops.attention_decode_f16_paged(
        flambeau_ops::AttnPagedDecodeBuffers {
            q_batched: q_batched.ptr,
            k_pool,
            v_pool,
            block_tables,
            out_batched: out_batched.ptr,
            n_tokens_kv_ptrs: n_tokens_kv,
        },
        flambeau_ops::AttnDecodePagedShape {
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_slots,
            page_size,
            max_pages_per_slot,
        },
        scale,
    )
}
