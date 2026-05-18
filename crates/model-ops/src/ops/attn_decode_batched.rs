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
#[allow(clippy::too_many_arguments)]
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
        k_src.ptr,
        v_src.ptr,
        slot_k_dst_ptrs,
        slot_v_dst_ptrs,
        slot_write_pos,
        n_slots,
        kv_width,
    )
}

/// Single-launch GQA decode attention over N slots. Each slot owns
/// its own K/V cache base; per-slot KV length comes from the
/// `n_tokens_kv` device array. `q_batched` is `[n_slots, n_heads_q,
/// head_dim]` slot-major F16; `out_batched` matches. `k_cache_ptrs` /
/// `v_cache_ptrs` are `[n_slots] u64` device pointer arrays.
#[allow(clippy::too_many_arguments)]
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
    ops: &HipOps<'_>,
) -> Result<()> {
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16_batched: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_slots == 0 || n_slots > 32 {
        bail!("attn_decode_f16_batched: n_slots {n_slots} out of range [1, 32]");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_f16_batched: head counts invalid (q={n_heads_q}, kv={n_heads_kv})"
        );
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
        q_batched.ptr,
        k_cache_ptrs,
        v_cache_ptrs,
        out_batched.ptr,
        n_tokens_kv,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_slots,
        scale,
    )
}
