//! Per-head V unit RMSNorm in place over
//! `[n_tokens, n_kv_heads, head_dim]` F16. Unit weights (no learnable
//! gamma) — matches gemma4's `attn_v_unit_norm = true` semantics.
//! Composes with `kv_append_f16` (prefill) and
//! `kv_append_f16_batched_slots` (batched decode) on the mixed-batch
//! path where the fused `kv_append_v_unit_norm_f16` of the legacy
//! prefill_shape branch cannot fan out to N distinct slots.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// In-place per-head unit-RMSNorm on V. `v` is treated as
/// `[n_tokens, n_kv_heads, head_dim]` row-major; each
/// `(token, head)` group of `head_dim` F16 elements is independently
/// normalized.
///
/// # Errors
///
/// Returns an error if `v` is too small for the requested shape, if
/// `head_dim` is outside `[64, 512]` or not a multiple of 64, or if
/// the kernel launch fails.
pub fn v_unit_norm_per_head_f16(
    v: &mut Tensor<F16>,
    n_tokens: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    ops: &HipOps<'_>,
) -> Result<()> {
    let need = n_tokens * n_kv_heads * head_dim;
    if v.n_elems < need {
        bail!(
            "v_unit_norm_per_head_f16: v has {} F16 elems, need >= {} \
             ({n_tokens}*{n_kv_heads}*{head_dim})",
            v.n_elems,
            need,
        );
    }
    ops.v_unit_norm_per_head_f16(v.ptr, n_tokens, n_kv_heads, head_dim, eps)
}
