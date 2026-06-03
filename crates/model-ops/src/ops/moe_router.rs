//! MoE router: per-token top-k expert selection + softmax-over-k.
//! Algebraically equivalent to llama.cpp's
//! `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` with `norm_w=true`
//! (softmax-over-all then top-k then renormalise).

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F32, I32};
use crate::error::Result;
use crate::tensor::Tensor;

/// Inputs:
/// - `logits[n_tokens, n_experts]` F32 — router-projection output.
///   Outputs:
/// - `out_ids[n_tokens, k]` I32 — selected expert indices.
/// - `out_weights[n_tokens, k]` F32 — normalised weights summing to 1
///   over `k` per token.
pub fn topk_f32(
    logits: &Tensor<F32>,
    out_ids: &mut Tensor<I32>,
    out_weights: &mut Tensor<F32>,
    n_tokens: usize,
    n_experts: usize,
    k: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if k == 0 || k > n_experts {
        bail!("moe_router::topk_f32: k {k} invalid for n_experts {n_experts}");
    }
    let l_need = n_tokens * n_experts;
    if logits.n_elems < l_need {
        bail!(
            "moe_router::topk_f32: logits has {} F32 elems, need >= {l_need}",
            logits.n_elems
        );
    }
    let out_need = n_tokens * k;
    if out_ids.n_elems < out_need {
        bail!(
            "moe_router::topk_f32: out_ids has {} I32 elems, need >= {out_need}",
            out_ids.n_elems
        );
    }
    if out_weights.n_elems < out_need {
        bail!(
            "moe_router::topk_f32: out_weights has {} F32 elems, need >= {out_need}",
            out_weights.n_elems
        );
    }
    ops.topk_f32(
        logits.ptr,
        out_ids.ptr,
        out_weights.ptr,
        n_tokens,
        n_experts,
        k,
    )
}
