//! GQA prefill attention reading a Q8_0 KV cache. F16 Q, F16 out.
//! Same shape contract as `attn_prefill_f16`, swapping the K/V cache
//! dtype. Supports `head_dim ∈ {64, 128, 256, 512}`.

use anyhow::bail;
use flambeau_ops::Ops;

use crate::dtype::{F16, Q8_0};
use crate::error::Result;
use crate::tensor::Tensor;

pub fn attn_prefill_q8_kv(
    q: &Tensor<F16>,
    k_cache: &Tensor<Q8_0>,
    v_cache: &Tensor<Q8_0>,
    out: &mut Tensor<F16>,
    shape: flambeau_ops::AttnPrefillShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &impl Ops,
) -> Result<()> {
    let flambeau_ops::AttnPrefillShape {
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset: _,
    } = shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!(
            "attn_prefill_q8_kv: head_dim {head_dim} not in {{64, 128, 256, 512}}"
        );
    }
    if n_heads_q == 0 || n_heads_kv == 0 {
        bail!(
            "attn_prefill_q8_kv: head counts must be > 0 (got q={n_heads_q}, kv={n_heads_kv})"
        );
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_prefill_q8_kv: n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})"
        );
    }
    let q_need = n_q_tokens * n_heads_q * head_dim;
    let cache_rows = crate::ops::kv_append::ring_cache_rows(n_k_tokens, knobs.ring_depth as usize);
    let cache_need = cache_rows * n_heads_kv * head_dim;
    let out_need = n_q_tokens * n_heads_q * head_dim;
    if q.n_elems < q_need {
        bail!(
            "attn_prefill_q8_kv: q has {} F16 elems, need >= {q_need}",
            q.n_elems
        );
    }
    if k_cache.n_elems < cache_need {
        bail!(
            "attn_prefill_q8_kv: k_cache has {} logical Q8_0 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "attn_prefill_q8_kv: v_cache has {} logical Q8_0 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    if out.n_elems < out_need {
        bail!(
            "attn_prefill_q8_kv: out has {} F16 elems, need >= {out_need}",
            out.n_elems
        );
    }
    ops.attention_prefill_q8_kv(
        flambeau_ops::AttnBuffers {
            q: q.ptr,
            k: k_cache.ptr,
            v: v_cache.ptr,
            out: out.ptr,
        },
        shape,
        knobs,
    )
}
