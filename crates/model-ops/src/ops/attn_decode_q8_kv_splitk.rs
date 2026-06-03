//! Split-K flash-decoding with Q8_0 KV cache. Q8 sibling of
//! `attn_decode_f16_splitk`. Fans out blocks over `n_chunks =
//! ceil(n_tokens_kv / chunk_size)`, restoring the inter-block
//! parallelism that the single-block `attn_decode_q8_kv` loses
//! at long context. `head_dim ∈ {64, 128, 256}`; no SWA.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, Q8_0};
use crate::error::Result;
use crate::tensor::Tensor;

pub fn attn_decode_q8_kv_splitk(
    q: &Tensor<F16>,
    k_cache: &Tensor<Q8_0>,
    v_cache: &Tensor<Q8_0>,
    outputs: crate::ops::attn_decode_splitk::SplitkOutputs<'_>,
    shape: flambeau_ops::AttnSplitkShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &HipOps<'_>,
) -> Result<()> {
    let crate::ops::attn_decode_splitk::SplitkOutputs {
        out,
        m: partials_m,
        s: partials_s,
        o: partials_o,
    } = outputs;
    let flambeau_ops::AttnSplitkShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
    } = shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_q8_kv_splitk: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || chunk_size == 0 {
        bail!(
            "attn_decode_q8_kv_splitk: zero counts (n_heads_q={n_heads_q}, \
             n_heads_kv={n_heads_kv}, chunk_size={chunk_size})"
        );
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_q8_kv_splitk: n_heads_q ({n_heads_q}) must be divisible by \
             n_heads_kv ({n_heads_kv})"
        );
    }
    let n_chunks = n_tokens_kv.div_ceil(chunk_size);
    let q_need = n_heads_q * head_dim;
    let cache_need = n_tokens_kv * n_heads_kv * head_dim;
    let ms_need = n_heads_q * n_chunks;
    let o_need = n_heads_q * n_chunks * head_dim;
    if q.n_elems < q_need {
        bail!(
            "attn_decode_q8_kv_splitk: q has {} F16 elems, need >= {q_need}",
            q.n_elems
        );
    }
    if k_cache.n_elems < cache_need {
        bail!(
            "attn_decode_q8_kv_splitk: k_cache has {} logical Q8_0 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "attn_decode_q8_kv_splitk: v_cache has {} logical Q8_0 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    if out.n_elems < q_need {
        bail!(
            "attn_decode_q8_kv_splitk: out has {} F16 elems, need >= {q_need}",
            out.n_elems
        );
    }
    if partials_m.n_elems < ms_need {
        bail!(
            "attn_decode_q8_kv_splitk: partials_m has {} F32 elems, need >= {ms_need}",
            partials_m.n_elems
        );
    }
    if partials_s.n_elems < ms_need {
        bail!(
            "attn_decode_q8_kv_splitk: partials_s has {} F32 elems, need >= {ms_need}",
            partials_s.n_elems
        );
    }
    if partials_o.n_elems < o_need {
        bail!(
            "attn_decode_q8_kv_splitk: partials_o has {} F32 elems, need >= {o_need}",
            partials_o.n_elems
        );
    }
    ops.attention_decode_q8_kv_splitk(
        flambeau_ops::AttnBuffers {
            q: q.ptr,
            k: k_cache.ptr,
            v: v_cache.ptr,
            out: out.ptr,
        },
        flambeau_ops::AttnSplitkPartials {
            partials_m: partials_m.ptr,
            partials_s: partials_s.ptr,
            partials_o: partials_o.ptr,
        },
        shape,
        knobs,
    )
}
