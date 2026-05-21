//! Split-K (flash-decoding) decode attention, F16. Partitions the KV
//! context across `grid.y = n_chunks` to attack single-pass
//! occupancy starvation at long contexts. `chunk_size` controls the
//! partition; pick via `splitk_chunk_size(n_tokens_kv)`. F16 KV only.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, F32};
use crate::error::Result;
use crate::tensor::Tensor;

/// Caller must size partials at `n_heads_q * n_chunks` (m, s) and
/// `n_heads_q * n_chunks * head_dim` (o), where `n_chunks =
/// ceil(n_tokens_kv / chunk_size)`.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_f16_splitk(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
    out: &mut Tensor<F16>,
    partials_m: &mut Tensor<F32>,
    partials_s: &mut Tensor<F32>,
    partials_o: &mut Tensor<F32>,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_tokens_kv: usize,
    chunk_size: usize,
    scale: f32,
    window_size: i32,
    ops: &HipOps<'_>,
) -> Result<()> {
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16_splitk: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 || chunk_size == 0 {
        bail!(
            "attn_decode_f16_splitk: zero counts (n_heads_q={n_heads_q}, \
             n_heads_kv={n_heads_kv}, chunk_size={chunk_size})"
        );
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_f16_splitk: n_heads_q ({n_heads_q}) must be divisible by \
             n_heads_kv ({n_heads_kv}) for GQA"
        );
    }
    let n_chunks = n_tokens_kv.div_ceil(chunk_size);
    let q_need = n_heads_q * head_dim;
    let cache_need = n_tokens_kv * n_heads_kv * head_dim;
    let ms_need = n_heads_q * n_chunks;
    let o_need = n_heads_q * n_chunks * head_dim;
    if q.n_elems < q_need {
        bail!(
            "attn_decode_f16_splitk: q has {} F16 elems, need >= {q_need}",
            q.n_elems
        );
    }
    if k_cache.n_elems < cache_need {
        bail!(
            "attn_decode_f16_splitk: k_cache has {} F16 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "attn_decode_f16_splitk: v_cache has {} F16 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    if out.n_elems < q_need {
        bail!(
            "attn_decode_f16_splitk: out has {} F16 elems, need >= {q_need}",
            out.n_elems
        );
    }
    if partials_m.n_elems < ms_need {
        bail!(
            "attn_decode_f16_splitk: partials_m has {} F32 elems, need >= {ms_need}",
            partials_m.n_elems
        );
    }
    if partials_s.n_elems < ms_need {
        bail!(
            "attn_decode_f16_splitk: partials_s has {} F32 elems, need >= {ms_need}",
            partials_s.n_elems
        );
    }
    if partials_o.n_elems < o_need {
        bail!(
            "attn_decode_f16_splitk: partials_o has {} F32 elems, need >= {o_need}",
            partials_o.n_elems
        );
    }
    ops.attention_decode_f16_splitk(
        q.ptr,
        k_cache.ptr,
        v_cache.ptr,
        out.ptr,
        partials_m.ptr,
        partials_s.ptr,
        partials_o.ptr,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_tokens_kv,
        chunk_size,
        scale,
        window_size,
    )
}

/// Same threshold as `flambeau_ops::hip::attention::splitk_chunk_size`
/// — re-exported for callers that need to size scratch.
pub fn splitk_chunk_size(n_tokens_kv: usize) -> usize {
    flambeau_ops::hip::attention::splitk_chunk_size(n_tokens_kv)
}
