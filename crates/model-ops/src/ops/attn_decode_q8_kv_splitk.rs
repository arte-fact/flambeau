//! Split-K flash-decoding with Q8_0 KV cache. Q8 sibling of
//! `attn_decode_f16_splitk`. Fans out blocks over `n_chunks =
//! ceil(n_tokens_kv / chunk_size)`, restoring the inter-block
//! parallelism that the single-block `attn_decode_q8_kv` loses
//! at long context. `head_dim ∈ {64, 128, 256}`; no SWA.

use anyhow::bail;
use flambeau_ops::Ops;

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
    ops: &impl Ops,
) -> Result<()> {
    attn_decode_q8_kv_splitk_slots(q, k_cache, v_cache, outputs, shape, knobs, ops, None)
}

/// Graph-captureable variant of [`attn_decode_q8_kv_splitk`]. `slots=None` is
/// byte-identical to the wrapper; `Some` tags the chunk kernel's `n_tokens_kv`
/// + `chunk_base` scalars so a captured graph rebinds them per replay.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_q8_kv_splitk_slots(
    q: &Tensor<F16>,
    k_cache: &Tensor<Q8_0>,
    v_cache: &Tensor<Q8_0>,
    outputs: crate::ops::attn_decode_splitk::SplitkOutputs<'_>,
    shape: flambeau_ops::AttnSplitkShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &impl Ops,
    slots: Option<flambeau_ops::AttnDecodeQ8SplitkSlots>,
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
        chunk_base,
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
    let n_chunks = (n_tokens_kv - chunk_base).div_ceil(chunk_size);
    let q_need = n_heads_q * head_dim;
    let cache_rows = crate::ops::kv_append::ring_cache_rows(n_tokens_kv, knobs.ring_depth as usize);
    let cache_need = cache_rows * n_heads_kv * head_dim;
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
    ops.attention_decode_q8_kv_splitk_slots(
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
        slots,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::F32;
    use crate::ops::attn_decode_splitk::SplitkOutputs;
    use crate::ops::kv_append_f16_to_q8::kv_append_f16_to_q8;
    use crate::testing::{alloc, download, free, test_device, test_ops_registry, upload};
    use flambeau_core::{Device, Stream};
    use flambeau_ops::HipOps;
    use half::f16;

    /// `attn_decode_q8_kv_splitk_slots(None)` must produce the exact same
    /// device output as `attn_decode_q8_kv_splitk` — transparency contract for
    /// the graph-capture variant outside capture.
    #[test]
    fn q8_splitk_slots_none_bit_identical() {
        const N_HEADS_Q: usize = 4;
        const N_HEADS_KV: usize = 2;
        const HEAD_DIM: usize = 128;
        const N_TOKENS_KV: usize = 12;
        const KV_WIDTH: usize = N_HEADS_KV * HEAD_DIM;
        const CHUNK_SIZE: usize = 4;
        const CHUNK_BASE: usize = 0;
        let scale = (HEAD_DIM as f32).sqrt().recip();

        let device = test_device();
        device.bind().expect("bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let n_chunks = (N_TOKENS_KV - CHUNK_BASE).div_ceil(CHUNK_SIZE);
        let q_n = N_HEADS_Q * HEAD_DIM;
        let src_n = N_TOKENS_KV * KV_WIDTH;
        let ms_n = N_HEADS_Q * n_chunks;
        let o_n = ms_n * HEAD_DIM;

        let q_host: Vec<f16> =
            (0..q_n).map(|i| f16::from_f32(((i as f32) * 0.017 + 0.3).sin() * 0.5)).collect();
        let k_host: Vec<f16> =
            (0..src_n).map(|i| f16::from_f32(((i as f32) * 0.023 - 0.4).cos() * 0.5)).collect();
        let v_host: Vec<f16> =
            (0..src_n).map(|i| f16::from_f32(((i as f32) * 0.031 + 0.1).sin() * 0.5)).collect();

        let (q_t, q_ptr) = upload::<F16, f16>(&device, &q_host, q_n);
        let (k_src_t, k_src_ptr) = upload::<F16, f16>(&device, &k_host, src_n);
        let (v_src_t, v_src_ptr) = upload::<F16, f16>(&device, &v_host, src_n);

        let q8_bytes = N_TOKENS_KV * (KV_WIDTH / 32) * 34;
        let q8_init = vec![0u8; q8_bytes];
        let (mut k_cache_q8, k_cache_ptr) = upload::<Q8_0, u8>(&device, &q8_init, N_TOKENS_KV * KV_WIDTH);
        let (mut v_cache_q8, v_cache_ptr) = upload::<Q8_0, u8>(&device, &q8_init, N_TOKENS_KV * KV_WIDTH);
        kv_append_f16_to_q8(
            &k_src_t,
            &v_src_t,
            &mut k_cache_q8,
            &mut v_cache_q8,
            crate::ops::kv_append::KvAppendSpec {
                n_tokens: N_TOKENS_KV,
                kv_width: KV_WIDTH,
                write_pos: 0,
                max_seq_len: N_TOKENS_KV,
                ring_depth: 0,
            },
            &ops,
        )
        .expect("kv_append_f16_to_q8");

        let shape = flambeau_ops::AttnSplitkShape {
            n_heads_q: N_HEADS_Q,
            n_heads_kv: N_HEADS_KV,
            head_dim: HEAD_DIM,
            n_tokens_kv: N_TOKENS_KV,
            chunk_size: CHUNK_SIZE,
            chunk_base: CHUNK_BASE,
        };
        let knobs = flambeau_ops::AttnKnobs { scale, window_size: 0, ring_depth: 0 };

        let (mut out_a, out_a_ptr) = alloc::<F16>(&device, q_n);
        let (mut m_a, m_a_ptr) = alloc::<F32>(&device, ms_n);
        let (mut s_a, s_a_ptr) = alloc::<F32>(&device, ms_n);
        let (mut o_a, o_a_ptr) = alloc::<F32>(&device, o_n);
        attn_decode_q8_kv_splitk(
            &q_t,
            &k_cache_q8,
            &v_cache_q8,
            SplitkOutputs { out: &mut out_a, m: &mut m_a, s: &mut s_a, o: &mut o_a },
            shape,
            knobs,
            &ops,
        )
        .expect("plain q8 split-K");

        let (mut out_b, out_b_ptr) = alloc::<F16>(&device, q_n);
        let (mut m_b, m_b_ptr) = alloc::<F32>(&device, ms_n);
        let (mut s_b, s_b_ptr) = alloc::<F32>(&device, ms_n);
        let (mut o_b, o_b_ptr) = alloc::<F32>(&device, o_n);
        attn_decode_q8_kv_splitk_slots(
            &q_t,
            &k_cache_q8,
            &v_cache_q8,
            SplitkOutputs { out: &mut out_b, m: &mut m_b, s: &mut s_b, o: &mut o_b },
            shape,
            knobs,
            &ops,
            None,
        )
        .expect("q8 split-K slots=None");

        stream.synchronize().expect("sync");
        let a: Vec<f16> = download::<F16, f16>(&device, &out_a);
        let b: Vec<f16> = download::<F16, f16>(&device, &out_b);
        let a_bits: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
        let b_bits: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
        assert_eq!(a_bits, b_bits, "slots=None output must be bit-identical to plain");

        free(&device, q_ptr, q_t.bytes());
        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_ptr, k_cache_q8.bytes());
        free(&device, v_cache_ptr, v_cache_q8.bytes());
        free(&device, out_a_ptr, out_a.bytes());
        free(&device, m_a_ptr, m_a.bytes());
        free(&device, s_a_ptr, s_a.bytes());
        free(&device, o_a_ptr, o_a.bytes());
        free(&device, out_b_ptr, out_b.bytes());
        free(&device, m_b_ptr, m_b.bytes());
        free(&device, s_b_ptr, s_b.bytes());
        free(&device, o_b_ptr, o_b.bytes());
    }
}
