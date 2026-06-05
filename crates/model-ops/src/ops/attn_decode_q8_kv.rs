//! GQA flash-decoding with Q8_0 KV cache. Same shape contract as
//! `attn_decode_f16`; the cache argument carries Q8_0 blocks instead
//! of F16 elements. Q stays F16 (per-call cost too small to amortise
//! a quantize launch). Kernel supports `head_dim ∈ {64, 128, 256}` —
//! gemma4 global head_dim=512 bails back to the F16 path until that
//! kernel is extended.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, Q8_0};
use crate::error::Result;
use crate::tensor::Tensor;

pub fn attn_decode_q8_kv(
    q: &Tensor<F16>,
    k_cache: &Tensor<Q8_0>,
    v_cache: &Tensor<Q8_0>,
    out: &mut Tensor<F16>,
    shape: flambeau_ops::AttnDecodeShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &HipOps<'_>,
) -> Result<()> {
    let flambeau_ops::AttnDecodeShape { n_heads_q, n_heads_kv, head_dim, n_tokens_kv } = shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_q8_kv: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 {
        bail!("attn_decode_q8_kv: head counts must be > 0 (got q={n_heads_q}, kv={n_heads_kv})");
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_q8_kv: n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})"
        );
    }
    let q_need = n_heads_q * head_dim;
    let cache_need = n_tokens_kv * n_heads_kv * head_dim;
    if q.n_elems < q_need {
        bail!(
            "attn_decode_q8_kv: q has {} F16 elems, need >= {q_need}",
            q.n_elems
        );
    }
    if k_cache.n_elems < cache_need {
        bail!(
            "attn_decode_q8_kv: k_cache has {} logical Q8_0 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "attn_decode_q8_kv: v_cache has {} logical Q8_0 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    if out.n_elems < q_need {
        bail!(
            "attn_decode_q8_kv: out has {} F16 elems, need >= {q_need}",
            out.n_elems
        );
    }
    ops.attention_decode_q8_kv(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attn_decode::attn_decode_f16;
    use crate::ops::kv_append_f16_to_q8::kv_append_f16_to_q8;
    use crate::ops::kv_append::kv_append_f16;
    use crate::testing::{alloc, download, free, test_device, test_ops_registry, upload};
    use flambeau_core::{Device, Stream};
    use half::f16;

    /// Compute F16 vs Q8 attention outputs on identical synthetic K/V
    /// data; assert that Q8 output stays within Q8-noise tolerance of
    /// F16. Q quantization noise is per-block (32 elems), so the
    /// elementwise tolerance scales with abs(amax_per_block) / 127.
    #[test]
    fn attn_decode_q8_kv_matches_f16_within_q8_noise() {
        const N_HEADS_Q: usize = 4;
        const N_HEADS_KV: usize = 2;
        const HEAD_DIM: usize = 64;
        const N_TOKENS_KV: usize = 5;
        const MAX_SEQ_LEN: usize = 8;
        const KV_WIDTH: usize = N_HEADS_KV * HEAD_DIM;
        const SCALE: f32 = 0.125;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let q_host: Vec<f16> = (0..N_HEADS_Q * HEAD_DIM)
            .map(|i| {
                let t = (i as f32) / (N_HEADS_Q * HEAD_DIM) as f32;
                f16::from_f32(0.5 * (t - 0.5))
            })
            .collect();
        let k_host: Vec<f16> = (0..N_TOKENS_KV * KV_WIDTH)
            .map(|i| {
                let t = (i as f32) / (N_TOKENS_KV * KV_WIDTH) as f32;
                f16::from_f32(0.6 * (t.sin() - 0.3))
            })
            .collect();
        let v_host: Vec<f16> = (0..N_TOKENS_KV * KV_WIDTH)
            .map(|i| {
                let t = (i as f32) / (N_TOKENS_KV * KV_WIDTH) as f32;
                f16::from_f32(0.4 * (t.cos() + 0.1))
            })
            .collect();

        let (q_t, q_ptr) = upload::<F16, f16>(&device, &q_host, q_host.len());
        let (k_src_t, k_src_ptr) = upload::<F16, f16>(&device, &k_host, k_host.len());
        let (v_src_t, v_src_ptr) = upload::<F16, f16>(&device, &v_host, v_host.len());

        // F16 reference cache + decode.
        let f16_cache_init = vec![f16::ZERO; MAX_SEQ_LEN * KV_WIDTH];
        let (mut k_cache_f16, k_cache_f16_ptr) =
            upload::<F16, f16>(&device, &f16_cache_init, f16_cache_init.len());
        let (mut v_cache_f16, v_cache_f16_ptr) =
            upload::<F16, f16>(&device, &f16_cache_init, f16_cache_init.len());
        kv_append_f16(
            &k_src_t,
            &v_src_t,
            &mut k_cache_f16,
            &mut v_cache_f16,
            crate::ops::kv_append::KvAppendSpec {
                n_tokens: N_TOKENS_KV,
                kv_width: KV_WIDTH,
                write_pos: 0,
                max_seq_len: MAX_SEQ_LEN,
            },
            &device,
            stream,
        )
        .expect("kv_append_f16");
        let (mut out_f16, out_f16_ptr) = alloc::<F16>(&device, N_HEADS_Q * HEAD_DIM);
        let dec_shape = flambeau_ops::AttnDecodeShape {
            n_heads_q: N_HEADS_Q,
            n_heads_kv: N_HEADS_KV,
            head_dim: HEAD_DIM,
            n_tokens_kv: N_TOKENS_KV,
        };
        let dec_knobs = flambeau_ops::AttnKnobs { scale: SCALE, window_size: 0, ring_depth: 0 };
        attn_decode_f16(
            &q_t,
            &k_cache_f16,
            &v_cache_f16,
            &mut out_f16,
            dec_shape,
            dec_knobs,
            &ops,
        )
        .expect("attn_decode_f16");

        // Q8 cache + decode on same data.
        const Q8_CACHE_BYTES: usize =
            MAX_SEQ_LEN * (KV_WIDTH / 32) * 34;
        let q8_cache_init = vec![0u8; Q8_CACHE_BYTES];
        let (mut k_cache_q8, k_cache_q8_ptr) =
            upload::<Q8_0, u8>(&device, &q8_cache_init, MAX_SEQ_LEN * KV_WIDTH);
        let (mut v_cache_q8, v_cache_q8_ptr) =
            upload::<Q8_0, u8>(&device, &q8_cache_init, MAX_SEQ_LEN * KV_WIDTH);
        kv_append_f16_to_q8(
            &k_src_t,
            &v_src_t,
            &mut k_cache_q8,
            &mut v_cache_q8,
            crate::ops::kv_append::KvAppendSpec {
                n_tokens: N_TOKENS_KV,
                kv_width: KV_WIDTH,
                write_pos: 0,
                max_seq_len: MAX_SEQ_LEN,
            },
            &ops,
        )
        .expect("kv_append_f16_to_q8");
        let (mut out_q8, out_q8_ptr) = alloc::<F16>(&device, N_HEADS_Q * HEAD_DIM);
        attn_decode_q8_kv(
            &q_t,
            &k_cache_q8,
            &v_cache_q8,
            &mut out_q8,
            dec_shape,
            dec_knobs,
            &ops,
        )
        .expect("attn_decode_q8_kv");
        stream.synchronize().expect("stream sync");

        let f16_vals: Vec<f16> = download::<F16, f16>(&device, &out_f16);
        let q8_vals: Vec<f16> = download::<F16, f16>(&device, &out_q8);
        // Q8_0 quantization noise on attention output: roughly
        // sqrt(n_tokens × head_dim) × amax/127 amplified through the
        // softmax-weighted sum. For our small synth data with amax ≈
        // 0.6, tolerance of 0.05 (~ 8 % of amax) covers Q8 round-trip
        // + softmax-weighted accumulation; tighter than the typical
        // F16 epsilon (~1e-3) but far below output amplitude.
        for i in 0..N_HEADS_Q * HEAD_DIM {
            let f = f16_vals[i].to_f32();
            let q = q8_vals[i].to_f32();
            assert!(
                (f - q).abs() < 0.05,
                "out[{i}]: F16={f} Q8={q} delta={}",
                (f - q).abs()
            );
        }

        free(&device, q_ptr, q_t.bytes());
        free(&device, k_src_ptr, k_src_t.bytes());
        free(&device, v_src_ptr, v_src_t.bytes());
        free(&device, k_cache_f16_ptr, k_cache_f16.bytes());
        free(&device, v_cache_f16_ptr, v_cache_f16.bytes());
        free(&device, out_f16_ptr, out_f16.bytes());
        free(&device, k_cache_q8_ptr, k_cache_q8.bytes());
        free(&device, v_cache_q8_ptr, v_cache_q8.bytes());
        free(&device, out_q8_ptr, out_q8.bytes());
    }
}
