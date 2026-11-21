//! Flash-attention-v2 decode (1 Q token, full K/V), F16. GQA via
//! `kv_head_of(q) = q / (n_heads_q / n_heads_kv)`. `head_dim` ∈
//! {64,128,256,512}. `window_size`: 0 = unbounded causal; >0 = SWA
//! over the last N tokens.

use anyhow::bail;
use flambeau_ops::Ops;

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// Q has 1 token, K/V cover `n_tokens_kv`. Output `[n_heads_q, head_dim]` F16.
pub fn attn_decode_f16(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
    out: &mut Tensor<F16>,
    shape: flambeau_ops::AttnDecodeShape,
    knobs: flambeau_ops::AttnKnobs,
    ops: &impl Ops,
) -> Result<()> {
    let flambeau_ops::AttnDecodeShape { n_heads_q, n_heads_kv, head_dim, n_tokens_kv } = shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_decode_f16: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 {
        bail!("attn_decode_f16: head counts must be > 0 (got q={n_heads_q}, kv={n_heads_kv})");
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_decode_f16: n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv}) for GQA"
        );
    }
    let q_need = n_heads_q * head_dim;
    let cache_rows = crate::ops::kv_append::ring_cache_rows(n_tokens_kv, knobs.ring_depth as usize);
    let cache_need = cache_rows * n_heads_kv * head_dim;
    if q.n_elems < q_need {
        bail!("attn_decode_f16: q has {} F16 elems, need >= {q_need}", q.n_elems);
    }
    if k_cache.n_elems < cache_need {
        bail!("attn_decode_f16: k_cache has {} F16 elems, need >= {cache_need}", k_cache.n_elems);
    }
    if v_cache.n_elems < cache_need {
        bail!("attn_decode_f16: v_cache has {} F16 elems, need >= {cache_need}", v_cache.n_elems);
    }
    if out.n_elems < q_need {
        bail!("attn_decode_f16: out has {} F16 elems, need >= {q_need}", out.n_elems);
    }
    ops.attention_decode_f16(
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
fn cpu_attn_decode(
    q: &[f32],       // [n_heads_q, head_dim]
    k_cache: &[f32], // [n_tokens, n_heads_kv, head_dim]
    v_cache: &[f32], // [n_tokens, n_heads_kv, head_dim]
    shape: flambeau_ops::AttnDecodeShape,
    knobs: flambeau_ops::AttnKnobs,
) -> Vec<f32> {
    let flambeau_ops::AttnDecodeShape { n_heads_q, n_heads_kv, head_dim, n_tokens_kv: n_tokens } = shape;
    let flambeau_ops::AttnKnobs { scale, window_size, ring_depth: _ } = knobs;
    let group = n_heads_q / n_heads_kv;
    let mut out = vec![0.0_f32; n_heads_q * head_dim];
    for q_head in 0..n_heads_q {
        let kv_head = q_head / group;

        // SWA: keep the last `window_size` tokens; 0 = unbounded.
        let lo = if window_size > 0 {
            n_tokens.saturating_sub(window_size as usize)
        } else {
            0
        };

        let mut scores = vec![0.0_f32; n_tokens - lo];
        for (i, t) in (lo..n_tokens).enumerate() {
            let mut s = 0.0_f32;
            for d in 0..head_dim {
                let q_v = q[q_head * head_dim + d];
                let k_v = k_cache[(t * n_heads_kv + kv_head) * head_dim + d];
                s += q_v * k_v;
            }
            scores[i] = s * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for (i, t) in (lo..n_tokens).enumerate() {
                let p = exps[i] / sum;
                acc += p * v_cache[(t * n_heads_kv + kv_head) * head_dim + d];
            }
            out[q_head * head_dim + d] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use flambeau_ops::HipOps;
    use crate::testing::{
        alloc, assert_close_f16, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;
    use half::f16;

    fn run_case(
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_tokens: usize,
        window_size: i32,
    ) {
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);
        let scale = (head_dim as f32).sqrt().recip();

        let q_n = n_heads_q * head_dim;
        let cache_n = n_tokens * n_heads_kv * head_dim;

        let q_host_f32: Vec<f32> = (0..q_n)
            .map(|i| ((i as f32) * 0.017 + 0.3).sin() * 0.5)
            .collect();
        let k_host_f32: Vec<f32> = (0..cache_n)
            .map(|i| ((i as f32) * 0.023 - 0.4).cos() * 0.5)
            .collect();
        let v_host_f32: Vec<f32> = (0..cache_n)
            .map(|i| ((i as f32) * 0.031 + 0.1).sin() * 0.5)
            .collect();

        let q_host_f16: Vec<f16> = q_host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let k_host_f16: Vec<f16> = k_host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let v_host_f16: Vec<f16> = v_host_f32.iter().map(|&v| f16::from_f32(v)).collect();

        // F32 the kernel sees post F16 round-trip.
        let q_kernel_f32: Vec<f32> = q_host_f16.iter().map(|v| v.to_f32()).collect();
        let k_kernel_f32: Vec<f32> = k_host_f16.iter().map(|v| v.to_f32()).collect();
        let v_kernel_f32: Vec<f32> = v_host_f16.iter().map(|v| v.to_f32()).collect();

        let shape = flambeau_ops::AttnDecodeShape {
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_tokens_kv: n_tokens,
        };
        let knobs = flambeau_ops::AttnKnobs { scale, window_size, ring_depth: 0 };
        let expected_f32 = cpu_attn_decode(&q_kernel_f32, &k_kernel_f32, &v_kernel_f32, shape, knobs);

        let (q_t, q_ptr) = upload::<F16, f16>(&device, &q_host_f16, q_n);
        let (k_t, k_ptr) = upload::<F16, f16>(&device, &k_host_f16, cache_n);
        let (v_t, v_ptr) = upload::<F16, f16>(&device, &v_host_f16, cache_n);
        let (mut out_t, out_ptr) = alloc::<F16>(&device, q_n);

        attn_decode_f16(&q_t, &k_t, &v_t, &mut out_t, shape, knobs, &ops)
            .expect("attn_decode_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        // Bound scales with n_tokens; 5e-3 covers n_tokens ≤ 64.
        assert_close_f16(&got, &expected_f32, 5e-3, 1e-2);

        free(&device, q_ptr, q_t.bytes());
        free(&device, k_ptr, k_t.bytes());
        free(&device, v_ptr, v_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn attn_decode_f16_gqa_head_dim_128_no_swa() {
        run_case(4, 2, 128, 16, 0);
    }

    #[test]
    fn attn_decode_f16_gqa_head_dim_64_with_swa() {
        run_case(4, 2, 64, 20, 5);
    }

    #[test]
    fn attn_decode_f16_head_dim_256() {
        run_case(2, 1, 256, 8, 0);
    }
}
