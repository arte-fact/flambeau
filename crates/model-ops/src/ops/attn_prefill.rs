//! Multi-Q-token attention with causal mask + optional SWA, F16.
//! `q_offset` = global position of `Q[0]` (0 for fresh prefill;
//! prior-cache length for continuation). Each (q_token, q_head)
//! attends to `[t_start, limit)` where
//! `limit = min(q_offset + q_token + 1, n_k_tokens)` and
//! `t_start = max(0, q_offset + q_token - window_size + 1)` if
//! `window_size > 0`, else 0.

use anyhow::bail;
use flambeau_ops::Ops;

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// `q`/`out` are `[n_q_tokens, n_heads_q, head_dim]` F16;
/// `k_cache`/`v_cache` are `[n_k_tokens, n_heads_kv, head_dim]` F16.
pub fn attn_prefill_f16(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
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
        q_offset,
    } = shape;
    if !matches!(head_dim, 64 | 128 | 256 | 512) {
        bail!("attn_prefill_f16: head_dim {head_dim} not in {{64, 128, 256, 512}}");
    }
    if n_heads_q == 0 || n_heads_kv == 0 {
        bail!("attn_prefill_f16: head counts must be > 0");
    }
    if n_heads_q % n_heads_kv != 0 {
        bail!(
            "attn_prefill_f16: n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})"
        );
    }
    if q_offset + n_q_tokens > n_k_tokens {
        bail!(
            "attn_prefill_f16: q_offset ({q_offset}) + n_q_tokens ({n_q_tokens}) > n_k_tokens ({n_k_tokens}); \
             caller must append Q's keys before calling"
        );
    }
    let q_need = n_q_tokens * n_heads_q * head_dim;
    let cache_rows = crate::ops::kv_append::ring_cache_rows(n_k_tokens, knobs.ring_depth as usize);
    let cache_need = cache_rows * n_heads_kv * head_dim;
    if q.n_elems < q_need {
        bail!(
            "attn_prefill_f16: q has {} F16 elems, need >= {q_need}",
            q.n_elems
        );
    }
    if k_cache.n_elems < cache_need {
        bail!(
            "attn_prefill_f16: k_cache has {} F16 elems, need >= {cache_need}",
            k_cache.n_elems
        );
    }
    if v_cache.n_elems < cache_need {
        bail!(
            "attn_prefill_f16: v_cache has {} F16 elems, need >= {cache_need}",
            v_cache.n_elems
        );
    }
    if out.n_elems < q_need {
        bail!(
            "attn_prefill_f16: out has {} F16 elems, need >= {q_need}",
            out.n_elems
        );
    }
    ops.attention_prefill_f16(
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
fn cpu_attn_prefill(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    shape: flambeau_ops::AttnPrefillShape,
    knobs: flambeau_ops::AttnKnobs,
) -> Vec<f32> {
    let flambeau_ops::AttnPrefillShape {
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset,
    } = shape;
    let flambeau_ops::AttnKnobs { scale, window_size, ring_depth: _ } = knobs;
    let group = n_heads_q / n_heads_kv;
    let mut out = vec![0.0_f32; n_q_tokens * n_heads_q * head_dim];
    for q_token in 0..n_q_tokens {
        let qpos = q_offset + q_token;
        let limit = (qpos + 1).min(n_k_tokens);
        let t_start = if window_size > 0 {
            (qpos as i32 - window_size + 1).max(0) as usize
        } else {
            0
        };
        for q_head in 0..n_heads_q {
            let kv_head = q_head / group;
            let q_base = (q_token * n_heads_q + q_head) * head_dim;

            let mut scores = vec![0.0_f32; limit.saturating_sub(t_start)];
            for (i, t) in (t_start..limit).enumerate() {
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[q_base + d] * k_cache[(t * n_heads_kv + kv_head) * head_dim + d];
                }
                scores[i] = s * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for (i, t) in (t_start..limit).enumerate() {
                    let p = exps[i] / sum;
                    acc += p * v_cache[(t * n_heads_kv + kv_head) * head_dim + d];
                }
                out[q_base + d] = acc;
            }
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
        n_q_tokens: usize,
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        n_k_tokens: usize,
        q_offset: usize,
        window_size: i32,
    ) {
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);
        let scale = (head_dim as f32).sqrt().recip();

        let q_n = n_q_tokens * n_heads_q * head_dim;
        let cache_n = n_k_tokens * n_heads_kv * head_dim;

        let q_host_f32: Vec<f32> = (0..q_n)
            .map(|i| ((i as f32) * 0.019 - 0.5).sin() * 0.5)
            .collect();
        let k_host_f32: Vec<f32> = (0..cache_n)
            .map(|i| ((i as f32) * 0.021 + 0.2).cos() * 0.5)
            .collect();
        let v_host_f32: Vec<f32> = (0..cache_n)
            .map(|i| ((i as f32) * 0.029 - 0.1).sin() * 0.5)
            .collect();

        let q_host_f16: Vec<f16> = q_host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let k_host_f16: Vec<f16> = k_host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let v_host_f16: Vec<f16> = v_host_f32.iter().map(|&v| f16::from_f32(v)).collect();

        let q_kernel_f32: Vec<f32> = q_host_f16.iter().map(|v| v.to_f32()).collect();
        let k_kernel_f32: Vec<f32> = k_host_f16.iter().map(|v| v.to_f32()).collect();
        let v_kernel_f32: Vec<f32> = v_host_f16.iter().map(|v| v.to_f32()).collect();

        let shape = flambeau_ops::AttnPrefillShape {
            n_q_tokens,
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_k_tokens,
            q_offset,
        };
        let knobs = flambeau_ops::AttnKnobs { scale, window_size, ring_depth: 0 };
        let expected_f32 = cpu_attn_prefill(
            &q_kernel_f32,
            &k_kernel_f32,
            &v_kernel_f32,
            shape,
            knobs,
        );

        let (q_t, q_ptr) = upload::<F16, f16>(&device, &q_host_f16, q_n);
        let (k_t, k_ptr) = upload::<F16, f16>(&device, &k_host_f16, cache_n);
        let (v_t, v_ptr) = upload::<F16, f16>(&device, &v_host_f16, cache_n);
        let (mut out_t, out_ptr) = alloc::<F16>(&device, q_n);

        attn_prefill_f16(&q_t, &k_t, &v_t, &mut out_t, shape, knobs, &ops)
            .expect("attn_prefill_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        assert_close_f16(&got, &expected_f32, 5e-3, 1e-2);

        free(&device, q_ptr, q_t.bytes());
        free(&device, k_ptr, k_t.bytes());
        free(&device, v_ptr, v_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn attn_prefill_f16_fresh_no_swa() {
        run_case(8, 4, 2, 128, 8, 0, 0);
    }

    #[test]
    fn attn_prefill_f16_continuation() {
        run_case(4, 4, 2, 128, 8, 4, 0);
    }

    #[test]
    fn attn_prefill_f16_with_swa() {
        run_case(3, 4, 2, 64, 16, 13, 5);
    }

    #[test]
    fn attn_prefill_f16_flash_tile_swa_unaligned_window() {
        // flash_tile path (n_q_tokens >= 4) with window_size that
        // makes block_swa_min unaligned to BC. Regression test for the
        // NaN bug where the first chunk's first row is masked and the
        // online softmax initializer (m_i = s_j = -INFINITY) produced
        // NaN through alpha = exp(-inf - -inf).
        run_case(8, 4, 2, 64, 600, 520, 512);
    }
}
