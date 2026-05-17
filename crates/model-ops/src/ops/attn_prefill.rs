//! Prefill attention (multi-token Q, causal mask, optional SWA), F16.
//!
//! Generalises [`crate::attn_decode_f16`] to `n_q_tokens > 1`. Each
//! (q_token, q_head) pair attends to `[t_start, limit)` of K/V where:
//! - `limit = min(q_offset + q_token + 1, n_k_tokens)` (causal mask)
//! - `t_start = max(0, q_offset + q_token - window_size + 1)` if
//!   `window_size > 0`, else 0
//!
//! `q_offset` is the global position of `Q[0]` in the sequence; for a
//! fresh prefill it's 0, for a continuation prefill (KV already
//! populated from a previous turn) it's the number of K/V tokens
//! already in the cache.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// Prefill attention. `q` is `[n_q_tokens, n_heads_q, head_dim]` F16;
/// `k_cache`/`v_cache` are `[n_k_tokens, n_heads_kv, head_dim]` F16;
/// `out` is `[n_q_tokens, n_heads_q, head_dim]` F16. Each Q token
/// applies a causal mask anchored at global position `q_offset + qi`.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_f16(
    q: &Tensor<F16>,
    k_cache: &Tensor<F16>,
    v_cache: &Tensor<F16>,
    out: &mut Tensor<F16>,
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    scale: f32,
    window_size: i32,
    ops: &HipOps<'_>,
) -> Result<()> {
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
    let cache_need = n_k_tokens * n_heads_kv * head_dim;
    if q.n_elems < q_need {
        bail!("attn_prefill_f16: q has {} F16 elems, need >= {q_need}", q.n_elems);
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
        bail!("attn_prefill_f16: out has {} F16 elems, need >= {q_need}", out.n_elems);
    }
    ops.attention_prefill_f16(
        q.ptr,
        k_cache.ptr,
        v_cache.ptr,
        out.ptr,
        n_q_tokens,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_k_tokens,
        q_offset,
        scale,
        window_size,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cpu_attn_prefill(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_q_tokens: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_k_tokens: usize,
    q_offset: usize,
    scale: f32,
    window_size: i32,
) -> Vec<f32> {
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
    use crate::testing::{
        alloc, assert_close_f16, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;
    use half::f16;

    #[allow(clippy::too_many_arguments)]
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

        let expected_f32 = cpu_attn_prefill(
            &q_kernel_f32,
            &k_kernel_f32,
            &v_kernel_f32,
            n_q_tokens,
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_k_tokens,
            q_offset,
            scale,
            window_size,
        );

        let (q_t, q_ptr) = upload::<F16, f16>(&device, &q_host_f16, q_n);
        let (k_t, k_ptr) = upload::<F16, f16>(&device, &k_host_f16, cache_n);
        let (v_t, v_ptr) = upload::<F16, f16>(&device, &v_host_f16, cache_n);
        let (mut out_t, out_ptr) = alloc::<F16>(&device, q_n);

        attn_prefill_f16(
            &q_t,
            &k_t,
            &v_t,
            &mut out_t,
            n_q_tokens,
            n_heads_q,
            n_heads_kv,
            head_dim,
            n_k_tokens,
            q_offset,
            scale,
            window_size,
            &ops,
        )
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
        // 8 Q tokens, 8 K tokens (fresh prefill), GQA 4/2, head_dim 128.
        run_case(8, 4, 2, 128, 8, 0, 0);
    }

    #[test]
    fn attn_prefill_f16_continuation() {
        // Continuation prefill: 4 new Q tokens, KV already has 4 + 4 = 8.
        run_case(4, 4, 2, 128, 8, 4, 0);
    }

    #[test]
    fn attn_prefill_f16_with_swa() {
        // Continuation at the tail of a 16-token context, 3 new Q tokens at
        // positions 13..15, SWA radius 5. `n_q_tokens < 4` routes through
        // the oracle prefill kernel (the flash_tile kernel's SWA path has
        // a NaN-init issue when `block_swa_min` is not a multiple of BC).
        run_case(3, 4, 2, 64, 16, 13, 5);
    }
}
