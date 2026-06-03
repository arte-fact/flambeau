//! In-place RoPE on F16 `[n_tokens, n_heads, head_dim]`.
//!
//! - `rope_f16` — interleaved pairs `(x[2i], x[2i+1])` (Gemma4).
//! - `rope_neox_partial_f16` — split pairs `(x[i], x[i + rot/2])`
//!   over the first `rotated_dims` of `head_dim` (Qwen3.x / Next
//!   full-attn). Positions are `I32`, one per token. Caller invokes
//!   separately for Q and K.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, I32};
use crate::error::Result;
use crate::tensor::Tensor;

/// `head_dim` must be even.
pub fn rope_f16(
    x: &mut Tensor<F16>,
    positions: &Tensor<I32>,
    theta_base: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if head_dim % 2 != 0 {
        bail!("rope_f16: head_dim ({head_dim}) must be even");
    }
    let need = n_tokens * n_heads * head_dim;
    if x.n_elems < need {
        bail!("rope_f16: x has {} F16 elems, need >= {need}", x.n_elems);
    }
    if positions.n_elems < n_tokens {
        bail!(
            "rope_f16: positions has {} I32 elems, need >= {n_tokens}",
            positions.n_elems
        );
    }
    ops.rope_f16(
        flambeau_ops::RopeBuffers {
            x: x.ptr,
            positions: positions.ptr,
        },
        flambeau_ops::RopeShape {
            n_tokens,
            n_heads,
            head_dim,
        },
        theta_base,
    )
}

/// Dims `rotated_dims..head_dim` pass through. `rotated_dims` must
/// be even and ≤ `head_dim`.
pub fn rope_neox_partial_f16(
    x: &mut Tensor<F16>,
    positions: &Tensor<I32>,
    theta_base: f32,
    shape: flambeau_ops::RopePartialShape,
    ops: &HipOps<'_>,
) -> Result<()> {
    let flambeau_ops::RopePartialShape { n_tokens, n_heads, head_dim, rotated_dims } = shape;
    if rotated_dims % 2 != 0 {
        bail!("rope_neox_partial_f16: rotated_dims ({rotated_dims}) must be even");
    }
    if rotated_dims > head_dim {
        bail!("rope_neox_partial_f16: rotated_dims ({rotated_dims}) > head_dim ({head_dim})");
    }
    let need = n_tokens * n_heads * head_dim;
    if x.n_elems < need {
        bail!(
            "rope_neox_partial_f16: x has {} F16 elems, need >= {need}",
            x.n_elems
        );
    }
    if positions.n_elems < n_tokens {
        bail!(
            "rope_neox_partial_f16: positions has {} I32 elems, need >= {n_tokens}",
            positions.n_elems
        );
    }
    ops.rope_neox_partial_f16(
        flambeau_ops::RopeBuffers {
            x: x.ptr,
            positions: positions.ptr,
        },
        shape,
        theta_base,
    )
}

#[cfg(test)]
fn cpu_rope_interleaved(
    x: &mut [f32],
    positions: &[i32],
    theta_base: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    for (t, &pos) in positions.iter().enumerate().take(n_tokens) {
        for h in 0..n_heads {
            for pair in 0..head_dim / 2 {
                let exponent = 2.0_f32 * (pair as f32) / (head_dim as f32);
                let inv_freq = 1.0_f32 / theta_base.powf(exponent);
                let angle = (pos as f32) * inv_freq;
                let c = angle.cos();
                let s = angle.sin();
                let base = (t * n_heads + h) * head_dim + 2 * pair;
                let x0 = x[base];
                let x1 = x[base + 1];
                x[base] = x0 * c - x1 * s;
                x[base + 1] = x0 * s + x1 * c;
            }
        }
    }
}

#[cfg(test)]
fn cpu_rope_neox_partial(
    x: &mut [f32],
    positions: &[i32],
    theta_base: f32,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rotated_dims: usize,
) {
    let half = rotated_dims / 2;
    for (t, &pos) in positions.iter().enumerate().take(n_tokens) {
        for h in 0..n_heads {
            let head_base = (t * n_heads + h) * head_dim;
            for pair in 0..half {
                let exponent = 2.0_f32 * (pair as f32) / (rotated_dims as f32);
                let inv_freq = 1.0_f32 / theta_base.powf(exponent);
                let angle = (pos as f32) * inv_freq;
                let c = angle.cos();
                let s = angle.sin();
                let lo = head_base + pair;
                let hi = head_base + pair + half;
                let x0 = x[lo];
                let x1 = x[hi];
                x[lo] = x0 * c - x1 * s;
                x[hi] = x0 * s + x1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        assert_close_f16, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;
    use half::f16;

    #[test]
    fn rope_f16_matches_cpu_reference() {
        const N_TOKENS: usize = 4;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        const THETA: f32 = 10000.0;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let total = N_TOKENS * N_HEADS * HEAD_DIM;
        let x_host_f32: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.01 - 0.5).sin())
            .collect();
        let x_host_f16: Vec<f16> = x_host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let positions: Vec<i32> = (0..N_TOKENS as i32).collect();

        let mut expected_f32 = x_host_f32.clone();
        let expected_inputs: Vec<f32> = x_host_f16.iter().map(|v| v.to_f32()).collect();
        expected_f32.copy_from_slice(&expected_inputs);
        cpu_rope_interleaved(
            &mut expected_f32,
            &positions,
            THETA,
            N_TOKENS,
            N_HEADS,
            HEAD_DIM,
        );

        let (mut x_t, x_ptr) = upload::<F16, f16>(&device, &x_host_f16, total);
        let (pos_t, pos_ptr) = upload::<I32, i32>(&device, &positions, N_TOKENS);

        rope_f16(&mut x_t, &pos_t, THETA, N_TOKENS, N_HEADS, HEAD_DIM, &ops).expect("rope_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &x_t);
        assert_close_f16(&got, &expected_f32, 3e-3, 3e-3);

        free(&device, x_ptr, x_t.bytes());
        free(&device, pos_ptr, pos_t.bytes());
    }

    #[test]
    fn rope_neox_partial_f16_matches_cpu_reference() {
        const N_TOKENS: usize = 4;
        const N_HEADS: usize = 2;
        const HEAD_DIM: usize = 128;
        const ROTATED_DIMS: usize = 64;
        const THETA: f32 = 1_000_000.0;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let total = N_TOKENS * N_HEADS * HEAD_DIM;
        let x_host_f32: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.013 - 0.7).cos())
            .collect();
        let x_host_f16: Vec<f16> = x_host_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let positions: Vec<i32> = vec![0, 7, 31, 128];

        let mut expected_f32: Vec<f32> = x_host_f16.iter().map(|v| v.to_f32()).collect();
        cpu_rope_neox_partial(
            &mut expected_f32,
            &positions,
            THETA,
            N_TOKENS,
            N_HEADS,
            HEAD_DIM,
            ROTATED_DIMS,
        );

        let (mut x_t, x_ptr) = upload::<F16, f16>(&device, &x_host_f16, total);
        let (pos_t, pos_ptr) = upload::<I32, i32>(&device, &positions, N_TOKENS);

        rope_neox_partial_f16(
            &mut x_t,
            &pos_t,
            THETA,
            flambeau_ops::RopePartialShape {
                n_tokens: N_TOKENS,
                n_heads: N_HEADS,
                head_dim: HEAD_DIM,
                rotated_dims: ROTATED_DIMS,
            },
            &ops,
        )
        .expect("rope_neox_partial_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &x_t);
        assert_close_f16(&got, &expected_f32, 3e-3, 3e-3);

        free(&device, x_ptr, x_t.bytes());
        free(&device, pos_ptr, pos_t.bytes());
    }
}
