//! Gated-attention helpers: deinterleave a fused `[Q | gate]` per-head
//! projection, and apply a sigmoid gate to the post-attention output.
//! Used by qwen3.5 / qwen3.6 / qwen3-Next full-attention layers.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// Deinterleave a fused per-head `[Q | gate]` F16 projection into two
/// contiguous tensors. `fused_qg` has shape
/// `[n_tokens, n_heads, 2 * head_dim]`; `q_out` and `gate_out` are
/// `[n_tokens, n_heads, head_dim]`.
pub fn split_q_gate_f16(
    fused_qg: &Tensor<F16>,
    q_out: &mut Tensor<F16>,
    gate_out: &mut Tensor<F16>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    let fused_need = n_tokens * n_heads * 2 * head_dim;
    let split_need = n_tokens * n_heads * head_dim;
    if fused_qg.n_elems < fused_need {
        bail!(
            "split_q_gate_f16: fused has {} elems, need >= {fused_need}",
            fused_qg.n_elems
        );
    }
    if q_out.n_elems < split_need {
        bail!(
            "split_q_gate_f16: q_out has {} elems, need >= {split_need}",
            q_out.n_elems
        );
    }
    if gate_out.n_elems < split_need {
        bail!(
            "split_q_gate_f16: gate_out has {} elems, need >= {split_need}",
            gate_out.n_elems
        );
    }
    ops.split_q_gate_f16(
        fused_qg.ptr,
        q_out.ptr,
        gate_out.ptr,
        n_tokens,
        n_heads,
        head_dim,
    )
}

/// In-place sigmoid gate: `y[i] = sigmoid(gate[i]) * x[i]` in F16.
/// `y` may alias `x` (used by the gated-attention path to write the
/// gated output back into `attn_out_f16`).
pub fn sigmoid_mul_f16(
    gate: &Tensor<F16>,
    x: &Tensor<F16>,
    y: &mut Tensor<F16>,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if gate.n_elems < n {
        bail!(
            "sigmoid_mul_f16: gate has {} elems, need >= {n}",
            gate.n_elems
        );
    }
    if x.n_elems < n {
        bail!("sigmoid_mul_f16: x has {} elems, need >= {n}", x.n_elems);
    }
    if y.n_elems < n {
        bail!("sigmoid_mul_f16: y has {} elems, need >= {n}", y.n_elems);
    }
    ops.sigmoid_mul_f16(gate.ptr, x.ptr, y.ptr, n)
}

#[cfg(test)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
fn cpu_sigmoid_mul_f16(gate: &[half::f16], x: &[half::f16]) -> Vec<f32> {
    gate.iter()
        .zip(x.iter())
        .map(|(g, x)| sigmoid(g.to_f32()) * x.to_f32())
        .collect()
}

#[cfg(test)]
fn cpu_split_q_gate_f16(
    fused: &[half::f16],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> (Vec<half::f16>, Vec<half::f16>) {
    let split_n = n_tokens * n_heads * head_dim;
    let mut q = Vec::with_capacity(split_n);
    let mut gate = Vec::with_capacity(split_n);
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let fused_base = (t * n_heads + h) * (2 * head_dim);
            for d in 0..head_dim {
                q.push(fused[fused_base + d]);
                gate.push(fused[fused_base + head_dim + d]);
            }
        }
    }
    (q, gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        alloc, assert_close_f16, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;
    use half::f16;

    #[test]
    fn sigmoid_mul_f16_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let gate_host: Vec<f16> = (0..N)
            .map(|i| f16::from_f32((i as f32) * 0.03 - 4.0))
            .collect();
        let x_host: Vec<f16> = (0..N)
            .map(|i| f16::from_f32(1.0 + (i as f32) * 0.005))
            .collect();
        let expected = cpu_sigmoid_mul_f16(&gate_host, &x_host);

        let (gate_t, gate_ptr) = upload::<F16, f16>(&device, &gate_host, gate_host.len());
        let (x_t, x_ptr) = upload::<F16, f16>(&device, &x_host, x_host.len());
        let (mut y_t, y_ptr) = alloc::<F16>(&device, N);

        sigmoid_mul_f16(&gate_t, &x_t, &mut y_t, N, &ops).expect("sigmoid_mul_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &y_t);
        assert_close_f16(&got, &expected, 3e-3, 3e-3);

        free(&device, gate_ptr, gate_t.bytes());
        free(&device, x_ptr, x_t.bytes());
        free(&device, y_ptr, y_t.bytes());
    }

    #[test]
    fn split_q_gate_f16_matches_cpu_reference() {
        const N_TOKENS: usize = 2;
        const N_HEADS: usize = 4;
        const HEAD_DIM: usize = 16;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let fused_n = N_TOKENS * N_HEADS * 2 * HEAD_DIM;
        let fused_host: Vec<f16> = (0..fused_n)
            .map(|i| f16::from_f32((i as f32) * 0.01 - 1.0))
            .collect();
        let (exp_q, exp_gate) =
            cpu_split_q_gate_f16(&fused_host, N_TOKENS, N_HEADS, HEAD_DIM);

        let (fused_t, fused_ptr) = upload::<F16, f16>(&device, &fused_host, fused_n);
        let split_n = N_TOKENS * N_HEADS * HEAD_DIM;
        let (mut q_t, q_ptr) = alloc::<F16>(&device, split_n);
        let (mut gate_t, gate_ptr) = alloc::<F16>(&device, split_n);

        split_q_gate_f16(
            &fused_t, &mut q_t, &mut gate_t, N_TOKENS, N_HEADS, HEAD_DIM, &ops,
        )
        .expect("split_q_gate_f16");

        let got_q: Vec<f16> = download::<F16, f16>(&device, &q_t);
        let got_gate: Vec<f16> = download::<F16, f16>(&device, &gate_t);
        // Deinterleave is bit-exact (no math); F16 → F16 with same bits.
        let exp_q_f32: Vec<f32> = exp_q.iter().map(|v| v.to_f32()).collect();
        let exp_gate_f32: Vec<f32> = exp_gate.iter().map(|v| v.to_f32()).collect();
        assert_close_f16(&got_q, &exp_q_f32, 0.0, 0.0);
        assert_close_f16(&got_gate, &exp_gate_f32, 0.0, 0.0);

        free(&device, fused_ptr, fused_t.bytes());
        free(&device, q_ptr, q_t.bytes());
        free(&device, gate_ptr, gate_t.bytes());
    }
}
