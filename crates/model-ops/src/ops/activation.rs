//! SwiGLU `silu(gate)*up` (Qwen) and gated-GELU-tanh `gelu(gate)*up`
//! (Gemma4) for FFN paths.

use anyhow::bail;
use flambeau_ops::Ops;

use crate::dtype::{F16, F32, Q8_1};
use crate::error::Result;
use crate::tensor::Tensor;

/// `output[i] = silu(gate[i]) * up[i]`, all F16.
pub fn swiglu_f16(
    gate: &Tensor<F16>,
    up: &Tensor<F16>,
    output: &mut Tensor<F16>,
    n: usize,
    ops: &impl Ops,
) -> Result<()> {
    if gate.n_elems < n {
        bail!("swiglu_f16: gate has {} elems, need >= {n}", gate.n_elems);
    }
    if up.n_elems < n {
        bail!("swiglu_f16: up has {} elems, need >= {n}", up.n_elems);
    }
    if output.n_elems < n {
        bail!(
            "swiglu_f16: output has {} elems, need >= {n}",
            output.n_elems
        );
    }
    ops.swiglu_f16(gate.ptr, up.ptr, output.ptr, n)
}

/// `output[i] = silu(gate[i]) * up[i]`, F32 gate/up → F16 output.
pub fn swiglu_f32_to_f16(
    gate: &Tensor<F32>,
    up: &Tensor<F32>,
    output: &mut Tensor<F16>,
    n: usize,
    ops: &impl Ops,
) -> Result<()> {
    if gate.n_elems < n {
        bail!(
            "swiglu_f32_to_f16: gate has {} elems, need >= {n}",
            gate.n_elems
        );
    }
    if up.n_elems < n {
        bail!(
            "swiglu_f32_to_f16: up has {} elems, need >= {n}",
            up.n_elems
        );
    }
    if output.n_elems < n {
        bail!(
            "swiglu_f32_to_f16: output has {} elems, need >= {n}",
            output.n_elems
        );
    }
    ops.swiglu_f32_to_f16(gate.ptr, up.ptr, output.ptr, n)
}

/// `output[i] = gelu_tanh(gate[i]) * up[i]`, F32 gate/up → F16 output.
/// Tanh approximation matches `ggml_gelu_inplace`.
pub fn gelu_mul_f32_to_f16(
    gate: &Tensor<F32>,
    up: &Tensor<F32>,
    output: &mut Tensor<F16>,
    n: usize,
    ops: &impl Ops,
) -> Result<()> {
    if gate.n_elems < n {
        bail!(
            "gelu_mul_f32_to_f16: gate has {} elems, need >= {n}",
            gate.n_elems
        );
    }
    if up.n_elems < n {
        bail!(
            "gelu_mul_f32_to_f16: up has {} elems, need >= {n}",
            up.n_elems
        );
    }
    if output.n_elems < n {
        bail!(
            "gelu_mul_f32_to_f16: output has {} elems, need >= {n}",
            output.n_elems
        );
    }
    ops.gelu_f32_to_f16(gate.ptr, up.ptr, output.ptr, n)
}

/// `output = quantize_row_q8_1(gelu_tanh(gate) * up)`, F32 gate/up → Q8_1.
/// Fused twin of [`gelu_mul_f32_to_f16`] + `quantize_f16_to_q8_1`: skips the
/// F16 gated intermediate. `n` must be a multiple of 32 (QK8_1).
pub fn gelu_mul_f32_to_q8_1(
    gate: &Tensor<F32>,
    up: &Tensor<F32>,
    output: &mut Tensor<Q8_1>,
    n: usize,
    ops: &impl Ops,
) -> Result<()> {
    if gate.n_elems < n {
        bail!(
            "gelu_mul_f32_to_q8_1: gate has {} elems, need >= {n}",
            gate.n_elems
        );
    }
    if up.n_elems < n {
        bail!("gelu_mul_f32_to_q8_1: up has {} elems, need >= {n}", up.n_elems);
    }
    if output.n_elems == 0 {
        bail!("gelu_mul_f32_to_q8_1: output tensor unallocated");
    }
    if n % 32 != 0 {
        bail!("gelu_mul_f32_to_q8_1: n={n} must be a multiple of 32");
    }
    ops.gelu_mul_f32_to_q8_1(gate.ptr, up.ptr, output.ptr, n)
}

/// `output = quantize_q8_1_mmq(gelu_tanh(gate) * up)`, F32 gate/up → the DS4
/// 144-B MMQ activation layout. MMQ-layout twin of [`gelu_mul_f32_to_q8_1`]:
/// feeds the prefill down-projection MMQ directly, skipping the F16 gated
/// intermediate. `ncols` (intermediate width) must be a multiple of 128;
/// `total_b` is the token count. Inputs are `[total_b, ncols]` row-major.
pub fn gelu_mul_f32_to_q8_1_mmq(
    gate: &Tensor<F32>,
    up: &Tensor<F32>,
    output: &mut Tensor<Q8_1>,
    ncols: usize,
    total_b: usize,
    ops: &impl Ops,
) -> Result<()> {
    let n = ncols * total_b;
    if gate.n_elems < n {
        bail!(
            "gelu_mul_f32_to_q8_1_mmq: gate has {} elems, need >= {n}",
            gate.n_elems
        );
    }
    if up.n_elems < n {
        bail!("gelu_mul_f32_to_q8_1_mmq: up has {} elems, need >= {n}", up.n_elems);
    }
    if output.n_elems == 0 {
        bail!("gelu_mul_f32_to_q8_1_mmq: output tensor unallocated");
    }
    if ncols % 128 != 0 {
        bail!("gelu_mul_f32_to_q8_1_mmq: ncols={ncols} must be a multiple of 128");
    }
    ops.gelu_mul_f32_to_q8_1_mmq(gate.ptr, up.ptr, output.ptr, ncols, total_b)
}

#[cfg(test)]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
fn cpu_swiglu_f16(gate: &[half::f16], up: &[half::f16]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| silu_f32(g.to_f32()) * u.to_f32())
        .collect()
}

#[cfg(test)]
fn cpu_swiglu_f32(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu_f32(g) * u)
        .collect()
}

#[cfg(test)]
fn gelu_tanh_approx(x: f32) -> f32 {
    let k = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
fn cpu_gelu_mul_f32(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| gelu_tanh_approx(g) * u)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flambeau_ops::HipOps;
    use crate::testing::{
        alloc, assert_close_f16, assert_close_f32, download, free, test_device, test_ops_registry,
        upload,
    };
    use flambeau_core::Device;
    use half::f16;

    #[test]
    fn swiglu_f16_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let gate_host: Vec<f16> = (0..N)
            .map(|i| f16::from_f32((i as f32) * 0.02 - 2.5))
            .collect();
        let up_host: Vec<f16> = (0..N)
            .map(|i| f16::from_f32(1.0 + (i as f32) * 0.005))
            .collect();
        let expected = cpu_swiglu_f16(&gate_host, &up_host);

        let (gate_t, gate_ptr) = upload::<F16, f16>(&device, &gate_host, gate_host.len());
        let (up_t, up_ptr) = upload::<F16, f16>(&device, &up_host, up_host.len());
        let (mut out_t, out_ptr) = alloc::<F16>(&device, N);

        swiglu_f16(&gate_t, &up_t, &mut out_t, N, &ops).expect("swiglu_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        assert_close_f16(&got, &expected, 2e-3, 2e-3);

        free(&device, gate_ptr, gate_t.bytes());
        free(&device, up_ptr, up_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn swiglu_f32_to_f16_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let gate_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.02 - 2.5).collect();
        let up_host: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 0.005).collect();
        let expected = cpu_swiglu_f32(&gate_host, &up_host);

        let (gate_t, gate_ptr) = upload::<F32, f32>(&device, &gate_host, gate_host.len());
        let (up_t, up_ptr) = upload::<F32, f32>(&device, &up_host, up_host.len());
        let (mut out_t, out_ptr) = alloc::<F16>(&device, N);

        swiglu_f32_to_f16(&gate_t, &up_t, &mut out_t, N, &ops).expect("swiglu_f32_to_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        assert_close_f16(&got, &expected, 2e-3, 2e-3);

        free(&device, gate_ptr, gate_t.bytes());
        free(&device, up_ptr, up_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn gelu_mul_f32_to_f16_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let gate_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.02 - 2.5).collect();
        let up_host: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 0.005).collect();
        let expected = cpu_gelu_mul_f32(&gate_host, &up_host);

        let (gate_t, gate_ptr) = upload::<F32, f32>(&device, &gate_host, gate_host.len());
        let (up_t, up_ptr) = upload::<F32, f32>(&device, &up_host, up_host.len());
        let (mut out_t, out_ptr) = alloc::<F16>(&device, N);

        gelu_mul_f32_to_f16(&gate_t, &up_t, &mut out_t, N, &ops).expect("gelu_mul_f32_to_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        // GELU tanh approximation has slightly more tolerance than
        // SwiGLU due to the cubic term + tanh; 4e-3 covers F16
        // round-trip + approximation noise.
        assert_close_f16(&got, &expected, 4e-3, 4e-3);

        free(&device, gate_ptr, gate_t.bytes());
        free(&device, up_ptr, up_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn gelu_mul_f32_to_q8_1_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let gate_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.02 - 2.5).collect();
        let up_host: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 0.005).collect();
        let expected = cpu_gelu_mul_f32(&gate_host, &up_host);

        let (gate_t, gate_ptr) = upload::<F32, f32>(&device, &gate_host, gate_host.len());
        let (up_t, up_ptr) = upload::<F32, f32>(&device, &up_host, up_host.len());
        let (mut out_t, out_ptr) = alloc::<Q8_1>(&device, N);

        gelu_mul_f32_to_q8_1(&gate_t, &up_t, &mut out_t, N, &ops).expect("gelu_mul_f32_to_q8_1");

        let raw: Vec<u8> = download::<Q8_1, u8>(&device, &out_t);
        let got = flambeau_quant::dequantize_to_vec(flambeau_quant::GgmlDType::Q8_1, &raw, N)
            .expect("dequant q8_1");
        // Q8_1 step (~2*amax/255 per 32-block) + tanh-approx GELU noise.
        assert_close_f32(&got, &expected, 2e-2, 5e-2);

        free(&device, gate_ptr, gate_t.bytes());
        free(&device, up_ptr, up_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn gelu_mul_f32_to_q8_1_mmq_matches_cpu_reference() {
        const NCOLS: usize = 256;
        const TOTAL_B: usize = 3;
        let n = NCOLS * TOTAL_B;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let gate_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.017 - 2.1).collect();
        let up_host: Vec<f32> = (0..n).map(|i| 0.9 + ((i as f32) * 0.021).sin()).collect();
        let expected = cpu_gelu_mul_f32(&gate_host, &up_host);

        let (gate_t, gate_ptr) = upload::<F32, f32>(&device, &gate_host, gate_host.len());
        let (up_t, up_ptr) = upload::<F32, f32>(&device, &up_host, up_host.len());
        let n_big = NCOLS / 128;
        // DS4 mmq: 144 B / 128 elems == Q8_1's 36 B / 32 elems byte ratio, so
        // an `n`-element Q8_1 allocation is exactly the mmq block capacity.
        let (mut out_t, out_ptr) = alloc::<Q8_1>(&device, n);

        gelu_mul_f32_to_q8_1_mmq(&gate_t, &up_t, &mut out_t, NCOLS, TOTAL_B, &ops)
            .expect("gelu_mul_f32_to_q8_1_mmq");

        let raw: Vec<u8> = download::<Q8_1, u8>(&device, &out_t);
        let mut got = vec![0.0f32; n];
        for c in 0..TOTAL_B {
            for b in 0..n_big {
                let base = (b * TOTAL_B + c) * 144;
                for ki in 0..128 {
                    let sub = ki / 32;
                    let d = half::f16::from_le_bytes([raw[base + sub * 4], raw[base + sub * 4 + 1]])
                        .to_f32();
                    let q = raw[base + 16 + ki] as i8;
                    got[c * NCOLS + b * 128 + ki] = d * q as f32;
                }
            }
        }
        assert_close_f32(&got, &expected, 2e-2, 5e-2);

        free(&device, gate_ptr, gate_t.bytes());
        free(&device, up_ptr, up_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
