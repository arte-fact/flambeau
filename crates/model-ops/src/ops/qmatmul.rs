//! Quantised-weight matmul. One function per weight dtype; each
//! forwards to `flambeau_ops::Ops::qmatmul` with the right `QDtype`
//! tag.
//!
//! Shape convention: weight is `[n, k]` (one row per output channel),
//! activation is `[m, k]` Q8_1-quantised, output is `[m, n]` F32.
//! `act_q8_1` is the standard 36-B/block layout for MMVQ + 4-warp MMQ;
//! `act_q8_1_mmq` is the DS4 144-B/block layout for the LDS-tiled MMQ
//! kernel that wins at m >= 32. Decode-path callers (m = 1) can pass
//! `act_q8_1_mmq` as a null tensor; the dispatcher routes through
//! MMVQ and never reads it.

use anyhow::bail;
use flambeau_core::op::QDtype;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1};
use crate::error::Result;
use crate::tensor::Tensor;

/// `output[m, n] = weight[n, k] @ act[m, k].T` (Q8_0 weights).
pub fn qmatmul_q8_0(
    weight: &Tensor<Q8_0>,
    act_q8_1: &Tensor<Q8_1>,
    act_q8_1_mmq: &Tensor<Q8_1>,
    output: &mut Tensor<F32>,
    m: usize,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    qmatmul_dispatch(
        weight.ptr,
        act_q8_1.ptr,
        act_q8_1_mmq.ptr,
        output,
        m,
        k,
        n,
        QDtype::Q8_0,
        ops,
    )
}

/// `output[m, n] = weight[n, k] @ act[m, k].T` (Q4_0 weights).
pub fn qmatmul_q4_0(
    weight: &Tensor<Q4_0>,
    act_q8_1: &Tensor<Q8_1>,
    act_q8_1_mmq: &Tensor<Q8_1>,
    output: &mut Tensor<F32>,
    m: usize,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    qmatmul_dispatch(
        weight.ptr,
        act_q8_1.ptr,
        act_q8_1_mmq.ptr,
        output,
        m,
        k,
        n,
        QDtype::Q4_0,
        ops,
    )
}

/// `output[m, n] = weight[n, k] @ act[m, k].T` (Q4_1 weights).
pub fn qmatmul_q4_1(
    weight: &Tensor<Q4_1>,
    act_q8_1: &Tensor<Q8_1>,
    act_q8_1_mmq: &Tensor<Q8_1>,
    output: &mut Tensor<F32>,
    m: usize,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    qmatmul_dispatch(
        weight.ptr,
        act_q8_1.ptr,
        act_q8_1_mmq.ptr,
        output,
        m,
        k,
        n,
        QDtype::Q4_1,
        ops,
    )
}

/// `output[m, n] = weight[n, k] @ act[m, k].T` (Q5_0 weights).
pub fn qmatmul_q5_0(
    weight: &Tensor<Q5_0>,
    act_q8_1: &Tensor<Q8_1>,
    act_q8_1_mmq: &Tensor<Q8_1>,
    output: &mut Tensor<F32>,
    m: usize,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    qmatmul_dispatch(
        weight.ptr,
        act_q8_1.ptr,
        act_q8_1_mmq.ptr,
        output,
        m,
        k,
        n,
        QDtype::Q5_0,
        ops,
    )
}

/// `output[m, n] = weight[n, k] @ act[m, k].T` (Q5_1 weights).
pub fn qmatmul_q5_1(
    weight: &Tensor<Q5_1>,
    act_q8_1: &Tensor<Q8_1>,
    act_q8_1_mmq: &Tensor<Q8_1>,
    output: &mut Tensor<F32>,
    m: usize,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    qmatmul_dispatch(
        weight.ptr,
        act_q8_1.ptr,
        act_q8_1_mmq.ptr,
        output,
        m,
        k,
        n,
        QDtype::Q5_1,
        ops,
    )
}

#[allow(clippy::too_many_arguments)]
fn qmatmul_dispatch(
    weight_ptr: flambeau_core::DevicePtr,
    act_q8_1_ptr: flambeau_core::DevicePtr,
    act_q8_1_mmq_ptr: flambeau_core::DevicePtr,
    output: &mut Tensor<F32>,
    m: usize,
    k: usize,
    n: usize,
    dtype: QDtype,
    ops: &HipOps<'_>,
) -> Result<()> {
    if output.n_elems < m * n {
        bail!(
            "qmatmul ({dtype:?}): output has {} F32 elems, need >= {m}*{n}={}",
            output.n_elems,
            m * n
        );
    }
    ops.qmatmul(
        weight_ptr,
        act_q8_1_ptr,
        act_q8_1_mmq_ptr,
        output.ptr,
        m,
        k,
        n,
        dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quantize::quantize_f32_to_q8_1;
    use crate::testing::{
        alloc, assert_close_f32, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::{Device, DevicePtr};
    use flambeau_quant::quantize_k::quantize_row_q8_0;

    /// Q8_0 weight + Q8_1 activation. Host quantises both, uploads,
    /// runs `qmatmul_q8_0`, downloads F32 output, compares to CPU
    /// reference (dequant(weight) @ dequant(act)).
    #[test]
    fn qmatmul_q8_0_matches_cpu_reference_at_m1() {
        const M: usize = 1;
        const K: usize = 64;
        const N: usize = 8;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        // Weight: deterministic linspace, [N, K] row-major. Values
        // span [-0.5, 0.5] to exercise Q8_0's signed range.
        let weight_f32: Vec<f32> = (0..N * K)
            .map(|i| {
                let t = (i as f32) / (N * K - 1) as f32;
                t - 0.5
            })
            .collect();
        let mut weight_q8_0_bytes: Vec<u8> = Vec::new();
        for row in 0..N {
            quantize_row_q8_0(&weight_f32[row * K..(row + 1) * K], &mut weight_q8_0_bytes);
        }
        // Host-side dequant gives the CPU reference matmul's weight side.
        let weight_dequant = flambeau_quant::dequantize_to_vec(
            flambeau_quant::GgmlDType::Q8_0,
            &weight_q8_0_bytes,
            N * K,
        )
        .expect("dequant Q8_0 weight");

        // Activation: 1 row × K elems, F32.
        let act_f32: Vec<f32> = (0..M * K).map(|i| (i as f32) * 0.01 - 0.32).collect();

        // CPU reference: dequant(weight) @ act.T → [M, N] F32.
        let mut expected = vec![0.0f32; M * N];
        for mi in 0..M {
            for ni in 0..N {
                let mut acc = 0.0f32;
                for ki in 0..K {
                    acc += weight_dequant[ni * K + ki] * act_f32[mi * K + ki];
                }
                expected[mi * N + ni] = acc;
            }
        }

        // Upload weight bytes and quantise activation on device.
        let (weight_t, weight_ptr) =
            upload::<Q8_0, u8>(&device, &weight_q8_0_bytes, N * K);
        let (act_f32_t, act_f32_ptr) = upload::<F32, f32>(&device, &act_f32, act_f32.len());
        let (mut act_q8_1_t, act_q8_1_ptr) = alloc::<Q8_1>(&device, M * K);

        quantize_f32_to_q8_1(&act_f32_t, &mut act_q8_1_t, M * K, &ops).expect("quant act");

        // m=1 → MMVQ path; act_q8_1_mmq unused, pass a null tensor.
        // SAFETY: ptr is NULL and never dereferenced by the MMVQ launch.
        let act_mmq_null = unsafe { Tensor::<Q8_1>::from_raw(DevicePtr::NULL, 0) };

        let (mut out_t, out_ptr) = alloc::<F32>(&device, M * N);

        qmatmul_q8_0(
            &weight_t,
            &act_q8_1_t,
            &act_mmq_null,
            &mut out_t,
            M,
            K,
            N,
            &ops,
        )
        .expect("qmatmul_q8_0");

        let got: Vec<f32> = download::<F32, f32>(&device, &out_t);

        // Tolerance: Q8_0 weight quantum (~ max(|w_row|)/127) + Q8_1
        // activation quantum (~ 2*max(|a|)/255), accumulated over K
        // multiplies. For our scales this lands around 1e-2 abs.
        assert_close_f32(&got, &expected, 5e-2, 5e-2);

        free(&device, weight_ptr, weight_t.bytes());
        free(&device, act_f32_ptr, act_f32_t.bytes());
        free(&device, act_q8_1_ptr, act_q8_1_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
