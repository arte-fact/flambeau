//! Quantised matmul, one fn per weight dtype.
//!
//! Shape: weight `[n, k]`, act `[m, k]` Q8_1, output `[m, n]` F32.
//! `act_q8_1` is the 36-B/block layout (MMVQ + 4-warp MMQ);
//! `act_q8_1_mmq` is the 144-B/block layout (LDS-tiled MMQ, m ≥ 32).
//! Decode (m=1) may pass `act_q8_1_mmq` as a null tensor — the
//! dispatcher routes through MMVQ and never reads it.

use anyhow::bail;
use flambeau_core::op::QDtype;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1};
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

/// Fused gate+up decode for Q4_0 dense FFN — one launch produces both
/// projections, sharing the Q8_1 activation HBM read. Caller guarantees
/// `n_rows_gate == n_rows_up == n` and `m == 1`.
#[allow(clippy::too_many_arguments)]
pub fn mmvq_q4_0_gate_up_t128_decode(
    gate_w: &Tensor<Q4_0>,
    up_w: &Tensor<Q4_0>,
    act_q8_1: &Tensor<Q8_1>,
    gate_out: &mut Tensor<F32>,
    up_out: &mut Tensor<F32>,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if gate_out.n_elems < n || up_out.n_elems < n {
        bail!(
            "mmvq_q4_0_gate_up_t128_decode: outputs too small (gate={}, up={}, need {n})",
            gate_out.n_elems,
            up_out.n_elems
        );
    }
    ops.mmvq_q4_0_gate_up_t128(
        gate_w.ptr,
        up_w.ptr,
        act_q8_1.ptr,
        gate_out.ptr,
        up_out.ptr,
        n,
        n,
        k,
    )
}

/// Fused K+V decode for Q4_0 attention — one launch produces both
/// projections in F16 directly, sharing the Q8_1 activation HBM read.
/// K and V must share shape `[n, k]`; caller guarantees `m == 1`.
#[allow(clippy::too_many_arguments)]
pub fn mmvq_q4_0_kv_decode_f16(
    k_w: &Tensor<Q4_0>,
    v_w: &Tensor<Q4_0>,
    act_q8_1: &Tensor<Q8_1>,
    k_out: &mut Tensor<F16>,
    v_out: &mut Tensor<F16>,
    k: usize,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if k_out.n_elems < n || v_out.n_elems < n {
        bail!(
            "mmvq_q4_0_kv_decode_f16: outputs too small (k_out={}, v_out={}, need {n})",
            k_out.n_elems,
            v_out.n_elems
        );
    }
    ops.mmvq_q4_0_kv_f16dst(k_w.ptr, v_w.ptr, act_q8_1.ptr, k_out.ptr, v_out.ptr, n, k)
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

        // Linspace [-0.5, 0.5] exercises Q8_0's signed range.
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
        let weight_dequant = flambeau_quant::dequantize_to_vec(
            flambeau_quant::GgmlDType::Q8_0,
            &weight_q8_0_bytes,
            N * K,
        )
        .expect("dequant Q8_0 weight");

        let act_f32: Vec<f32> = (0..M * K).map(|i| (i as f32) * 0.01 - 0.32).collect();

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

        let (weight_t, weight_ptr) = upload::<Q8_0, u8>(&device, &weight_q8_0_bytes, N * K);
        let (act_f32_t, act_f32_ptr) = upload::<F32, f32>(&device, &act_f32, act_f32.len());
        let (mut act_q8_1_t, act_q8_1_ptr) = alloc::<Q8_1>(&device, M * K);

        quantize_f32_to_q8_1(&act_f32_t, &mut act_q8_1_t, M * K, &ops).expect("quant act");

        // m=1 → MMVQ; `act_q8_1_mmq` unused.
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

        // Bound covers Q8_0 + Q8_1 quanta accumulated over K multiplies.
        assert_close_f32(&got, &expected, 5e-2, 5e-2);

        free(&device, weight_ptr, weight_t.bytes());
        free(&device, act_f32_ptr, act_f32_t.bytes());
        free(&device, act_q8_1_ptr, act_q8_1_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
