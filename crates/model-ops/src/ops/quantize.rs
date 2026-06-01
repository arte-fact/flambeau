//! F32 / F16 → Q8_1 (MMVQ decode layout). Q8_1 block:
//! `[d (fp16), s (fp16), qs[32] (i8)]`, 36 B / 32 elems.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, F32, Q8_1};
use crate::error::Result;
use crate::tensor::Tensor;

pub fn quantize_f32_to_q8_1(
    input: &Tensor<F32>,
    output: &mut Tensor<Q8_1>,
    n_elems: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if input.n_elems < n_elems {
        bail!(
            "quantize_f32_to_q8_1: input has {} F32 elems, need >= {n_elems}",
            input.n_elems
        );
    }
    if output.n_elems == 0 {
        bail!("quantize_f32_to_q8_1: output tensor unallocated");
    }
    ops.quantize_q8_1(input.ptr, output.ptr, n_elems)
}

pub fn quantize_f16_to_q8_1(
    input: &Tensor<F16>,
    output: &mut Tensor<Q8_1>,
    n_elems: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if input.n_elems < n_elems {
        bail!(
            "quantize_f16_to_q8_1: input has {} F16 elems, need >= {n_elems}",
            input.n_elems
        );
    }
    if output.n_elems == 0 {
        bail!("quantize_f16_to_q8_1: output tensor unallocated");
    }
    ops.quantize_f16_q8_1(input.ptr, output.ptr, n_elems)
}

/// MMQ-layout quantize: `ncols` columns per row, `total_b` rows. Output
/// has 144 B per 128-element block, row-major over the `(ncols/128, total_b)` grid.
/// `ncols` must be a multiple of 128.
pub fn quantize_f16_to_q8_1_mmq(
    input: &Tensor<F16>,
    output: &mut Tensor<Q8_1>,
    ncols: usize,
    total_b: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if input.n_elems < ncols * total_b {
        bail!(
            "quantize_f16_to_q8_1_mmq: input has {} F16 elems, need >= {}",
            input.n_elems,
            ncols * total_b
        );
    }
    if output.n_elems == 0 {
        bail!("quantize_f16_to_q8_1_mmq: output tensor unallocated");
    }
    ops.quantize_f16_q8_1_mmq(input.ptr, output.ptr, ncols, total_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        alloc, assert_close_f32, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;
    use half::f16;

    #[test]
    fn quantize_f32_to_q8_1_matches_gguf_dequant() {
        const N: usize = 128;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let input_host: Vec<f32> = (0..N)
            .map(|i| {
                let t = (i as f32) / (N - 1) as f32;
                4.0 * t - 2.0
            })
            .collect();

        let (input_t, input_ptr) = upload::<F32, f32>(&device, &input_host, input_host.len());
        let (mut out_t, out_ptr) = alloc::<Q8_1>(&device, N);

        quantize_f32_to_q8_1(&input_t, &mut out_t, N, &ops).expect("quantize_f32_to_q8_1");

        let raw: Vec<u8> = download::<Q8_1, u8>(&device, &out_t);
        let got = flambeau_quant::dequantize_to_vec(flambeau_quant::GgmlDType::Q8_1, &raw, N)
            .expect("dequant q8_1");

        // Q8_1 round-trip tolerance: ~ 2*max(|input|)/255 per block.
        // For inputs spanning [-2, 2], that's ~ 0.016 abs. Allow a bit
        // more for boundary effects on the small block count (4 blocks).
        assert_close_f32(&got, &input_host, 2e-2, 5e-2);

        free(&device, input_ptr, input_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn quantize_f16_to_q8_1_matches_gguf_dequant() {
        const N: usize = 128;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let input_host: Vec<f16> = (0..N)
            .map(|i| {
                let t = (i as f32) / (N - 1) as f32;
                f16::from_f32(4.0 * t - 2.0)
            })
            .collect();
        let expected_f32: Vec<f32> = input_host.iter().map(|x| x.to_f32()).collect();

        let (input_t, input_ptr) = upload::<F16, f16>(&device, &input_host, input_host.len());
        let (mut out_t, out_ptr) = alloc::<Q8_1>(&device, N);

        quantize_f16_to_q8_1(&input_t, &mut out_t, N, &ops).expect("quantize_f16_to_q8_1");

        let raw: Vec<u8> = download::<Q8_1, u8>(&device, &out_t);
        let got = flambeau_quant::dequantize_to_vec(flambeau_quant::GgmlDType::Q8_1, &raw, N)
            .expect("dequant q8_1");

        assert_close_f32(&got, &expected_f32, 2e-2, 5e-2);

        free(&device, input_ptr, input_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
