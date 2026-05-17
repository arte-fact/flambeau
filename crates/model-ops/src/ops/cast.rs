//! F16 ↔ F32 casts.

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, F32};
use crate::error::Result;
use crate::tensor::Tensor;

/// Saturating cast: Inf/NaN preserved; out-of-range clamps.
pub fn cast_f32_to_f16(
    input: &Tensor<F32>,
    output: &mut Tensor<F16>,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if input.n_elems < n {
        bail!(
            "cast_f32_to_f16: input has {} F32 elems, need >= {n}",
            input.n_elems
        );
    }
    if output.n_elems < n {
        bail!(
            "cast_f32_to_f16: output has {} F16 elems, need >= {n}",
            output.n_elems
        );
    }
    ops.cast_f32_to_f16(input.ptr, output.ptr, n)
}

/// `output[i] = input[i] as f32`.
pub fn cast_f16_to_f32(
    input: &Tensor<F16>,
    output: &mut Tensor<F32>,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if input.n_elems < n {
        bail!(
            "cast_f16_to_f32: input has {} F16 elems, need >= {n}",
            input.n_elems
        );
    }
    if output.n_elems < n {
        bail!(
            "cast_f16_to_f32: output has {} F32 elems, need >= {n}",
            output.n_elems
        );
    }
    ops.cast_f16_to_f32(input.ptr, output.ptr, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        alloc, assert_close_f16, assert_close_f32, download, free, test_device, test_ops_registry,
        upload,
    };
    use flambeau_core::Device;
    use half::f16;

    #[test]
    fn cast_f32_to_f16_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let input_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.05 - 6.4).collect();
        let expected: Vec<f32> = input_host.iter().map(|x| f16::from_f32(*x).to_f32()).collect();

        let (input_t, input_ptr) = upload::<F32, f32>(&device, &input_host, input_host.len());
        let (mut out_t, out_ptr) = alloc::<F16>(&device, N);

        cast_f32_to_f16(&input_t, &mut out_t, N, &ops).expect("cast_f32_to_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        // Cast is exact within F16 precision; allow only F16 quantum.
        assert_close_f16(&got, &expected, 1e-3, 0.0);

        free(&device, input_ptr, input_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn cast_f16_to_f32_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let input_host: Vec<f16> = (0..N)
            .map(|i| f16::from_f32((i as f32) * 0.05 - 6.4))
            .collect();
        let expected: Vec<f32> = input_host.iter().map(|x| x.to_f32()).collect();

        let (input_t, input_ptr) = upload::<F16, f16>(&device, &input_host, input_host.len());
        let (mut out_t, out_ptr) = alloc::<F32>(&device, N);

        cast_f16_to_f32(&input_t, &mut out_t, N, &ops).expect("cast_f16_to_f32");

        let got: Vec<f32> = download::<F32, f32>(&device, &out_t);
        // F16 → F32 widening is exact.
        assert_close_f32(&got, &expected, 0.0, 0.0);

        free(&device, input_ptr, input_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
