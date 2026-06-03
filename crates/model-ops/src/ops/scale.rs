//! Elementwise `y[i] = x[i] * scale`, F16. In-place safe (output may
//! alias input).

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

pub fn scale_f16(
    input: &Tensor<F16>,
    output: &mut Tensor<F16>,
    n: usize,
    scale: f32,
    ops: &HipOps<'_>,
) -> Result<()> {
    if input.n_elems < n {
        bail!("scale_f16: input has {} elems, need >= {n}", input.n_elems);
    }
    if output.n_elems < n {
        bail!(
            "scale_f16: output has {} elems, need >= {n}",
            output.n_elems
        );
    }
    ops.scale_f16(input.ptr, output.ptr, n, scale)
}

#[cfg(test)]
fn cpu_scale_f16(input: &[half::f16], scale: f32) -> Vec<f32> {
    input.iter().map(|x| x.to_f32() * scale).collect()
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
    fn scale_f16_matches_cpu_reference() {
        const N: usize = 256;
        const SCALE: f32 = std::f32::consts::SQRT_2;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let input_host: Vec<f16> = (0..N)
            .map(|i| {
                let t = (i as f32) / (N - 1) as f32;
                f16::from_f32(2.0 * t - 1.0)
            })
            .collect();
        let expected = cpu_scale_f16(&input_host, SCALE);

        let (input_t, input_ptr) = upload::<F16, f16>(&device, &input_host, input_host.len());
        let (mut out_t, out_ptr) = alloc::<F16>(&device, N);

        scale_f16(&input_t, &mut out_t, N, SCALE, &ops).expect("scale_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        assert_close_f16(&got, &expected, 1e-3, 1e-3);

        free(&device, input_ptr, input_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
