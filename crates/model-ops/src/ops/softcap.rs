//! Final-logit softcap: `y[i] = tanh(x[i] / cap) * cap`. In-place safe.

use anyhow::bail;
use flambeau_ops::Ops;

use crate::dtype::F32;
use crate::error::Result;
use crate::tensor::Tensor;

pub fn apply_softcap_f32(
    input: &Tensor<F32>,
    output: &mut Tensor<F32>,
    n: usize,
    cap: f32,
    ops: &impl Ops,
) -> Result<()> {
    if input.n_elems < n {
        bail!(
            "apply_softcap_f32: input has {} elems, need >= {n}",
            input.n_elems
        );
    }
    if output.n_elems < n {
        bail!(
            "apply_softcap_f32: output has {} elems, need >= {n}",
            output.n_elems
        );
    }
    ops.apply_softcap_f32(input.ptr, output.ptr, n, cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flambeau_ops::HipOps;
    use crate::testing::{
        alloc, assert_close_f32, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;

    #[test]
    fn apply_softcap_f32_matches_cpu_reference() {
        const N: usize = 128;
        const CAP: f32 = 30.0;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let input_host: Vec<f32> = (0..N)
            .map(|i| {
                let t = (i as f32) / (N - 1) as f32;
                100.0 * t - 50.0
            })
            .collect();
        let expected: Vec<f32> = input_host.iter().map(|&x| (x / CAP).tanh() * CAP).collect();

        let (input_t, input_ptr) = upload::<F32, f32>(&device, &input_host, input_host.len());
        let (mut out_t, out_ptr) = alloc::<F32>(&device, N);

        apply_softcap_f32(&input_t, &mut out_t, N, CAP, &ops).expect("apply_softcap_f32");

        let got: Vec<f32> = download::<F32, f32>(&device, &out_t);
        assert_close_f32(&got, &expected, 1e-5, 1e-5);

        free(&device, input_ptr, input_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
