//! Elementwise add: `y[i] = a[i] + b[i]`. F16 / F32 variants.
//!
//! Used in residual paths. Caller-allocated output `Tensor` (no in-
//! place variant exposed; if needed later, callers can pass the same
//! tensor as `a` and `output` since the underlying kernel accepts
//! aliasing — but that contract goes on the API surface explicitly).

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::{F16, F32};
use crate::error::Result;
use crate::tensor::Tensor;

/// `output[i] = a[i] + b[i]` for `i in 0..n`, F16.
pub fn add_f16(
    a: &Tensor<F16>,
    b: &Tensor<F16>,
    output: &mut Tensor<F16>,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if a.n_elems < n {
        bail!("add_f16: a has {} elems, need >= {n}", a.n_elems);
    }
    if b.n_elems < n {
        bail!("add_f16: b has {} elems, need >= {n}", b.n_elems);
    }
    if output.n_elems < n {
        bail!("add_f16: output has {} elems, need >= {n}", output.n_elems);
    }
    ops.add_f16(a.ptr, b.ptr, output.ptr, n)
}

/// `output[i] = a[i] + b[i]` for `i in 0..n`, F32.
pub fn add_f32(
    a: &Tensor<F32>,
    b: &Tensor<F32>,
    output: &mut Tensor<F32>,
    n: usize,
    ops: &HipOps<'_>,
) -> Result<()> {
    if a.n_elems < n {
        bail!("add_f32: a has {} elems, need >= {n}", a.n_elems);
    }
    if b.n_elems < n {
        bail!("add_f32: b has {} elems, need >= {n}", b.n_elems);
    }
    if output.n_elems < n {
        bail!("add_f32: output has {} elems, need >= {n}", output.n_elems);
    }
    ops.add_f32(a.ptr, b.ptr, output.ptr, n)
}

#[cfg(test)]
fn cpu_add_f16(a: &[half::f16], b: &[half::f16]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.to_f32() + y.to_f32())
        .collect()
}

#[cfg(test)]
fn cpu_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
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
    fn add_f16_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let a_host: Vec<f16> = (0..N).map(|i| f16::from_f32((i as f32) * 0.1)).collect();
        let b_host: Vec<f16> = (0..N).map(|i| f16::from_f32((i as f32) * -0.05 + 1.0)).collect();
        let expected = cpu_add_f16(&a_host, &b_host);

        let (a_t, a_ptr) = upload::<F16, f16>(&device, &a_host);
        let (b_t, b_ptr) = upload::<F16, f16>(&device, &b_host);
        let (mut out_t, out_ptr) = alloc::<F16>(&device, N);

        add_f16(&a_t, &b_t, &mut out_t, N, &ops).expect("add_f16");

        let got: Vec<f16> = download::<F16, f16>(&device, &out_t);
        assert_close_f16(&got, &expected, 1e-3, 1e-3);

        free(&device, a_ptr, a_t.bytes());
        free(&device, b_ptr, b_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }

    #[test]
    fn add_f32_matches_cpu_reference() {
        const N: usize = 256;
        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let a_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.1).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i as f32) * -0.05 + 1.0).collect();
        let expected = cpu_add_f32(&a_host, &b_host);

        let (a_t, a_ptr) = upload::<F32, f32>(&device, &a_host);
        let (b_t, b_ptr) = upload::<F32, f32>(&device, &b_host);
        let (mut out_t, out_ptr) = alloc::<F32>(&device, N);

        add_f32(&a_t, &b_t, &mut out_t, N, &ops).expect("add_f32");

        let got: Vec<f32> = download::<F32, f32>(&device, &out_t);
        assert_close_f32(&got, &expected, 1e-6, 1e-6);

        free(&device, a_ptr, a_t.bytes());
        free(&device, b_ptr, b_t.bytes());
        free(&device, out_ptr, out_t.bytes());
    }
}
