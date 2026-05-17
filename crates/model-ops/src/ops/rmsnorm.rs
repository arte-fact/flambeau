//! `rmsnorm_f16` — row-wise RMSNorm with F16 input/weight/output.
//!
//! `y[r, c] = x[r, c] / sqrt(mean(x[r, ·]²) + eps) * weight[c]`
//!
//! All tensors row-major, contiguous, F16. One thread block per row,
//! 256 threads/row (matches the underlying kernel's launch shape;
//! caller doesn't need to know).

use anyhow::bail;
use flambeau_ops::{HipOps, Ops};

use crate::dtype::F16;
use crate::error::Result;
use crate::tensor::Tensor;

/// `output[r, c] = input[r, c] / sqrt(mean(input[r, ·]²) + eps) * weight[c]`.
///
/// Shapes:
/// - `input`:  `[n_rows, hidden]`, F16
/// - `weight`: `[hidden]`,         F16
/// - `output`: `[n_rows, hidden]`, F16 (caller-allocated; written in place)
pub fn rmsnorm_f16(
    input: &Tensor<F16>,
    weight: &Tensor<F16>,
    output: &mut Tensor<F16>,
    n_rows: usize,
    hidden: usize,
    eps: f32,
    ops: &HipOps<'_>,
) -> Result<()> {
    let need = n_rows * hidden;
    if input.n_elems < need {
        bail!(
            "rmsnorm_f16: input has {} F16 elems, need >= {} ({n_rows}*{hidden})",
            input.n_elems,
            need
        );
    }
    if weight.n_elems < hidden {
        bail!(
            "rmsnorm_f16: weight has {} F16 elems, need >= {hidden}",
            weight.n_elems,
        );
    }
    if output.n_elems < need {
        bail!(
            "rmsnorm_f16: output has {} F16 elems, need >= {}",
            output.n_elems,
            need
        );
    }
    ops.rmsnorm_f16(input.ptr, weight.ptr, output.ptr, n_rows, hidden, eps)
}

/// CPU reference. Plain Rust over F16 inputs, F32 accumulation. Used
/// by the parity test below. Not exported — tests are the only consumer.
#[cfg(test)]
fn cpu_rmsnorm_f16(
    input: &[half::f16],
    weight: &[half::f16],
    n_rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * hidden];
    for r in 0..n_rows {
        let row = &input[r * hidden..(r + 1) * hidden];
        let sum_sq: f32 = row.iter().map(|x| {
            let v = x.to_f32();
            v * v
        }).sum();
        let rms = (sum_sq / hidden as f32 + eps).sqrt();
        for c in 0..hidden {
            out[r * hidden + c] = row[c].to_f32() / rms * weight[c].to_f32();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        alloc, assert_close_f16, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_backend_hip::HipStream;
    use flambeau_core::Device;
    use half::f16;

    /// Deterministic synthetic input on the device, parity vs CPU
    /// reference. Small dims to keep the test cheap and the failure
    /// mode (if any) easy to localise.
    #[test]
    fn rmsnorm_f16_matches_cpu_reference() {
        const N_ROWS: usize = 4;
        const HIDDEN: usize = 64;
        const EPS: f32 = 1e-5;

        let device = test_device();
        device.bind().expect("device bind");
        let stream: &HipStream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        // Deterministic input: linspace [-1, 1] across the flat buffer,
        // wrapped to F16. Avoids zeros (which would make RMS ill-defined)
        // and stays within F16 dynamic range.
        let input_host: Vec<f16> = (0..N_ROWS * HIDDEN)
            .map(|i| {
                let t = (i as f32) / (N_ROWS * HIDDEN - 1) as f32;
                f16::from_f32(2.0 * t - 1.0)
            })
            .collect();
        // Weight near 1.0 + small variation, so failures localise to
        // rmsnorm logic vs weight scaling.
        let weight_host: Vec<f16> = (0..HIDDEN)
            .map(|c| f16::from_f32(1.0 + (c as f32) * 0.01))
            .collect();

        let expected = cpu_rmsnorm_f16(&input_host, &weight_host, N_ROWS, HIDDEN, EPS);

        let (input_t, input_ptr) = upload::<F16, f16>(&device, &input_host);
        let (weight_t, weight_ptr) = upload::<F16, f16>(&device, &weight_host);
        let (mut output_t, output_ptr) = alloc::<F16>(&device, N_ROWS * HIDDEN);

        rmsnorm_f16(
            &input_t,
            &weight_t,
            &mut output_t,
            N_ROWS,
            HIDDEN,
            EPS,
            &ops,
        )
        .expect("rmsnorm_f16 launch");

        let got: Vec<f16> = download::<F16, f16>(&device, &output_t);

        // F16 tolerance: rmsnorm involves a sqrt + division, so we
        // allow ~1e-3 absolute or ~1e-3 relative. Tighter than that
        // and F16 round-trip noise dominates.
        assert_close_f16(&got, &expected, 1e-3, 1e-3);

        free(&device, input_ptr, input_t.bytes());
        free(&device, weight_ptr, weight_t.bytes());
        free(&device, output_ptr, output_t.bytes());
    }
}
