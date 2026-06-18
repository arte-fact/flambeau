//! Cross-backend (BenchBackend-generic) correctness sweeps for the Stream F
//! fixed pointwise kernels. The legacy `sweep_f32_pointwise` is HIP-only
//! (`open_kernel` + hardcoded `_gfx906` certs); this module is the P3 spine —
//! the same sweeps over `BenchBackend` so each kernel certs on HIP gfx906 AND
//! CUDA sm_86 from one driver. Pilot: `add_f16`.
#![cfg(any(feature = "hip", feature = "cuda"))]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "sweep harness — every unsafe block is a memcpy_async over buffers \
              allocated locally in the same function and freed before return."
)]

use std::path::Path;

use anyhow::{anyhow, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, LaunchCfg, Stream};
use half::f16;

use crate::backend::{BenchBackend, LaunchArgs};
use crate::cert::{now_utc_iso8601, Cert, ShapeResult, SCHEMA_VERSION};
use crate::harness::{alloc_and_upload, max_rel_err_with_floor, rig_for, seeded_f32_range};

/// Route a Stream F pointwise `--op` to its generic sweep, for either backend.
/// Returns `None` when `op` is not a pointwise-spine kernel (the caller then
/// falls through to its other sweep tables). One table, both rigs — the cli's
/// HIP and CUDA arms share this so a newly-ported kernel is wired in one place.
pub fn run_pointwise_sweep<B: BenchBackend>(
    op: &str,
    repo_root: &Path,
    arch: &str,
) -> Option<Result<Cert>> {
    let cert = match op {
        "add_f16" => run_add_f16_sweep::<B>(repo_root, arch),
        "cast_f16_f32" => run_cast_f16_f32_sweep::<B>(repo_root, arch),
        "cast_f32_f16" => run_cast_f32_f16_sweep::<B>(repo_root, arch),
        "scale_f32" => run_scale_f32_sweep::<B>(repo_root, arch),
        "scale_f16" => run_scale_f16_sweep::<B>(repo_root, arch),
        "silu_f32" => run_silu_f32_sweep::<B>(repo_root, arch),
        "apply_softcap_f32" => run_apply_softcap_f32_sweep::<B>(repo_root, arch),
        _ => return None,
    };
    Some(cert)
}

/// `add_f16` — pointwise `y[i] = (f32)a[i] + (f32)b[i] → f16`. Bit-exact vs the
/// F32-add-then-F16-cast host reference.
pub fn run_add_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("add_f16").ok_or_else(|| anyhow!("add_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_add_f16")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [64usize, 256, 2048, 4096, 8192];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64) * 83 + 23);
        let a_f32 = seeded_f32_range(seed, n, -0.5, 0.5);
        let b_f32 = seeded_f32_range(seed ^ 0xBE, n, -0.5, 0.5);
        let a_f16: Vec<f16> = a_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let b_f16: Vec<f16> = b_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let reference: Vec<f16> = a_f16
            .iter()
            .zip(&b_f16)
            .map(|(a, b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect();

        let d_a = alloc_and_upload(&dev, &a_f16);
        let d_b = alloc_and_upload(&dev, &b_f16);
        let d_y = dev.alloc(n * 2)?;
        {
            let stream = dev.default_stream();
            let n_i = n as i32;
            let a_ptr: u64 = d_a.as_usize() as u64;
            let b_ptr: u64 = d_b.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&a_ptr);
            args.push_arg(&b_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&n_i);
            B::launch(&kernel, stream, LaunchCfg::one_d((n as u32).div_ceil(256), 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![f16::from_f32(0.0); n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * 2,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_a, n * 2)?;
            dev.dealloc(d_b, n * 2)?;
            dev.dealloc(d_y, n * 2)?;
        }

        let mismatches = got
            .iter()
            .zip(&reference)
            .filter(|(g, r)| g.to_bits() != r.to_bits())
            .count();
        let err = mismatches as f32;
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 0.0,
            pass: err == 0.0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "add_f16".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "add_f16".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact bit match vs (f32) a + (f32) b → f16".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// Shared driver for the single-input pointwise leaves: one input buffer, one
/// output buffer, an optional trailing `f32` scalar, `ceil(n/256)×256` launch.
/// `IE`/`OE` are the input/output element types (`f16` or `f32`); `make_input`
/// builds the device input, `reference` the host oracle, `score` returns
/// `(err, tolerance, pass)` for one shape.
#[expect(clippy::too_many_arguments, reason = "explicit cert-metadata spine; \
         each field is a distinct cert column, not a struct worth naming.")]
fn run_unary_sweep<B, IE, OE>(
    repo_root: &Path,
    arch: &str,
    stem: &'static str,
    entry: &'static str,
    impl_id: &str,
    op: &str,
    dtype_weight: &str,
    dtype_activation: &str,
    tolerance_formula: &str,
    shapes: &[usize],
    salt: u64,
    scalar: Option<f32>,
    make_input: impl Fn(u64, usize) -> Vec<IE>,
    reference: impl Fn(&[IE]) -> Vec<OE>,
    score: impl Fn(&[OE], &[OE]) -> (f32, f32, bool),
) -> Result<Cert>
where
    B: BenchBackend,
    IE: Copy,
    OE: Copy + Default,
{
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let in_sz = core::mem::size_of::<IE>();
    let out_sz = core::mem::size_of::<OE>();
    let mut results = Vec::new();
    for &n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64).wrapping_mul(131).wrapping_add(salt));
        let input = make_input(seed, n);
        let reference_out = reference(&input);

        let d_x = alloc_and_upload(&dev, &input);
        let d_y = dev.alloc(n * out_sz)?;
        {
            let stream = dev.default_stream();
            let n_i = n as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&n_i);
            if let Some(s) = scalar.as_ref() {
                args.push_arg(s);
            }
            B::launch(&kernel, stream, LaunchCfg::one_d((n as u32).div_ceil(256), 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![OE::default(); n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                n * out_sz,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, n * in_sz)?;
            dev.dealloc(d_y, n * out_sz)?;
        }

        let (max_rel_err, tolerance, pass) = score(&got, &reference_out);
        results.push(ShapeResult { m: n, k: 1, n: 1, seed, max_rel_err, tolerance, pass });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: op.to_string(),
        dtype_weight: dtype_weight.to_string(),
        dtype_activation: dtype_activation.to_string(),
        tolerance_formula: tolerance_formula.to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

fn f16_input(seed: u64, n: usize) -> Vec<f16> {
    seeded_f32_range(seed, n, -0.5, 0.5)
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect()
}

fn bits_mismatch_f32(got: &[f32], reference: &[f32]) -> (f32, f32, bool) {
    let err = got
        .iter()
        .zip(reference)
        .filter(|(g, r)| g.to_bits() != r.to_bits())
        .count() as f32;
    (err, 0.0, err == 0.0)
}

fn bits_mismatch_f16(got: &[f16], reference: &[f16]) -> (f32, f32, bool) {
    let err = got
        .iter()
        .zip(reference)
        .filter(|(g, r)| g.to_bits() != r.to_bits())
        .count() as f32;
    (err, 0.0, err == 0.0)
}

/// `cast_f16_f32` — widening cast, bit-exact on both backends.
pub fn run_cast_f16_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_unary_sweep::<B, f16, f32>(
        repo_root, arch, "cast_f16_f32", "flambeau_cast_f16_f32", "cast_f16_f32",
        "cast_f16_f32", "F16", "F32", "exact bit match vs half::f16::to_f32",
        &[64, 1024, 2048, 4096, 8192], 0x13, None,
        f16_input,
        |x| x.iter().map(|v| v.to_f32()).collect(),
        bits_mismatch_f32,
    )
}

/// `cast_f32_f16` — narrowing cast, saturating at ±F16_MAX, bit-exact.
pub fn run_cast_f32_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_unary_sweep::<B, f32, f16>(
        repo_root, arch, "cast_f32_f16", "flambeau_cast_f32_f16", "cast_f32_f16",
        "cast_f32_f16", "F32", "F16", "exact bit match vs half::f16::from_f32 (saturated)",
        &[64, 1024, 2048, 4096, 8192], 0x27, None,
        |seed, n| seeded_f32_range(seed, n, -0.5, 0.5),
        |x| {
            x.iter()
                .map(|&v| {
                    let v = v.clamp(-65504.0, 65504.0);
                    f16::from_f32(v)
                })
                .collect()
        },
        bits_mismatch_f16,
    )
}

/// `scale_f32` — pointwise scalar multiply (GDN Q-scale 1/sqrt(head_k_dim)).
pub fn run_scale_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let scale = 1.0f32 / 128.0f32.sqrt();
    run_unary_sweep::<B, f32, f32>(
        repo_root, arch, "scale_f32", "flambeau_scale_f32", "scale_f32",
        "scale_f32", "F32", "F32", "exact (single f32 multiply, deterministic both backends)",
        &[64, 1024, 2048, 4096], 0x0b, Some(scale),
        |seed, n| seeded_f32_range(seed, n, -0.5, 0.5),
        move |x| x.iter().map(|v| v * scale).collect(),
        bits_mismatch_f32,
    )
}

/// `scale_f16` — F16 scalar multiply (Gemma4 per-layer output scale).
pub fn run_scale_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let scale = 0.7235f32;
    run_unary_sweep::<B, f16, f16>(
        repo_root, arch, "scale_f16", "flambeau_scale_f16", "scale_f16",
        "scale_f16", "F16", "F16", "exact bit match vs (f16)((f32)x * scale)",
        &[64, 1024, 2048, 4096], 0x1d, Some(scale),
        f16_input,
        move |x| x.iter().map(|v| f16::from_f32(v.to_f32() * scale)).collect(),
        bits_mismatch_f16,
    )
}

/// `silu_f32` — SiLU/swish (GDN silu(conv_out)). `__expf` is a fast intrinsic,
/// so the cross-backend envelope is `1e-5` rel vs the F64 host oracle.
pub fn run_silu_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_unary_sweep::<B, f32, f32>(
        repo_root, arch, "silu_f32", "flambeau_silu_f32", "silu_f32",
        "silu_f32", "F32", "F32", "|err| <= 1e-5 * max(|ref|, 1) vs F64 oracle (fast __expf)",
        &[64, 1024, 4096, 8192], 0x03, None,
        |seed, n| seeded_f32_range(seed, n, -0.5, 0.5),
        |x| {
            x.iter()
                .map(|&xi| {
                    let d = xi as f64;
                    (d / (1.0 + (-d).exp())) as f32
                })
                .collect()
        },
        |got, reference| {
            let err = max_rel_err_with_floor(got, reference, 1.0);
            (err, 1e-5, err <= 1e-5)
        },
    )
}

/// `apply_softcap_f32` — final-logit softcap `tanh(x/cap)*cap` (Gemma4, cap=30).
pub fn run_apply_softcap_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let cap = 30.0f32;
    run_unary_sweep::<B, f32, f32>(
        repo_root, arch, "apply_softcap_f32", "flambeau_apply_softcap_f32", "apply_softcap_f32",
        "apply_softcap_f32", "F32", "F32",
        "|err| <= 1e-5 * max(|ref|, 1) vs F64 oracle",
        &[64, 1024, 4096, 8192], 0x2f, Some(cap),
        // logits span a wide range; use a wider sampling interval than the
        // ±0.5 default so tanh's saturating region is exercised.
        |seed, n| seeded_f32_range(seed, n, -64.0, 64.0),
        move |x| {
            x.iter()
                .map(|&xi| {
                    let d = xi as f64;
                    ((d / cap as f64).tanh() * cap as f64) as f32
                })
                .collect()
        },
        |got, reference| {
            let err = max_rel_err_with_floor(got, reference, 1.0);
            (err, 1e-5, err <= 1e-5)
        },
    )
}
