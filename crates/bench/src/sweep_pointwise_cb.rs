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
        "swiglu_f16" => run_swiglu_f16_sweep::<B>(repo_root, arch),
        "sigmoid_mul_f16" => run_sigmoid_mul_f16_sweep::<B>(repo_root, arch),
        "swiglu_f32_to_q8_1" => run_swiglu_f32_to_q8_1_sweep::<B>(repo_root, arch),
        "split_q_gate_f16" => run_split_q_gate_f16_sweep::<B>(repo_root, arch),
        "rmsnorm_f32" => run_rmsnorm_f32_sweep::<B>(repo_root, arch),
        "rmsnorm_f16" => run_rmsnorm_f16_sweep::<B>(repo_root, arch),
        "l2_norm_f32" => run_l2_norm_f32_sweep::<B>(repo_root, arch),
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

/// Shared driver for the two-input F16 elementwise leaves (`silu`/`sigmoid`
/// gating): two F16 input buffers, one F16 output, `ceil(n/256)×256`. `oracle`
/// is the F64 host math `(a, b) -> result`; the result is rounded to F16 and
/// compared in F32 with a `5e-3` rel envelope (F16 step + fast-`__expf` drift).
fn run_binary_f16_sweep<B: BenchBackend>(
    repo_root: &Path,
    arch: &str,
    stem: &'static str,
    entry: &'static str,
    impl_id: &str,
    op: &str,
    tolerance_formula: &str,
    shapes: &[usize],
    salt: u64,
    oracle: impl Fn(f64, f64) -> f64,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for &n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64).wrapping_mul(151).wrapping_add(salt));
        let a = f16_input(seed, n);
        let b = f16_input(seed ^ 0xAB, n);
        let reference: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(av, bv)| {
                f16::from_f32(oracle(av.to_f32() as f64, bv.to_f32() as f64) as f32).to_f32()
            })
            .collect();

        let d_a = alloc_and_upload(&dev, &a);
        let d_b = alloc_and_upload(&dev, &b);
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
        let mut got_f16 = vec![f16::from_f32(0.0); n];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize),
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

        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, 1e-2);
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 5e-3,
            pass: err <= 5e-3,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: op.to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
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

/// `swiglu_f16` — FFN gate+up activation `silu(gate)*up`, F16 I/O.
pub fn run_swiglu_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_binary_f16_sweep::<B>(
        repo_root, arch, "swiglu_f16", "flambeau_swiglu_f16", "swiglu_f16", "swiglu_f16",
        "|err| <= 5e-3 * max(|ref|, 1e-2) vs F64 oracle (fast __expf)",
        &[2048, 15360, 4096], 0x41,
        |g, u| (g / (1.0 + (-g).exp())) * u,
    )
}

/// `sigmoid_mul_f16` — full-attention output gate `sigmoid(gate)*x`, F16 I/O.
pub fn run_sigmoid_mul_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_binary_f16_sweep::<B>(
        repo_root, arch, "sigmoid_mul_f16", "flambeau_sigmoid_mul_f16", "sigmoid_mul_f16",
        "sigmoid_mul_f16", "|err| <= 5e-3 * max(|ref|, 1e-2) vs F64 oracle (logistic)",
        &[256, 4096, 8192], 0x53,
        |g, x| (1.0 / (1.0 + (-g).exp())) * x,
    )
}

/// `swiglu_f32_to_q8_1` — fused `silu(a)*b` (F32) → Q8_1 blocks (36 B: d, s,
/// qs[32]). 32 threads/block = one 32-lane group; the amax/sum reduce is a
/// width-32 `__shfl_xor` (one warp on wave32, a sub-warp on wave64). Scored by
/// dequantizing the GPU output and comparing to the F64 oracle `silu(a)*b`.
pub fn run_swiglu_f32_to_q8_1_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("swiglu_f32_to_q8_1").ok_or_else(|| anyhow!("swiglu_f32_to_q8_1 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_swiglu_f32_to_q8_1")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [32usize, 256, 4096];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64).wrapping_mul(167).wrapping_add(0x61));
        let a = seeded_f32_range(seed, n, -0.5, 0.5);
        let b = seeded_f32_range(seed ^ 0xCD, n, -0.5, 0.5);
        // F64 oracle: the un-quantized fused activation per element.
        let xi: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(&av, &bv)| {
                let d = av as f64;
                ((d / (1.0 + (-d).exp())) * bv as f64) as f32
            })
            .collect();

        let d_a = alloc_and_upload(&dev, &a);
        let d_b = alloc_and_upload(&dev, &b);
        let n_blocks = n / 32;
        let out_bytes = n_blocks * 36;
        let d_y = dev.alloc(out_bytes)?;
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
            B::launch(&kernel, stream, LaunchCfg::one_d(n_blocks as u32, 32), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0u8; out_bytes];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                out_bytes,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_a, n * 4)?;
            dev.dealloc(d_b, n * 4)?;
            dev.dealloc(d_y, out_bytes)?;
        }

        // Dequantize GPU output (d * qs) and compare to the oracle xi.
        let mut dequant = vec![0.0f32; n];
        for (blk_idx, block) in got.chunks(36).enumerate() {
            let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for (lane, &qb) in block[4..36].iter().enumerate() {
                dequant[blk_idx * 32 + lane] = d * (qb as i8) as f32;
            }
        }
        let err = max_rel_err_with_floor(&dequant, &xi, 1.0);
        results.push(ShapeResult {
            m: n,
            k: 1,
            n: 1,
            seed,
            max_rel_err: err,
            tolerance: 5e-3,
            pass: err <= 5e-3,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "swiglu_f32_to_q8_1".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "swiglu_f32_to_q8_1".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "Q8_1".to_string(),
        tolerance_formula: "|dequant - oracle| <= 5e-3 * max(|oracle|, 1) (Q8_1 step + __expf)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// `rmsnorm_f32` — per-row RMSNorm (GDN ssm_norm). One block per row; the
/// block-wide sum-of-squares reduce rides the neutral `block_reduce_sum` over
/// the `WARP_SIZE` seam (correct on wave64 and wave32).
pub fn run_rmsnorm_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("rmsnorm_f32").ok_or_else(|| anyhow!("rmsnorm_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_rmsnorm_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(32usize, 128usize), (16, 128), (1, 512)];
    let eps = 1e-6f32;
    let mut results = Vec::new();
    for (m, k) in shapes {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 11);
        let x = seeded_f32_range(seed, m * k, -0.5, 0.5);
        let w = seeded_f32_range(seed ^ 0xEE, k, -0.5, 0.5);
        let mut reference = vec![0.0f32; m * k];
        for r in 0..m {
            let row = &x[r * k..(r + 1) * k];
            let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let rsqrt = 1.0 / ((sum_sq / (k as f64)) + eps as f64).sqrt();
            for i in 0..k {
                reference[r * k + i] = (row[i] as f64 * w[i] as f64 * rsqrt) as f32;
            }
        }

        let d_x = alloc_and_upload(&dev, &x);
        let d_w = alloc_and_upload(&dev, &w);
        let d_y = dev.alloc(m * k * 4)?;
        {
            let stream = dev.default_stream();
            let m_i = m as i32;
            let k_i = k as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&w_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&m_i);
            args.push_arg(&k_i);
            args.push_arg(&eps);
            B::launch(&kernel, stream, LaunchCfg::one_d(m as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; m * k];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                m * k * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, m * k * 4)?;
            dev.dealloc(d_w, k * 4)?;
            dev.dealloc(d_y, m * k * 4)?;
        }

        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 1e-5, pass: err <= 1e-5 });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rmsnorm_f32".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "rmsnorm_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 * max(|ref|, 1) vs F64 oracle".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// `rmsnorm_f16` — per-row RMSNorm, F16 I/O (every-layer decode norm). Same
/// neutral block-reduce seam as `rmsnorm_f32`; F32 accumulation, F16 output
/// compared in F32 with the `sqrt(k)` noise floor.
pub fn run_rmsnorm_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("rmsnorm_f16").ok_or_else(|| anyhow!("rmsnorm_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_rmsnorm_f16")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 2048usize), (8, 2048), (1, 5120)];
    let eps = 1e-6f32;
    let mut results = Vec::new();
    for (m, k) in shapes {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 11);
        let x = f16_input(seed, m * k);
        let w = f16_input(seed ^ 0xEE, k);
        let mut reference = vec![0.0f32; m * k];
        for r in 0..m {
            let sum_sq: f64 = (0..k).map(|i| (x[r * k + i].to_f32() as f64).powi(2)).sum();
            let rsqrt = 1.0 / ((sum_sq / (k as f64)) + eps as f64).sqrt();
            for i in 0..k {
                let v = x[r * k + i].to_f32() as f64 * w[i].to_f32() as f64 * rsqrt;
                reference[r * k + i] = f16::from_f32(v as f32).to_f32();
            }
        }

        let d_x = alloc_and_upload(&dev, &x);
        let d_w = alloc_and_upload(&dev, &w);
        let d_y = dev.alloc(m * k * 2)?;
        {
            let stream = dev.default_stream();
            let m_i = m as i32;
            let k_i = k as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&w_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&m_i);
            args.push_arg(&k_i);
            args.push_arg(&eps);
            B::launch(&kernel, stream, LaunchCfg::one_d(m as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); m * k];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize),
                d_y,
                m * k * 2,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, m * k * 2)?;
            dev.dealloc(d_w, k * 2)?;
            dev.dealloc(d_y, m * k * 2)?;
        }

        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "rmsnorm_f16".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "rmsnorm_f16".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "|err| <= 1e-2 * max(|ref|, sqrt(k))".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// `l2_norm_f32` — per-row L2 normalization (GDN Q/K). Same neutral block-reduce
/// seam as `rmsnorm_f32`.
pub fn run_l2_norm_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("l2_norm_f32").ok_or_else(|| anyhow!("l2_norm_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_l2_norm_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(16usize, 128usize), (2048, 128), (16, 256), (1, 512)];
    let eps = 1e-6f32;
    let mut results = Vec::new();
    for (m, k) in shapes {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 11);
        let x = seeded_f32_range(seed, m * k, -0.5, 0.5);
        let mut reference = vec![0.0f32; m * k];
        for r in 0..m {
            let row = &x[r * k..(r + 1) * k];
            let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv = 1.0 / (sum_sq + eps as f64).sqrt();
            for i in 0..k {
                reference[r * k + i] = (row[i] as f64 * inv) as f32;
            }
        }

        let d_x = alloc_and_upload(&dev, &x);
        let d_y = dev.alloc(m * k * 4)?;
        {
            let stream = dev.default_stream();
            let m_i = m as i32;
            let k_i = k as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&m_i);
            args.push_arg(&k_i);
            args.push_arg(&eps);
            B::launch(&kernel, stream, LaunchCfg::one_d(m as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; m * k];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                d_y,
                m * k * 4,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, m * k * 4)?;
            dev.dealloc(d_y, m * k * 4)?;
        }

        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 1e-5, pass: err <= 1e-5 });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "l2_norm_f32".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "l2_norm_f32".to_string(),
        dtype_weight: "F32".to_string(),
        dtype_activation: "F32".to_string(),
        tolerance_formula: "|err| <= 1e-5 * max(|ref|, 1) vs F64 oracle".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}

/// `split_q_gate_f16` — strided split of fused `[tok, head, 2*head_dim]` into
/// contiguous `q` + `gate`. Pure copy, no arithmetic → bit-exact both backends.
/// 3D grid `(n_tokens, n_head, ceil(head_dim/128))`.
pub fn run_split_q_gate_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("split_q_gate_f16").ok_or_else(|| anyhow!("split_q_gate_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_split_q_gate_f16")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 16usize, 256usize), (128, 16, 256), (1, 2, 256)];
    let mut results = Vec::new();
    for (n_tokens, n_head, head_dim) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (n_head as u64) * 53 + (head_dim as u64) * 11);
        let total_fused = n_tokens * n_head * 2 * head_dim;
        let total_split = n_tokens * n_head * head_dim;
        let fused = f16_input(seed, total_fused);
        let mut ref_q = vec![f16::from_f32(0.0); total_split];
        let mut ref_g = vec![f16::from_f32(0.0); total_split];
        for t in 0..n_tokens {
            for h in 0..n_head {
                let fb = (t * n_head + h) * (2 * head_dim);
                let sb = (t * n_head + h) * head_dim;
                for d in 0..head_dim {
                    ref_q[sb + d] = fused[fb + d];
                    ref_g[sb + d] = fused[fb + head_dim + d];
                }
            }
        }

        let d_f = alloc_and_upload(&dev, &fused);
        let d_q = dev.alloc(total_split * 2)?;
        let d_g = dev.alloc(total_split * 2)?;
        {
            let stream = dev.default_stream();
            let n_tokens_i = n_tokens as i32;
            let n_head_i = n_head as i32;
            let head_dim_i = head_dim as i32;
            let f_ptr: u64 = d_f.as_usize() as u64;
            let q_ptr: u64 = d_q.as_usize() as u64;
            let g_ptr: u64 = d_g.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&f_ptr);
            args.push_arg(&q_ptr);
            args.push_arg(&g_ptr);
            args.push_arg(&n_tokens_i);
            args.push_arg(&n_head_i);
            args.push_arg(&head_dim_i);
            let threads = 128u32;
            let cfg = LaunchCfg {
                grid: (n_tokens as u32, n_head as u32, (head_dim as u32).div_ceil(threads)),
                block: (threads, 1, 1),
                shared_bytes: 0,
            };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_q = vec![f16::from_f32(0.0); total_split];
        let mut got_g = vec![f16::from_f32(0.0); total_split];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_q.as_mut_ptr() as usize),
                d_q,
                total_split * 2,
            )?;
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(got_g.as_mut_ptr() as usize),
                d_g,
                total_split * 2,
            )?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_f, total_fused * 2)?;
            dev.dealloc(d_q, total_split * 2)?;
            dev.dealloc(d_g, total_split * 2)?;
        }

        let mismatches = got_q
            .iter()
            .zip(&ref_q)
            .chain(got_g.iter().zip(&ref_g))
            .filter(|(g, r)| g.to_bits() != r.to_bits())
            .count() as f32;
        results.push(ShapeResult {
            m: n_tokens,
            k: n_head,
            n: head_dim,
            seed,
            max_rel_err: mismatches,
            tolerance: 0.0,
            pass: mismatches == 0.0,
        });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: "split_q_gate_f16".to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: "split_q_gate_f16".to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: "F16".to_string(),
        tolerance_formula: "exact match (strided copy, no arithmetic)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
}
