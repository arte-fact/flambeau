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
        "quantize_f16_q8_1" => run_quantize_f16_q8_sweep::<B>(
            repo_root, arch, "quantize_f16_q8_1", "flambeau_quantize_row_f16_q8_1", "Q8_1", 36, 4,
        ),
        "quantize_f16_q8_0" => run_quantize_f16_q8_sweep::<B>(
            repo_root, arch, "quantize_f16_q8_0", "flambeau_quantize_row_f16_q8_0", "Q8_0", 34, 2,
        ),
        "rmsnorm_f32_to_f16" => run_rmsnorm_f32_to_f16_sweep::<B>(repo_root, arch),
        "rmsnorm_f32_to_f16_add_residual" => {
            run_rmsnorm_add_residual_sweep::<B>(repo_root, arch, false)
        }
        "rmsnorm_f16_to_f16_add_residual" => {
            run_rmsnorm_add_residual_sweep::<B>(repo_root, arch, true)
        }
        "rmsnorm_f16_add_residual" => run_rmsnorm_f16_add_residual_sweep::<B>(repo_root, arch),
        "rmsnorm_q8_1_fused" => run_rmsnorm_q8_1_fused_sweep::<B>(repo_root, arch),
        "rope_f16" => run_rope_f16_sweep::<B>(repo_root, arch, false),
        "rope_neox_partial_f16" => run_rope_f16_sweep::<B>(repo_root, arch, true),
        "moe_combine_no_residual_f16" => run_moe_combine_sweep::<B>(repo_root, arch, 0),
        "moe_combine_f16" => run_moe_combine_sweep::<B>(repo_root, arch, 1),
        "moe_combine_two_residuals_f16" => run_moe_combine_sweep::<B>(repo_root, arch, 2),
        "apply_per_expert_scale_f32" => run_apply_per_expert_scale_sweep::<B>(repo_root, arch),
        "shared_expert_scale_f32" => run_shared_expert_scale_sweep::<B>(repo_root, arch),
        "topk_f32" => run_topk_f32_sweep::<B>(repo_root, arch),
        "sampler_apply_penalties_f32" => run_sampler_apply_penalties_sweep::<B>(repo_root, arch),
        "softmax_masked_f16" => run_softmax_masked_f16_sweep::<B>(repo_root, arch),
        "sampler_topk_softmax_f32" => run_sampler_topk_softmax_sweep::<B>(repo_root, arch),
        "kv_append_f16_batched_slots" => run_kv_append_f16_batched_slots_sweep::<B>(repo_root, arch),
        "attention_decode_f16" => run_attention_decode_f16_sweep::<B>(repo_root, arch),
        "attention_decode_f16_splitk" => run_attention_decode_splitk_sweep::<B>(repo_root, arch),
        "moe_sort_by_expert" => run_moe_sort_by_expert_sweep::<B>(repo_root, arch),
        "swiglu_f32" => run_binary_act_sweep::<B>(repo_root, arch, "swiglu_f32", "flambeau_swiglu_f32", false, true),
        "swiglu_f32_to_f16" => run_binary_act_sweep::<B>(repo_root, arch, "swiglu_f32_to_f16", "flambeau_swiglu_f32_to_f16", true, true),
        "gelu_mul_f32" => run_binary_act_sweep::<B>(repo_root, arch, "gelu_mul_f32", "flambeau_gelu_mul_f32", false, false),
        "gelu_f32_to_f16" => run_binary_act_sweep::<B>(repo_root, arch, "gelu_f32_to_f16", "flambeau_gelu_f32_to_f16", true, false),
        "moe_combine_no_residual_f32" => run_moe_combine_no_residual_f32_sweep::<B>(repo_root, arch),
        "dense_gemv_f32_f16" => run_dense_gemv_sweep::<B>(repo_root, arch, "dense_gemv_f32_f16", "flambeau_dense_gemv_f32_f16", false, 0),
        "dense_gemv_f16_f16" => run_dense_gemv_sweep::<B>(repo_root, arch, "dense_gemv_f16_f16", "flambeau_dense_gemv_f16_f16", true, 0),
        "dense_gemv_f32_f16_batched" => run_dense_gemv_sweep::<B>(repo_root, arch, "dense_gemv_f32_f16_batched", "flambeau_dense_gemv_f32_f16_batched", false, 4),
        "dense_gemv_f16_f16_batched" => run_dense_gemv_sweep::<B>(repo_root, arch, "dense_gemv_f16_f16_batched", "flambeau_dense_gemv_f16_f16_batched", true, 4),
        "v_unit_norm_per_head_f16" => run_v_unit_norm_sweep::<B>(repo_root, arch),
        "kv_append_v_unit_norm_f16" => run_kv_append_v_unit_norm_sweep::<B>(repo_root, arch),
        "rmsnorm_rope_neox_partial_f16" => run_rmsnorm_rope_sweep::<B>(repo_root, arch),
        "gdn_alpha_beta_f32" => run_gdn_alpha_beta_sweep::<B>(repo_root, arch),
        "gdn_split_qkv_f32" => run_gdn_split_qkv_sweep::<B>(repo_root, arch),
        "gdn_assemble_conv_input_f32" => run_gdn_assemble_conv_input_sweep::<B>(repo_root, arch),
        "causal_conv1d_f32" => run_causal_conv1d_sweep::<B>(repo_root, arch),
        "gdn_conv_trio_decode_f32_batched_slots" => run_gdn_conv_trio_sweep::<B>(repo_root, arch),
        "gdn_state_step_f32" => run_gdn_state_step_sweep::<B>(repo_root, arch),
        "gdn_state_step_alphabeta_f32" => run_gdn_alphabeta_sweep::<B>(repo_root, arch, false),
        "gdn_state_step_alphabeta_f32_batched_slots" => run_gdn_alphabeta_sweep::<B>(repo_root, arch, true),
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

const RMS_SHAPES: [(usize, usize); 3] = [(1, 2048), (8, 2048), (1, 5120)];
const RMS_EPS: f32 = 1e-6;

/// Assemble + write a cert from finished per-shape results. Shared tail for the
/// row-shaped (norm / fused) sweeps whose only variation is metadata.
#[expect(clippy::too_many_arguments, reason = "each arg is a distinct cert column.")]
fn finish_cert<D: Device>(
    dev: &D,
    repo_root: &Path,
    arch: &str,
    impl_id: &str,
    op: &str,
    dtype_weight: &str,
    dtype_activation: &str,
    tolerance_formula: &str,
    results: Vec<ShapeResult>,
    pmc: crate::cert::PmcSnapshot,
) -> Result<Cert> {
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

fn rms_rsqrt(row: &[f32], eps: f32) -> f64 {
    let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
    1.0 / ((sum_sq / (row.len() as f64)) + eps as f64).sqrt()
}

/// `rmsnorm_f32_to_f16` — F32 input, F16 weight + output (gemma4 cast+norm).
pub fn run_rmsnorm_f32_to_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("rmsnorm_f32_to_f16").ok_or_else(|| anyhow!("rmsnorm_f32_to_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_rmsnorm_f32_to_f16")?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for (m, k) in RMS_SHAPES {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 11);
        let x = seeded_f32_range(seed, m * k, -0.5, 0.5);
        let w = f16_input(seed ^ 0xEE, k);
        let mut reference = vec![0.0f32; m * k];
        for r in 0..m {
            let rsqrt = rms_rsqrt(&x[r * k..(r + 1) * k], RMS_EPS);
            for i in 0..k {
                let v = x[r * k + i] as f64 * w[i].to_f32() as f64 * rsqrt;
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
            let eps = RMS_EPS;
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
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_y, m * k * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, m * k * 4)?;
            dev.dealloc(d_w, k * 2)?;
            dev.dealloc(d_y, m * k * 2)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }
    finish_cert(&dev, repo_root, arch, "rmsnorm_f32_to_f16", "rmsnorm_f32_to_f16", "F32", "F16",
        "|err| <= 1e-2 * max(|ref|, sqrt(k))", results, pmc)
}

/// `rmsnorm_{f32,f16}_to_f16_add_residual` — norm a delta, add into the F16
/// residual stream (saturated). `f16_delta` selects the delta input dtype.
pub fn run_rmsnorm_add_residual_sweep<B: BenchBackend>(
    repo_root: &Path,
    arch: &str,
    f16_delta: bool,
) -> Result<Cert> {
    let (stem, entry, dw) = if f16_delta {
        ("rmsnorm_f32_to_f16", "flambeau_rmsnorm_f16_to_f16_add_residual", "F16")
    } else {
        ("rmsnorm_f32_to_f16", "flambeau_rmsnorm_f32_to_f16_add_residual", "F32")
    };
    let impl_id = if f16_delta { "rmsnorm_f16_to_f16_add_residual" } else { "rmsnorm_f32_to_f16_add_residual" };
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for (m, k) in RMS_SHAPES {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 11) ^ if f16_delta { 0xF16 } else { 0 };
        // Delta values; for the f16 path they round-trip through f16 first.
        let delta_f32 = seeded_f32_range(seed, m * k, -0.5, 0.5);
        let delta: Vec<f32> = if f16_delta {
            delta_f32.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
        } else {
            delta_f32.clone()
        };
        let w = f16_input(seed ^ 0xEE, k);
        let resid = f16_input(seed ^ 0x5A, m * k);
        let mut reference = vec![0.0f32; m * k];
        for r in 0..m {
            let rsqrt = rms_rsqrt(&delta[r * k..(r + 1) * k], RMS_EPS);
            for i in 0..k {
                let normed = delta[r * k + i] as f64 * w[i].to_f32() as f64 * rsqrt;
                let sum = (resid[r * k + i].to_f32() as f64 + normed) as f32;
                reference[r * k + i] = f16::from_f32(sum.clamp(-65504.0, 65504.0)).to_f32();
            }
        }

        // Upload delta as f16 or f32 depending on the kernel's input type.
        let d_x = if f16_delta {
            let d16: Vec<f16> = delta.iter().map(|v| f16::from_f32(*v)).collect();
            alloc_and_upload(&dev, &d16)
        } else {
            alloc_and_upload(&dev, &delta)
        };
        let x_bytes = if f16_delta { m * k * 2 } else { m * k * 4 };
        let d_w = alloc_and_upload(&dev, &w);
        let d_resid = alloc_and_upload(&dev, &resid);
        let d_out = dev.alloc(m * k * 2)?;
        {
            let stream = dev.default_stream();
            let m_i = m as i32;
            let k_i = k as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let ri_ptr: u64 = d_resid.as_usize() as u64;
            let ro_ptr: u64 = d_out.as_usize() as u64;
            let eps = RMS_EPS;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&w_ptr);
            args.push_arg(&ri_ptr);
            args.push_arg(&ro_ptr);
            args.push_arg(&m_i);
            args.push_arg(&k_i);
            args.push_arg(&eps);
            B::launch(&kernel, stream, LaunchCfg::one_d(m as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); m * k];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_out, m * k * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, x_bytes)?;
            dev.dealloc(d_w, k * 2)?;
            dev.dealloc(d_resid, m * k * 2)?;
            dev.dealloc(d_out, m * k * 2)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, (k as f32).sqrt() * 0.01);
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }
    finish_cert(&dev, repo_root, arch, impl_id, impl_id, dw, "F16",
        "|err| <= 1e-2 * max(|ref|, sqrt(k))", results, pmc)
}

/// `rmsnorm_f16_add_residual` — fused add (`mid = x_in + delta`) + rmsnorm
/// (`mid_norm = rmsnorm(mid) * weight`); both outputs emitted. Scored on the
/// worse of the two outputs.
pub fn run_rmsnorm_f16_add_residual_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("rmsnorm_f16_add_residual").ok_or_else(|| anyhow!("rmsnorm_f16_add_residual not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_rmsnorm_f16_add_residual")?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for (m, k) in RMS_SHAPES {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 17);
        let x_in = f16_input(seed, m * k);
        let delta = f16_input(seed ^ 0xD0, m * k);
        let w = f16_input(seed ^ 0xEE, k);
        // mid uses the unrounded sum for the reduce, the f16-rounded mid for norm.
        let mut ref_mid = vec![0.0f32; m * k];
        let mut ref_norm = vec![0.0f32; m * k];
        for r in 0..m {
            let mid_unrounded: Vec<f32> = (0..k)
                .map(|i| x_in[r * k + i].to_f32() + delta[r * k + i].to_f32())
                .collect();
            let mid_f16: Vec<f32> = mid_unrounded.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
            for i in 0..k {
                ref_mid[r * k + i] = mid_f16[i];
            }
            let rsqrt = rms_rsqrt(&mid_unrounded, RMS_EPS);
            for i in 0..k {
                let v = mid_f16[i] as f64 * w[i].to_f32() as f64 * rsqrt;
                ref_norm[r * k + i] = f16::from_f32(v as f32).to_f32();
            }
        }

        let d_xin = alloc_and_upload(&dev, &x_in);
        let d_delta = alloc_and_upload(&dev, &delta);
        let d_w = alloc_and_upload(&dev, &w);
        let d_mid = dev.alloc(m * k * 2)?;
        let d_norm = dev.alloc(m * k * 2)?;
        {
            let stream = dev.default_stream();
            let m_i = m as i32;
            let k_i = k as i32;
            let xin_ptr: u64 = d_xin.as_usize() as u64;
            let dl_ptr: u64 = d_delta.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let mid_ptr: u64 = d_mid.as_usize() as u64;
            let norm_ptr: u64 = d_norm.as_usize() as u64;
            let eps = RMS_EPS;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&xin_ptr);
            args.push_arg(&dl_ptr);
            args.push_arg(&w_ptr);
            args.push_arg(&mid_ptr);
            args.push_arg(&norm_ptr);
            args.push_arg(&m_i);
            args.push_arg(&k_i);
            args.push_arg(&eps);
            B::launch(&kernel, stream, LaunchCfg::one_d(m as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got_mid = vec![f16::from_f32(0.0); m * k];
        let mut got_norm = vec![f16::from_f32(0.0); m * k];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_mid.as_mut_ptr() as usize), d_mid, m * k * 2)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_norm.as_mut_ptr() as usize), d_norm, m * k * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_xin, m * k * 2)?;
            dev.dealloc(d_delta, m * k * 2)?;
            dev.dealloc(d_w, k * 2)?;
            dev.dealloc(d_mid, m * k * 2)?;
            dev.dealloc(d_norm, m * k * 2)?;
        }
        let gm: Vec<f32> = got_mid.iter().map(|v| v.to_f32()).collect();
        let gn: Vec<f32> = got_norm.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&gm, &ref_mid, (k as f32).sqrt() * 0.01)
            .max(max_rel_err_with_floor(&gn, &ref_norm, (k as f32).sqrt() * 0.01));
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }
    finish_cert(&dev, repo_root, arch, "rmsnorm_f16_add_residual", "rmsnorm_f16_add_residual",
        "F16", "F16", "|err| <= 1e-2 * max(|ref|, sqrt(k)) on both outputs", results, pmc)
}

/// `rmsnorm_q8_1_fused` — RMSNorm → Q8_1 quantize (decode hot edge). Scored by
/// dequantizing the output and comparing to the F64 rmsnorm oracle.
pub fn run_rmsnorm_q8_1_fused_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("rmsnorm_q8_1_fused").ok_or_else(|| anyhow!("rmsnorm_q8_1_fused not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_rmsnorm_q8_1_fused")?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    for (m, k) in RMS_SHAPES {
        let seed = 0xC0FFEE ^ ((m as u64) * 1049 + (k as u64) * 23);
        let x = f16_input(seed, m * k);
        let w = f16_input(seed ^ 0xEE, k);
        let mut normed = vec![0.0f32; m * k];
        for r in 0..m {
            let row: Vec<f32> = (0..k).map(|i| x[r * k + i].to_f32()).collect();
            let rsqrt = rms_rsqrt(&row, RMS_EPS);
            for i in 0..k {
                normed[r * k + i] = (row[i] as f64 * w[i].to_f32() as f64 * rsqrt) as f32;
            }
        }

        let d_x = alloc_and_upload(&dev, &x);
        let d_w = alloc_and_upload(&dev, &w);
        let n_blocks = m * (k / 32);
        let out_bytes = n_blocks * 36;
        let d_y = dev.alloc(out_bytes)?;
        {
            let stream = dev.default_stream();
            let m_i = m as i32;
            let k_i = k as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let eps = RMS_EPS;
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
        let mut got = vec![0u8; out_bytes];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_y, out_bytes)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, m * k * 2)?;
            dev.dealloc(d_w, k * 2)?;
            dev.dealloc(d_y, out_bytes)?;
        }
        let mut dequant = vec![0.0f32; m * k];
        for (blk_idx, block) in got.chunks(36).enumerate() {
            let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for lane in 0..32 {
                dequant[blk_idx * 32 + lane] = d * (block[4 + lane] as i8) as f32;
            }
        }
        let err = max_rel_err_with_floor(&dequant, &normed, 1.0);
        results.push(ShapeResult { m, k, n: 1, seed, max_rel_err: err, tolerance: 2e-2, pass: err <= 2e-2 });
    }
    finish_cert(&dev, repo_root, arch, "rmsnorm_q8_1_fused", "rmsnorm_q8_1_fused", "F16", "Q8_1",
        "|dequant - rmsnorm oracle| <= 2e-2 * max(|ref|, 1)", results, pmc)
}

/// `rope_f16` / `rope_neox_partial_f16` — rotary embedding, F16 in-place. Pure
/// index math (no reduce); `neox` selects the split-pair partial layout vs the
/// interleaved-pair full-head layout. F64 angle oracle, 5e-3 F16 envelope.
pub fn run_rope_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str, neox: bool) -> Result<Cert> {
    let (stem, entry, impl_id) = if neox {
        ("rope_neox_partial_f16", "flambeau_rope_neox_partial_f16", "rope_neox_partial_f16")
    } else {
        ("rope_f16", "flambeau_rope_f16", "rope_f16")
    };
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let theta_base = 10000.0f32;
    // (n_tokens, n_heads, head_dim, rotated_dims). Interleaved rotates the whole
    // head_dim; neox rotates only rotated_dims with split pairs.
    let shapes: [(usize, usize, usize, usize); 3] = if neox {
        [(1, 16, 256, 64), (4, 8, 256, 64), (1, 2, 128, 64)]
    } else {
        [(1, 16, 128, 128), (4, 8, 128, 128), (1, 2, 64, 64)]
    };
    let mut results = Vec::new();
    for (n_tokens, n_heads, head_dim, rotated) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (n_heads as u64) * 53 + (head_dim as u64) * 11);
        let total = n_tokens * n_heads * head_dim;
        let x = f16_input(seed, total);
        let positions: Vec<i32> = (0..n_tokens).map(|t| (t as i32) * 7 + 1).collect();

        // Host oracle (F64 angles), applied to the f16-upcast input.
        let mut reference: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        for t in 0..n_tokens {
            for h in 0..n_heads {
                let base = (t * n_heads + h) * head_dim;
                let half = rotated / 2;
                for p in 0..half {
                    let exponent = 2.0 * p as f64 / rotated as f64;
                    let inv_freq = 1.0 / (theta_base as f64).powf(exponent);
                    let angle = positions[t] as f64 * inv_freq;
                    let (s, c) = angle.sin_cos();
                    let (lo, hi) = if neox { (base + p, base + p + half) } else { (base + 2 * p, base + 2 * p + 1) };
                    let x0 = x[lo].to_f32() as f64;
                    let x1 = x[hi].to_f32() as f64;
                    reference[lo] = f16::from_f32((x0 * c - x1 * s) as f32).to_f32();
                    reference[hi] = f16::from_f32((x0 * s + x1 * c) as f32).to_f32();
                }
            }
        }

        let d_x = alloc_and_upload(&dev, &x);
        let d_pos = alloc_and_upload(&dev, &positions);
        {
            let stream = dev.default_stream();
            let nh = n_heads as i32;
            let hd = head_dim as i32;
            let rd = rotated as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let pos_ptr: u64 = d_pos.as_usize() as u64;
            let block_x = if neox { (rotated / 2) as u32 } else { (head_dim / 2) as u32 };
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&pos_ptr);
            args.push_arg(&theta_base);
            args.push_arg(&nh);
            args.push_arg(&hd);
            if neox {
                args.push_arg(&rd);
            }
            let cfg = LaunchCfg {
                grid: (n_tokens as u32, n_heads as u32, 1),
                block: (block_x, 1, 1),
                shared_bytes: 0,
            };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); total];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_x, total * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, total * 2)?;
            dev.dealloc(d_pos, n_tokens * 4)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, 1e-2);
        results.push(ShapeResult { m: n_tokens, k: n_heads, n: head_dim, seed, max_rel_err: err, tolerance: 5e-3, pass: err <= 5e-3 });
    }
    finish_cert(&dev, repo_root, arch, impl_id, impl_id, "F16", "F16",
        "|err| <= 5e-3 * max(|ref|, 1e-2) vs F64 angle oracle", results, pmc)
}

/// `gdn_state_step_alphabeta_f32[_batched_slots]` — GDN recurrent step with
/// inline alpha/beta. `batched` selects per-slot state pointer arrays. F64
/// recurrence oracle (n_rep=1), 1e-3.
pub fn run_gdn_alphabeta_sweep<B: BenchBackend>(repo_root: &Path, arch: &str, batched: bool) -> Result<Cert> {
    let (stem, entry) = if batched {
        ("gdn_state_step_alphabeta_f32_batched_slots", "flambeau_gdn_state_step_alphabeta_f32_s128_batched_slots")
    } else {
        ("gdn_state_step_alphabeta_f32", "flambeau_gdn_state_step_alphabeta_f32_s128")
    };
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let warp_size: u32 = if arch == "sm_86" { 32 } else { 64 };
    let warps_per_block: u32 = 4;
    const S_V: usize = 128;
    let cases = [(1usize, 2usize, 1usize), (2, 2, 2)];
    let mut results = Vec::new();
    for (b_n, h_n, l_n) in cases {
        let seed = 0xC0FFEE ^ ((h_n as u64) * 41 + (l_n as u64) * 19 + if batched { 1 } else { 0 });
        let q = seeded_f32_range(seed, b_n * l_n * h_n * S_V, -0.3, 0.3);
        let k = seeded_f32_range(seed ^ 0x11, b_n * l_n * h_n * S_V, -0.3, 0.3);
        let v = seeded_f32_range(seed ^ 0x22, b_n * l_n * h_n * S_V, -0.3, 0.3);
        let alpha = seeded_f32_range(seed ^ 0x33, b_n * l_n * h_n, -1.0, 1.0);
        let beta_in = seeded_f32_range(seed ^ 0x44, b_n * l_n * h_n, -1.0, 1.0);
        let dt_bias = seeded_f32_range(seed ^ 0x55, h_n, -0.3, 0.3);
        let ssm_a: Vec<f32> = (0..h_n).map(|i| -(0.05 + 0.01 * i as f32)).collect();
        let state_in = seeded_f32_range(seed ^ 0x66, b_n * h_n * S_V * S_V, -0.1, 0.1);

        let mut attn_ref = vec![0.0f32; b_n * l_n * h_n * S_V];
        let mut state_ref = vec![0.0f32; b_n * h_n * S_V * S_V];
        for b in 0..b_n {
            for h in 0..h_n {
                let bh = b * h_n + h;
                let mut st = vec![0.0f64; S_V * S_V];
                for x in 0..S_V * S_V { st[x] = state_in[bh * S_V * S_V + x] as f64; }
                for t in 0..l_n {
                    let a_raw = (alpha[(b * l_n + t) * h_n + h] + dt_bias[h]) as f64;
                    let sp = a_raw.max(0.0) + (1.0 + (-a_raw.abs()).exp()).ln();
                    let g = (sp * ssm_a[h] as f64).exp();
                    let br = beta_in[(b * l_n + t) * h_n + h] as f64;
                    let bv = 1.0 / (1.0 + (-br).exp());
                    let qb = ((b * l_n + t) * h_n + h) * S_V;
                    for col in 0..S_V {
                        let sk: f64 = (0..S_V).map(|i| st[col * S_V + i] * k[qb + i] as f64).sum();
                        let delta = (v[qb + col] as f64 - g * sk) * bv;
                        let mut attn = 0.0f64;
                        for i in 0..S_V {
                            st[col * S_V + i] = g * st[col * S_V + i] + k[qb + i] as f64 * delta;
                            attn += st[col * S_V + i] * q[qb + i] as f64;
                        }
                        attn_ref[(((b * l_n + t) * h_n) + h) * S_V + col] = attn as f32;
                    }
                }
                for x in 0..S_V * S_V { state_ref[bh * S_V * S_V + x] = st[x] as f32; }
            }
        }

        let d_q = alloc_and_upload(&dev, &q);
        let d_k = alloc_and_upload(&dev, &k);
        let d_v = alloc_and_upload(&dev, &v);
        let d_al = alloc_and_upload(&dev, &alpha);
        let d_be = alloc_and_upload(&dev, &beta_in);
        let d_dt = alloc_and_upload(&dev, &dt_bias);
        let d_sa = alloc_and_upload(&dev, &ssm_a);
        let d_ao = dev.alloc(b_n * l_n * h_n * S_V * 4)?;
        // State: contiguous or per-slot buffers.
        let per_batch = h_n * S_V * S_V;
        let (d_si, d_so, d_sip, d_sop, mut slot_in, mut slot_out) =
            if batched {
                let mut si = Vec::with_capacity(b_n);
                let mut so = Vec::with_capacity(b_n);
                let mut sip = vec![0u64; b_n];
                let mut sop = vec![0u64; b_n];
                for b in 0..b_n {
                    let sb = alloc_and_upload(&dev, &state_in[b * per_batch..(b + 1) * per_batch]);
                    let ob = dev.alloc(per_batch * 4)?;
                    sip[b] = sb.as_usize() as u64;
                    sop[b] = ob.as_usize() as u64;
                    si.push(sb);
                    so.push(ob);
                }
                let d_sip = alloc_and_upload(&dev, &sip);
                let d_sop = alloc_and_upload(&dev, &sop);
                (None, None, Some(d_sip), Some(d_sop), si, so)
            } else {
                let si = alloc_and_upload(&dev, &state_in);
                let so = dev.alloc(b_n * per_batch * 4)?;
                (Some(si), Some(so), None, None, Vec::new(), Vec::new())
            };
        {
            let stream = dev.default_stream();
            let (bb, hh, ll, nr, ril) = (b_n as i32, h_n as i32, l_n as i32, 1i32, 0i32);
            let qp: u64 = d_q.as_usize() as u64;
            let kp: u64 = d_k.as_usize() as u64;
            let vp: u64 = d_v.as_usize() as u64;
            let alp: u64 = d_al.as_usize() as u64;
            let bep: u64 = d_be.as_usize() as u64;
            let dtp: u64 = d_dt.as_usize() as u64;
            let sap: u64 = d_sa.as_usize() as u64;
            let aop: u64 = d_ao.as_usize() as u64;
            let s1: u64 = if batched { d_sip.as_ref().unwrap().as_usize() as u64 } else { d_si.as_ref().unwrap().as_usize() as u64 };
            let s2: u64 = if batched { d_sop.as_ref().unwrap().as_usize() as u64 } else { d_so.as_ref().unwrap().as_usize() as u64 };
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&qp); args.push_arg(&kp); args.push_arg(&vp);
            args.push_arg(&alp); args.push_arg(&bep); args.push_arg(&dtp); args.push_arg(&sap);
            args.push_arg(&s1); args.push_arg(&s2); args.push_arg(&aop);
            args.push_arg(&bb); args.push_arg(&hh); args.push_arg(&ll); args.push_arg(&nr); args.push_arg(&ril);
            let grid_z = S_V as u32 / warps_per_block;
            let cfg = LaunchCfg { grid: (h_n as u32, b_n as u32, grid_z), block: (warp_size, warps_per_block, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_a = vec![0.0f32; b_n * l_n * h_n * S_V];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_a.as_mut_ptr() as usize), d_ao, b_n * l_n * h_n * S_V * 4)?;
        }
        dev.default_stream().synchronize()?;
        let mut got_s = vec![0.0f32; b_n * per_batch];
        if batched {
            for b in 0..b_n {
                let mut sb = vec![0.0f32; per_batch];
                unsafe {
                    dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                        DevicePtr(sb.as_mut_ptr() as usize), slot_out[b], per_batch * 4)?;
                }
                dev.default_stream().synchronize()?;
                got_s[b * per_batch..(b + 1) * per_batch].copy_from_slice(&sb);
            }
        } else {
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(got_s.as_mut_ptr() as usize), d_so.as_ref().unwrap().clone(), b_n * per_batch * 4)?;
            }
            dev.default_stream().synchronize()?;
        }
        unsafe {
            dev.dealloc(d_q, b_n * l_n * h_n * S_V * 4)?; dev.dealloc(d_k, b_n * l_n * h_n * S_V * 4)?;
            dev.dealloc(d_v, b_n * l_n * h_n * S_V * 4)?; dev.dealloc(d_al, b_n * l_n * h_n * 4)?;
            dev.dealloc(d_be, b_n * l_n * h_n * 4)?; dev.dealloc(d_dt, h_n * 4)?; dev.dealloc(d_sa, h_n * 4)?;
            dev.dealloc(d_ao, b_n * l_n * h_n * S_V * 4)?;
            if batched {
                dev.dealloc(d_sip.unwrap(), b_n * 8)?; dev.dealloc(d_sop.unwrap(), b_n * 8)?;
                for b in 0..b_n { dev.dealloc(slot_in[b], per_batch * 4)?; dev.dealloc(slot_out[b], per_batch * 4)?; }
            } else {
                dev.dealloc(d_si.unwrap(), b_n * per_batch * 4)?; dev.dealloc(d_so.unwrap(), b_n * per_batch * 4)?;
            }
        }
        slot_in.clear();
        slot_out.clear();
        let err = max_rel_err_with_floor(&got_a, &attn_ref, 1.0).max(max_rel_err_with_floor(&got_s, &state_ref, 1.0));
        results.push(ShapeResult { m: b_n, k: h_n, n: l_n, seed, max_rel_err: err, tolerance: 1e-3, pass: err <= 1e-3 });
    }
    finish_cert(&dev, repo_root, arch, stem, stem, "F32", "F32",
        "attn_out + state_out |err| <= 1e-3 vs F64 recurrence", results, pmc)
}

/// `gdn_state_step_f32` — GDN recurrent step (S_v=128). Each warp owns a state
/// column (register-sharded); block = (WARP_SIZE, GDN_WARPS_PER_BLOCK), so the
/// sweep launches 64- or 32-wide per backend. F64 recurrence oracle (n_rep=1),
/// compares attn_out + state_out at 1e-3 (cross-backend reduce partition).
pub fn run_gdn_state_step_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("gdn_state_step_f32").ok_or_else(|| anyhow!("gdn_state_step_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_gdn_state_step_f32_s128")?;
    let pmc = B::attributes(&kernel)?;

    let warp_size: u32 = if arch == "sm_86" { 32 } else { 64 };
    let warps_per_block: u32 = 4;
    const S_V: usize = 128;
    let cases = [(1usize, 2usize, 1usize), (1, 2, 4)]; // (B, H, L), n_rep=1
    let mut results = Vec::new();
    for (b_n, h_n, l_n) in cases {
        let n_rep = 1usize;
        let seed = 0xC0FFEE ^ ((h_n as u64) * 41 + (l_n as u64) * 19);
        let q = seeded_f32_range(seed, b_n * l_n * h_n * S_V, -0.3, 0.3);
        let k = seeded_f32_range(seed ^ 0x11, b_n * l_n * h_n * S_V, -0.3, 0.3);
        let v = seeded_f32_range(seed ^ 0x22, b_n * l_n * h_n * S_V, -0.3, 0.3);
        let gate = seeded_f32_range(seed ^ 0x33, b_n * l_n * h_n, -1.0, 0.0);
        let beta = seeded_f32_range(seed ^ 0x44, b_n * l_n * h_n, 0.0, 1.0);
        let state_in = seeded_f32_range(seed ^ 0x55, b_n * h_n * S_V * S_V, -0.1, 0.1);

        // F64 recurrence oracle.
        let mut attn_ref = vec![0.0f32; b_n * l_n * h_n * S_V];
        let mut state_ref = vec![0.0f32; b_n * h_n * S_V * S_V];
        for b in 0..b_n {
            for h in 0..h_n {
                let bh = b * h_n + h;
                // state[col][i], col-outer.
                let mut st = vec![0.0f64; S_V * S_V];
                for x in 0..S_V * S_V {
                    st[x] = state_in[bh * S_V * S_V + x] as f64;
                }
                for t in 0..l_n {
                    let g = (gate[(b * l_n + t) * h_n + h] as f64).exp();
                    let bv = beta[(b * l_n + t) * h_n + h] as f64;
                    let qb = ((b * l_n + t) * h_n + h) * S_V;
                    let kb = qb;
                    let vb = qb;
                    for col in 0..S_V {
                        for i in 0..S_V {
                            st[col * S_V + i] *= g;
                        }
                        let sk: f64 = (0..S_V).map(|i| st[col * S_V + i] * k[kb + i] as f64).sum();
                        let delta = (v[vb + col] as f64 - sk) * bv;
                        let mut attn = 0.0f64;
                        for i in 0..S_V {
                            st[col * S_V + i] += k[kb + i] as f64 * delta;
                            attn += st[col * S_V + i] * q[qb + i] as f64;
                        }
                        attn_ref[(((b * l_n + t) * h_n) + h) * S_V + col] = attn as f32;
                    }
                }
                for x in 0..S_V * S_V {
                    state_ref[bh * S_V * S_V + x] = st[x] as f32;
                }
                let _ = n_rep;
            }
        }

        let d_q = alloc_and_upload(&dev, &q);
        let d_k = alloc_and_upload(&dev, &k);
        let d_v = alloc_and_upload(&dev, &v);
        let d_g = alloc_and_upload(&dev, &gate);
        let d_b = alloc_and_upload(&dev, &beta);
        let d_si = alloc_and_upload(&dev, &state_in);
        let d_so = dev.alloc(b_n * h_n * S_V * S_V * 4)?;
        let d_ao = dev.alloc(b_n * l_n * h_n * S_V * 4)?;
        {
            let stream = dev.default_stream();
            let (bb, hh, ll, nr, ril) = (b_n as i32, h_n as i32, l_n as i32, n_rep as i32, 0i32);
            let qp: u64 = d_q.as_usize() as u64;
            let kp: u64 = d_k.as_usize() as u64;
            let vp: u64 = d_v.as_usize() as u64;
            let gp: u64 = d_g.as_usize() as u64;
            let bp: u64 = d_b.as_usize() as u64;
            let sip: u64 = d_si.as_usize() as u64;
            let sop: u64 = d_so.as_usize() as u64;
            let aop: u64 = d_ao.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&qp); args.push_arg(&kp); args.push_arg(&vp);
            args.push_arg(&gp); args.push_arg(&bp); args.push_arg(&sip);
            args.push_arg(&sop); args.push_arg(&aop);
            args.push_arg(&bb); args.push_arg(&hh); args.push_arg(&ll);
            args.push_arg(&nr); args.push_arg(&ril);
            let grid_z = S_V as u32 / warps_per_block;
            let cfg = LaunchCfg { grid: (h_n as u32, b_n as u32, grid_z), block: (warp_size, warps_per_block, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_a = vec![0.0f32; b_n * l_n * h_n * S_V];
        let mut got_s = vec![0.0f32; b_n * h_n * S_V * S_V];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_a.as_mut_ptr() as usize), d_ao, b_n * l_n * h_n * S_V * 4)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_s.as_mut_ptr() as usize), d_so, b_n * h_n * S_V * S_V * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_q, b_n * l_n * h_n * S_V * 4)?; dev.dealloc(d_k, b_n * l_n * h_n * S_V * 4)?;
            dev.dealloc(d_v, b_n * l_n * h_n * S_V * 4)?; dev.dealloc(d_g, b_n * l_n * h_n * 4)?;
            dev.dealloc(d_b, b_n * l_n * h_n * 4)?; dev.dealloc(d_si, b_n * h_n * S_V * S_V * 4)?;
            dev.dealloc(d_so, b_n * h_n * S_V * S_V * 4)?; dev.dealloc(d_ao, b_n * l_n * h_n * S_V * 4)?;
        }
        let err = max_rel_err_with_floor(&got_a, &attn_ref, 1.0).max(max_rel_err_with_floor(&got_s, &state_ref, 1.0));
        results.push(ShapeResult { m: b_n, k: h_n, n: l_n, seed, max_rel_err: err, tolerance: 1e-3, pass: err <= 1e-3 });
    }
    finish_cert(&dev, repo_root, arch, "gdn_state_step_f32", "gdn_state_step_f32", "F32", "F32",
        "attn_out + state_out |err| <= 1e-3 * max(|ref|, 1) vs F64 recurrence", results, pmc)
}

/// `gdn_alpha_beta_f32` — GDN gate = softplus(alpha+dt_bias)*ssm_a, beta =
/// sigmoid(beta_in). Pointwise; F64 oracle, 1e-4.
pub fn run_gdn_alpha_beta_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("gdn_alpha_beta_f32").ok_or_else(|| anyhow!("gdn_alpha_beta_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_gdn_alpha_beta_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 32usize), (8, 32), (4, 64)];
    let mut results = Vec::new();
    for (n_tokens, nvh) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 41 + (nvh as u64) * 19);
        let alpha = seeded_f32_range(seed, n_tokens * nvh, -2.0, 2.0);
        let beta_in = seeded_f32_range(seed ^ 0xB2, n_tokens * nvh, -2.0, 2.0);
        let dt_bias = seeded_f32_range(seed ^ 0xD3, nvh, -0.5, 0.5);
        let ssm_a: Vec<f32> = (0..nvh).map(|i| -(0.05 + 0.01 * i as f32)).collect();
        let mut ref_g = vec![0.0f32; n_tokens * nvh];
        let mut ref_b = vec![0.0f32; n_tokens * nvh];
        for t in 0..n_tokens {
            for i in 0..nvh {
                let a = (alpha[t * nvh + i] + dt_bias[i]) as f64;
                let sp = a.max(0.0) + (1.0 + (-a.abs()).exp()).ln();
                ref_g[t * nvh + i] = (sp * ssm_a[i] as f64) as f32;
                let b = beta_in[t * nvh + i] as f64;
                ref_b[t * nvh + i] = (1.0 / (1.0 + (-b).exp())) as f32;
            }
        }

        let d_a = alloc_and_upload(&dev, &alpha);
        let d_b = alloc_and_upload(&dev, &beta_in);
        let d_dt = alloc_and_upload(&dev, &dt_bias);
        let d_sa = alloc_and_upload(&dev, &ssm_a);
        let d_g = dev.alloc(n_tokens * nvh * 4)?;
        let d_bo = dev.alloc(n_tokens * nvh * 4)?;
        {
            let stream = dev.default_stream();
            let nv = nvh as i32;
            let nt = n_tokens as i32;
            let ap: u64 = d_a.as_usize() as u64;
            let bp: u64 = d_b.as_usize() as u64;
            let dtp: u64 = d_dt.as_usize() as u64;
            let sap: u64 = d_sa.as_usize() as u64;
            let gp: u64 = d_g.as_usize() as u64;
            let bop: u64 = d_bo.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&ap);
            args.push_arg(&bp);
            args.push_arg(&dtp);
            args.push_arg(&sap);
            args.push_arg(&gp);
            args.push_arg(&bop);
            args.push_arg(&nv);
            args.push_arg(&nt);
            let cfg = LaunchCfg { grid: (n_tokens as u32, 1, 1), block: (nvh as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_g = vec![0.0f32; n_tokens * nvh];
        let mut got_b = vec![0.0f32; n_tokens * nvh];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_g.as_mut_ptr() as usize), d_g, n_tokens * nvh * 4)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_b.as_mut_ptr() as usize), d_bo, n_tokens * nvh * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_a, n_tokens * nvh * 4)?; dev.dealloc(d_b, n_tokens * nvh * 4)?;
            dev.dealloc(d_dt, nvh * 4)?; dev.dealloc(d_sa, nvh * 4)?;
            dev.dealloc(d_g, n_tokens * nvh * 4)?; dev.dealloc(d_bo, n_tokens * nvh * 4)?;
        }
        let err = max_rel_err_with_floor(&got_g, &ref_g, 1.0).max(max_rel_err_with_floor(&got_b, &ref_b, 1.0));
        results.push(ShapeResult { m: n_tokens, k: nvh, n: 1, seed, max_rel_err: err, tolerance: 1e-4, pass: err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "gdn_alpha_beta_f32", "gdn_alpha_beta_f32", "F32", "F32",
        "|err| <= 1e-4 * max(|ref|, 1) on gate + beta", results, pmc)
}

/// `gdn_split_qkv_f32` / `gdn_assemble_conv_input_f32` — pure F32 copy/split
/// kernels; bit-exact. `assemble=false` runs the qkv split, `true` the conv
/// assemble.
fn run_gdn_copy_sweep<B: BenchBackend>(repo_root: &Path, arch: &str, assemble: bool) -> Result<Cert> {
    let (stem, entry) = if assemble {
        ("gdn_assemble_conv_input_f32", "flambeau_gdn_assemble_conv_input_f32")
    } else {
        ("gdn_split_qkv_f32", "flambeau_gdn_split_qkv_f32")
    };
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let mut results = Vec::new();
    if assemble {
        let cases = [(512usize, 4usize), (2048, 4), (1536, 3)];
        for (conv_channels, conv_kernel) in cases {
            let seed = 0xC0FFEE ^ ((conv_channels as u64) * 13 + (conv_kernel as u64) * 7);
            let hist_elems = conv_channels * (conv_kernel - 1);
            let history = seeded_f32_range(seed, hist_elems, -1.0, 1.0);
            let current = seeded_f32_range(seed ^ 0xCC, conv_channels, -1.0, 1.0);
            let n = conv_channels * conv_kernel;
            let mut reference = vec![0.0f32; n];
            reference[..hist_elems].copy_from_slice(&history);
            reference[hist_elems..].copy_from_slice(&current);

            let d_h = alloc_and_upload(&dev, &history);
            let d_c = alloc_and_upload(&dev, &current);
            let d_out = dev.alloc(n * 4)?;
            {
                let stream = dev.default_stream();
                let cc = conv_channels as i32;
                let ck = conv_kernel as i32;
                let hp: u64 = d_h.as_usize() as u64;
                let cp: u64 = d_c.as_usize() as u64;
                let op: u64 = d_out.as_usize() as u64;
                let mut args = <B::Args<'_>>::default();
                args.push_arg(&hp);
                args.push_arg(&cp);
                args.push_arg(&op);
                args.push_arg(&cc);
                args.push_arg(&ck);
                B::launch(&kernel, stream, LaunchCfg::one_d((n as u32).div_ceil(256), 256), args)?;
                stream.synchronize()?;
            }
            let mut got = vec![0.0f32; n];
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(got.as_mut_ptr() as usize), d_out, n * 4)?;
            }
            dev.default_stream().synchronize()?;
            unsafe { dev.dealloc(d_h, hist_elems * 4)?; dev.dealloc(d_c, conv_channels * 4)?; dev.dealloc(d_out, n * 4)?; }
            let (err, tol, pass) = bits_mismatch_f32(&got, &reference);
            results.push(ShapeResult { m: conv_channels, k: conv_kernel, n: 1, seed, max_rel_err: err, tolerance: tol, pass });
        }
    } else {
        let cases = [(4usize, 256usize, 128usize), (16, 512, 256)];
        for (n_tokens, qk_size, v_size) in cases {
            let seed = 0xC0FFEE ^ ((n_tokens as u64) * 13 + (qk_size as u64) * 7);
            let conv_channels = 2 * qk_size + v_size;
            let silu = seeded_f32_range(seed, n_tokens * conv_channels, -1.0, 1.0);
            let mut ref_q = vec![0.0f32; n_tokens * qk_size];
            let mut ref_k = vec![0.0f32; n_tokens * qk_size];
            let mut ref_v = vec![0.0f32; n_tokens * v_size];
            for t in 0..n_tokens {
                for c in 0..conv_channels {
                    let val = silu[t * conv_channels + c];
                    if c < qk_size { ref_q[t * qk_size + c] = val; }
                    else if c < 2 * qk_size { ref_k[t * qk_size + (c - qk_size)] = val; }
                    else { ref_v[t * v_size + (c - 2 * qk_size)] = val; }
                }
            }
            let d_s = alloc_and_upload(&dev, &silu);
            let d_q = dev.alloc(n_tokens * qk_size * 4)?;
            let d_k = dev.alloc(n_tokens * qk_size * 4)?;
            let d_v = dev.alloc(n_tokens * v_size * 4)?;
            {
                let stream = dev.default_stream();
                let nt = n_tokens as i32;
                let qk = qk_size as i32;
                let vs = v_size as i32;
                let sp: u64 = d_s.as_usize() as u64;
                let qp: u64 = d_q.as_usize() as u64;
                let kp: u64 = d_k.as_usize() as u64;
                let vp: u64 = d_v.as_usize() as u64;
                let mut args = <B::Args<'_>>::default();
                args.push_arg(&sp); args.push_arg(&qp); args.push_arg(&kp); args.push_arg(&vp);
                args.push_arg(&nt); args.push_arg(&qk); args.push_arg(&vs);
                let total = n_tokens * conv_channels;
                B::launch(&kernel, stream, LaunchCfg::one_d((total as u32).div_ceil(256), 256), args)?;
                stream.synchronize()?;
            }
            let mut gq = vec![0.0f32; n_tokens * qk_size];
            let mut gk = vec![0.0f32; n_tokens * qk_size];
            let mut gv = vec![0.0f32; n_tokens * v_size];
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(gq.as_mut_ptr() as usize), d_q, n_tokens * qk_size * 4)?;
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(gk.as_mut_ptr() as usize), d_k, n_tokens * qk_size * 4)?;
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(gv.as_mut_ptr() as usize), d_v, n_tokens * v_size * 4)?;
            }
            dev.default_stream().synchronize()?;
            unsafe { dev.dealloc(d_s, n_tokens * conv_channels * 4)?; dev.dealloc(d_q, n_tokens * qk_size * 4)?; dev.dealloc(d_k, n_tokens * qk_size * 4)?; dev.dealloc(d_v, n_tokens * v_size * 4)?; }
            let bad = bits_mismatch_f32(&gq, &ref_q).0 + bits_mismatch_f32(&gk, &ref_k).0 + bits_mismatch_f32(&gv, &ref_v).0;
            results.push(ShapeResult { m: n_tokens, k: qk_size, n: v_size, seed, max_rel_err: bad, tolerance: 0.0, pass: bad == 0.0 });
        }
    }
    finish_cert(&dev, repo_root, arch, stem, stem, "F32", "F32", "exact (copy/split)", results, pmc)
}

pub fn run_gdn_split_qkv_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_gdn_copy_sweep::<B>(repo_root, arch, false)
}
pub fn run_gdn_assemble_conv_input_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    run_gdn_copy_sweep::<B>(repo_root, arch, true)
}

/// `causal_conv1d_f32` — depthwise causal 1D conv. F64 oracle, 1e-4.
pub fn run_causal_conv1d_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("causal_conv1d_f32").ok_or_else(|| anyhow!("causal_conv1d_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_causal_conv1d_f32")?;
    let pmc = B::attributes(&kernel)?;

    let cases = [(1usize, 512usize, 4usize), (8, 2048, 4), (4, 1536, 3)];
    let mut results = Vec::new();
    for (n_new, conv_channels, conv_kernel) in cases {
        let seed = 0xC0FFEE ^ ((n_new as u64) * 13 + (conv_channels as u64) * 7);
        let n_total = (conv_kernel - 1) + n_new;
        let conv_input = seeded_f32_range(seed, n_total * conv_channels, -1.0, 1.0);
        let weight = seeded_f32_range(seed ^ 0xAB, conv_channels * conv_kernel, -0.5, 0.5);
        let mut reference = vec![0.0f32; n_new * conv_channels];
        for t in 0..n_new {
            for c in 0..conv_channels {
                let acc: f64 = (0..conv_kernel)
                    .map(|k| conv_input[(t + k) * conv_channels + c] as f64 * weight[c * conv_kernel + k] as f64)
                    .sum();
                reference[t * conv_channels + c] = acc as f32;
            }
        }

        let d_ci = alloc_and_upload(&dev, &conv_input);
        let d_w = alloc_and_upload(&dev, &weight);
        let d_y = dev.alloc(n_new * conv_channels * 4)?;
        {
            let stream = dev.default_stream();
            let nn = n_new as i32;
            let cc = conv_channels as i32;
            let ck = conv_kernel as i32;
            let cip: u64 = d_ci.as_usize() as u64;
            let wp: u64 = d_w.as_usize() as u64;
            let yp: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&cip);
            args.push_arg(&wp);
            args.push_arg(&yp);
            args.push_arg(&nn);
            args.push_arg(&cc);
            args.push_arg(&ck);
            let cfg = LaunchCfg { grid: ((conv_channels as u32).div_ceil(256), n_new as u32, 1), block: (256, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; n_new * conv_channels];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_y, n_new * conv_channels * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe { dev.dealloc(d_ci, n_total * conv_channels * 4)?; dev.dealloc(d_w, conv_channels * conv_kernel * 4)?; dev.dealloc(d_y, n_new * conv_channels * 4)?; }
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_new, k: conv_channels, n: conv_kernel, seed, max_rel_err: err, tolerance: 1e-4, pass: err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "causal_conv1d_f32", "causal_conv1d_f32", "F32", "F32",
        "|err| <= 1e-4 * max(|ref|, 1) vs F64 oracle", results, pmc)
}

/// `gdn_conv_trio_decode_f32_batched_slots` — fused per-slot conv + history
/// shift across N decode slots (per-slot history pointers). Checks conv_out +
/// the shifted history. F64 oracle, 1e-4.
pub fn run_gdn_conv_trio_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("gdn_conv_trio_decode_f32_batched_slots").ok_or_else(|| anyhow!("gdn_conv_trio not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_gdn_conv_trio_decode_f32_batched_slots")?;
    let pmc = B::attributes(&kernel)?;

    let cases = [(1usize, 512usize, 4usize), (4, 2048, 4), (8, 1024, 3)];
    let mut results = Vec::new();
    for (n_slots, conv_channels, conv_kernel) in cases {
        let seed = 0xC0FFEE ^ ((n_slots as u64) * 13 + (conv_channels as u64) * 7);
        let hist_rows = conv_kernel - 1;
        let qkv = seeded_f32_range(seed, n_slots * conv_channels, -1.0, 1.0);
        let weight = seeded_f32_range(seed ^ 0xAB, conv_channels * conv_kernel, -0.5, 0.5);
        let hist0: Vec<Vec<f32>> = (0..n_slots)
            .map(|s| seeded_f32_range(seed ^ (0x100 + s as u64), hist_rows * conv_channels, -1.0, 1.0))
            .collect();
        // Oracle: conv_out + shifted history per slot.
        let mut ref_out = vec![0.0f32; n_slots * conv_channels];
        let mut ref_hist: Vec<Vec<f32>> = hist0.clone();
        for s in 0..n_slots {
            for c in 0..conv_channels {
                let nv = qkv[s * conv_channels + c] as f64;
                let mut acc = 0.0f64;
                for k in 0..hist_rows {
                    acc += hist0[s][k * conv_channels + c] as f64 * weight[c * conv_kernel + k] as f64;
                }
                acc += nv * weight[c * conv_kernel + hist_rows] as f64;
                ref_out[s * conv_channels + c] = acc as f32;
                for k in 0..hist_rows {
                    let v = if k + 1 < hist_rows { hist0[s][(k + 1) * conv_channels + c] } else { qkv[s * conv_channels + c] };
                    ref_hist[s][k * conv_channels + c] = v;
                }
            }
        }

        let d_qkv = alloc_and_upload(&dev, &qkv);
        let d_w = alloc_and_upload(&dev, &weight);
        let d_out = dev.alloc(n_slots * conv_channels * 4)?;
        let hbytes = hist_rows * conv_channels * 4;
        let mut hist_bufs = Vec::with_capacity(n_slots);
        let mut hptrs = vec![0u64; n_slots];
        for s in 0..n_slots {
            let hb = alloc_and_upload(&dev, &hist0[s]);
            hptrs[s] = hb.as_usize() as u64;
            hist_bufs.push(hb);
        }
        let d_hptrs = alloc_and_upload(&dev, &hptrs);
        {
            let stream = dev.default_stream();
            let ns = n_slots as i32;
            let cc = conv_channels as i32;
            let ck = conv_kernel as i32;
            let hpp: u64 = d_hptrs.as_usize() as u64;
            let qp: u64 = d_qkv.as_usize() as u64;
            let wp: u64 = d_w.as_usize() as u64;
            let op: u64 = d_out.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&hpp);
            args.push_arg(&qp);
            args.push_arg(&wp);
            args.push_arg(&op);
            args.push_arg(&ns);
            args.push_arg(&cc);
            args.push_arg(&ck);
            let cfg = LaunchCfg { grid: ((conv_channels as u32).div_ceil(256), n_slots as u32, 1), block: (256, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_out = vec![0.0f32; n_slots * conv_channels];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_out.as_mut_ptr() as usize), d_out, n_slots * conv_channels * 4)?;
        }
        dev.default_stream().synchronize()?;
        let mut out_err = max_rel_err_with_floor(&got_out, &ref_out, 1.0);
        for s in 0..n_slots {
            let mut gh = vec![0.0f32; hist_rows * conv_channels];
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(gh.as_mut_ptr() as usize), hist_bufs[s], hbytes)?;
            }
            dev.default_stream().synchronize()?;
            out_err = out_err.max(max_rel_err_with_floor(&gh, &ref_hist[s], 1.0));
        }
        unsafe {
            dev.dealloc(d_qkv, n_slots * conv_channels * 4)?;
            dev.dealloc(d_w, conv_channels * conv_kernel * 4)?;
            dev.dealloc(d_out, n_slots * conv_channels * 4)?;
            dev.dealloc(d_hptrs, n_slots * 8)?;
            for s in 0..n_slots { dev.dealloc(hist_bufs[s], hbytes)?; }
        }
        results.push(ShapeResult { m: n_slots, k: conv_channels, n: conv_kernel, seed, max_rel_err: out_err, tolerance: 1e-4, pass: out_err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "gdn_conv_trio_decode_f32_batched_slots", "gdn_conv_trio_decode_f32_batched_slots",
        "F32", "F32", "conv_out + shifted history |err| <= 1e-4 vs F64 oracle", results, pmc)
}

/// `rmsnorm_rope_neox_partial_f16` — fused per-head RMSNorm + partial NeoX RoPE,
/// F16 in place. HEAD_DIM-templated entries (each with __launch_bounds__).
pub fn run_rmsnorm_rope_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("rmsnorm_rope_neox_partial_f16").ok_or_else(|| anyhow!("rmsnorm_rope_neox_partial_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let pmc = B::attributes(&B::kernel(&module, "flambeau_rmsnorm_rope_neox_partial_f16_d128")?)?;

    let theta = 10000.0f32;
    let eps = 1e-6f32;
    // (n_tokens, n_heads, head_dim, rotated_dims, entry)
    let cases = [
        (1usize, 4usize, 128usize, 64usize, "flambeau_rmsnorm_rope_neox_partial_f16_d128"),
        (8, 2, 256, 128, "flambeau_rmsnorm_rope_neox_partial_f16_d256"),
        (2, 4, 64, 32, "flambeau_rmsnorm_rope_neox_partial_f16_d64"),
    ];
    let mut results = Vec::new();
    for (n_tokens, n_heads, head_dim, rotated, entry) in cases {
        let kernel = B::kernel(&module, entry)?;
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (head_dim as u64) * 11);
        let total = n_tokens * n_heads * head_dim;
        let x = f16_input(seed, total);
        let norm_w = f16_input(seed ^ 0xEE, head_dim);
        let positions: Vec<i32> = (0..n_tokens).map(|t| (t as i32) * 5 + 1).collect();

        let mut reference: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        for t in 0..n_tokens {
            for h in 0..n_heads {
                let base = (t * n_heads + h) * head_dim;
                let sum_sq: f64 = (0..head_dim).map(|i| (x[base + i].to_f32() as f64).powi(2)).sum();
                let inv = 1.0 / (sum_sq / head_dim as f64 + eps as f64).sqrt();
                let normed: Vec<f64> = (0..head_dim)
                    .map(|i| x[base + i].to_f32() as f64 * norm_w[i].to_f32() as f64 * inv)
                    .collect();
                let half = rotated / 2;
                for i in 0..head_dim {
                    reference[base + i] = if i < half {
                        let exponent = 2.0 * i as f64 / rotated as f64;
                        let inv_freq = 1.0 / (theta as f64).powf(exponent);
                        let angle = positions[t] as f64 * inv_freq;
                        let (s, c) = angle.sin_cos();
                        f16::from_f32((normed[i] * c - normed[i + half] * s) as f32).to_f32()
                    } else if i < rotated {
                        let lo = i - half;
                        let exponent = 2.0 * lo as f64 / rotated as f64;
                        let inv_freq = 1.0 / (theta as f64).powf(exponent);
                        let angle = positions[t] as f64 * inv_freq;
                        let (s, c) = angle.sin_cos();
                        f16::from_f32((normed[lo] * s + normed[i] * c) as f32).to_f32()
                    } else {
                        f16::from_f32(normed[i] as f32).to_f32()
                    };
                }
            }
        }

        let d_x = alloc_and_upload(&dev, &x);
        let d_w = alloc_and_upload(&dev, &norm_w);
        let d_pos = alloc_and_upload(&dev, &positions);
        {
            let stream = dev.default_stream();
            let nh = n_heads as i32;
            let rd = rotated as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let p_ptr: u64 = d_pos.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
            args.push_arg(&w_ptr);
            args.push_arg(&p_ptr);
            args.push_arg(&nh);
            args.push_arg(&rd);
            args.push_arg(&theta);
            args.push_arg(&eps);
            let cfg = LaunchCfg { grid: (n_tokens as u32, n_heads as u32, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); total];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_x, total * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_x, total * 2)?;
            dev.dealloc(d_w, head_dim * 2)?;
            dev.dealloc(d_pos, n_tokens * 4)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, (head_dim as f32).sqrt() * 0.01);
        results.push(ShapeResult { m: n_tokens, k: n_heads, n: head_dim, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }
    finish_cert(&dev, repo_root, arch, "rmsnorm_rope_neox_partial_f16", "rmsnorm_rope_neox_partial_f16",
        "F16", "F16", "|err| <= 1e-2 * max(|ref|, sqrt(head_dim)) vs F64 oracle", results, pmc)
}

/// `v_unit_norm_per_head_f16` — per-head V RMSNorm (unit weights) in place.
/// One block per (kv_head, token), blockDim = head_dim. F64 oracle, 1e-2.
pub fn run_v_unit_norm_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("v_unit_norm_per_head_f16").ok_or_else(|| anyhow!("v_unit_norm_per_head_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_v_unit_norm_per_head_f16")?;
    let pmc = B::attributes(&kernel)?;

    let eps = 1e-6f32;
    let shapes = [(1usize, 2usize, 128usize), (8, 2, 256), (4, 8, 64)];
    let mut results = Vec::new();
    for (n_tokens, n_kv_heads, head_dim) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (head_dim as u64) * 11);
        let total = n_tokens * n_kv_heads * head_dim;
        let v = f16_input(seed, total);
        let mut reference = vec![0.0f32; total];
        for t in 0..n_tokens {
            for h in 0..n_kv_heads {
                let base = (t * n_kv_heads + h) * head_dim;
                let sum_sq: f64 = (0..head_dim).map(|i| (v[base + i].to_f32() as f64).powi(2)).sum();
                let inv = 1.0 / (sum_sq / head_dim as f64 + eps as f64).sqrt();
                for i in 0..head_dim {
                    reference[base + i] = f16::from_f32((v[base + i].to_f32() as f64 * inv) as f32).to_f32();
                }
            }
        }

        let d_v = alloc_and_upload(&dev, &v);
        {
            let stream = dev.default_stream();
            let nkv = n_kv_heads as i32;
            let hd = head_dim as i32;
            let v_ptr: u64 = d_v.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&v_ptr);
            args.push_arg(&nkv);
            args.push_arg(&hd);
            args.push_arg(&eps);
            let cfg = LaunchCfg { grid: (n_kv_heads as u32, n_tokens as u32, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); total];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_v, total * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe { dev.dealloc(d_v, total * 2)?; }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, (head_dim as f32).sqrt() * 0.01);
        results.push(ShapeResult { m: n_tokens, k: n_kv_heads, n: head_dim, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }
    finish_cert(&dev, repo_root, arch, "v_unit_norm_per_head_f16", "v_unit_norm_per_head_f16",
        "F16", "F16", "|err| <= 1e-2 * max(|ref|, sqrt(head_dim))", results, pmc)
}

/// `kv_append_v_unit_norm_f16` — fused per-head V RMSNorm + K/V append. One
/// HEAD_DIM-templated entry per head_dim (each with its own __launch_bounds__).
/// K is a plain copy; V is unit-RMS-normalized. Compares the written cache rows.
pub fn run_kv_append_v_unit_norm_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("kv_append_v_unit_norm_f16").ok_or_else(|| anyhow!("kv_append_v_unit_norm_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let pmc = B::attributes(&B::kernel(&module, "flambeau_kv_append_v_unit_norm_f16_d128")?)?;

    let eps = 1e-6f32;
    // (n_tokens, n_kv_heads, head_dim, entry)
    let cases = [
        (4usize, 2usize, 64usize, "flambeau_kv_append_v_unit_norm_f16_d64"),
        (8, 2, 128, "flambeau_kv_append_v_unit_norm_f16_d128"),
        (4, 8, 256, "flambeau_kv_append_v_unit_norm_f16_d256"),
    ];
    let max_seq = 64usize;
    let mut results = Vec::new();
    for (n_tokens, n_kv_heads, head_dim, entry) in cases {
        let kernel = B::kernel(&module, entry)?;
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (head_dim as u64) * 11);
        let src_elems = n_tokens * n_kv_heads * head_dim;
        let k_src = f16_input(seed, src_elems);
        let v_src = f16_input(seed ^ 0xAA, src_elems);
        let write_pos = 0i32;
        // Reference cache rows (absolute addressing, ring_depth=0).
        let cache_elems = max_seq * n_kv_heads * head_dim;
        let mut ref_k = vec![f16::from_f32(0.0); cache_elems];
        let mut ref_v = vec![f16::from_f32(0.0); cache_elems];
        for t in 0..n_tokens {
            for h in 0..n_kv_heads {
                let s = (t * n_kv_heads + h) * head_dim;
                let d = ((write_pos as usize + t) * n_kv_heads + h) * head_dim;
                let sum_sq: f64 = (0..head_dim).map(|i| (v_src[s + i].to_f32() as f64).powi(2)).sum();
                let inv = 1.0 / (sum_sq / head_dim as f64 + eps as f64).sqrt();
                for i in 0..head_dim {
                    ref_k[d + i] = k_src[s + i];
                    ref_v[d + i] = f16::from_f32((v_src[s + i].to_f32() as f64 * inv) as f32);
                }
            }
        }

        let d_ks = alloc_and_upload(&dev, &k_src);
        let d_vs = alloc_and_upload(&dev, &v_src);
        let d_kc = dev.alloc(cache_elems * 2)?;
        let d_vc = dev.alloc(cache_elems * 2)?;
        {
            let stream = dev.default_stream();
            let nkv = n_kv_heads as i32;
            let wp = write_pos;
            let rd = 0i32;
            let ks: u64 = d_ks.as_usize() as u64;
            let vs: u64 = d_vs.as_usize() as u64;
            let kc: u64 = d_kc.as_usize() as u64;
            let vc: u64 = d_vc.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&ks);
            args.push_arg(&vs);
            args.push_arg(&kc);
            args.push_arg(&vc);
            args.push_arg(&nkv);
            args.push_arg(&wp);
            args.push_arg(&eps);
            args.push_arg(&rd);
            let cfg = LaunchCfg { grid: (n_tokens as u32, n_kv_heads as u32, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_k = vec![f16::from_f32(0.0); cache_elems];
        let mut got_v = vec![f16::from_f32(0.0); cache_elems];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_k.as_mut_ptr() as usize), d_kc, cache_elems * 2)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_v.as_mut_ptr() as usize), d_vc, cache_elems * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_ks, src_elems * 2)?;
            dev.dealloc(d_vs, src_elems * 2)?;
            dev.dealloc(d_kc, cache_elems * 2)?;
            dev.dealloc(d_vc, cache_elems * 2)?;
        }
        // K bit-exact; V compared in f32 (RMS + f16 round).
        let mut k_bad = 0u32;
        for t in 0..n_tokens {
            for h in 0..n_kv_heads {
                let d = ((write_pos as usize + t) * n_kv_heads + h) * head_dim;
                for i in 0..head_dim {
                    if got_k[d + i].to_bits() != ref_k[d + i].to_bits() { k_bad += 1; }
                }
            }
        }
        let gv: Vec<f32> = got_v.iter().map(|v| v.to_f32()).collect();
        let rv: Vec<f32> = ref_v.iter().map(|v| v.to_f32()).collect();
        let v_err = max_rel_err_with_floor(&gv, &rv, (head_dim as f32).sqrt() * 0.01);
        let err = (k_bad as f32).max(v_err);
        results.push(ShapeResult { m: n_tokens, k: n_kv_heads, n: head_dim, seed, max_rel_err: err, tolerance: 1e-2, pass: err <= 1e-2 });
    }
    finish_cert(&dev, repo_root, arch, "kv_append_v_unit_norm_f16", "kv_append_v_unit_norm_f16",
        "F16", "F16", "K bit-exact; V |err| <= 1e-2 * max(|ref|, sqrt(head_dim))", results, pmc)
}

/// `dense_gemv_{f32,f16}_f16[_batched]` — per-row dense GEMV, F16 activation,
/// F32 output. `w_f16` selects the weight dtype; `n_tokens_b > 0` selects the
/// batched variant. F64 oracle, 1e-3.
pub fn run_dense_gemv_sweep<B: BenchBackend>(
    repo_root: &Path,
    arch: &str,
    stem: &'static str,
    entry: &'static str,
    w_f16: bool,
    n_tokens_b: usize,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;
    let batched = n_tokens_b > 0;
    let n_tokens = if batched { n_tokens_b } else { 1 };

    let shapes = [(4usize, 256usize), (16, 1024), (128, 2048)];
    let mut results = Vec::new();
    for (n_rows, k) in shapes {
        let seed = 0xC0FFEE ^ ((n_rows as u64) * 1049 + (k as u64) * 11 + (n_tokens as u64) * 7);
        let w_f32 = seeded_f32_range(seed, n_rows * k, -0.5, 0.5);
        let w_eff: Vec<f32> = if w_f16 {
            w_f32.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
        } else {
            w_f32.clone()
        };
        let x = f16_input(seed ^ 0xAA, n_tokens * k);
        let mut reference = vec![0.0f32; n_tokens * n_rows];
        for t in 0..n_tokens {
            for r in 0..n_rows {
                let acc: f64 = (0..k)
                    .map(|i| w_eff[r * k + i] as f64 * x[t * k + i].to_f32() as f64)
                    .sum();
                reference[t * n_rows + r] = acc as f32;
            }
        }

        let d_w = if w_f16 {
            let w16: Vec<f16> = w_eff.iter().map(|v| f16::from_f32(*v)).collect();
            alloc_and_upload(&dev, &w16)
        } else {
            alloc_and_upload(&dev, &w_f32)
        };
        let w_bytes = n_rows * k * if w_f16 { 2 } else { 4 };
        let d_x = alloc_and_upload(&dev, &x);
        let d_y = dev.alloc(n_tokens * n_rows * 4)?;
        {
            let stream = dev.default_stream();
            let nr = n_rows as i32;
            let kk = k as i32;
            let nt = n_tokens as i32;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&w_ptr);
            args.push_arg(&x_ptr);
            args.push_arg(&y_ptr);
            args.push_arg(&nr);
            args.push_arg(&kk);
            if batched {
                args.push_arg(&nt);
            }
            let grid = if batched { (n_rows as u32, n_tokens as u32, 1) } else { (n_rows as u32, 1, 1) };
            B::launch(&kernel, stream, LaunchCfg { grid, block: (256, 1, 1), shared_bytes: 0 }, args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; n_tokens * n_rows];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_y, n_tokens * n_rows * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_w, w_bytes)?;
            dev.dealloc(d_x, n_tokens * k * 2)?;
            dev.dealloc(d_y, n_tokens * n_rows * 4)?;
        }
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_rows, k, n: n_tokens, seed, max_rel_err: err, tolerance: 1e-3, pass: err <= 1e-3 });
    }
    let dw = if w_f16 { "F16" } else { "F32" };
    finish_cert(&dev, repo_root, arch, stem, stem, dw, "F16",
        "|err| <= 1e-3 * max(|ref|, 1) vs F64 oracle", results, pmc)
}

/// Shared driver for the F32-input gated activations `act(a)*b` (silu/gelu),
/// output F32 or saturated F16. `silu` selects silu vs tanh-approx gelu;
/// `out_f16` selects the output dtype. F64 oracle.
pub fn run_binary_act_sweep<B: BenchBackend>(
    repo_root: &Path,
    arch: &str,
    stem: &'static str,
    entry: &'static str,
    out_f16: bool,
    silu: bool,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let act = |x: f64| -> f64 {
        if silu {
            x / (1.0 + (-x).exp())
        } else {
            const C: f64 = 0.797_884_560_802_865_4;
            const A: f64 = 0.044715;
            0.5 * x * (1.0 + (C * x * (1.0 + A * x * x)).tanh())
        }
    };
    let out_sz = if out_f16 { 2 } else { 4 };
    let shapes = [256usize, 1024, 4096];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64).wrapping_mul(191).wrapping_add(if out_f16 { 1 } else { 0 }));
        let a = seeded_f32_range(seed, n, -2.0, 2.0);
        let b = seeded_f32_range(seed ^ 0xBE, n, -1.0, 1.0);
        let reference: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(&av, &bv)| {
                let v = (act(av as f64) * bv as f64) as f32;
                if out_f16 { f16::from_f32(v.clamp(-65504.0, 65504.0)).to_f32() } else { v }
            })
            .collect();

        let d_a = alloc_and_upload(&dev, &a);
        let d_b = alloc_and_upload(&dev, &b);
        let d_y = dev.alloc(n * out_sz)?;
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
        let got: Vec<f32> = if out_f16 {
            let mut g = vec![f16::from_f32(0.0); n];
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(g.as_mut_ptr() as usize), d_y, n * 2)?;
            }
            dev.default_stream().synchronize()?;
            g.iter().map(|v| v.to_f32()).collect()
        } else {
            let mut g = vec![0.0f32; n];
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(g.as_mut_ptr() as usize), d_y, n * 4)?;
            }
            dev.default_stream().synchronize()?;
            g
        };
        unsafe {
            dev.dealloc(d_a, n * 4)?;
            dev.dealloc(d_b, n * 4)?;
            dev.dealloc(d_y, n * out_sz)?;
        }
        let (floor, tol) = if out_f16 { (1e-2, 5e-3) } else { (1.0, 1e-4) };
        let err = max_rel_err_with_floor(&got, &reference, floor);
        results.push(ShapeResult { m: n, k: 1, n: 1, seed, max_rel_err: err, tolerance: tol, pass: err <= tol });
    }
    let (dw, da) = if out_f16 { ("F32", "F16") } else { ("F32", "F32") };
    finish_cert(&dev, repo_root, arch, stem, stem, dw, da,
        "|err| <= tol * max(|ref|, floor) vs F64 oracle", results, pmc)
}

/// `moe_combine_no_residual_f32` — F32 expert fan-in (TP-sharded). Pointwise.
pub fn run_moe_combine_no_residual_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("moe_combine_no_residual_f32").ok_or_else(|| anyhow!("moe_combine_no_residual_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_moe_combine_no_residual_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 8usize, 2048usize), (8, 8, 2048), (4, 4, 1024)];
    let mut results = Vec::new();
    for (n_tokens, top_k, hidden) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (top_k as u64) * 53 + (hidden as u64) * 11);
        let total = n_tokens * hidden;
        let expert_outs = seeded_f32_range(seed, n_tokens * top_k * hidden, -0.5, 0.5);
        let weights = seeded_f32_range(seed ^ 0x11, n_tokens * top_k, -0.5, 0.5);
        let mut reference = vec![0.0f32; total];
        for token in 0..n_tokens {
            for d in 0..hidden {
                let mut acc = 0.0f64;
                for k in 0..top_k {
                    acc += weights[token * top_k + k] as f64
                        * expert_outs[(token * top_k + k) * hidden + d] as f64;
                }
                reference[token * hidden + d] = acc as f32;
            }
        }

        let d_eo = alloc_and_upload(&dev, &expert_outs);
        let d_w = alloc_and_upload(&dev, &weights);
        let d_out = dev.alloc(total * 4)?;
        {
            let stream = dev.default_stream();
            let nt = n_tokens as i32;
            let tk = top_k as i32;
            let h = hidden as i32;
            let eo: u64 = d_eo.as_usize() as u64;
            let w: u64 = d_w.as_usize() as u64;
            let o: u64 = d_out.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&eo);
            args.push_arg(&w);
            args.push_arg(&o);
            args.push_arg(&nt);
            args.push_arg(&tk);
            args.push_arg(&h);
            B::launch(&kernel, stream, LaunchCfg::one_d((total as u32).div_ceil(256), 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; total];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_out, total * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_eo, n_tokens * top_k * hidden * 4)?;
            dev.dealloc(d_w, n_tokens * top_k * 4)?;
            dev.dealloc(d_out, total * 4)?;
        }
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_tokens, k: top_k, n: hidden, seed, max_rel_err: err, tolerance: 1e-4, pass: err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "moe_combine_no_residual_f32", "moe_combine_no_residual_f32",
        "F32", "F32", "|err| <= 1e-4 * max(|ref|, 1) vs F64 oracle", results, pmc)
}

/// `moe_sort_by_expert` — group (token,slot) pairs by expert via histogram +
/// prefix-sum + deterministic scatter. Wave-agnostic (atomics + LDS scan, no
/// warp ops) → MIRROR. The deterministic pipeline (zero→count→scan→scatter_det)
/// is bit-reproducible, so counts/offsets/sorted_pair_idx match the CPU oracle
/// exactly on both rigs.
pub fn run_moe_sort_by_expert_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("moe_sort_by_expert").ok_or_else(|| anyhow!("moe_sort_by_expert not compiled"))?;
    let module = B::load(&dev, image)?;
    let zero_k = B::kernel(&module, "flambeau_moe_sort_zero_counts")?;
    let count_k = B::kernel(&module, "flambeau_moe_sort_count")?;
    let scan_k = B::kernel(&module, "flambeau_moe_sort_scan_offsets")?;
    let scatter_k = B::kernel(&module, "flambeau_moe_sort_scatter_det")?;
    let pmc = B::attributes(&count_k)?;

    let shapes = [(8usize, 8usize, 64usize), (64, 8, 128), (128, 8, 256)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_experts) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (n_experts as u64) * 11);
        let total = n_tokens * top_k;
        let expert_ids: Vec<i32> = (0..total)
            .map(|i| (i.wrapping_mul(2654435761) ^ (seed as usize)) as i32 % n_experts as i32)
            .map(|e| e.rem_euclid(n_experts as i32))
            .collect();
        // CPU oracle: counts, exclusive offsets, stable scatter by input order.
        let mut counts_ref = vec![0i32; n_experts];
        for &e in &expert_ids {
            counts_ref[e as usize] += 1;
        }
        let mut offsets_ref = vec![0i32; n_experts + 1];
        for e in 0..n_experts {
            offsets_ref[e + 1] = offsets_ref[e] + counts_ref[e];
        }
        let mut cursor = offsets_ref[..n_experts].to_vec();
        let mut sorted_ref = vec![0i32; total];
        for (i, &e) in expert_ids.iter().enumerate() {
            let pos = cursor[e as usize];
            cursor[e as usize] += 1;
            sorted_ref[pos as usize] = i as i32;
        }

        let d_ids = alloc_and_upload(&dev, &expert_ids);
        let d_counts = dev.alloc(n_experts * 4)?;
        let d_offsets = dev.alloc((n_experts + 1) * 4)?;
        let d_cursors = dev.alloc(n_experts * 4)?;
        let d_sorted = dev.alloc(total * 4)?;
        {
            let stream = dev.default_stream();
            let ne = n_experts as i32;
            let tot = total as i32;
            let ids: u64 = d_ids.as_usize() as u64;
            let cnt: u64 = d_counts.as_usize() as u64;
            let off: u64 = d_offsets.as_usize() as u64;
            let cur: u64 = d_cursors.as_usize() as u64;
            let srt: u64 = d_sorted.as_usize() as u64;

            let mut a0 = <B::Args<'_>>::default();
            a0.push_arg(&cnt);
            a0.push_arg(&ne);
            B::launch(&zero_k, stream, LaunchCfg { grid: (1, 1, 1), block: (512, 1, 1), shared_bytes: 0 }, a0)?;

            let mut a1 = <B::Args<'_>>::default();
            a1.push_arg(&ids);
            a1.push_arg(&cnt);
            a1.push_arg(&tot);
            B::launch(&count_k, stream, LaunchCfg::one_d((total as u32).div_ceil(256), 256), a1)?;

            let mut a2 = <B::Args<'_>>::default();
            a2.push_arg(&cnt);
            a2.push_arg(&off);
            a2.push_arg(&cur);
            a2.push_arg(&ne);
            B::launch(&scan_k, stream, LaunchCfg { grid: (1, 1, 1), block: (512, 1, 1), shared_bytes: 0 }, a2)?;

            let mut a3 = <B::Args<'_>>::default();
            a3.push_arg(&ids);
            a3.push_arg(&cur);
            a3.push_arg(&srt);
            a3.push_arg(&tot);
            B::launch(&scatter_k, stream, LaunchCfg { grid: (1, 1, 1), block: (1, 1, 1), shared_bytes: 0 }, a3)?;
            stream.synchronize()?;
        }
        let mut got_counts = vec![0i32; n_experts];
        let mut got_offsets = vec![0i32; n_experts + 1];
        let mut got_sorted = vec![0i32; total];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_counts.as_mut_ptr() as usize), d_counts, n_experts * 4)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_offsets.as_mut_ptr() as usize), d_offsets, (n_experts + 1) * 4)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_sorted.as_mut_ptr() as usize), d_sorted, total * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_ids, total * 4)?;
            dev.dealloc(d_counts, n_experts * 4)?;
            dev.dealloc(d_offsets, (n_experts + 1) * 4)?;
            dev.dealloc(d_cursors, n_experts * 4)?;
            dev.dealloc(d_sorted, total * 4)?;
        }
        let mismatches = (got_counts != counts_ref) as u32
            + (got_offsets != offsets_ref) as u32
            + (got_sorted != sorted_ref) as u32;
        let err = mismatches as f32;
        results.push(ShapeResult { m: n_tokens, k: n_experts, n: total, seed, max_rel_err: err, tolerance: 0.0, pass: err == 0.0 });
    }
    finish_cert(&dev, repo_root, arch, "moe_sort_by_expert", "moe_sort_by_expert", "I32", "I32",
        "exact (counts + exclusive offsets + deterministic scatter vs CPU oracle)", results, pmc)
}

/// `attention_decode_f16_splitk` — flash-decoding split-K decode attention.
/// Two-pass: chunk (grid n_heads_q × n_chunks → partial m/s/o) then combine
/// (grid n_heads_q → out). Same F64 attention oracle as the single-pass kernel.
pub fn run_attention_decode_splitk_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("attention_decode_f16_splitk").ok_or_else(|| anyhow!("attention_decode_f16_splitk not compiled"))?;
    let module = B::load(&dev, image)?;
    let chunk_k = B::kernel(&module, "flambeau_attention_decode_f16_splitk_chunk")?;
    let combine_k = B::kernel(&module, "flambeau_attention_decode_f16_splitk_combine")?;
    let pmc = B::attributes(&chunk_k)?;

    // (n_heads_q, n_heads_kv, head_dim, n_tokens, n_chunks)
    let shapes = [(16usize, 2usize, 256usize, 512usize, 8usize), (8, 8, 128, 1024, 8), (4, 4, 64, 300, 4)];
    let mut results = Vec::new();
    for (n_heads_q, n_heads_kv, head_dim, n_tokens, n_chunks) in shapes {
        let seed = 0xC0FFEE ^ ((n_heads_q as u64) * 1049 + (head_dim as u64) * 53 + (n_tokens as u64) * 11);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let chunk_size = n_tokens.div_ceil(n_chunks);
        let q = f16_input(seed, n_heads_q * head_dim);
        let k = f16_input(seed ^ 0xAA, n_tokens * n_heads_kv * head_dim);
        let v = f16_input(seed ^ 0xBB, n_tokens * n_heads_kv * head_dim);
        let group = n_heads_q / n_heads_kv;

        let mut reference = vec![0.0f32; n_heads_q * head_dim];
        for qh in 0..n_heads_q {
            let kvh = qh / group;
            let scores: Vec<f64> = (0..n_tokens)
                .map(|t| {
                    let dot: f64 = (0..head_dim)
                        .map(|d| q[qh * head_dim + d].to_f32() as f64
                            * k[(t * n_heads_kv + kvh) * head_dim + d].to_f32() as f64)
                        .sum();
                    dot * scale as f64
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores.iter().map(|&s| (s - m).exp()).sum();
            for d in 0..head_dim {
                let acc: f64 = (0..n_tokens)
                    .map(|t| (scores[t] - m).exp() * v[(t * n_heads_kv + kvh) * head_dim + d].to_f32() as f64)
                    .sum();
                reference[qh * head_dim + d] = (acc / denom) as f32;
            }
        }

        let d_q = alloc_and_upload(&dev, &q);
        let d_k = alloc_and_upload(&dev, &k);
        let d_v = alloc_and_upload(&dev, &v);
        let d_pm = dev.alloc(n_heads_q * n_chunks * 4)?;
        let d_ps = dev.alloc(n_heads_q * n_chunks * 4)?;
        let d_po = dev.alloc(n_heads_q * n_chunks * head_dim * 4)?;
        let d_out = dev.alloc(n_heads_q * head_dim * 2)?;
        {
            let stream = dev.default_stream();
            let nhq = n_heads_q as i32;
            let nhkv = n_heads_kv as i32;
            let hd = head_dim as i32;
            let nt = n_tokens as i32;
            let nc = n_chunks as i32;
            let cs = chunk_size as i32;
            let (ws, rd, cb) = (0i32, 0i32, 0i32);
            let q_ptr: u64 = d_q.as_usize() as u64;
            let k_ptr: u64 = d_k.as_usize() as u64;
            let v_ptr: u64 = d_v.as_usize() as u64;
            let pm: u64 = d_pm.as_usize() as u64;
            let ps: u64 = d_ps.as_usize() as u64;
            let po: u64 = d_po.as_usize() as u64;
            let mut a = <B::Args<'_>>::default();
            a.push_arg(&q_ptr);
            a.push_arg(&k_ptr);
            a.push_arg(&v_ptr);
            a.push_arg(&pm);
            a.push_arg(&ps);
            a.push_arg(&po);
            a.push_arg(&nhq);
            a.push_arg(&nhkv);
            a.push_arg(&hd);
            a.push_arg(&nt);
            a.push_arg(&nc);
            a.push_arg(&cs);
            a.push_arg(&scale);
            a.push_arg(&ws);
            a.push_arg(&rd);
            a.push_arg(&cb);
            let cfg = LaunchCfg { grid: (n_heads_q as u32, n_chunks as u32, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
            B::launch(&chunk_k, stream, cfg, a)?;
            stream.synchronize()?;

            let out_ptr: u64 = d_out.as_usize() as u64;
            let mut c = <B::Args<'_>>::default();
            c.push_arg(&pm);
            c.push_arg(&ps);
            c.push_arg(&po);
            c.push_arg(&out_ptr);
            c.push_arg(&nhq);
            c.push_arg(&nc);
            c.push_arg(&hd);
            let cfg2 = LaunchCfg { grid: (n_heads_q as u32, 1, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
            B::launch(&combine_k, stream, cfg2, c)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); n_heads_q * head_dim];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_out, n_heads_q * head_dim * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_q, n_heads_q * head_dim * 2)?;
            dev.dealloc(d_k, n_tokens * n_heads_kv * head_dim * 2)?;
            dev.dealloc(d_v, n_tokens * n_heads_kv * head_dim * 2)?;
            dev.dealloc(d_pm, n_heads_q * n_chunks * 4)?;
            dev.dealloc(d_ps, n_heads_q * n_chunks * 4)?;
            dev.dealloc(d_po, n_heads_q * n_chunks * head_dim * 4)?;
            dev.dealloc(d_out, n_heads_q * head_dim * 2)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_heads_q, k: head_dim, n: n_tokens, seed, max_rel_err: err, tolerance: 5e-3, pass: err <= 5e-3 });
    }
    finish_cert(&dev, repo_root, arch, "attention_decode_f16_splitk", "attention_decode_f16_splitk",
        "F16", "F16", "|err| <= 5e-3 * max(|ref|, 1) vs F64 attention oracle", results, pmc)
}

/// `attention_decode_f16` — GQA decode attention (1 Q token, full-context K/V),
/// flash-attn-v2 online softmax. The Q·K dot is a block sum over the WARP_SIZE
/// seam; the rest is per-thread. Full causal (window_size=0), F64 oracle, 5e-3.
pub fn run_attention_decode_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("attention_decode_f16").ok_or_else(|| anyhow!("attention_decode_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_attention_decode_f16")?;
    let pmc = B::attributes(&kernel)?;

    // (n_heads_q, n_heads_kv, head_dim, n_tokens)
    let shapes = [(4usize, 4usize, 128usize, 32usize), (16, 2, 256, 64), (8, 8, 64, 16)];
    let mut results = Vec::new();
    for (n_heads_q, n_heads_kv, head_dim, n_tokens) in shapes {
        let seed = 0xC0FFEE ^ ((n_heads_q as u64) * 1049 + (head_dim as u64) * 53 + (n_tokens as u64) * 11);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let q = f16_input(seed, n_heads_q * head_dim);
        let k = f16_input(seed ^ 0xAA, n_tokens * n_heads_kv * head_dim);
        let v = f16_input(seed ^ 0xBB, n_tokens * n_heads_kv * head_dim);
        let group = n_heads_q / n_heads_kv;

        let mut reference = vec![0.0f32; n_heads_q * head_dim];
        for qh in 0..n_heads_q {
            let kvh = qh / group;
            let scores: Vec<f64> = (0..n_tokens)
                .map(|t| {
                    let dot: f64 = (0..head_dim)
                        .map(|d| q[qh * head_dim + d].to_f32() as f64
                            * k[(t * n_heads_kv + kvh) * head_dim + d].to_f32() as f64)
                        .sum();
                    dot * scale as f64
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores.iter().map(|&s| (s - m).exp()).sum();
            for d in 0..head_dim {
                let acc: f64 = (0..n_tokens)
                    .map(|t| (scores[t] - m).exp() * v[(t * n_heads_kv + kvh) * head_dim + d].to_f32() as f64)
                    .sum();
                reference[qh * head_dim + d] = (acc / denom) as f32;
            }
        }

        let d_q = alloc_and_upload(&dev, &q);
        let d_k = alloc_and_upload(&dev, &k);
        let d_v = alloc_and_upload(&dev, &v);
        let d_out = dev.alloc(n_heads_q * head_dim * 2)?;
        {
            let stream = dev.default_stream();
            let nhq = n_heads_q as i32;
            let nhkv = n_heads_kv as i32;
            let hd = head_dim as i32;
            let nt = n_tokens as i32;
            let ws = 0i32;
            let rd = 0i32;
            let q_ptr: u64 = d_q.as_usize() as u64;
            let k_ptr: u64 = d_k.as_usize() as u64;
            let v_ptr: u64 = d_v.as_usize() as u64;
            let o_ptr: u64 = d_out.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&q_ptr);
            args.push_arg(&k_ptr);
            args.push_arg(&v_ptr);
            args.push_arg(&o_ptr);
            args.push_arg(&nhq);
            args.push_arg(&nhkv);
            args.push_arg(&hd);
            args.push_arg(&nt);
            args.push_arg(&scale);
            args.push_arg(&ws);
            args.push_arg(&rd);
            let cfg = LaunchCfg { grid: (n_heads_q as u32, 1, 1), block: (head_dim as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); n_heads_q * head_dim];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_out, n_heads_q * head_dim * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_q, n_heads_q * head_dim * 2)?;
            dev.dealloc(d_k, n_tokens * n_heads_kv * head_dim * 2)?;
            dev.dealloc(d_v, n_tokens * n_heads_kv * head_dim * 2)?;
            dev.dealloc(d_out, n_heads_q * head_dim * 2)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_heads_q, k: head_dim, n: n_tokens, seed, max_rel_err: err, tolerance: 5e-3, pass: err <= 5e-3 });
    }
    finish_cert(&dev, repo_root, arch, "attention_decode_f16", "attention_decode_f16", "F16", "F16",
        "|err| <= 5e-3 * max(|ref|, 1) vs F64 attention oracle", results, pmc)
}

/// `kv_append_f16_batched_slots` — append one K+V token row into each of N
/// per-slot KV caches (per-slot base pointers + write positions). Strided copy;
/// bit-exact both rigs. Builds N independent cache buffers + a device array of
/// their pointers.
pub fn run_kv_append_f16_batched_slots_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("kv_append_f16_batched_slots").ok_or_else(|| anyhow!("kv_append_f16_batched_slots not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_kv_append_f16_batched_slots")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 1024usize, 16usize), (4, 2048, 32), (8, 512, 8)];
    let ring_depth = 0i32;
    let mut results = Vec::new();
    for (n_slots, kv_width, max_seq) in shapes {
        let seed = 0xC0FFEE ^ ((n_slots as u64) * 1049 + (kv_width as u64) * 11);
        let k_src = f16_input(seed, n_slots * kv_width);
        let v_src = f16_input(seed ^ 0xAA, n_slots * kv_width);
        let write_pos: Vec<i32> = (0..n_slots).map(|s| (s % max_seq) as i32).collect();

        let d_ksrc = alloc_and_upload(&dev, &k_src);
        let d_vsrc = alloc_and_upload(&dev, &v_src);
        let d_wp = alloc_and_upload(&dev, &write_pos);
        // Per-slot cache buffers + device arrays of their base pointers.
        let cache_elems = max_seq * kv_width;
        let mut k_caches = Vec::with_capacity(n_slots);
        let mut v_caches = Vec::with_capacity(n_slots);
        let mut k_ptrs = vec![0u64; n_slots];
        let mut v_ptrs = vec![0u64; n_slots];
        for s in 0..n_slots {
            let kc = dev.alloc(cache_elems * 2)?;
            let vc = dev.alloc(cache_elems * 2)?;
            k_ptrs[s] = kc.as_usize() as u64;
            v_ptrs[s] = vc.as_usize() as u64;
            k_caches.push(kc);
            v_caches.push(vc);
        }
        let d_kptrs = alloc_and_upload(&dev, &k_ptrs);
        let d_vptrs = alloc_and_upload(&dev, &v_ptrs);
        {
            let stream = dev.default_stream();
            let ns = n_slots as i32;
            let kw = kv_width as i32;
            let rd = ring_depth;
            let ks: u64 = d_ksrc.as_usize() as u64;
            let vs: u64 = d_vsrc.as_usize() as u64;
            let kp: u64 = d_kptrs.as_usize() as u64;
            let vp: u64 = d_vptrs.as_usize() as u64;
            let wp: u64 = d_wp.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&ks);
            args.push_arg(&vs);
            args.push_arg(&kp);
            args.push_arg(&vp);
            args.push_arg(&wp);
            args.push_arg(&ns);
            args.push_arg(&kw);
            args.push_arg(&rd);
            let cfg = LaunchCfg { grid: (n_slots as u32, 1, 1), block: (128, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        // Read back each slot's written row and compare to the source row.
        let mut mismatches = 0u32;
        for s in 0..n_slots {
            let dst_pos = write_pos[s] as usize; // ring_depth=0 → absolute
            let mut got_k = vec![f16::from_f32(0.0); kv_width];
            let mut got_v = vec![f16::from_f32(0.0); kv_width];
            let row_off = dst_pos * kv_width * 2;
            unsafe {
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(got_k.as_mut_ptr() as usize), DevicePtr(k_caches[s].as_usize() + row_off), kv_width * 2)?;
                dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                    DevicePtr(got_v.as_mut_ptr() as usize), DevicePtr(v_caches[s].as_usize() + row_off), kv_width * 2)?;
            }
            dev.default_stream().synchronize()?;
            for i in 0..kv_width {
                if got_k[i].to_bits() != k_src[s * kv_width + i].to_bits() { mismatches += 1; }
                if got_v[i].to_bits() != v_src[s * kv_width + i].to_bits() { mismatches += 1; }
            }
        }
        unsafe {
            dev.dealloc(d_ksrc, n_slots * kv_width * 2)?;
            dev.dealloc(d_vsrc, n_slots * kv_width * 2)?;
            dev.dealloc(d_wp, n_slots * 4)?;
            dev.dealloc(d_kptrs, n_slots * 8)?;
            dev.dealloc(d_vptrs, n_slots * 8)?;
            for s in 0..n_slots {
                dev.dealloc(k_caches[s], cache_elems * 2)?;
                dev.dealloc(v_caches[s], cache_elems * 2)?;
            }
        }
        let err = mismatches as f32;
        results.push(ShapeResult { m: n_slots, k: kv_width, n: max_seq, seed, max_rel_err: err, tolerance: 0.0, pass: err == 0.0 });
    }
    finish_cert(&dev, repo_root, arch, "kv_append_f16_batched_slots", "kv_append_f16_batched_slots",
        "F16", "F16", "exact (strided gather/scatter copy)", results, pmc)
}

/// `sampler_topk_softmax_f32` — single-block top-K + full-vocab softmax over a
/// large vocab. Phase-1 online reduce rides `block_reduce_softmax`; phases 2-5
/// (per-thread heap + bitonic sort) are wave-agnostic. Indices match the CPU
/// oracle exactly (lower-id tiebreak); full-vocab probs to 1e-4.
pub fn run_sampler_topk_softmax_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("sampler_topk_softmax_f32").ok_or_else(|| anyhow!("sampler_topk_softmax_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_sampler_topk_softmax_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1000usize, 50usize, 1.0f32), (50000, 256, 0.8), (151000, 100, 1.25)];
    let mut results = Vec::new();
    for (vocab, k, inv_temp) in shapes {
        let seed = 0xC0FFEE ^ ((vocab as u64) * 17 + (k as u64) * 7);
        let logits = seeded_f32_range(seed, vocab, -8.0, 8.0);
        // CPU oracle: top-k by (value desc, id asc); full-vocab softmax probs.
        let v: Vec<f64> = logits.iter().map(|&l| l as f64 * inv_temp as f64).collect();
        let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = v.iter().map(|&x| (x - m).exp()).sum();
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
        let ref_ids: Vec<i32> = order[..k].iter().map(|&i| i as i32).collect();
        let ref_probs: Vec<f32> = order[..k].iter().map(|&i| ((v[i] - m).exp() / sum) as f32).collect();

        let d_logits = alloc_and_upload(&dev, &logits);
        let d_ids = dev.alloc(k * 4)?;
        let d_probs = dev.alloc(k * 4)?;
        {
            let stream = dev.default_stream();
            let vv = vocab as i32;
            let kk = k as i32;
            let l_ptr: u64 = d_logits.as_usize() as u64;
            let id_ptr: u64 = d_ids.as_usize() as u64;
            let p_ptr: u64 = d_probs.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&l_ptr);
            args.push_arg(&id_ptr);
            args.push_arg(&p_ptr);
            args.push_arg(&vv);
            args.push_arg(&kk);
            args.push_arg(&inv_temp);
            let cfg = LaunchCfg { grid: (1, 1, 1), block: (256, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_ids = vec![0i32; k];
        let mut got_probs = vec![0.0f32; k];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_ids.as_mut_ptr() as usize), d_ids, k * 4)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_probs.as_mut_ptr() as usize), d_probs, k * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_logits, vocab * 4)?;
            dev.dealloc(d_ids, k * 4)?;
            dev.dealloc(d_probs, k * 4)?;
        }
        let idx_mismatch = got_ids.iter().zip(&ref_ids).filter(|(g, r)| g != r).count() as f32;
        let p_err = max_rel_err_with_floor(&got_probs, &ref_probs, 1.0);
        let err = idx_mismatch.max(p_err);
        results.push(ShapeResult { m: vocab, k, n: 1, seed, max_rel_err: err, tolerance: 1e-4, pass: err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "sampler_topk_softmax_f32", "sampler_topk_softmax_f32",
        "F32", "F32", "indices exact + full-vocab probs |err| <= 1e-4 vs CPU oracle", results, pmc)
}

/// `softmax_masked_f16` — per-row masked attention softmax. Two-pass online
/// softmax; the (max, sum) reduce rides `block_reduce_softmax` over the
/// WARP_SIZE seam. Tested with a causal-style F16 mask (0 / large-negative).
pub fn run_softmax_masked_f16_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("softmax_masked_f16").ok_or_else(|| anyhow!("softmax_masked_f16 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_softmax_masked_f16")?;
    let pmc = B::attributes(&kernel)?;

    let scale = 0.125f32;
    let shapes = [(1usize, 128usize), (8, 512), (32, 1024)];
    let mut results = Vec::new();
    for (n_rows, k) in shapes {
        let seed = 0xC0FFEE ^ ((n_rows as u64) * 1049 + (k as u64) * 11);
        let scores = f16_input(seed, n_rows * k);
        // Causal-style mask: position i in a row is masked above a per-row
        // cutoff (mask = -65504), else 0.
        let mut mask = vec![f16::from_f32(0.0); n_rows * k];
        for r in 0..n_rows {
            let cutoff = (k * (r + 1) / (n_rows + 1)).max(1);
            for i in cutoff..k {
                mask[r * k + i] = f16::from_f32(-65504.0);
            }
        }
        let mut reference = vec![0.0f32; n_rows * k];
        for r in 0..n_rows {
            let v: Vec<f64> = (0..k)
                .map(|i| scale as f64 * scores[r * k + i].to_f32() as f64 + mask[r * k + i].to_f32() as f64)
                .collect();
            let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = v.iter().map(|&x| (x - m).exp()).sum();
            for i in 0..k {
                reference[r * k + i] = f16::from_f32(((v[i] - m).exp() / sum) as f32).to_f32();
            }
        }

        let d_s = alloc_and_upload(&dev, &scores);
        let d_m = alloc_and_upload(&dev, &mask);
        let d_o = dev.alloc(n_rows * k * 2)?;
        {
            let stream = dev.default_stream();
            let nr = n_rows as i32;
            let kk = k as i32;
            let s_ptr: u64 = d_s.as_usize() as u64;
            let m_ptr: u64 = d_m.as_usize() as u64;
            let o_ptr: u64 = d_o.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&s_ptr);
            args.push_arg(&m_ptr);
            args.push_arg(&o_ptr);
            args.push_arg(&nr);
            args.push_arg(&kk);
            args.push_arg(&scale);
            B::launch(&kernel, stream, LaunchCfg::one_d(n_rows as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); n_rows * k];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_o, n_rows * k * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_s, n_rows * k * 2)?;
            dev.dealloc(d_m, n_rows * k * 2)?;
            dev.dealloc(d_o, n_rows * k * 2)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_rows, k, n: 1, seed, max_rel_err: err, tolerance: 5e-3, pass: err <= 5e-3 });
    }
    finish_cert(&dev, repo_root, arch, "softmax_masked_f16", "softmax_masked_f16", "F16", "F16",
        "|err| <= 5e-3 * max(|ref|, 1) vs F64 online-softmax oracle", results, pmc)
}

/// `sampler_apply_penalties_f32` — repetition/presence/frequency penalties on
/// F32 logits, one thread per deduped (tok, count) pair. Exact f32 arithmetic,
/// bit-exact both rigs.
pub fn run_sampler_apply_penalties_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("sampler_apply_penalties_f32").ok_or_else(|| anyhow!("sampler_apply_penalties_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_sampler_apply_penalties_f32")?;
    let pmc = B::attributes(&kernel)?;

    let (rep, pres, freq) = (1.1f32, 0.5f32, 0.3f32);
    let shapes = [(1000usize, 50usize), (50000, 200), (151000, 500)];
    let mut results = Vec::new();
    for (vocab, n_pairs) in shapes {
        let seed = 0xC0FFEE ^ ((vocab as u64) * 17 + (n_pairs as u64) * 7);
        let logits = seeded_f32_range(seed, vocab, -8.0, 8.0);
        // Unique token ids (stride through the vocab) + small counts.
        let stride = (vocab / n_pairs).max(1);
        let mut token_counts = vec![0u32; n_pairs * 2];
        for p in 0..n_pairs {
            let tok = ((p * stride) % vocab) as u32;
            token_counts[p * 2] = tok;
            token_counts[p * 2 + 1] = ((p % 5) + 1) as u32;
        }
        // F64 oracle — the device may contract `l - freq*count` into an FMA,
        // so compare to a higher-precision reference with a small rel tol
        // rather than bit-exact.
        let mut reference = logits.clone();
        for p in 0..n_pairs {
            let tok = token_counts[p * 2] as usize;
            let count = token_counts[p * 2 + 1] as f64;
            let mut l = reference[tok] as f64;
            if l > 0.0 { l /= rep as f64; } else { l *= rep as f64; }
            l -= pres as f64;
            l -= freq as f64 * count;
            reference[tok] = l as f32;
        }

        let d_logits = alloc_and_upload(&dev, &logits);
        let d_tc = alloc_and_upload(&dev, &token_counts);
        {
            let stream = dev.default_stream();
            let np = n_pairs as i32;
            let v = vocab as i32;
            let l_ptr: u64 = d_logits.as_usize() as u64;
            let tc_ptr: u64 = d_tc.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&l_ptr);
            args.push_arg(&tc_ptr);
            args.push_arg(&np);
            args.push_arg(&v);
            args.push_arg(&rep);
            args.push_arg(&pres);
            args.push_arg(&freq);
            B::launch(&kernel, stream, LaunchCfg::one_d((n_pairs as u32).div_ceil(256), 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; vocab];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_logits, vocab * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_logits, vocab * 4)?;
            dev.dealloc(d_tc, n_pairs * 2 * 4)?;
        }
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: vocab, k: n_pairs, n: 1, seed, max_rel_err: err, tolerance: 1e-5, pass: err <= 1e-5 });
    }
    finish_cert(&dev, repo_root, arch, "sampler_apply_penalties_f32", "sampler_apply_penalties_f32",
        "F32", "F32", "|err| <= 1e-5 * max(|ref|, 1) vs F64 oracle (device FMA contraction)", results, pmc)
}

/// `topk_f32` — per-token router top-k + softmax. One block per token,
/// blockDim = n_experts; the per-round block argmax rides the WARP_SIZE seam.
/// Indices must match the CPU oracle exactly (lower-index tiebreak); weights to
/// 1e-4 (fast `__expf`).
pub fn run_topk_f32_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("topk_f32").ok_or_else(|| anyhow!("topk_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_topk_softmax_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 128usize, 8usize), (8, 256, 8), (1, 512, 10)];
    let mut results = Vec::new();
    for (n_tokens, n_experts, k) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (n_experts as u64) * 53 + (k as u64) * 11);
        let logits = seeded_f32_range(seed, n_tokens * n_experts, -4.0, 4.0);
        // CPU oracle: top-k by (value desc, index asc), softmax of k raw values.
        let mut ref_idx = vec![0i32; n_tokens * k];
        let mut ref_w = vec![0.0f32; n_tokens * k];
        for t in 0..n_tokens {
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| {
                let (va, vb) = (logits[t * n_experts + a], logits[t * n_experts + b]);
                vb.partial_cmp(&va).unwrap().then(a.cmp(&b))
            });
            let top: Vec<usize> = order[..k].to_vec();
            let m = logits[t * n_experts + top[0]];
            let sum: f64 = top.iter().map(|&e| ((logits[t * n_experts + e] - m) as f64).exp()).sum();
            for (r, &e) in top.iter().enumerate() {
                ref_idx[t * k + r] = e as i32;
                ref_w[t * k + r] = (((logits[t * n_experts + e] - m) as f64).exp() / sum) as f32;
            }
        }

        let d_logits = alloc_and_upload(&dev, &logits);
        let d_idx = dev.alloc(n_tokens * k * 4)?;
        let d_w = dev.alloc(n_tokens * k * 4)?;
        {
            let stream = dev.default_stream();
            let nt = n_tokens as i32;
            let ne = n_experts as i32;
            let kk = k as i32;
            let l_ptr: u64 = d_logits.as_usize() as u64;
            let idx_ptr: u64 = d_idx.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&l_ptr);
            args.push_arg(&idx_ptr);
            args.push_arg(&w_ptr);
            args.push_arg(&nt);
            args.push_arg(&ne);
            args.push_arg(&kk);
            let cfg = LaunchCfg { grid: (n_tokens as u32, 1, 1), block: (n_experts as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got_idx = vec![0i32; n_tokens * k];
        let mut got_w = vec![0.0f32; n_tokens * k];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_idx.as_mut_ptr() as usize), d_idx, n_tokens * k * 4)?;
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_w.as_mut_ptr() as usize), d_w, n_tokens * k * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_logits, n_tokens * n_experts * 4)?;
            dev.dealloc(d_idx, n_tokens * k * 4)?;
            dev.dealloc(d_w, n_tokens * k * 4)?;
        }
        let idx_mismatch = got_idx.iter().zip(&ref_idx).filter(|(g, r)| g != r).count() as f32;
        let w_err = max_rel_err_with_floor(&got_w, &ref_w, 1.0);
        let err = idx_mismatch.max(w_err);
        results.push(ShapeResult { m: n_tokens, k: n_experts, n: k, seed, max_rel_err: err, tolerance: 1e-4, pass: err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "topk_f32", "topk_f32", "F32", "F32",
        "indices exact + weights |err| <= 1e-4 vs CPU oracle", results, pmc)
}

/// `moe_combine_*_f16` — expert fan-in `Σ_k w_k·e_k` (+ 0/1/2 F16 residuals).
/// Pure pointwise per (token, d). `n_residuals` selects the variant.
pub fn run_moe_combine_sweep<B: BenchBackend>(repo_root: &Path, arch: &str, n_residuals: usize) -> Result<Cert> {
    let (stem, entry, impl_id) = match n_residuals {
        0 => ("moe_combine_no_residual_f16", "flambeau_moe_combine_no_residual_f16", "moe_combine_no_residual_f16"),
        1 => ("moe_combine_f16", "flambeau_moe_combine_f16", "moe_combine_f16"),
        _ => ("moe_combine_two_residuals_f16", "flambeau_moe_combine_two_residuals_f16", "moe_combine_two_residuals_f16"),
    };
    let dev = B::new_device(0)?;
    let image = B::image(stem).ok_or_else(|| anyhow!("{stem} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 8usize, 2048usize), (8, 8, 2048), (4, 4, 1024)];
    let mut results = Vec::new();
    for (n_tokens, top_k, hidden) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (top_k as u64) * 53 + (hidden as u64) * 11);
        let total = n_tokens * hidden;
        let expert_outs = f16_input(seed, n_tokens * top_k * hidden);
        let weights = seeded_f32_range(seed ^ 0x11, n_tokens * top_k, -0.5, 0.5);
        let r1 = f16_input(seed ^ 0x22, total);
        let r2 = f16_input(seed ^ 0x33, total);

        let mut reference = vec![0.0f32; total];
        for token in 0..n_tokens {
            for d in 0..hidden {
                let idx = token * hidden + d;
                let mut acc = match n_residuals {
                    1 => r1[idx].to_f32() as f64,
                    2 => r1[idx].to_f32() as f64 + r2[idx].to_f32() as f64,
                    _ => 0.0,
                };
                for k in 0..top_k {
                    let w = weights[token * top_k + k] as f64;
                    let e = expert_outs[(token * top_k + k) * hidden + d].to_f32() as f64;
                    acc += w * e;
                }
                reference[idx] = f16::from_f32(acc as f32).to_f32();
            }
        }

        let d_eo = alloc_and_upload(&dev, &expert_outs);
        let d_w = alloc_and_upload(&dev, &weights);
        let d_r1 = alloc_and_upload(&dev, &r1);
        let d_r2 = alloc_and_upload(&dev, &r2);
        let d_out = dev.alloc(total * 2)?;
        {
            let stream = dev.default_stream();
            let nt = n_tokens as i32;
            let tk = top_k as i32;
            let h = hidden as i32;
            let eo_ptr: u64 = d_eo.as_usize() as u64;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let r1_ptr: u64 = d_r1.as_usize() as u64;
            let r2_ptr: u64 = d_r2.as_usize() as u64;
            let out_ptr: u64 = d_out.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&eo_ptr);
            args.push_arg(&w_ptr);
            if n_residuals >= 1 {
                args.push_arg(&r1_ptr);
            }
            if n_residuals >= 2 {
                args.push_arg(&r2_ptr);
            }
            args.push_arg(&out_ptr);
            args.push_arg(&nt);
            args.push_arg(&tk);
            args.push_arg(&h);
            B::launch(&kernel, stream, LaunchCfg::one_d((total as u32).div_ceil(256), 256), args)?;
            stream.synchronize()?;
        }
        let mut got_f16 = vec![f16::from_f32(0.0); total];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got_f16.as_mut_ptr() as usize), d_out, total * 2)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_eo, n_tokens * top_k * hidden * 2)?;
            dev.dealloc(d_w, n_tokens * top_k * 4)?;
            dev.dealloc(d_r1, total * 2)?;
            dev.dealloc(d_r2, total * 2)?;
            dev.dealloc(d_out, total * 2)?;
        }
        let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err_with_floor(&got, &reference, 1e-2);
        results.push(ShapeResult { m: n_tokens, k: top_k, n: hidden, seed, max_rel_err: err, tolerance: 5e-3, pass: err <= 5e-3 });
    }
    finish_cert(&dev, repo_root, arch, impl_id, impl_id, "F16", "F16",
        "|err| <= 5e-3 * max(|ref|, 1e-2) vs F64 oracle", results, pmc)
}

/// `apply_per_expert_scale_f32` — fold expert scalars into routing weights in
/// place. One thread per (token, k); exact f32 multiply (bit-exact both rigs).
pub fn run_apply_per_expert_scale_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("apply_per_expert_scale_f32").ok_or_else(|| anyhow!("apply_per_expert_scale_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_apply_per_expert_scale_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 8usize, 128usize), (8, 8, 128), (4, 4, 32)];
    let mut results = Vec::new();
    for (n_tokens, top_k, n_experts) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (top_k as u64) * 53 + (n_experts as u64) * 11);
        let weights = seeded_f32_range(seed, n_tokens * top_k, -0.5, 0.5);
        let scales = seeded_f32_range(seed ^ 0x44, n_experts, 0.1, 2.0);
        // expert_ids: pseudo-random valid ids.
        let expert_ids: Vec<i32> = (0..n_tokens * top_k)
            .map(|i| ((i.wrapping_mul(2654435761)) % n_experts) as i32)
            .collect();
        let reference: Vec<f32> = (0..n_tokens * top_k)
            .map(|i| weights[i] * scales[expert_ids[i] as usize])
            .collect();

        let d_w = alloc_and_upload(&dev, &weights);
        let d_ids = alloc_and_upload(&dev, &expert_ids);
        let d_scales = alloc_and_upload(&dev, &scales);
        {
            let stream = dev.default_stream();
            let tk = top_k as i32;
            let w_ptr: u64 = d_w.as_usize() as u64;
            let ids_ptr: u64 = d_ids.as_usize() as u64;
            let sc_ptr: u64 = d_scales.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&w_ptr);
            args.push_arg(&ids_ptr);
            args.push_arg(&sc_ptr);
            args.push_arg(&tk);
            let cfg = LaunchCfg { grid: (n_tokens as u32, 1, 1), block: (top_k as u32, 1, 1), shared_bytes: 0 };
            B::launch(&kernel, stream, cfg, args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; n_tokens * top_k];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_w, n_tokens * top_k * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_w, n_tokens * top_k * 4)?;
            dev.dealloc(d_ids, n_tokens * top_k * 4)?;
            dev.dealloc(d_scales, n_experts * 4)?;
        }
        let (err, tol, pass) = bits_mismatch_f32(&got, &reference);
        results.push(ShapeResult { m: n_tokens, k: top_k, n: n_experts, seed, max_rel_err: err, tolerance: tol, pass });
    }
    finish_cert(&dev, repo_root, arch, "apply_per_expert_scale_f32", "apply_per_expert_scale_f32",
        "F32", "F32", "exact (per-element f32 multiply)", results, pmc)
}

/// `shared_expert_scale_f32` — per-token gate `sigmoid(dot(x, gate_w))`, scale
/// `shared_out` in place. Block-reduce dot rides `block_reduce_sum`.
pub fn run_shared_expert_scale_sweep<B: BenchBackend>(repo_root: &Path, arch: &str) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image("shared_expert_scale_f32").ok_or_else(|| anyhow!("shared_expert_scale_f32 not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, "flambeau_shared_expert_scale_f32")?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [(1usize, 2048usize), (8, 2048), (1, 5120)];
    let mut results = Vec::new();
    for (n_tokens, hidden) in shapes {
        let seed = 0xC0FFEE ^ ((n_tokens as u64) * 1049 + (hidden as u64) * 11);
        let x = seeded_f32_range(seed, n_tokens * hidden, -0.25, 0.25);
        let gate_w = seeded_f32_range(seed ^ 0x55, hidden, -0.25, 0.25);
        let shared = seeded_f32_range(seed ^ 0x66, n_tokens * hidden, -0.5, 0.5);
        let mut reference = vec![0.0f32; n_tokens * hidden];
        for t in 0..n_tokens {
            let dot: f64 = (0..hidden).map(|i| gate_w[i] as f64 * x[t * hidden + i] as f64).sum();
            let g = 1.0 / (1.0 + (-dot).exp());
            for i in 0..hidden {
                reference[t * hidden + i] = (shared[t * hidden + i] as f64 * g) as f32;
            }
        }

        let d_out = alloc_and_upload(&dev, &shared);
        let d_x = alloc_and_upload(&dev, &x);
        let d_gw = alloc_and_upload(&dev, &gate_w);
        {
            let stream = dev.default_stream();
            let nt = n_tokens as i32;
            let h = hidden as i32;
            let out_ptr: u64 = d_out.as_usize() as u64;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let gw_ptr: u64 = d_gw.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&out_ptr);
            args.push_arg(&x_ptr);
            args.push_arg(&gw_ptr);
            args.push_arg(&nt);
            args.push_arg(&h);
            B::launch(&kernel, stream, LaunchCfg::one_d(n_tokens as u32, 256), args)?;
            stream.synchronize()?;
        }
        let mut got = vec![0.0f32; n_tokens * hidden];
        unsafe {
            dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize), d_out, n_tokens * hidden * 4)?;
        }
        dev.default_stream().synchronize()?;
        unsafe {
            dev.dealloc(d_out, n_tokens * hidden * 4)?;
            dev.dealloc(d_x, n_tokens * hidden * 4)?;
            dev.dealloc(d_gw, hidden * 4)?;
        }
        let err = max_rel_err_with_floor(&got, &reference, 1.0);
        results.push(ShapeResult { m: n_tokens, k: hidden, n: 1, seed, max_rel_err: err, tolerance: 1e-4, pass: err <= 1e-4 });
    }
    finish_cert(&dev, repo_root, arch, "shared_expert_scale_f32", "shared_expert_scale_f32",
        "F32", "F32", "|err| <= 1e-4 * max(|ref|, 1) vs F64 oracle (fast __expf)", results, pmc)
}

/// `quantize_f16_q8_{0,1}` — F16 activation → Q8 blocks (32 threads/block,
/// width-32 `__shfl_xor` amax/sum). `block_bytes` = 34 (Q8_0) or 36 (Q8_1);
/// `qs_off` = 2 or 4. Scored by dequantizing (`d * qs`) and comparing to the
/// F16 input upcast — cross-backend the integer `rintf`/shuffle path is exact,
/// so this is the Q8 step error.
fn run_quantize_f16_q8_sweep<B: BenchBackend>(
    repo_root: &Path,
    arch: &str,
    impl_id: &'static str,
    entry: &'static str,
    dst_dtype: &str,
    block_bytes: usize,
    qs_off: usize,
) -> Result<Cert> {
    let dev = B::new_device(0)?;
    let image = B::image(impl_id).ok_or_else(|| anyhow!("{impl_id} not compiled"))?;
    let module = B::load(&dev, image)?;
    let kernel = B::kernel(&module, entry)?;
    let pmc = B::attributes(&kernel)?;

    let shapes = [32usize, 256, 4096];
    let mut results = Vec::new();
    for n in shapes {
        let seed = 0xC0FFEE ^ ((n as u64).wrapping_mul(173).wrapping_add(qs_off as u64));
        let x = f16_input(seed, n);
        let xi: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();

        let d_x = alloc_and_upload(&dev, &x);
        let n_blocks = n / 32;
        let out_bytes = n_blocks * block_bytes;
        let d_y = dev.alloc(out_bytes)?;
        {
            let stream = dev.default_stream();
            let n_i = n as i32;
            let x_ptr: u64 = d_x.as_usize() as u64;
            let y_ptr: u64 = d_y.as_usize() as u64;
            let mut args = <B::Args<'_>>::default();
            args.push_arg(&x_ptr);
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
            dev.dealloc(d_x, n * 2)?;
            dev.dealloc(d_y, out_bytes)?;
        }

        let mut dequant = vec![0.0f32; n];
        for (blk_idx, block) in got.chunks(block_bytes).enumerate() {
            let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for lane in 0..32 {
                dequant[blk_idx * 32 + lane] = d * (block[qs_off + lane] as i8) as f32;
            }
        }
        let err = max_rel_err_with_floor(&dequant, &xi, 1.0);
        results.push(ShapeResult { m: n, k: 1, n: 1, seed, max_rel_err: err, tolerance: 5e-3, pass: err <= 5e-3 });
    }

    let pass = results.iter().all(|r| r.pass);
    let cert = Cert {
        schema_version: SCHEMA_VERSION,
        impl_id: impl_id.to_string(),
        backend: dev.backend().to_string(),
        arch: arch.to_string(),
        op: impl_id.to_string(),
        dtype_weight: "F16".to_string(),
        dtype_activation: dst_dtype.to_string(),
        tolerance_formula: "|dequant - input| <= 5e-3 * max(|input|, 1) (Q8 step)".to_string(),
        results,
        pass,
        emitted_at: now_utc_iso8601(),
        rig: rig_for(arch),
        pmc: Some(pmc),
    };
    cert.write_to_disk(repo_root)?;
    Ok(cert)
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
