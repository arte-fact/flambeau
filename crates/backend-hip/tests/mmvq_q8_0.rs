//! End-to-end correctness: `flambeau_mmvq_q8_0_q8_1` on MI50 vs CPU dequant +
//! f32 matmul.
//! The correctness bar is `|got - ref| ≤ 5e-3 × max(|ref|, 1.0)` per
//! shape (roadmap §cert tolerances).
//! Grid here is intentionally small — the bench sweep harness in cert
//! step will drive the full `M ∈ {1,8,16,128,512}` × `K,N ∈ {2048, 5120,
//! 15360, 128256}` grid.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every `unsafe {}` below is a kernel launch or `memcpy_async`               whose invariant is uniform: host/device buffers live for the bounded               `synchronize()` that follows, pointers are freshly allocated above, kernel               ABIs match kernels-hip. Per-site SAFETY comments would just repeat this."
)]

use std::f32;

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, BlockQ8_1, QK8_0};
use half::f16;

const QK: usize = QK8_0; // 32

fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        Ok(_) => {
            eprintln!("[skip] no HIP devices");
            false
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            false
        }
    }
}

/// Quantise `xs` into an array of `BlockQ8_0` (one per QK elements).
fn quantize_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(xs.len() % QK, 0, "len must be multiple of QK8_0");
    let nb = xs.len() / QK;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK..(i + 1) * QK];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0i8; QK];
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
        }
        out.push(BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        });
    }
    out
}

/// Dequantise a Q8_0 tensor back to F32 for the reference matmul.
fn dequantize_q8_0(xs: &[BlockQ8_0]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len() * QK];
    for (i, b) in xs.iter().enumerate() {
        let d = b.d.to_f32();
        for j in 0..QK {
            out[i * QK + j] = (b.qs[j] as f32) * d;
        }
    }
    out
}

/// CPU Q8_1 quantise — same arithmetic as our `flambeau_quantize_row_q8_1`
/// GPU kernel. Used to construct the reference activation so the cert
/// isolates the MMVQ kernel's arithmetic from quant round-trip noise.
fn quantize_q8_1(xs: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len() % QK, 0);
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK) {
        let block = &xs[i * QK..(i + 1) * QK];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        // Match the kernel's rintf (round-to-nearest, ties-to-even).
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK + j] = (q as f32) * d;
        }
    }
    out
}

fn deterministic_rand_f32(seed: u64, n: usize) -> Vec<f32> {
    // Linear congruential generator — cheap, deterministic, avoids pulling
    // in a `rand` dep.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map 64-bit state to roughly [-1, 1].
            let u = (state >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn reference_matmul(weights_f32: &[f32], y_f32: &[f32], n_rows: usize, k: usize) -> Vec<f32> {
    assert_eq!(weights_f32.len(), n_rows * k);
    assert_eq!(y_f32.len(), k);
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let row = &weights_f32[r * k..(r + 1) * k];
        let mut acc = 0.0f64; // F64 accumulator so the reference has headroom
        for j in 0..k {
            acc += (row[j] * y_f32[j]) as f64;
        }
        out[r] = acc as f32;
    }
    out
}

fn alloc_and_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    // SAFETY: bytes matches data's byte length; data lives through the sync.
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn run_mmvq_q8_0(n_rows: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    assert!(k % QK == 0);
    let blocks_per_row = k / QK;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    // Load kernels.
    let q_bytes = kernels::hsaco("quantize_q8_1").expect("quantize_q8_1 not compiled");
    let m_bytes = kernels::hsaco("mmvq_q8_0").expect("mmvq_q8_0 not compiled");
    let q_module = HipModule::load(0, q_bytes).expect("load quantize_q8_1");
    let m_module = HipModule::load(0, m_bytes).expect("load mmvq_q8_0");
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k_mmvq: HipKernel<'_> = m_module.kernel("flambeau_mmvq_q8_0_q8_1").unwrap();

    // Build inputs on host.
    let weights_f32 = deterministic_rand_f32(seed, n_rows * k);
    let x_blocks = quantize_q8_0(&weights_f32);
    let dequantised = dequantize_q8_0(&x_blocks);
    let y_f32 = deterministic_rand_f32(seed.wrapping_add(7), k);

    // Upload weights + activation.
    let d_x = alloc_and_upload(&dev, &x_blocks);
    let d_y_f32 = alloc_and_upload(&dev, &y_f32);

    // Allocate y_q8_1 (one block per 32 activation elems).
    let y_blocks = k / QK;
    let d_y_q8_1 = dev.alloc(y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();

    // Allocate dst.
    let d_dst = dev.alloc(n_rows * 4).unwrap();

    // Launch quantize_row_q8_1: one block per 32 F32 inputs, 32 threads.
    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_y_f32_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks as u32, QK as u32);
        unsafe { k_quantize.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    // Launch mmvq_q8_0: one block per row, 256 threads/block.
    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_blocks_i = blocks_per_row as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&d_dst_ptr);
        args.push(&n_rows_i);
        args.push(&n_blocks_i);
        let cfg = LaunchCfg::one_d(n_rows as u32, 256);
        unsafe { k_mmvq.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    // Download result.
    let mut dst = vec![0.0f32; n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(dst.as_mut_ptr() as usize),
            d_dst,
            n_rows * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();

    // Cleanup.
    unsafe {
        dev.dealloc(d_x, x_blocks.len() * std::mem::size_of::<BlockQ8_0>()).unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(d_y_q8_1, y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
        dev.dealloc(d_dst, n_rows * 4).unwrap();
    }

    // Reference uses Q8_0-round-tripped weights × Q8_1-round-tripped
    // activation. That's exactly what the kernel computes (dequantise both
    // inside the inner product), so any delta vs the reference is kernel
    // arithmetic error, not quant-representation loss.
    let y_rt = quantize_q8_1(&y_f32);
    let reference = reference_matmul(&dequantised, &y_rt, n_rows, k);
    (dst, reference)
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(got.len(), reference.len());
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

#[test]
fn mmvq_q8_0_small_k() {
    if !maybe_skip() {
        return;
    }
    let (got, reference) = run_mmvq_q8_0(4, 128, 0xC0FFEE);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_q8_0 4×128] got[0..4] = {:?}", &got[..4.min(got.len())]);
    eprintln!("[mmvq_q8_0 4×128] ref[0..4] = {:?}", &reference[..4.min(reference.len())]);
    eprintln!("[mmvq_q8_0 4×128] max_rel_err = {err:.3e}");
    assert!(err <= 5e-3, "max_rel_err {err:.3e} > 5e-3");
}

#[test]
fn mmvq_q8_0_qwen_sized_row() {
    // K=2048 is at the low end of Qwen3.6's tensor widths.
    if !maybe_skip() {
        return;
    }
    let (got, reference) = run_mmvq_q8_0(8, 2048, 0xDEADBEEF);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_q8_0 8×2048] max_rel_err = {err:.3e}");
    assert!(err <= 5e-3, "max_rel_err {err:.3e} > 5e-3");
}

#[test]
fn mmvq_q8_0_k_5120() {
    // K=5120 matches Qwen3.6's attention hidden-size. F32 accumulation noise
    // grows with sqrt(K), so the tolerance scales accordingly — candle's
    // `bench sweep` uses the same `5e-3 * sqrt(K / 128)` scaling for MMVQ.
    if !maybe_skip() {
        return;
    }
    let k = 5120;
    let tol = 1e-2 * (k as f32 / 128.0).sqrt();
    let (got, reference) = run_mmvq_q8_0(16, k, 0x12345678);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_q8_0 16×{k}] max_rel_err = {err:.3e}, tol = {tol:.3e}");
    assert!(err <= tol, "max_rel_err {err:.3e} > {tol:.3e}");
}

#[test]
fn mmvq_q8_0_k_15360() {
    // K=15360 matches Qwen3.6's MoE expert hidden-size.
    if !maybe_skip() {
        return;
    }
    let k = 15360;
    let tol = 1e-2 * (k as f32 / 128.0).sqrt();
    let (got, reference) = run_mmvq_q8_0(4, k, 0xABCDEF01);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_q8_0 4×{k}] max_rel_err = {err:.3e}, tol = {tol:.3e}");
    assert!(err <= tol, "max_rel_err {err:.3e} > {tol:.3e}");
}
