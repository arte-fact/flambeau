//! End-to-end correctness for `flambeau_mmvq_iq4_xs_q8_1` on MI50 vs CPU
//! dequant + f32 matmul (with Q8_1 activation round-trip to isolate kernel
//! arithmetic from quant-representation noise).
//!
//! Generates IQ4_XS blocks by direct byte randomisation. The 6-bit signed
//! sub-block scales (low4+high2) can land anywhere in [-32, 31] including
//! 0 — the kernel branchlessly multiplies by `ls`, so a 0 scale produces a
//! 0 contribution and the dot product still matches the CPU reference.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every `unsafe {}` is a kernel launch or `memcpy_async` whose               invariant is uniform: host/device buffers live for the bounded               `synchronize()` that follows, pointers are freshly allocated above, kernel               ABIs match kernels-hip."
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "usize → i32 casts here are kernel-shape math (row/col/block indices \
              bounded by GGUF dims); wrap is not possible for any shape we ever run."
)]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockIq4Xs, BlockQ8_1, QK8_0, QK_K};
use half::f16;

const QK8: usize = QK8_0; // 32

fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        _ => {
            eprintln!("[skip] no HIP device");
            false
        }
    }
}

fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 24) as u8
        })
        .collect()
}

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn random_iq4_xs_block(seed: u64, idx: usize) -> BlockIq4Xs {
    // 136 B/block: d(2) + scales_h(2) + scales_l[4] + qs[128].
    let bytes = seeded_bytes(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)), 136);
    let d_scalar = (bytes[0] as f32 / 255.0) * 0.1 + 0.01;
    let d = f16::from_f32(d_scalar);
    let scales_h = u16::from_le_bytes([bytes[2], bytes[3]]);
    let mut scales_l = [0u8; QK_K / 64];
    scales_l.copy_from_slice(&bytes[4..4 + QK_K / 64]);
    let mut qs = [0u8; QK_K / 2];
    qs.copy_from_slice(&bytes[8..8 + QK_K / 2]);
    BlockIq4Xs { d, scales_h, scales_l, qs }
}

fn quantize_q8_1_roundtrip(xs: &[f32]) -> Vec<f32> {
    assert_eq!(xs.len() % QK8, 0);
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..(xs.len() / QK8) {
        let block = &xs[i * QK8..(i + 1) * QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i32;
            out[i * QK8 + j] = (q as f32) * d;
        }
    }
    out
}

fn reference_matmul(weights_f32: &[f32], y_f32: &[f32], n_rows: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let row = &weights_f32[r * k..(r + 1) * k];
        let mut acc = 0.0f64;
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

fn run_mmvq_iq4_xs(n_rows: usize, k: usize, seed: u64, kernel_stem: &'static str, entry: &'static str, rows_per_block: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(k % QK_K, 0, "k must be a multiple of QK_K (256)");
    let superblocks_per_row = k / QK_K;

    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let q_bytes = kernels::hsaco("quantize_q8_1").unwrap();
    let m_bytes = kernels::hsaco(kernel_stem).unwrap();
    let q_module = HipModule::load(0, q_bytes).unwrap();
    let m_module = HipModule::load(0, m_bytes).unwrap();
    let k_quantize: HipKernel<'_> = q_module.kernel("flambeau_quantize_row_q8_1").unwrap();
    let k_mmvq: HipKernel<'_> = m_module.kernel(entry).unwrap();

    let mut x_blocks: Vec<BlockIq4Xs> = Vec::with_capacity(n_rows * superblocks_per_row);
    for i in 0..(n_rows * superblocks_per_row) {
        x_blocks.push(random_iq4_xs_block(seed, i));
    }
    let total_elems = n_rows * k;
    let mut x_dequant = vec![0.0f32; total_elems];
    {
        let raw: &[u8] = bytemuck::cast_slice(&x_blocks);
        flambeau_quant::dequantize_into(
            flambeau_quant::GgmlDType::Iq4Xs,
            raw,
            &mut x_dequant,
        )
        .unwrap();
    }

    let y_f32 = seeded_f32(seed.wrapping_add(31), k);

    let d_x = alloc_and_upload(&dev, &x_blocks);
    let d_y_f32 = alloc_and_upload(&dev, &y_f32);
    let y_blocks = k / QK8;
    let d_y_q8_1 = dev.alloc(y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
    let d_dst = dev.alloc(n_rows * 4).unwrap();

    {
        let stream = dev.default_stream();
        let n_elems = k as i32;
        let d_y_f32_ptr: u64 = d_y_f32.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_y_f32_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&n_elems);
        let cfg = LaunchCfg::one_d(y_blocks as u32, QK8 as u32);
        unsafe { k_quantize.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    {
        let stream = dev.default_stream();
        let n_rows_i = n_rows as i32;
        let n_sb_i = superblocks_per_row as i32;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_y_q8_1_ptr: u64 = d_y_q8_1.as_usize() as u64;
        let d_dst_ptr: u64 = d_dst.as_usize() as u64;
        let mut args = KernelArgs::new();
        args.push(&d_x_ptr);
        args.push(&d_y_q8_1_ptr);
        args.push(&d_dst_ptr);
        args.push(&n_rows_i);
        args.push(&n_sb_i);
        let n_blocks_grid = n_rows.div_ceil(rows_per_block);
        let cfg = LaunchCfg::one_d(n_blocks_grid as u32, 64);
        unsafe { k_mmvq.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

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

    unsafe {
        dev.dealloc(d_x, x_blocks.len() * std::mem::size_of::<BlockIq4Xs>()).unwrap();
        dev.dealloc(d_y_f32, y_f32.len() * 4).unwrap();
        dev.dealloc(d_y_q8_1, y_blocks * std::mem::size_of::<BlockQ8_1>()).unwrap();
        dev.dealloc(d_dst, n_rows * 4).unwrap();
    }

    let y_rt = quantize_q8_1_roundtrip(&y_f32);
    let reference = reference_matmul(&x_dequant, &y_rt, n_rows, k);
    (dst, reference)
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

fn tol_for(k: usize) -> f32 {
    1e-2 * (k as f32 / 128.0).sqrt()
}

#[test]
fn mmvq_iq4_xs_small() {
    if !maybe_skip() { return; }
    let (got, reference) = run_mmvq_iq4_xs(4, QK_K, 0xC0FFEE, "mmvq_iq4_xs", "flambeau_mmvq_iq4_xs_q8_1", 1);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_iq4_xs 4×{QK_K}] err={err:.3e}, tol={:.3e}", tol_for(QK_K));
    assert!(err <= tol_for(QK_K), "err {err:.3e}");
}

#[test]
fn mmvq_iq4_xs_2048() {
    if !maybe_skip() { return; }
    let k = 2048;
    let (got, reference) = run_mmvq_iq4_xs(8, k, 0xFEEDFACE, "mmvq_iq4_xs", "flambeau_mmvq_iq4_xs_q8_1", 1);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_iq4_xs 8×{k}] err={err:.3e}, tol={:.3e}", tol_for(k));
    assert!(err <= tol_for(k), "err {err:.3e}");
}

#[test]
fn mmvq_iq4_xs_5120() {
    if !maybe_skip() { return; }
    let k = 5120;
    let (got, reference) = run_mmvq_iq4_xs(16, k, 0x12345678, "mmvq_iq4_xs", "flambeau_mmvq_iq4_xs_q8_1", 1);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_iq4_xs 16×{k}] err={err:.3e}, tol={:.3e}", tol_for(k));
    assert!(err <= tol_for(k), "err {err:.3e}");
}

#[test]
fn mmvq_iq4_xs_r2_small() {
    if !maybe_skip() { return; }
    let (got, reference) = run_mmvq_iq4_xs(4, QK_K, 0xC0FFEE, "mmvq_iq4_xs_r2", "flambeau_mmvq_iq4_xs_r2_q8_1", 2);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_iq4_xs_r2 4×{QK_K}] err={err:.3e}, tol={:.3e}", tol_for(QK_K));
    assert!(err <= tol_for(QK_K), "err {err:.3e}");
}

#[test]
fn mmvq_iq4_xs_r2_2048() {
    if !maybe_skip() { return; }
    let k = 2048;
    let (got, reference) = run_mmvq_iq4_xs(8, k, 0xFEEDFACE, "mmvq_iq4_xs_r2", "flambeau_mmvq_iq4_xs_r2_q8_1", 2);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_iq4_xs_r2 8×{k}] err={err:.3e}, tol={:.3e}", tol_for(k));
    assert!(err <= tol_for(k), "err {err:.3e}");
}

#[test]
fn mmvq_iq4_xs_r2_odd_rows() {
    if !maybe_skip() { return; }
    let k = 1024;
    let (got, reference) = run_mmvq_iq4_xs(7, k, 0xDEADBEEF, "mmvq_iq4_xs_r2", "flambeau_mmvq_iq4_xs_r2_q8_1", 2);
    let err = max_rel_err(&got, &reference);
    eprintln!("[mmvq_iq4_xs_r2 7×{k}] err={err:.3e}, tol={:.3e}", tol_for(k));
    assert!(err <= tol_for(k), "err {err:.3e}");
}
