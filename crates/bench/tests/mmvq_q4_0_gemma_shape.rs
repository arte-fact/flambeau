//! Targeted parity test for `flambeau_mmvq_q4_0_q8_1` at the exact shape
//! that breaks gemma-4-31B-it-Q4_0 inference: k=8192 (q_width = 32 heads
//! × 256 head_dim for the SWA layers), n=5376 (hidden). Activation is a
//! single row of Q8_1.
//!
//! Compares the kernel output to a CPU dequant+matmul reference at one
//! row only (m=1). Diff is expected within F16+Q8_1 quantize noise
//! (typical: |Δ| < 0.05 per element). Significant divergence at any
//! element identifies a kernel bug.
//!
//! Run: `cargo test --release -p flambeau-bench --features hip --test
//! mmvq_q4_0_gemma_shape -- --nocapture`.

#![cfg(feature = "hip")]
#![allow(clippy::undocumented_unsafe_blocks)]

use flambeau_backend_hip::{device_count, HipDevice, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ4_0, BlockQ8_1, QK4_0, QK8_1};
use half::f16;

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn quantize_row_q4_0(xs: &[f32]) -> Vec<BlockQ4_0> {
    assert_eq!(xs.len() % QK4_0, 0);
    let nb = xs.len() / QK4_0;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK4_0..(i + 1) * QK4_0];
        let amax_signed = block.iter().fold(0.0f32, |m, &v| {
            if v.abs() > m.abs() {
                v
            } else {
                m
            }
        });
        let d = amax_signed / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0u8; QK4_0 / 2];
        for j in 0..(QK4_0 / 2) {
            let x0 = block[j];
            let x1 = block[j + QK4_0 / 2];
            let q0 = (x0 * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            let q1 = (x1 * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            qs[j] = (q1 << 4) | q0;
        }
        out.push(BlockQ4_0 {
            d: f16::from_f32(d),
            qs,
        });
    }
    out
}

fn quantize_row_q8_1(xs: &[f32]) -> Vec<BlockQ8_1> {
    assert_eq!(xs.len() % QK8_1, 0);
    let nb = xs.len() / QK8_1;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK8_1..(i + 1) * QK8_1];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_1];
        let mut sum_i: i32 = 0;
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
            sum_i += q as i32;
        }
        let s = d * sum_i as f32;
        out.push(BlockQ8_1 {
            d: f16::from_f32(d),
            s: f16::from_f32(s),
            qs,
        });
    }
    out
}

fn dequant_q4_0_row(blocks: &[BlockQ4_0]) -> Vec<f32> {
    let mut out = vec![0.0_f32; blocks.len() * QK4_0];
    for (i, b) in blocks.iter().enumerate() {
        let d = b.d.to_f32();
        for j in 0..(QK4_0 / 2) {
            let q0 = (b.qs[j] & 0x0F) as i16 - 8;
            let q1 = (b.qs[j] >> 4) as i16 - 8;
            out[i * QK4_0 + j] = q0 as f32 * d;
            out[i * QK4_0 + j + QK4_0 / 2] = q1 as f32 * d;
        }
    }
    out
}

fn dequant_q8_1_row(blocks: &[BlockQ8_1]) -> Vec<f32> {
    let mut out = vec![0.0_f32; blocks.len() * QK8_1];
    for (i, b) in blocks.iter().enumerate() {
        let d = b.d.to_f32();
        for j in 0..QK8_1 {
            out[i * QK8_1 + j] = b.qs[j] as f32 * d;
        }
    }
    out
}

fn alloc_upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
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

fn download_f32(dev: &HipDevice, ptr: DevicePtr, n: usize) -> Vec<f32> {
    let mut host = vec![0.0_f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            ptr,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

#[test]
fn mmvq_q4_0_at_gemma4_31b_attn_output_shape() {
    if device_count().unwrap_or(0) == 0 {
        eprintln!("SKIP: no HIP devices");
        return;
    }

    // gemma4-31B SWA layer's attn_output projection: k=8192 (n_heads × head_dim_swa = 32×256),
    // n=5376 (hidden). Q4_0 weight, Q8_1 activation, F32 output, single row (m=1).
    const K: usize = 8192;
    const N: usize = 5376;
    const SEED_W: u64 = 0xA110_C0A7;
    const SEED_A: u64 = 0xFEED_FACE;

    let dev = HipDevice::new(0).expect("HipDevice 0");
    dev.bind().expect("bind");
    let stream = dev.default_stream();

    // Build N rows of Q4_0 weights. Each row is K elements quantized to K/32
    // blocks of 18 bytes each.
    let mut weight_blocks: Vec<BlockQ4_0> = Vec::with_capacity(N * (K / QK4_0));
    let mut weight_f32 = Vec::with_capacity(N * K);
    for row in 0..N {
        let row_f32 = seeded_f32(SEED_W.wrapping_add(row as u64), K);
        let blocks = quantize_row_q4_0(&row_f32);
        // Re-dequantize for the CPU reference (the round-tripped values
        // are what the kernel actually multiplies against).
        let row_dq = dequant_q4_0_row(&blocks);
        weight_f32.extend_from_slice(&row_dq);
        weight_blocks.extend_from_slice(&blocks);
    }

    // Activation: 1 row of K elements quantized to K/32 Q8_1 blocks.
    let act_f32 = seeded_f32(SEED_A, K);
    let act_blocks = quantize_row_q8_1(&act_f32);
    let act_dq = dequant_q8_1_row(&act_blocks);

    // CPU reference: dst[row] = sum_k (act_dq[k] * weight_dq[row*K + k]).
    let mut expected = vec![0.0_f32; N];
    for row in 0..N {
        let row_off = row * K;
        let s: f32 = (0..K).map(|k| act_dq[k] * weight_f32[row_off + k]).sum();
        expected[row] = s;
    }

    // Upload.
    let weight_dev = alloc_upload(&dev, &weight_blocks);
    let act_dev = alloc_upload(&dev, &act_blocks);
    let dst_dev = dev.alloc(N * 4).unwrap();

    // Launch flambeau_mmvq_q4_0_q8_1 directly (256 threads, n=N blocks).
    let kernel_bin = kernels::hsaco("mmvq_q4_0").expect("mmvq_q4_0 hsaco");
    let module = HipModule::load(0, kernel_bin).expect("module load");
    let kernel = module
        .kernel("flambeau_mmvq_q4_0_q8_1")
        .expect("kernel symbol");

    let n_blocks_per_row = (K / QK4_0) as i32;
    let n_rows = N as i32;
    let w_ptr: u64 = weight_dev.as_usize() as u64;
    let a_ptr: u64 = act_dev.as_usize() as u64;
    let d_ptr: u64 = dst_dev.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&a_ptr);
    args.push(&d_ptr);
    args.push(&n_rows);
    args.push(&n_blocks_per_row);
    let cfg = LaunchCfg::one_d(N as u32, 256);
    unsafe { kernel.launch(stream, cfg, args).expect("kernel launch") };
    stream.synchronize().expect("sync");

    let got = download_f32(&dev, dst_dev, N);

    // Compare: count elements with |Δ| > tol. With K=8192 Q4_0/Q8_1
    // round-trip noise, expected |Δ| ≈ 0.5-1.0 per row (sum of 8192 noisy
    // products averages out to ~sqrt(8192) × per-element noise of ~0.01
    // ≈ 0.9). Use a loose absolute tol of 2.0 and a relative tol of 5%.
    const ABS_TOL: f32 = 2.0;
    const REL_TOL: f32 = 0.05;
    let mut wrong = 0usize;
    let mut max_abs_err = 0.0_f32;
    let mut max_abs_err_row = 0usize;
    let mut first_wrong: Vec<(usize, f32, f32)> = Vec::new();
    for row in 0..N {
        let err = (got[row] - expected[row]).abs();
        let rel = err / expected[row].abs().max(1e-6);
        if err > max_abs_err {
            max_abs_err = err;
            max_abs_err_row = row;
        }
        if err > ABS_TOL && rel > REL_TOL {
            if first_wrong.len() < 10 {
                first_wrong.push((row, expected[row], got[row]));
            }
            wrong += 1;
        }
    }

    eprintln!(
        "mmvq_q4_0 (k={K}, n={N}, m=1) — max_abs_err={max_abs_err:.4} at row {max_abs_err_row} \
         (expected={:.4}, got={:.4})",
        expected[max_abs_err_row], got[max_abs_err_row]
    );
    if !first_wrong.is_empty() {
        eprintln!("First wrong elements:");
        for (row, e, g) in &first_wrong {
            eprintln!("  row {row}: expected {e:.4}, got {g:.4}, Δ={:.4}", g - e);
        }
    }
    eprintln!("Total wrong rows: {wrong}/{N} (tol abs={ABS_TOL} & rel={REL_TOL})");

    // Sample first 8 rows for visibility.
    eprintln!("First 8 rows expected={:?}", &expected[..8]);
    eprintln!("First 8 rows got     ={:?}", &got[..8]);

    assert_eq!(wrong, 0, "{wrong} rows exceed tolerance");
}
