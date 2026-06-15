//! Correctness of the templated `mmvq_q8_0_dp4a` instantiation on the GPU:
//! quantize Q8_0 weights × Q8_1 activations, run the kernel, compare against a
//! CPU reference of the same quantized dot. Skips without a cubin / CUDA device.

use flambeau_backend_cuda::{device_count, CudaDevice, CudaModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use half::f16;

const QK: usize = 32; // block size (QK8_0 == QK8_1)

/// Quantize one 32-element block to Q8_0 bytes: `[d: f16-le][qs: 32×i8]`.
/// Returns the `(d, qs)` actually written, so the reference uses identical data.
fn quant_q8_0(block: &[f32], out: &mut Vec<u8>) -> (f16, Vec<i8>) {
    let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let d = amax / 127.0;
    let dh = f16::from_f32(d);
    out.extend_from_slice(&dh.to_le_bytes());
    let mut qs = Vec::with_capacity(block.len());
    for &v in block {
        let q = if d > 0.0 { (v / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
        qs.push(q);
        out.push(q as u8);
    }
    (dh, qs)
}

/// Quantize one 32-element block to Q8_1 bytes: `[d: f16-le][s: f16-le][qs]`.
/// `s` is unused by the Q8_0 vec_dot, written as 0.
fn quant_q8_1(block: &[f32], out: &mut Vec<u8>) -> (f16, Vec<i8>) {
    let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let d = amax / 127.0;
    let dh = f16::from_f32(d);
    out.extend_from_slice(&dh.to_le_bytes());
    out.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
    let mut qs = Vec::with_capacity(QK);
    for &v in block {
        let q = if d > 0.0 { (v / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
        qs.push(q);
        out.push(q as u8);
    }
    (dh, qs)
}

#[test]
fn mmvq_q8_0_matches_cpu_reference() {
    let Some(cubin) = flambeau_kernels_cuda::cubin("mmvq_q8_0_dp4a") else {
        eprintln!("mmvq_q8_0_dp4a cubin absent (CUDA_SKIP_BUILD / no nvcc) — skipping");
        return;
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("no CUDA device — skipping");
        return;
    }

    const N_ROWS: usize = 8;
    const NB: usize = 8; // blocks per row

    // Deterministic pseudo-random inputs.
    let mut seed = 0x9e3779b9u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
    };

    // Weights: [N_ROWS][K] quantized to Q8_0, row-major blocks. Keep per-block
    // (d, qs) for the reference.
    let mut w_bytes = Vec::new();
    let mut w_d = vec![[f16::ZERO; NB]; N_ROWS];
    let mut w_q = vec![[[0i8; QK]; NB]; N_ROWS];
    for r in 0..N_ROWS {
        for b in 0..NB {
            let blk: Vec<f32> = (0..QK).map(|_| rnd() * (1.0 + r as f32 * 0.1)).collect();
            let (dh, qs) = quant_q8_0(&blk, &mut w_bytes);
            w_d[r][b] = dh;
            w_q[r][b].copy_from_slice(&qs);
        }
    }

    // Activation: [K] quantized to Q8_1.
    let mut a_bytes = Vec::new();
    let mut a_d = [f16::ZERO; NB];
    let mut a_q = [[0i8; QK]; NB];
    for b in 0..NB {
        let blk: Vec<f32> = (0..QK).map(|_| rnd()).collect();
        let (dh, qs) = quant_q8_1(&blk, &mut a_bytes);
        a_d[b] = dh;
        a_q[b].copy_from_slice(&qs);
    }

    // CPU reference: the same quantized dot the kernel computes.
    let mut dst_ref = vec![0f32; N_ROWS];
    for r in 0..N_ROWS {
        let mut acc = 0.0f32;
        for b in 0..NB {
            let mut sumi = 0i32;
            for i in 0..QK {
                sumi += w_q[r][b][i] as i32 * a_q[b][i] as i32;
            }
            acc += w_d[r][b].to_f32() * a_d[b].to_f32() * sumi as f32;
        }
        dst_ref[r] = acc;
    }

    // Run on the GPU.
    let dev = CudaDevice::new(0).expect("CudaDevice::new");
    let stream = dev.default_stream();
    let module = CudaModule::load(0, cubin).expect("load mmvq cubin");
    let kernel = module.kernel("flambeau_mmvq_q8_0_dp4a_q8_1").expect("resolve kernel");

    let dw = dev.alloc(w_bytes.len()).unwrap();
    let da = dev.alloc(a_bytes.len()).unwrap();
    let ddst = dev.alloc(N_ROWS * 4).unwrap();
    // SAFETY: host buffers outlive the sync; device allocs sized to match.
    unsafe {
        let hw = DevicePtr(w_bytes.as_ptr() as usize);
        let ha = DevicePtr(a_bytes.as_ptr() as usize);
        dev.memcpy_async(stream, CopyDirection::HostToDevice, dw, hw, w_bytes.len()).unwrap();
        dev.memcpy_async(stream, CopyDirection::HostToDevice, da, ha, a_bytes.len()).unwrap();
    }

    let w_ptr = dw.as_usize() as u64;
    let a_ptr = da.as_usize() as u64;
    let dst_ptr = ddst.as_usize() as u64;
    let n_rows_i = N_ROWS as i32;
    let nb_i = NB as i32;
    let mut args = KernelArgs::new();
    args.push(&w_ptr);
    args.push(&a_ptr);
    args.push(&dst_ptr);
    args.push(&n_rows_i);
    args.push(&nb_i);
    kernel
        .launch(stream, LaunchCfg::one_d(N_ROWS as u32, 256), args)
        .expect("launch mmvq");

    let mut got = vec![0f32; N_ROWS];
    // SAFETY: `got` outlives the sync; `ddst` is N_ROWS f32.
    unsafe {
        let hg = DevicePtr(got.as_mut_ptr() as usize);
        dev.memcpy_async(stream, CopyDirection::DeviceToHost, hg, ddst, N_ROWS * 4).unwrap();
    }
    stream.synchronize().unwrap();

    let mut max_rel = 0.0f32;
    for r in 0..N_ROWS {
        let denom = dst_ref[r].abs().max(1e-3);
        let rel = (got[r] - dst_ref[r]).abs() / denom;
        max_rel = max_rel.max(rel);
        eprintln!("row {r}: got {:.6}  ref {:.6}  rel {:.2e}", got[r], dst_ref[r], rel);
    }
    eprintln!("max relative error: {max_rel:.2e}");
    assert!(max_rel < 1e-3, "mmvq_q8_0 templated kernel diverged from reference (max_rel={max_rel:.2e})");

    // SAFETY: each ptr came from `alloc`; stream synced.
    unsafe {
        dev.dealloc(dw, w_bytes.len()).ok();
        dev.dealloc(da, a_bytes.len()).ok();
        dev.dealloc(ddst, N_ROWS * 4).ok();
    }
}
