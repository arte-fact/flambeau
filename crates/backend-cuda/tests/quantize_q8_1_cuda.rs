//! Correctness of the `quantize_q8_1` CUDA kernel (the shared device function's
//! instantiation): F32 activation -> Q8_1 blocks on the GPU, compared against a
//! CPU reference that mirrors the kernel math exactly (d = max|x|/127,
//! id = 1/d, qs = round_ties_even(x*id), s = d*sum(qs)). Skips without a cubin /
//! CUDA device.

use flambeau_backend_cuda::{device_count, CudaDevice, CudaModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use half::f16;

const QK: usize = 32; // QK8_1
const BLOCK_BYTES: usize = 2 + 2 + QK; // d:f16 + s:f16 + qs:32×i8

/// CPU reference for one block, mirroring the kernel's arithmetic bit-for-bit.
fn ref_block(block: &[f32]) -> (f16, f16, [i8; QK]) {
    let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0i8; QK];
    let mut sum = 0i32;
    for (j, &v) in block.iter().enumerate() {
        let q = (v * id).round_ties_even().clamp(-127.0, 127.0) as i32;
        qs[j] = q as i8;
        sum += q;
    }
    (f16::from_f32(d), f16::from_f32(d * sum as f32), qs)
}

#[test]
fn quantize_q8_1_matches_cpu_reference() {
    let Some(cubin) = flambeau_kernels_cuda::cubin("quantize_q8_1") else {
        eprintln!("quantize_q8_1 cubin absent (CUDA_SKIP_BUILD / no nvcc) — skipping");
        return;
    };
    if device_count().unwrap_or(0) < 1 {
        eprintln!("no CUDA device — skipping");
        return;
    }

    const NB: usize = 64; // blocks
    let n_elems = NB * QK;

    // Deterministic pseudo-random inputs in [-1, 1), scaled per block so the
    // amax / scale path is exercised across magnitudes.
    let mut seed = 0x9e3779b9u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    let mut x = vec![0f32; n_elems];
    for b in 0..NB {
        let scale = 1.0 + b as f32 * 0.05;
        for j in 0..QK {
            x[b * QK + j] = rnd() * scale;
        }
    }

    let mut ref_d = vec![f16::ZERO; NB];
    let mut ref_s = vec![f16::ZERO; NB];
    let mut ref_q = vec![[0i8; QK]; NB];
    for b in 0..NB {
        let (d, s, qs) = ref_block(&x[b * QK..(b + 1) * QK]);
        ref_d[b] = d;
        ref_s[b] = s;
        ref_q[b] = qs;
    }

    let dev = CudaDevice::new(0).expect("CudaDevice::new");
    let stream = dev.default_stream();
    let module = CudaModule::load(0, cubin).expect("load quantize cubin");
    let kernel = module
        .kernel("flambeau_quantize_row_q8_1")
        .expect("resolve kernel");

    let dx = dev.alloc(n_elems * 4).unwrap();
    let dy = dev.alloc(NB * BLOCK_BYTES).unwrap();
    // SAFETY: `x` outlives the sync below; `dx` sized to n_elems f32.
    unsafe {
        let hx = DevicePtr(x.as_ptr() as usize);
        dev.memcpy_async(stream, CopyDirection::HostToDevice, dx, hx, n_elems * 4)
            .unwrap();
    }

    let x_ptr = dx.as_usize() as u64;
    let y_ptr = dy.as_usize() as u64;
    let n_i = n_elems as i32;
    let mut args = KernelArgs::new();
    args.push(&x_ptr);
    args.push(&y_ptr);
    args.push(&n_i);
    kernel
        .launch(stream, LaunchCfg::one_d(NB as u32, QK as u32), args)
        .expect("launch quantize_q8_1");

    let mut got = vec![0u8; NB * BLOCK_BYTES];
    // SAFETY: `got` outlives the sync; `dy` is NB × BLOCK_BYTES.
    unsafe {
        let hg = DevicePtr(got.as_mut_ptr() as usize);
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            hg,
            dy,
            NB * BLOCK_BYTES,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    for b in 0..NB {
        let off = b * BLOCK_BYTES;
        let d = f16::from_le_bytes([got[off], got[off + 1]]);
        let s = f16::from_le_bytes([got[off + 2], got[off + 3]]);
        let qs = &got[off + 4..off + 4 + QK];
        assert_eq!(d.to_bits(), ref_d[b].to_bits(), "block {b}: d mismatch");
        assert_eq!(s.to_bits(), ref_s[b].to_bits(), "block {b}: s mismatch");
        for j in 0..QK {
            assert_eq!(
                qs[j] as i8, ref_q[b][j],
                "block {b} lane {j}: qs mismatch (got {}, ref {})",
                qs[j] as i8, ref_q[b][j]
            );
        }
    }
    eprintln!("quantize_q8_1: {NB} blocks bit-exact vs CPU reference");

    // SAFETY: each ptr came from `alloc`; stream synced.
    unsafe {
        dev.dealloc(dx, n_elems * 4).ok();
        dev.dealloc(dy, NB * BLOCK_BYTES).ok();
    }
}
