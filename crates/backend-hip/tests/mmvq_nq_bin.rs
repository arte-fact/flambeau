//! Correctness for the NanoQuant binary two-stage GEMV prototype
//! (`flambeau_nq_stage1_vt_x` / `flambeau_nq_stage2_u_t` in
//! `kernels-hip/src/kernels/mmvq_nq_bin.cu`) on MI50 vs a CPU f64 reference
//! that dequantises Ŵ = diag(s1)·U·Vᵀ·diag(s2) explicitly.
//!
//! U, V are random ±1 matrices packed lsb_first (bit=1 → +1, bit=0 → −1),
//! matching the paper's "−1→0, +1→1" packing; s1, s2 and x are fp16.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every `unsafe {}` is a kernel launch or memcpy whose invariant is uniform: buffers are freshly allocated and live until the synchronize() that follows."
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "usize → i32 casts are kernel-shape math bounded by the test shapes below."
)]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        _ => {
            eprintln!("[skip] no HIP device");
            false
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32_sym(&mut self) -> f32 {
        let u = (self.next() >> 32) as u32;
        (u as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn sign(&mut self) -> i8 {
        if self.next() & 1 == 0 { 1 } else { -1 }
    }
}

/// Pack a (rows × cols) ±1 matrix into row-major u32 words, lsb_first:
/// bit b of word kw = (col kw*32+b == +1) ? 1 : 0.
fn pack_signs(mat: &[i8], rows: usize, cols: usize) -> Vec<u32> {
    let wpr = cols.div_ceil(32);
    let mut out = vec![0u32; rows * wpr];
    for r in 0..rows {
        for c in 0..cols {
            if mat[r * cols + c] > 0 {
                out[r * wpr + (c >> 5)] |= 1u32 << (c & 31);
            }
        }
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

fn download_f32(dev: &HipDevice, d: DevicePtr, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            d,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

/// Run both stages on device; return (t, y) as f32 vectors.
fn run_nq_gemv(
    dev: &HipDevice,
    up: &[u32],
    vp: &[u32],
    u: &[i8],
    v: &[i8],
    x: &[f16],
    s1: &[f16],
    s2: &[f16],
    dout: usize,
    din: usize,
    r: usize,
) -> (Vec<f32>, Vec<f32>) {
    let wpr = r.div_ceil(32);
    let d_up = alloc_and_upload(dev, up);
    let d_vp = alloc_and_upload(dev, vp);
    let d_x = alloc_and_upload(dev, x);
    let d_s1 = alloc_and_upload(dev, s1);
    let d_s2 = alloc_and_upload(dev, s2);
    let d_t = dev.alloc(r * 4).unwrap();
    let d_y = dev.alloc(dout * 4).unwrap();

    let m_bytes = kernels::hsaco("mmvq_nq_bin").unwrap();
    let module = HipModule::load(0, m_bytes).unwrap();
    let k_zero: HipKernel<'_> = module.kernel("flambeau_nq_zero_f32").unwrap();
    let k_s1: HipKernel<'_> = module.kernel("flambeau_nq_stage1_vt_x").unwrap();
    let k_s2: HipKernel<'_> = module.kernel("flambeau_nq_stage2_u_t").unwrap();

    let stream = dev.default_stream();
    {
        let d_t_ptr: u64 = d_t.as_usize() as u64;
        let r_i = r as i32;
        let mut args = KernelArgs::new();
        args.push(&d_t_ptr);
        args.push(&r_i);
        let cfg = LaunchCfg::one_d((r as u32).div_ceil(256), 256);
        unsafe { k_zero.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }
    {
        let d_vp_ptr: u64 = d_vp.as_usize() as u64;
        let d_x_ptr: u64 = d_x.as_usize() as u64;
        let d_s2_ptr: u64 = d_s2.as_usize() as u64;
        let d_t_ptr: u64 = d_t.as_usize() as u64;
        let din_i = din as i32;
        let r_i = r as i32;
        let wpr_i = wpr as i32;
        let mut args = KernelArgs::new();
        args.push(&d_vp_ptr);
        args.push(&d_x_ptr);
        args.push(&d_s2_ptr);
        args.push(&d_t_ptr);
        args.push(&din_i);
        args.push(&r_i);
        args.push(&wpr_i);
        let gx = (r as u32).div_ceil(256);
        let gy = (din as u32).div_ceil(512);
        let cfg = LaunchCfg { grid: (gx, gy, 1), block: (256, 1, 1), shared_bytes: 18432 };
        unsafe { k_s1.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }
    {
        let d_up_ptr: u64 = d_up.as_usize() as u64;
        let d_t_ptr: u64 = d_t.as_usize() as u64;
        let d_s1_ptr: u64 = d_s1.as_usize() as u64;
        let d_y_ptr: u64 = d_y.as_usize() as u64;
        let r_i = r as i32;
        let wpr_i = wpr as i32;
        let mut args = KernelArgs::new();
        args.push(&d_up_ptr);
        args.push(&d_t_ptr);
        args.push(&d_s1_ptr);
        args.push(&d_y_ptr);
        args.push(&r_i);
        args.push(&wpr_i);
        let dout_i = dout as i32;
        args.push(&dout_i);
        let cfg = LaunchCfg::one_d((dout as u32).div_ceil(8), 256);
        unsafe { k_s2.launch(stream, cfg, args).unwrap() };
        stream.synchronize().unwrap();
    }

    let t = download_f32(dev, d_t, r);
    let y = download_f32(dev, d_y, dout);
    let _ = (u, v);
    unsafe {
        dev.dealloc(d_up, up.len() * 4).unwrap();
        dev.dealloc(d_vp, vp.len() * 4).unwrap();
        dev.dealloc(d_x, x.len() * 2).unwrap();
        dev.dealloc(d_s1, s1.len() * 2).unwrap();
        dev.dealloc(d_s2, s2.len() * 2).unwrap();
        dev.dealloc(d_t, r * 4).unwrap();
        dev.dealloc(d_y, dout * 4).unwrap();
    }
    (t, y)
}

/// CPU f64 reference: t = Vᵀ(s2⊙x), y = s1⊙(U t).
fn reference(u: &[i8], v: &[i8], x: &[f16], s1: &[f16], s2: &[f16], dout: usize, din: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let xf: Vec<f64> = x.iter().map(|&h| f16::to_f32(h) as f64).collect();
    let s2f: Vec<f64> = s2.iter().map(|&h| f16::to_f32(h) as f64).collect();
    let s1f: Vec<f64> = s1.iter().map(|&h| f16::to_f32(h) as f64).collect();
    let mut t = vec![0.0f64; r];
    for k in 0..r {
        let mut acc = 0.0f64;
        for j in 0..din {
            acc += (v[j * r + k] as f64) * s2f[j] * xf[j];
        }
        t[k] = acc;
    }
    let mut y = vec![0.0f64; dout];
    for i in 0..dout {
        let mut acc = 0.0f64;
        for k in 0..r {
            acc += (u[i * r + k] as f64) * t[k];
        }
        y[i] = s1f[i] * acc;
    }
    (t.iter().map(|&v| v as f32).collect(), y.iter().map(|&v| v as f32).collect())
}

fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    got.iter()
        .zip(reference)
        .map(|(g, r)| (g - r).abs() / r.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

fn run_case(dout: usize, din: usize, r: usize, seed: u64) {
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();
    let mut rng = Rng(seed);
    let u: Vec<i8> = (0..dout * r).map(|_| rng.sign()).collect();
    let v: Vec<i8> = (0..din * r).map(|_| rng.sign()).collect();
    let x: Vec<f16> = (0..din).map(|_| f16::from_f32(rng.f32_sym())).collect();
    let s1: Vec<f16> = (0..dout).map(|_| f16::from_f32(0.01 + 0.09 * (rng.next() >> 40) as f32 / 1024.0)).collect();
    let s2: Vec<f16> = (0..din).map(|_| f16::from_f32(0.01 + 0.09 * (rng.next() >> 40) as f32 / 1024.0)).collect();

    let up = pack_signs(&u, dout, r);
    let vp = pack_signs(&v, din, r);

    let (t_gpu, y_gpu) = run_nq_gemv(&dev, &up, &vp, &u, &v, &x, &s1, &s2, dout, din, r);
    let (t_ref, y_ref) = reference(&u, &v, &x, &s1, &s2, dout, din, r);

    let et = max_rel_err(&t_gpu, &t_ref);
    let ey = max_rel_err(&y_gpu, &y_ref);
    eprintln!("case dout={dout} din={din} r={r}: max_rel_err t={et:.3e} y={ey:.3e}");
    assert!(et < 2e-3, "stage1 mismatch: {et:.3e}");
    assert!(ey < 2e-3, "stage2 mismatch: {ey:.3e}");
}

#[test]
fn mmvq_nq_bin_small() {
    if !maybe_skip() {
        return;
    }
    run_case(128, 256, 64, 0xA001);
}

#[test]
fn mmvq_nq_bin_odd_rank() {
    if !maybe_skip() {
        return;
    }
    run_case(512, 1024, 100, 0xA002);
}

#[test]
fn mmvq_nq_bin_8b_attn_shape() {
    if !maybe_skip() {
        return;
    }
    // Qwen3-8B q_proj shape at 1 bpw: 4096×4096, r = 4096*4096*1.0/(4096+4096) − 16 = 2032.
    run_case(4096, 4096, 2032, 0xA003);
}
