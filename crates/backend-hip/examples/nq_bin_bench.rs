//! Micro-benchmark for the NanoQuant binary two-stage GEMV prototype
//! (`mmvq_nq_bin.cu`) on gfx906: per-shape achieved bandwidth and a projected
//! decode throughput (t/s) for Qwen3.8-27B-class shapes at 1 bpw, vs the Q4_K
//! path (4.5 bpw) used today in Flambeau.
//!
//! Usage:
//!   cargo run --release -p flambeau-backend-hip --example nq_bin_bench
//!
//! One GPU job at a time: check `rocm-smi --showuse` before running.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "bench example: every unsafe block is a kernel launch or memcpy with freshly allocated buffers that live until program exit."
)]

use std::time::Instant;

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use half::f16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn pack_random(rows: usize, cols: usize, rng: &mut Rng) -> Vec<u32> {
    let wpr = cols.div_ceil(32);
    let mut out = vec![0u32; rows * wpr];
    for w in out.iter_mut() {
        *w = (rng.next() >> 32) as u32;
    }
    let _ = (rows, cols);
    out
}

fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
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

/// Time one (stage1+stage2) pass for a layer of shape (dout, din, r).
/// Returns (median_us_streamed, weight_bytes).
fn bench_shape(dev: &HipDevice, module: &HipModule, dout: usize, din: usize, r: usize, iters: usize) -> (f64, usize) {
    let wpr = r.div_ceil(32);
    let mut rng = Rng(0xBEEF + dout as u64 * 131 + din as u64 * 7 + r as u64);
    let up = pack_random(dout, r, &mut rng);
    let vp = pack_random(din, r, &mut rng);
    let x: Vec<f16> = (0..din).map(|_| f16::from_f32(((rng.next() >> 40) as f32 / 1024.0) * 2.0 - 1.0)).collect();
    let s1: Vec<f16> = (0..dout).map(|_| f16::from_f32(0.05)).collect();
    let s2: Vec<f16> = (0..din).map(|_| f16::from_f32(0.05)).collect();

    let d_up = upload(dev, &up);
    let d_vp = upload(dev, &vp);
    let d_x = upload(dev, &x);
    let d_s1 = upload(dev, &s1);
    let d_s2 = upload(dev, &s2);
    let d_t = dev.alloc(r * 4).unwrap();
    let d_y = dev.alloc(dout * 4).unwrap();

    let stream = dev.default_stream();

    let launch_pair = |stream: &flambeau_backend_hip::HipStream, module: &HipModule| unsafe {
        let k_zero = module.kernel("flambeau_nq_zero_f32").unwrap();
        let k_s1 = module.kernel("flambeau_nq_stage1_vt_x").unwrap();
        let k_s2 = module.kernel("flambeau_nq_stage2_u_t").unwrap();
        let mut a = KernelArgs::new();
        let p_vp: u64 = d_vp.as_usize() as u64;
        let p_x: u64 = d_x.as_usize() as u64;
        let p_s2: u64 = d_s2.as_usize() as u64;
        let p_t: u64 = d_t.as_usize() as u64;
        let din_i = din as i32;
        let r_i = r as i32;
        let wpr_i = wpr as i32;
        a.push(&p_vp);
        a.push(&p_x);
        a.push(&p_s2);
        a.push(&p_t);
        a.push(&din_i);
        a.push(&r_i);
        a.push(&wpr_i);
        {
            let mut z = KernelArgs::new();
            let p_t0: u64 = d_t.as_usize() as u64;
            z.push(&p_t0);
            z.push(&r_i);
            k_zero.launch(stream, LaunchCfg::one_d((r as u32).div_ceil(256), 256), z).unwrap();
        }
        let gx = (r as u32).div_ceil(256);
        let gy = (din as u32).div_ceil(512);
        k_s1.launch(stream, LaunchCfg { grid: (gx, gy, 1), block: (256, 1, 1), shared_bytes: 18432 }, a)
            .unwrap();
        let mut b = KernelArgs::new();
        let p_up: u64 = d_up.as_usize() as u64;
        let p_t2: u64 = d_t.as_usize() as u64;
        let p_s1: u64 = d_s1.as_usize() as u64;
        let p_y: u64 = d_y.as_usize() as u64;
        b.push(&p_up);
        b.push(&p_t2);
        b.push(&p_s1);
        b.push(&p_y);
        b.push(&r_i);
        b.push(&wpr_i);
        let dout_i = dout as i32;
        b.push(&dout_i);
        k_s2.launch(stream, LaunchCfg::one_d((dout as u32).div_ceil(8), 256), b)
            .unwrap();
    };

    // warmup
    for _ in 0..10 {
        launch_pair(stream, &module);
    }
    stream.synchronize().unwrap();

    // streamed: iters back-to-back, one sync at the end
    let t0 = Instant::now();
    for _ in 0..iters {
        launch_pair(stream, &module);
    }
    stream.synchronize().unwrap();
    let streamed_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let weight_bytes = (din + dout) * wpr * 4 + (din + dout) * 2; // packed bits + fp16 scales
    unsafe {
        dev.dealloc(d_up, up.len() * 4).unwrap();
        dev.dealloc(d_vp, vp.len() * 4).unwrap();
        dev.dealloc(d_x, x.len() * 2).unwrap();
        dev.dealloc(d_s1, s1.len() * 2).unwrap();
        dev.dealloc(d_s2, s2.len() * 2).unwrap();
        dev.dealloc(d_t, r * 4).unwrap();
        dev.dealloc(d_y, dout * 4).unwrap();
    }
    (streamed_us, weight_bytes)
}

fn main() {
    let n = device_count().unwrap_or(0);
    if n == 0 {
        eprintln!("no HIP device — check rocm-smi");
        std::process::exit(1);
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();
    let module = HipModule::load(0, kernels::hsaco("mmvq_nq_bin").unwrap()).unwrap();

    // Qwen3.8-27B text config (Qwen3.5-27B): hidden 5120, ffn 17408, 64 layers,
    // 24 q heads / 4 kv heads × head_dim 256, vocab 248320.
    // rank at 1 bpw: r = dout*din/(dout+din) − 16.
    let rank = |dout: usize, din: usize| -> usize { (dout * din).div_ceil(dout + din) - 16 };
    let shapes: &[(&str, usize, usize)] = &[
        ("q_proj 5120x6144", 5120, 6144),
        ("k_proj 5120x1024", 5120, 1024),
        ("o_proj 5120x5120", 5120, 5120),
        ("gate_proj 5120x17408", 5120, 17408),
        ("down_proj 17408x5120", 17408, 5120),
        ("lm_head 248320x5120", 248320, 5120),
    ];

    println!("shape | r | us/layer | weight GB | GB/s (1 GPU)");
    let mut total_us = 0.0f64;
    let mut total_bytes = 0usize;
    let layers = 64usize;
    for (name, dout, din) in shapes {
        let r = rank(*dout, *din);
        let (us, bytes) = bench_shape(&dev, &module, *dout, *din, r, 100);
        let gb_s = bytes as f64 / (us * 1e-6) / 1e9;
        // per-block usage in the model: q×1, k×1, v×1, o×1, gate×1, up×1, down×1, lm_head×1
        let count = if *name == "lm_head 248320x5120" { 1 } else { layers };
        total_us += us * count as f64;
        total_bytes += bytes * count;
        println!("{name} | r={r} | {us:.1} us | {:.3} GB | {gb_s:.1} GB/s", bytes as f64 / 1e9);
    }
    let tps_1gpu = 1e6 / total_us;
    println!();
    println!("projected decode (m=1, weights only): {:.2} t/s on 1 MI50, {:.2} t/s on 2 MI50 (TP-2, perfect scaling)", tps_1gpu, tps_1gpu * 2.0);
    println!("weight bytes/token @1bpw: {:.2} GB (Q4_K @4.5bpw: {:.2} GB)", total_bytes as f64 / 1e9, 27e9 * 0.5625 / 1e9);
    println!("HBM2 ceiling 1 MI50 = 1024 GB/s -> {:.0} t/s ; 2 MI50 = 2048 GB/s -> {:.0} t/s", 1024.0 / (total_bytes as f64 / 1e9), 2048.0 / (total_bytes as f64 / 1e9));
    println!("reference: Flambeau Q4_K 27B TP-2 measured 29.6 t/s end-to-end (chantier ETAT); Q4_K ceiling TP-2 = {:.0} t/s", 2048.0 / (27e9 * 0.5625 / 1e9));
}
