//! V2.20.b parity — verify `mmvq_q8_0_gate_up_dp4a` output matches two
//! independent `mmvq_q8_0_dp4a_vdr2` calls within F16 round-trip noise at
//! the dense-FFN shape (27B-Q8_0: hidden=5120, inter=17408) and a smaller
//! full-attn-style shape (hidden=5120, n_kv_heads*head_dim=1024) for good
//! measure.
//!
//! Run: `cargo test --release -p flambeau-bench --features hip --test
//! mmvq_q8_0_gate_up_parity -- --nocapture`.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "bench A/B test — every unsafe block is a kernel.launch or memcpy_async \
              over locally-allocated buffers. Scope ends at synchronize + dealloc."
)]

use flambeau_backend_hip::{device_count, HipDevice, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, BlockQ8_1, QK8_0, QK8_1};
use half::f16;

fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            (u as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn quantize_row_q8_0(xs: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(xs.len() % QK8_0, 0);
    let nb = xs.len() / QK8_0;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        let block = &xs[i * QK8_0..(i + 1) * QK8_0];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_0];
        for (j, &v) in block.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
        }
        out.push(BlockQ8_0 { d: f16::from_f32(d), qs });
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
        out.push(BlockQ8_1 { d: f16::from_f32(d), s: f16::from_f32(s), qs });
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
    let mut out = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            ptr,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    out
}

#[test]
fn ab_parity_fused_vs_unfused_dense_ffn() {
    if device_count().unwrap_or(0) < 1 {
        eprintln!("no HIP — skip");
        return;
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    // 27B-Q8_0 dense FFN + 35B-A3B full-attn K+V (historical cert-by-use).
    let shapes: &[(usize, usize, &str)] = &[
        (5120, 17408, "27B dense FFN (hidden=5120, inter=17408)"),
        (2048, 1024, "35B-A3B full-attn K+V (hidden=2048, kv_rows=1024)"),
    ];

    let kb_single = kernels::hsaco("mmvq_q8_0_dp4a_vdr2").unwrap();
    let mod_single = HipModule::load(dev.id(), kb_single).unwrap();
    let k_single = mod_single.kernel("flambeau_mmvq_q8_0_dp4a_vdr2_q8_1").unwrap();

    let kb_fused = kernels::hsaco("mmvq_q8_0_gate_up_dp4a").unwrap();
    let mod_fused = HipModule::load(dev.id(), kb_fused).unwrap();
    let k_fused = mod_fused.kernel("flambeau_mmvq_q8_0_gate_up_dp4a_q8_1").unwrap();

    for &(hidden, inter, label) in shapes {
        let gate_f32 = seeded_f32(0x9419, inter * hidden);
        let up_f32 = seeded_f32(0x9419 ^ 0xAA, inter * hidden);
        let x_f32 = seeded_f32(0x9419 ^ 0x17, hidden);
        let gate_q = quantize_row_q8_0(&gate_f32);
        let up_q = quantize_row_q8_0(&up_f32);
        let x_q = quantize_row_q8_1(&x_f32);

        let d_gate = alloc_upload(&dev, &gate_q);
        let d_up = alloc_upload(&dev, &up_q);
        let d_x = alloc_upload(&dev, &x_q);
        let d_gate_out_single = dev.alloc(inter * 4).unwrap();
        let d_up_out_single = dev.alloc(inter * 4).unwrap();
        let d_gate_out_fused = dev.alloc(inter * 4).unwrap();
        let d_up_out_fused = dev.alloc(inter * 4).unwrap();

        let stream = dev.default_stream();

        // Single-row path — two independent launches.
        let n_rows = inter as i32;
        let n_blocks = (hidden / 32) as i32;
        {
            let gw: u64 = d_gate.as_usize() as u64;
            let y: u64 = d_x.as_usize() as u64;
            let out: u64 = d_gate_out_single.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&gw); args.push(&y); args.push(&out);
            args.push(&n_rows); args.push(&n_blocks);
            let cfg = LaunchCfg::one_d(inter as u32, 256);
            unsafe { k_single.launch(stream, cfg, args).unwrap() };
        }
        {
            let uw: u64 = d_up.as_usize() as u64;
            let y: u64 = d_x.as_usize() as u64;
            let out: u64 = d_up_out_single.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&uw); args.push(&y); args.push(&out);
            args.push(&n_rows); args.push(&n_blocks);
            let cfg = LaunchCfg::one_d(inter as u32, 256);
            unsafe { k_single.launch(stream, cfg, args).unwrap() };
        }

        // Fused path — one launch.
        {
            let gw: u64 = d_gate.as_usize() as u64;
            let uw: u64 = d_up.as_usize() as u64;
            let y: u64 = d_x.as_usize() as u64;
            let g_out: u64 = d_gate_out_fused.as_usize() as u64;
            let u_out: u64 = d_up_out_fused.as_usize() as u64;
            let mut args = KernelArgs::new();
            args.push(&gw); args.push(&uw); args.push(&y);
            args.push(&g_out); args.push(&u_out);
            args.push(&n_rows); args.push(&n_rows); args.push(&n_blocks);
            let cfg = LaunchCfg::one_d(inter as u32, 256);
            unsafe { k_fused.launch(stream, cfg, args).unwrap() };
        }
        stream.synchronize().unwrap();

        let gate_single = download_f32(&dev, d_gate_out_single, inter);
        let up_single = download_f32(&dev, d_up_out_single, inter);
        let gate_fused = download_f32(&dev, d_gate_out_fused, inter);
        let up_fused = download_f32(&dev, d_up_out_fused, inter);

        let max_rel = |a: &[f32], b: &[f32]| -> (f32, f32) {
            let floor = (hidden as f32).sqrt() * 0.01;
            let mut mrel = 0.0f32;
            let mut mabs = 0.0f32;
            for (x, y) in a.iter().zip(b) {
                let abs = (x - y).abs();
                let rel = abs / x.abs().max(floor);
                if rel > mrel { mrel = rel; }
                if abs > mabs { mabs = abs; }
            }
            (mrel, mabs)
        };
        let (g_rel, g_abs) = max_rel(&gate_single, &gate_fused);
        let (u_rel, u_abs) = max_rel(&up_single, &up_fused);

        println!(
            "  {label}: gate max_rel={g_rel:.2e} max_abs={g_abs:.2e} ; up max_rel={u_rel:.2e} max_abs={u_abs:.2e}"
        );

        // F32 accumulation inside the kernel, output stored as F32 → exact
        // arithmetic equivalence is expected. Bar at 1e-5 leaves room for
        // any addition-order shuffle inside the DPP reduction.
        assert!(
            g_rel < 1e-5 && u_rel < 1e-5,
            "fused vs single-row diverge beyond F32 noise: gate {g_rel} up {u_rel}"
        );

        unsafe {
            dev.dealloc(d_gate, gate_q.len() * std::mem::size_of::<BlockQ8_0>()).unwrap();
            dev.dealloc(d_up, up_q.len() * std::mem::size_of::<BlockQ8_0>()).unwrap();
            dev.dealloc(d_x, x_q.len() * std::mem::size_of::<BlockQ8_1>()).unwrap();
            dev.dealloc(d_gate_out_single, inter * 4).unwrap();
            dev.dealloc(d_up_out_single, inter * 4).unwrap();
            dev.dealloc(d_gate_out_fused, inter * 4).unwrap();
            dev.dealloc(d_up_out_fused, inter * 4).unwrap();
        }
    }
}
