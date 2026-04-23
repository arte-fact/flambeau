//! V2.19.b — A/B microbench: attention_decode_f16 vs attention_decode_q8_kv.
//!
//! V2.19.a profiling showed `attention_decode_f16` at ~2869 µs/call on
//! Qwen3.6 (head_dim=256, n_heads_q=16, n_heads_kv=2, n_tokens≈2048) —
//! ~700× off the 4 MiB / 1 TB/s ≈ 4 µs HBM roofline. grid.y=1 means 16
//! blocks on 60 CUs (27% occupancy), so the kernel is compute/occupancy-
//! starved, not HBM-bound. Before investing in the full `KvCache<Q8Contig>`
//! wiring through forward.rs, measure whether Q8-KV actually wins at this
//! shape with pre-populated caches.
//!
//! Run with: `cargo test --release -p flambeau-bench --features hip --test
//! attention_decode_ab -- --nocapture`.

#![cfg(feature = "hip")]

use std::time::Instant;

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockQ8_0, QK8_0};
use half::f16;

const HEAD_DIM: usize = 256;
const N_HEADS_Q: usize = 16;
const N_HEADS_KV: usize = 2;

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

fn launch_f16(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    d_q: DevicePtr,
    d_k: DevicePtr,
    d_v: DevicePtr,
    d_out: DevicePtr,
    n_tokens: usize,
) {
    let stream = dev.default_stream();
    let n_heads_q_i = N_HEADS_Q as i32;
    let n_heads_kv_i = N_HEADS_KV as i32;
    let head_dim_i = HEAD_DIM as i32;
    let n_tokens_i = n_tokens as i32;
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let d_q_ptr: u64 = d_q.as_usize() as u64;
    let d_k_ptr: u64 = d_k.as_usize() as u64;
    let d_v_ptr: u64 = d_v.as_usize() as u64;
    let d_out_ptr: u64 = d_out.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_q_ptr);
    args.push(&d_k_ptr);
    args.push(&d_v_ptr);
    args.push(&d_out_ptr);
    args.push(&n_heads_q_i);
    args.push(&n_heads_kv_i);
    args.push(&head_dim_i);
    args.push(&n_tokens_i);
    args.push(&scale);
    let cfg = LaunchCfg::one_d(N_HEADS_Q as u32, HEAD_DIM as u32);
    unsafe { kernel.launch(stream, cfg, args).unwrap() };
}

// Same signature as F16 (identical kernel args).
fn launch_q8(
    dev: &HipDevice,
    kernel: &HipKernel<'_>,
    d_q: DevicePtr,
    d_k: DevicePtr,
    d_v: DevicePtr,
    d_out: DevicePtr,
    n_tokens: usize,
) {
    launch_f16(dev, kernel, d_q, d_k, d_v, d_out, n_tokens);
}

fn time_kernel(
    dev: &HipDevice,
    name: &str,
    launch: impl Fn(),
    warmup: usize,
    iters: usize,
) -> (f64, f64, f64) {
    // warmup
    for _ in 0..warmup {
        launch();
    }
    dev.default_stream().synchronize().unwrap();

    let mut times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        launch();
        dev.default_stream().synchronize().unwrap();
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_us[0];
    let median = times_us[iters / 2];
    let max = times_us[iters - 1];
    let _ = name;
    (min, median, max)
}

fn launch_splitk(
    dev: &HipDevice,
    k_chunk: &HipKernel<'_>,
    k_combine: &HipKernel<'_>,
    d_q: DevicePtr,
    d_k: DevicePtr,
    d_v: DevicePtr,
    d_out: DevicePtr,
    d_part_m: DevicePtr,
    d_part_s: DevicePtr,
    d_part_o: DevicePtr,
    n_tokens: usize,
    chunk_size: usize,
) {
    let stream = dev.default_stream();
    let n_chunks = n_tokens.div_ceil(chunk_size);
    let n_heads_q_i = N_HEADS_Q as i32;
    let n_heads_kv_i = N_HEADS_KV as i32;
    let head_dim_i = HEAD_DIM as i32;
    let n_tokens_i = n_tokens as i32;
    let n_chunks_i = n_chunks as i32;
    let chunk_size_i = chunk_size as i32;
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let d_q_ptr: u64 = d_q.as_usize() as u64;
    let d_k_ptr: u64 = d_k.as_usize() as u64;
    let d_v_ptr: u64 = d_v.as_usize() as u64;
    let d_m_ptr: u64 = d_part_m.as_usize() as u64;
    let d_s_ptr: u64 = d_part_s.as_usize() as u64;
    let d_o_ptr: u64 = d_part_o.as_usize() as u64;
    let d_out_ptr: u64 = d_out.as_usize() as u64;

    // Pass 1 — chunk kernel.
    let mut a1 = KernelArgs::new();
    a1.push(&d_q_ptr);
    a1.push(&d_k_ptr);
    a1.push(&d_v_ptr);
    a1.push(&d_m_ptr);
    a1.push(&d_s_ptr);
    a1.push(&d_o_ptr);
    a1.push(&n_heads_q_i);
    a1.push(&n_heads_kv_i);
    a1.push(&head_dim_i);
    a1.push(&n_tokens_i);
    a1.push(&n_chunks_i);
    a1.push(&chunk_size_i);
    a1.push(&scale);
    let cfg1 = LaunchCfg {
        grid: (N_HEADS_Q as u32, n_chunks as u32, 1),
        block: (HEAD_DIM as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_chunk.launch(stream, cfg1, a1).unwrap() };

    // Pass 2 — combine.
    let mut a2 = KernelArgs::new();
    a2.push(&d_m_ptr);
    a2.push(&d_s_ptr);
    a2.push(&d_o_ptr);
    a2.push(&d_out_ptr);
    a2.push(&n_heads_q_i);
    a2.push(&n_chunks_i);
    a2.push(&head_dim_i);
    let cfg2 = LaunchCfg {
        grid: (N_HEADS_Q as u32, 1, 1),
        block: (HEAD_DIM as u32, 1, 1),
        shared_bytes: 0,
    };
    unsafe { k_combine.launch(stream, cfg2, a2).unwrap() };
}

#[test]
fn ab_decode_f16_vs_q8_qwen36_shape() {
    let n = device_count().unwrap();
    if n < 1 {
        eprintln!("no HIP device — skipping");
        return;
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let kb_f16 = kernels::hsaco("attention_decode_f16").unwrap();
    let mod_f16 = HipModule::load(dev.id(), kb_f16).unwrap();
    let k_f16 = mod_f16.kernel("flambeau_attention_decode_f16").unwrap();

    let kb_q8 = kernels::hsaco("attention_decode_q8_kv").unwrap();
    let mod_q8 = HipModule::load(dev.id(), kb_q8).unwrap();
    let k_q8 = mod_q8.kernel("flambeau_attention_decode_q8_kv").unwrap();

    let kb_sk = kernels::hsaco("attention_decode_f16_splitk").unwrap();
    let mod_sk = HipModule::load(dev.id(), kb_sk).unwrap();
    let k_sk_chunk = mod_sk
        .kernel("flambeau_attention_decode_f16_splitk_chunk")
        .unwrap();
    let k_sk_combine = mod_sk
        .kernel("flambeau_attention_decode_f16_splitk_combine")
        .unwrap();

    println!("\nV2.19.b A/B — attention decode (head_dim={HEAD_DIM}, heads_q={N_HEADS_Q}, heads_kv={N_HEADS_KV})");
    let a_f16 = k_f16.attributes().unwrap();
    let a_q8 = k_q8.attributes().unwrap();
    let a_sk = k_sk_chunk.attributes().unwrap();
    println!(
        "  f16         VGPR={:3}  waves_per_simd={}  LDS={}B",
        a_f16.num_regs, a_f16.gfx906_waves_per_simd(), a_f16.shared_size_bytes,
    );
    println!(
        "  q8          VGPR={:3}  waves_per_simd={}  LDS={}B",
        a_q8.num_regs, a_q8.gfx906_waves_per_simd(), a_q8.shared_size_bytes,
    );
    println!(
        "  splitk_chk  VGPR={:3}  waves_per_simd={}  LDS={}B",
        a_sk.num_regs, a_sk.gfx906_waves_per_simd(), a_sk.shared_size_bytes,
    );
    println!();
    println!(
        "{:>8}  {:>22}  {:>22}  {:>22}  {:>8}  {:>8}",
        "n_tok",
        "f16 min/med/max µs",
        "q8 min/med/max µs",
        "splitK min/med/max µs",
        "q8/f16",
        "f16/sK",
    );

    let contexts = [128usize, 512, 1024, 2048, 4096];
    // Chunk size tuned per context so n_chunks lands in [4, 16] range
    // (16 heads × 4..16 chunks = 64..256 blocks on 60 CUs = 1..4× saturation).
    let pick_chunk = |n: usize| -> usize {
        if n <= 256 { 128 }
        else if n <= 512 { 128 }
        else if n <= 1024 { 128 }
        else if n <= 2048 { 256 }
        else { 512 }
    };

    for &n_tokens in &contexts {
        let chunk_size = pick_chunk(n_tokens);
        let n_chunks = n_tokens.div_ceil(chunk_size);

        let q_f32 = seeded_f32(0x9419, N_HEADS_Q * HEAD_DIM);
        let k_f32 = seeded_f32(0x9419 ^ 0xA1, n_tokens * N_HEADS_KV * HEAD_DIM);
        let v_f32 = seeded_f32(0x9419 ^ 0xA2, n_tokens * N_HEADS_KV * HEAD_DIM);

        let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let k_f16buf: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let v_f16buf: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

        let nb_per_row = HEAD_DIM / QK8_0;
        let n_rows = n_tokens * N_HEADS_KV;
        let mut k_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * nb_per_row);
        let mut v_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * nb_per_row);
        for row in 0..n_rows {
            let k_row = &k_f32[row * HEAD_DIM..(row + 1) * HEAD_DIM];
            let v_row = &v_f32[row * HEAD_DIM..(row + 1) * HEAD_DIM];
            k_blocks.extend(quantize_row_q8_0(k_row));
            v_blocks.extend(quantize_row_q8_0(v_row));
        }

        let d_q = alloc_and_upload(&dev, &q_f16);
        let d_k_f16 = alloc_and_upload(&dev, &k_f16buf);
        let d_v_f16 = alloc_and_upload(&dev, &v_f16buf);
        let d_k_q8 = alloc_and_upload(&dev, &k_blocks);
        let d_v_q8 = alloc_and_upload(&dev, &v_blocks);
        let out_bytes = N_HEADS_Q * HEAD_DIM * 2;
        let d_out = dev.alloc(out_bytes).unwrap();

        // Partials scratch: f32 × [n_heads_q, n_chunks] + [n_heads_q, n_chunks, head_dim]
        let part_ms_floats = N_HEADS_Q * n_chunks;
        let part_o_floats = N_HEADS_Q * n_chunks * HEAD_DIM;
        let d_part_m = dev.alloc(part_ms_floats * 4).unwrap();
        let d_part_s = dev.alloc(part_ms_floats * 4).unwrap();
        let d_part_o = dev.alloc(part_o_floats * 4).unwrap();

        let (f16_min, f16_med, f16_max) = time_kernel(
            &dev,
            "f16",
            || launch_f16(&dev, &k_f16, d_q, d_k_f16, d_v_f16, d_out, n_tokens),
            5, 50,
        );
        let (q8_min, q8_med, q8_max) = time_kernel(
            &dev,
            "q8",
            || launch_q8(&dev, &k_q8, d_q, d_k_q8, d_v_q8, d_out, n_tokens),
            5, 50,
        );
        let (sk_min, sk_med, sk_max) = time_kernel(
            &dev,
            "splitk",
            || launch_splitk(
                &dev, &k_sk_chunk, &k_sk_combine,
                d_q, d_k_f16, d_v_f16, d_out,
                d_part_m, d_part_s, d_part_o,
                n_tokens, chunk_size,
            ),
            5, 50,
        );

        // --- Parity: F16 vs split-K on the same F16 inputs, same decode
        // math. Expected < 5e-3 max relative error (F16 rounding only).
        launch_f16(&dev, &k_f16, d_q, d_k_f16, d_v_f16, d_out, n_tokens);
        dev.default_stream().synchronize().unwrap();
        let mut out_ref: Vec<f16> = vec![f16::from_f32(0.0); N_HEADS_Q * HEAD_DIM];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(out_ref.as_mut_ptr() as usize),
                d_out,
                out_bytes,
            ).unwrap();
        }
        dev.default_stream().synchronize().unwrap();

        launch_splitk(
            &dev, &k_sk_chunk, &k_sk_combine,
            d_q, d_k_f16, d_v_f16, d_out,
            d_part_m, d_part_s, d_part_o,
            n_tokens, chunk_size,
        );
        dev.default_stream().synchronize().unwrap();
        let mut out_sk: Vec<f16> = vec![f16::from_f32(0.0); N_HEADS_Q * HEAD_DIM];
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::DeviceToHost,
                DevicePtr(out_sk.as_mut_ptr() as usize),
                d_out,
                out_bytes,
            ).unwrap();
        }
        dev.default_stream().synchronize().unwrap();

        let abs_floor = (HEAD_DIM as f32).sqrt() * 0.01;
        let max_rel = out_ref.iter().zip(out_sk.iter())
            .map(|(r, s)| {
                let rf = r.to_f32();
                let sf = s.to_f32();
                (rf - sf).abs() / rf.abs().max(abs_floor)
            })
            .fold(0.0f32, f32::max);
        assert!(max_rel < 5e-3, "split-K parity fail: max_rel={max_rel} at n_tokens={n_tokens}");
        let q8_vs_f16 = f16_med / q8_med;
        let sk_vs_f16 = f16_med / sk_med;
        println!(
            "{:>8}  {:>6.1}/{:>6.1}/{:>6.1}  {:>6.1}/{:>6.1}/{:>6.1}  {:>6.1}/{:>6.1}/{:>6.1}  {:>7.2}x  {:>7.2}x  (chunk={}, n_ch={})",
            n_tokens,
            f16_min, f16_med, f16_max,
            q8_min, q8_med, q8_max,
            sk_min, sk_med, sk_max,
            q8_vs_f16, sk_vs_f16,
            chunk_size, n_chunks,
        );

        unsafe {
            dev.dealloc(d_q, q_f16.len() * 2).unwrap();
            dev.dealloc(d_k_f16, k_f16buf.len() * 2).unwrap();
            dev.dealloc(d_v_f16, v_f16buf.len() * 2).unwrap();
            dev.dealloc(d_k_q8, k_blocks.len() * std::mem::size_of::<BlockQ8_0>()).unwrap();
            dev.dealloc(d_v_q8, v_blocks.len() * std::mem::size_of::<BlockQ8_0>()).unwrap();
            dev.dealloc(d_out, out_bytes).unwrap();
            dev.dealloc(d_part_m, part_ms_floats * 4).unwrap();
            dev.dealloc(d_part_s, part_ms_floats * 4).unwrap();
            dev.dealloc(d_part_o, part_o_floats * 4).unwrap();
        }
    }
    println!();
}
