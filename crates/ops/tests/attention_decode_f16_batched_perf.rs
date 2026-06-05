//! #266b microbench — wall-clock A/B between serial-per-slot and batched
//! attention decode at the Qwen3.6-27B / pp2tp2 / N∈{2,4,8} shape.
//! Not a perf gate (that's #266d's throughput cert); this just sanity-
//! checks the direction-of-win and rules out a regression at N=1.
//! Run with: `cargo test --release -p flambeau-ops --features hip --test
//! attention_decode_f16_batched_perf -- --nocapture --ignored`.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "perf bench fixture — every unsafe block is a memcpy or kernel \
              launch over locally-allocated buffers."
)]

use std::time::Instant;

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::attention::{attention_decode_f16_batched, attention_decode_f16_slots};
use flambeau_ops::OpsRegistry;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping perf bench");
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
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

fn alloc_zeroed(dev: &HipDevice, bytes: usize) -> DevicePtr {
    let d = dev.alloc(bytes).unwrap();
    let host_zero = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(host_zero.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn seeded_f16(seed: u64, n: usize) -> Vec<f16> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            f16::from_f32((u as f32 / u32::MAX as f32) - 0.5)
        })
        .collect()
}

fn time_us(dev: &HipDevice, iters: usize, mut launch: impl FnMut()) -> f64 {
    // warmup
    for _ in 0..2 {
        launch();
    }
    dev.default_stream().synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        launch();
    }
    dev.default_stream().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / (iters as f64)
}

#[test]
#[ignore]
fn ab_perf_qwen36_27b_tp2_local() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    // Qwen3.6-27B / pp2tp2 / per-rank shape.
    let n_heads_q = 16usize;
    let n_heads_kv = 2usize;
    let head_dim = 128usize;
    let kv_len = 1024usize; // representative chat-decode KV-tail
    let scale = (head_dim as f32).sqrt().recip();
    let q_per_slot = n_heads_q * head_dim;
    let kv_stride_per_token = n_heads_kv * head_dim;
    let iters = 400;

    println!("\n[perf] Qwen3.6-27B / TP=2 local shape (n_q={n_heads_q}, n_kv={n_heads_kv}, hd={head_dim}, kv_len={kv_len})");
    println!("       iters={iters} per measurement\n");

    for &n_slots in &[1usize, 2, 4, 8] {
        // Build batched Q + per-slot K/V.
        let all_q = seeded_f16(0xD0, n_slots * q_per_slot);
        let mut per_slot_k: Vec<Vec<f16>> = Vec::with_capacity(n_slots);
        let mut per_slot_v: Vec<Vec<f16>> = Vec::with_capacity(n_slots);
        for s in 0..n_slots {
            per_slot_k.push(seeded_f16(0xE0 + s as u64, kv_len * kv_stride_per_token));
            per_slot_v.push(seeded_f16(0xF0 + s as u64, kv_len * kv_stride_per_token));
        }
        let d_q = upload(&dev, &all_q);
        let d_k_per_slot: Vec<DevicePtr> = per_slot_k.iter().map(|v| upload(&dev, v)).collect();
        let d_v_per_slot: Vec<DevicePtr> = per_slot_v.iter().map(|v| upload(&dev, v)).collect();
        let k_ptrs_host: Vec<u64> = d_k_per_slot.iter().map(|p| p.as_usize() as u64).collect();
        let v_ptrs_host: Vec<u64> = d_v_per_slot.iter().map(|p| p.as_usize() as u64).collect();
        let n_kv_host: Vec<i32> = vec![kv_len as i32; n_slots];
        let d_k_ptrs = upload(&dev, &k_ptrs_host);
        let d_v_ptrs = upload(&dev, &v_ptrs_host);
        let d_n_kv = upload(&dev, &n_kv_host);
        let out_bytes = n_slots * q_per_slot * 2;
        let d_out = alloc_zeroed(&dev, out_bytes);

        // Serial: launch single-slot kernel N times per "iteration".
        let serial_us = time_us(&dev, iters, || {
            for s in 0..n_slots {
                let q_off = s * q_per_slot * 2;
                let q_row = DevicePtr(d_q.as_usize() + q_off);
                let out_row = DevicePtr(d_out.as_usize() + q_off);
                attention_decode_f16_slots(
                    flambeau_ops::OpCtx {
                        reg: &reg,
                        stream,
                    },
                    flambeau_ops::AttnBuffers {
                        q: q_row,
                        k: d_k_per_slot[s],
                        v: d_v_per_slot[s],
                        out: out_row,
                    },
                    flambeau_ops::AttnDecodeShape {
                        n_heads_q,
                        n_heads_kv,
                        head_dim,
                        n_tokens_kv: kv_len,
                    },
                    flambeau_ops::AttnKnobs {
                        scale,
                        window_size: 0,
                        ring_depth: 0,
                    },
                    None,
                )
                .unwrap();
            }
        });

        // Batched: single launch per "iteration".
        let batched_us = time_us(&dev, iters, || {
            attention_decode_f16_batched(
                flambeau_ops::OpCtx {
                    reg: &reg,
                    stream,
                },
                flambeau_ops::AttnBatchedBuffers {
                    q_batched: d_q,
                    k_cache_ptrs: d_k_ptrs,
                    v_cache_ptrs: d_v_ptrs,
                    out_batched: d_out,
                    n_tokens_kv_ptrs: d_n_kv,
                },
                flambeau_ops::AttnDecodeBatchedShape {
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    n_slots,
                },
                flambeau_ops::AttnKnobs {
                    scale,
                    window_size: 0,
                    ring_depth: 0,
                },
            )
            .unwrap();
        });

        let speedup = serial_us / batched_us;
        println!(
            "  N={n_slots}: serial={serial_us:7.2} µs   batched={batched_us:7.2} µs   \
             speedup={speedup:.2}×"
        );

        unsafe {
            dev.dealloc(d_q, n_slots * q_per_slot * 2)?;
            for s in 0..n_slots {
                dev.dealloc(d_k_per_slot[s], kv_len * kv_stride_per_token * 2)?;
                dev.dealloc(d_v_per_slot[s], kv_len * kv_stride_per_token * 2)?;
            }
            dev.dealloc(d_k_ptrs, n_slots * 8)?;
            dev.dealloc(d_v_ptrs, n_slots * 8)?;
            dev.dealloc(d_n_kv, n_slots * 4)?;
            dev.dealloc(d_out, out_bytes)?;
        }
    }
    Ok(())
}
