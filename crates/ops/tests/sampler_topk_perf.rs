//! Sampler-D microbench — measure GPU `flambeau_sampler_topk_softmax_f32`
//! at production-shape (V=151424, K=256) and compare to a CPU reference.
//! Not a parity test (separate file); just prints wall numbers.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel launch."
)]

use std::time::Instant;

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::sampling::topk_softmax_f32;
use flambeau_ops::OpsRegistry;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping sampler_topk_perf");
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
}

fn upload_f32(dev: &HipDevice, data: &[f32]) -> DevicePtr {
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

#[test]
fn perf_v151424_k256() -> Result<()> {
    let Some(dev) = dev_or_skip() else { return Ok(()); };
    let reg = OpsRegistry::new(&dev).expect("registry");

    let vocab = 151424usize;
    let k = 256usize;
    let inv_temp = 1.0f32 / 0.7;

    // Synthetic logits.
    let mut state = 0x12345678u64;
    let mut logits = vec![0.0f32; vocab];
    for slot in &mut logits {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let raw = (state as i64 >> 32) as f32;
        *slot = raw * (5.0 / (1u32 << 31) as f32);
    }
    let d_logits = upload_f32(&dev, &logits);
    let d_ids = alloc_zeroed(&dev, k * 4);
    let d_probs = alloc_zeroed(&dev, k * 4);

    // Warmup
    for _ in 0..5 {
        topk_softmax_f32(&reg, dev.default_stream(), d_logits, d_ids, d_probs, vocab, k, inv_temp)?;
    }
    dev.default_stream().synchronize()?;

    // GPU timing — 100 iterations, sync after each, take median.
    let n = 100;
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        topk_softmax_f32(&reg, dev.default_stream(), d_logits, d_ids, d_probs, vocab, k, inv_temp)?;
        dev.default_stream().synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[n / 2];
    let p10 = samples[n / 10];
    let p90 = samples[(n * 9) / 10];

    // CPU reference timing — 10 iterations (much slower).
    let mut cpu_samples: Vec<f64> = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        let mut max_l = f32::NEG_INFINITY;
        for &v in &logits {
            let s = v * inv_temp;
            if s > max_l {
                max_l = s;
            }
        }
        let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(vocab);
        let mut sum = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            let p = (v * inv_temp - max_l).exp();
            pairs.push((i as u32, p));
            sum += p;
        }
        for (_, p) in pairs.iter_mut() {
            *p /= sum;
        }
        pairs.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(k);
        let _kept = &pairs[..];
        cpu_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    cpu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_median = cpu_samples[5];

    eprintln!();
    eprintln!("Sampler-D microbench (V={vocab}, K={k}, inv_temp={inv_temp}):");
    eprintln!("  GPU median: {median:.3} ms (p10={p10:.3}, p90={p90:.3})");
    eprintln!("  CPU median: {cpu_median:.3} ms (full softmax + sort, no partial-sort opt)");
    eprintln!("  GPU speedup: {:.1}x", cpu_median / median);

    unsafe {
        dev.dealloc(d_logits, vocab * 4)?;
        dev.dealloc(d_ids, k * 4)?;
        dev.dealloc(d_probs, k * 4)?;
    }
    Ok(())
}
