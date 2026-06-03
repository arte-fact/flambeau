//! Sampler-D parity test — verify `flambeau_sampler_topk_softmax_f32`
//! produces the same top-K ids as a CPU full-sort softmax reference,
//! and probs that match the CPU reference within F32 epsilon.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel launch \
              over host/device buffers that live for the bounded synchronize."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::sampling::topk_softmax_f32;
use flambeau_ops::{OpCtx, OpsRegistry, SamplerTopkSoftmaxBuffers, SamplerTopkSoftmaxKnobs};

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping sampler_topk_parity");
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

fn download_f32(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut host = vec![0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

fn download_i32(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<i32> {
    let mut host = vec![0i32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

/// Reference: full softmax + sort. Returns top-k (id, prob) pairs
/// sorted descending. Probs are FULL-VOCAB softmax probs (NOT
/// renormalised over the kept k) — matches the kernel output shape
/// (top_p applied host-side relies on full-vocab semantics).
fn cpu_topk_softmax(logits: &[f32], k: usize, inv_temp: f32) -> Vec<(u32, f32)> {
    let mut max_l = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = v * inv_temp;
        if scaled > max_l {
            max_l = scaled;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v * inv_temp - max_l).exp();
    }
    let mut pairs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, (v * inv_temp - max_l).exp() / sum))
        .collect();
    pairs.sort_by(|a, b| {
        // Descending by prob, ties broken by ascending id (matches kernel
        // packing order).
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    pairs.truncate(k);
    pairs
}

fn run_one(vocab: usize, k: usize, inv_temp: f32, seed: u64) -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev).expect("registry");

    // Simple LCG to fill logits deterministically.
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut logits = vec![0.0f32; vocab];
    for slot in &mut logits {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to a small range so exp() doesn't blow up.
        let raw = (state as i64 >> 32) as f32;
        *slot = raw * (5.0 / (1u32 << 31) as f32);
    }

    let d_logits = upload_f32(&dev, &logits);
    let d_ids = alloc_zeroed(&dev, k * 4);
    let d_probs = alloc_zeroed(&dev, k * 4);

    topk_softmax_f32(
            OpCtx { reg: &reg, stream: dev.default_stream() },
            SamplerTopkSoftmaxBuffers { logits: d_logits, out_ids: d_ids, out_probs: d_probs },
            SamplerTopkSoftmaxKnobs { vocab, k, inv_temp },
        )?;
    dev.default_stream().synchronize()?;

    let gpu_ids = download_i32(&dev, d_ids, k);
    let gpu_probs = download_f32(&dev, d_probs, k);
    let cpu_pairs = cpu_topk_softmax(&logits, k, inv_temp);

    // Compare ids exactly (within K) and probs within F32 epsilon.
    let mut max_id_mismatch = 0usize;
    let mut max_prob_diff = 0.0f32;
    for i in 0..k {
        let (cpu_id, cpu_p) = cpu_pairs[i];
        let gpu_id = gpu_ids[i] as u32;
        let gpu_p = gpu_probs[i];
        if cpu_id != gpu_id {
            max_id_mismatch += 1;
            if max_id_mismatch <= 5 {
                eprintln!(
                    "mismatch at slot {i}: cpu=(id={cpu_id}, p={cpu_p:.6e}) gpu=(id={gpu_id}, p={gpu_p:.6e})"
                );
            }
        }
        let d = (cpu_p - gpu_p).abs();
        if d > max_prob_diff {
            max_prob_diff = d;
        }
    }

    eprintln!(
        "  V={vocab} K={k} inv_t={inv_temp} seed={seed}: \
         id_mismatch={max_id_mismatch}/{k}, max_prob_diff={max_prob_diff:.3e}"
    );
    assert_eq!(
        max_id_mismatch, 0,
        "GPU top-K produced different ids than CPU reference"
    );
    assert!(
        max_prob_diff < 1e-5,
        "max prob diff {max_prob_diff:.3e} > 1e-5"
    );

    unsafe {
        dev.dealloc(d_logits, vocab * 4)?;
        dev.dealloc(d_ids, k * 4)?;
        dev.dealloc(d_probs, k * 4)?;
    }
    Ok(())
}

#[test]
fn parity_v1024_k64_t1() -> Result<()> {
    run_one(1024, 64, 1.0, 1)
}

#[test]
fn parity_v8192_k128_t07() -> Result<()> {
    // inv_temp = 1/0.7 (the common chat sampling temperature)
    run_one(8192, 128, 1.0 / 0.7, 42)
}

#[test]
fn parity_v151424_k256_t07() -> Result<()> {
    // Production-shape Qwen3.6 vocab + K_OUT_MAX cap.
    run_one(151424, 256, 1.0 / 0.7, 9419)
}

#[test]
fn parity_v151424_k64_t10() -> Result<()> {
    // High temperature flattens the distribution — exercises ties and
    // mid-magnitude logits.
    run_one(151424, 64, 1.0, 12345)
}
