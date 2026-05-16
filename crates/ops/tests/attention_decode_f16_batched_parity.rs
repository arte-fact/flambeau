//! #266b parity test — verify `attention_decode_f16_batched` produces
//! output bit-identical to running `attention_decode_f16_slots` once
//! per slot.
//! Each (q_head, slot) block in the batched kernel reuses the same
//! flash-attn-v2 body as the single-slot kernel; the only difference
//! is per-slot pointer/length rebinding before the compute body runs.
//! Therefore parity should be **bit-identical** for any N — the floats
//! traverse the same operations in the same order.
//! Sweep:
//! - N ∈ {1, 2, 4, 8}
//! - n_kv_tokens per slot: heterogeneous within the batch (stress the
//! per-slot loop bound in the kernel)
//! - (n_heads_q, n_heads_kv): (32, 4) for Qwen3.5 GQA-32/4 and
//! (16, 2) for Qwen3.6 GQA-16/2
//! - head_dim ∈ {128, 256, 512} — 512 covers gemma4 full-attn layers

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel launch \
              over host/device buffers that live for the bounded synchronize."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::attention::{
    attention_decode_f16_batched, attention_decode_f16_slots,
};
use flambeau_ops::OpsRegistry;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping attention_decode_f16_batched_parity");
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

fn download_f16(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f16> {
    let mut host = vec![f16::from_f32(0.0); n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

/// Deterministic LCG → small-range f16 fill.
fn seeded_f16(seed: u64, n: usize) -> Vec<f16> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            // Map to roughly [-0.5, 0.5] so values stay well within f16 range
            // and softmax doesn't degenerate.
            f16::from_f32((u as f32 / u32::MAX as f32) - 0.5)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Shape {
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
}

fn run_parity(
    label: &str,
    shape: Shape,
    n_slots: usize,
    slot_kv_lens: &[usize],
    seed: u64,
) -> Result<()> {
    assert_eq!(slot_kv_lens.len(), n_slots);
    let Some(dev) = dev_or_skip() else { return Ok(()); };
    let reg = OpsRegistry::new(&dev).expect("registry");
    let stream = dev.default_stream();

    let Shape { n_heads_q, n_heads_kv, head_dim } = shape;
    let q_per_slot = n_heads_q * head_dim;
    let kv_stride_per_token = n_heads_kv * head_dim; // F16 elements
    let scale = (head_dim as f32).sqrt().recip();

    // 1. Build host-side Q (one row per slot) and per-slot K/V caches sized
    // to that slot's `n_kv_tokens`.
    let mut all_q: Vec<f16> = Vec::with_capacity(n_slots * q_per_slot);
    let mut per_slot_k: Vec<Vec<f16>> = Vec::with_capacity(n_slots);
    let mut per_slot_v: Vec<Vec<f16>> = Vec::with_capacity(n_slots);
    for s in 0..n_slots {
        all_q.extend(seeded_f16(seed.wrapping_add(s as u64 * 7 + 1), q_per_slot));
        let kv_n = slot_kv_lens[s];
        per_slot_k.push(seeded_f16(seed.wrapping_add(s as u64 * 7 + 2), kv_n * kv_stride_per_token));
        per_slot_v.push(seeded_f16(seed.wrapping_add(s as u64 * 7 + 3), kv_n * kv_stride_per_token));
    }

    // 2. Upload everything to the device.
    let d_q_batched = upload(&dev, &all_q);
    let d_k_per_slot: Vec<DevicePtr> = per_slot_k.iter().map(|v| upload(&dev, v)).collect();
    let d_v_per_slot: Vec<DevicePtr> = per_slot_v.iter().map(|v| upload(&dev, v)).collect();

    // 3. Build the device-side per-slot tables.
    let k_ptrs_host: Vec<u64> = d_k_per_slot.iter().map(|p| p.as_usize() as u64).collect();
    let v_ptrs_host: Vec<u64> = d_v_per_slot.iter().map(|p| p.as_usize() as u64).collect();
    let n_kv_host: Vec<i32> = slot_kv_lens.iter().map(|&n| n as i32).collect();
    let d_k_ptrs = upload(&dev, &k_ptrs_host);
    let d_v_ptrs = upload(&dev, &v_ptrs_host);
    let d_n_kv = upload(&dev, &n_kv_host);

    // 4. Output buffers — one for serial-baseline, one for batched.
    let out_bytes = n_slots * q_per_slot * 2;
    let d_out_serial = alloc_zeroed(&dev, out_bytes);
    let d_out_batched = alloc_zeroed(&dev, out_bytes);

    // 5. Serial baseline: per-slot single-slot kernel.
    for s in 0..n_slots {
        let q_off = s * q_per_slot * 2;
        let q_row = DevicePtr(d_q_batched.as_usize() + q_off);
        let out_row = DevicePtr(d_out_serial.as_usize() + q_off);
        attention_decode_f16_slots(
            &reg,
            stream,
            q_row,
            d_k_per_slot[s],
            d_v_per_slot[s],
            out_row,
            n_heads_q,
            n_heads_kv,
            head_dim,
            slot_kv_lens[s],
            scale,
            /* window_size = */ 0,
            None,
        )?;
    }
    stream.synchronize()?;

    // 6. Batched: single launch.
    attention_decode_f16_batched(
        &reg,
        stream,
        d_q_batched,
        d_k_ptrs,
        d_v_ptrs,
        d_out_batched,
        d_n_kv,
        n_heads_q,
        n_heads_kv,
        head_dim,
        n_slots,
        scale,
    )?;
    stream.synchronize()?;

    // 7. Compare element-wise.
    let h_serial = download_f16(&dev, d_out_serial, n_slots * q_per_slot);
    let h_batched = download_f16(&dev, d_out_batched, n_slots * q_per_slot);
    let mut max_abs = 0.0f32;
    let mut n_diff = 0usize;
    let mut max_idx = 0usize;
    for (i, (a, b)) in h_serial.iter().zip(h_batched.iter()).enumerate() {
        let af = a.to_f32();
        let bf = b.to_f32();
        let d = (af - bf).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
        if a.to_bits() != b.to_bits() {
            n_diff += 1;
        }
    }
    let total = h_serial.len();
    let bit_identical = n_diff == 0;
    let kv_min = slot_kv_lens.iter().min().copied().unwrap_or(0);
    let kv_max = slot_kv_lens.iter().max().copied().unwrap_or(0);
    eprintln!(
        "{label}: N={n_slots} (n_q,n_kv)=({n_heads_q},{n_heads_kv}) hd={head_dim} \
         kv∈[{kv_min},{kv_max}]  max_abs={max_abs:.3e}  bit_diff={n_diff}/{total}  \
         bit_id={bit_identical}  (sample idx={max_idx})"
    );
    // Bit-identity is the strict expectation: per-block math is identical
    // between the two paths. If this ever breaks (e.g. compiler reorders
    // FP ops), drop to the flash-attn tolerance bound: max_abs < 1e-3.
    assert_eq!(
        n_diff, 0,
        "{label}: batched output diverged from serial baseline at {n_diff}/{total} positions \
         (max_abs={max_abs:.3e}); kernel body should be bit-identical per (q_head, slot) block"
    );

    // Cleanup.
    unsafe {
        dev.dealloc(d_q_batched, n_slots * q_per_slot * 2)?;
        for s in 0..n_slots {
            dev.dealloc(d_k_per_slot[s], slot_kv_lens[s] * kv_stride_per_token * 2)?;
            dev.dealloc(d_v_per_slot[s], slot_kv_lens[s] * kv_stride_per_token * 2)?;
        }
        dev.dealloc(d_k_ptrs, n_slots * 8)?;
        dev.dealloc(d_v_ptrs, n_slots * 8)?;
        dev.dealloc(d_n_kv, n_slots * 4)?;
        dev.dealloc(d_out_serial, out_bytes)?;
        dev.dealloc(d_out_batched, out_bytes)?;
    }
    Ok(())
}

#[test]
fn parity_qwen35_gqa32_4_hd128() -> Result<()> {
    let shape = Shape { n_heads_q: 32, n_heads_kv: 4, head_dim: 128 };
    // N=1 — regression guard for the wiring task (#266c).
    run_parity("Qwen3.5/N=1/kv=128", shape, 1, &[128], 0xA5)?;
    // N=2 / homogeneous KV-len.
    run_parity("Qwen3.5/N=2/kv=512", shape, 2, &[512, 512], 0xA6)?;
    // N=4 / heterogeneous KV-lens — most realistic workload (each slot
    // is at its own request's position).
    run_parity("Qwen3.5/N=4/kv=mix", shape, 4, &[64, 256, 1024, 1500], 0xA7)?;
    // N=8 / mixed long context.
    run_parity(
        "Qwen3.5/N=8/kv=mix-long",
        shape,
        8,
        &[128, 512, 1024, 2048, 256, 768, 1536, 4096],
        0xA8,
    )?;
    Ok(())
}

#[test]
fn parity_qwen36_gqa16_2_hd256() -> Result<()> {
    let shape = Shape { n_heads_q: 16, n_heads_kv: 2, head_dim: 256 };
    // N=1 — regression guard.
    run_parity("Qwen3.6/N=1/kv=128", shape, 1, &[128], 0xB5)?;
    // N=2 / TP=2 local: local_n_heads=8, but the test uses full
    // n_heads_q=16 to also exercise the n_heads_q>n_heads_kv group.
    run_parity("Qwen3.6/N=2/kv=mix", shape, 2, &[256, 1024], 0xB6)?;
    // N=4 / Qwen3.6 prod shape — head_dim=256 stresses LDS budget.
    run_parity("Qwen3.6/N=4/kv=mix", shape, 4, &[128, 512, 1024, 2048], 0xB7)?;
    // N=8 / longer.
    run_parity(
        "Qwen3.6/N=8/kv=long",
        shape,
        8,
        &[256, 512, 1024, 2048, 1024, 512, 256, 4096],
        0xB8,
    )?;
    Ok(())
}

/// Gemma4 26B-A4B full-attention layer shape: head_dim=512, GQA-16/2.
/// The cap was 256 until Phase 13 extended splitk/batched/q8_kv to 512;
/// this case keeps that extension green and proves the batched path
/// works at the new max head_dim alongside the smaller models.
#[test]
fn parity_gemma4_full_attn_gqa16_2_hd512() -> Result<()> {
    let shape = Shape { n_heads_q: 16, n_heads_kv: 2, head_dim: 512 };
    run_parity("Gemma4-26B-A4B/N=1/kv=128", shape, 1, &[128], 0xD5)?;
    run_parity("Gemma4-26B-A4B/N=2/kv=mix", shape, 2, &[256, 1024], 0xD6)?;
    run_parity(
        "Gemma4-26B-A4B/N=4/kv=mix",
        shape,
        4,
        &[128, 512, 1024, 2048],
        0xD7,
    )?;
    Ok(())
}

/// TP-shape: when wired into `forward_full_attn_layer_decode_batched_tp`
/// at TP=2, each rank sees `local_n_heads = num_heads / 2`. Verify the
/// kernel handles the typical TP-sliced shapes (n_heads_q smaller than
/// the global model count).
#[test]
fn parity_qwen36_tp2_local() -> Result<()> {
    // Qwen3.6 27B: num_heads=32, num_kv_heads=4 globally.
    // TP=2 sliced: local_n_heads=16, local_n_kv_heads=2.
    let shape = Shape { n_heads_q: 16, n_heads_kv: 2, head_dim: 128 };
    run_parity("Qwen3.6-27B/TP=2/N=1", shape, 1, &[256], 0xC5)?;
    run_parity("Qwen3.6-27B/TP=2/N=2", shape, 2, &[256, 512], 0xC6)?;
    run_parity("Qwen3.6-27B/TP=2/N=4", shape, 4, &[128, 256, 512, 1024], 0xC7)?;
    Ok(())
}
