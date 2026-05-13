//! S3 parity tests — sliding-window attention (decode + splitk + prefill) and
//! final-logit softcap. CPU references rebuild the masked attention sum and
//! the elementwise `tanh(x/cap)*cap`; window=0 must reproduce the existing
//! causal kernel output (regression guard).

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel launch over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::attention::{
    attention_decode_f16, attention_decode_f16_splitk, attention_prefill_f16,
};
use flambeau_ops::hip::mlp::{gelu_f32_to_f16, gelu_mul_f32};
use flambeau_ops::hip::softcap::apply_softcap_f32;
use flambeau_ops::OpsRegistry;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping swa_softcap_parity");
        return None;
    }
    let dev = HipDevice::new(0).ok()?;
    dev.bind().ok()?;
    Some(dev)
}

fn upload_f16(dev: &HipDevice, data: &[f16]) -> DevicePtr {
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

fn download_f32(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; n];
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

/// Deterministic LCG → small-range f32 fill in [-lo..lo].
fn seeded_f32(seed: u64, n: usize, lo: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
            v * lo
        })
        .collect()
}

/// Reference single-query attention over a half-open key range
/// `[t_start, t_end)`. Returns the per-head output `[n_heads_q, head_dim]`.
fn cpu_decode_attn_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    t_start: usize,
    t_end: usize,
    scale: f32,
) -> Vec<f32> {
    let group = n_heads_q / n_heads_kv;
    let mut out = vec![0.0f32; n_heads_q * head_dim];
    for qh in 0..n_heads_q {
        let kvh = qh / group;
        // Scores.
        let span = t_end - t_start;
        let mut scores = vec![0.0f64; span];
        for (i, t) in (t_start..t_end).enumerate() {
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                let qv = q[qh * head_dim + d] as f64;
                let kv = k[(t * n_heads_kv + kvh) * head_dim + d] as f64;
                dot += qv * kv;
            }
            scores[i] = dot * scale as f64;
        }
        // Softmax.
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv = 1.0f64 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
        // Sum_t w_t * V[t].
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for (i, t) in (t_start..t_end).enumerate() {
                let vv = v[(t * n_heads_kv + kvh) * head_dim + d] as f64;
                acc += scores[i] * vv;
            }
            // Round through F16 to match GPU output's storage precision.
            out[qh * head_dim + d] = f16::from_f32(acc as f32).to_f32();
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn swa_decode_window_4_of_8() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let head_dim = 64usize;
    let n_heads_q = 2usize;
    let n_heads_kv = 1usize;
    let n_tokens = 8usize;
    let window = 4i32;

    let q_f32 = seeded_f32(0xDE0A, n_heads_q * head_dim, 0.5);
    let k_f32 = seeded_f32(0xDE0B, n_tokens * n_heads_kv * head_dim, 0.5);
    let v_f32 = seeded_f32(0xDE0C, n_tokens * n_heads_kv * head_dim, 0.5);

    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = upload_f16(&dev, &q_f16);
    let d_k = upload_f16(&dev, &k_f16);
    let d_v = upload_f16(&dev, &v_f16);
    let out_n = n_heads_q * head_dim;
    let d_out = dev.alloc(out_n * 2).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();
    attention_decode_f16(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out,
        n_heads_q, n_heads_kv, head_dim, n_tokens, scale, window,
    )
    .unwrap();
    dev.default_stream().synchronize().unwrap();
    let got_f16 = download_f16(&dev, d_out, out_n);
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();

    // Round Q/K/V through F16 to match the kernel's loaded precision.
    let q_ref: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let k_ref: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
    let v_ref: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();
    // qpos = n_tokens - 1 = 7; window = 4 → t_start = 7 - 4 + 1 = 4.
    let reference = cpu_decode_attn_ref(
        &q_ref, &k_ref, &v_ref, n_heads_q, n_heads_kv, head_dim, 4, n_tokens, scale,
    );

    let err = max_abs_diff(&got, &reference);
    assert!(err < 5e-3, "SWA decode max-abs-diff {err} too high");

    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2).unwrap();
        dev.dealloc(d_k, k_f16.len() * 2).unwrap();
        dev.dealloc(d_v, v_f16.len() * 2).unwrap();
        dev.dealloc(d_out, out_n * 2).unwrap();
    }
}

#[test]
fn swa_decode_window_geq_ntokens_matches_unbounded() {
    // window_size >= n_tokens should produce output identical to window_size=0.
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let head_dim = 64usize;
    let n_heads_q = 2usize;
    let n_heads_kv = 1usize;
    let n_tokens = 8usize;

    let q_f32 = seeded_f32(0xBADE0, n_heads_q * head_dim, 0.5);
    let k_f32 = seeded_f32(0xBADE1, n_tokens * n_heads_kv * head_dim, 0.5);
    let v_f32 = seeded_f32(0xBADE2, n_tokens * n_heads_kv * head_dim, 0.5);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = upload_f16(&dev, &q_f16);
    let d_k = upload_f16(&dev, &k_f16);
    let d_v = upload_f16(&dev, &v_f16);
    let out_n = n_heads_q * head_dim;
    let d_out_off = dev.alloc(out_n * 2).unwrap();
    let d_out_big = dev.alloc(out_n * 2).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();
    attention_decode_f16(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out_off,
        n_heads_q, n_heads_kv, head_dim, n_tokens, scale, 0,
    )
    .unwrap();
    attention_decode_f16(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out_big,
        n_heads_q, n_heads_kv, head_dim, n_tokens, scale, 16,
    )
    .unwrap();
    dev.default_stream().synchronize().unwrap();
    let off = download_f16(&dev, d_out_off, out_n);
    let big = download_f16(&dev, d_out_big, out_n);
    for (a, b) in off.iter().zip(big.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "window>=n_tokens must match window=0 bit-exactly");
    }
    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2).unwrap();
        dev.dealloc(d_k, k_f16.len() * 2).unwrap();
        dev.dealloc(d_v, v_f16.len() * 2).unwrap();
        dev.dealloc(d_out_off, out_n * 2).unwrap();
        dev.dealloc(d_out_big, out_n * 2).unwrap();
    }
}

#[test]
fn swa_decode_splitk_matches_window_4() {
    // splitk + SWA should produce the same output as single-pass + SWA.
    // Force splitk by using n_tokens > 256.
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let head_dim = 128usize;
    let n_heads_q = 4usize;
    let n_heads_kv = 2usize;
    let n_tokens = 512usize;
    let window = 64i32;

    let q_f32 = seeded_f32(0x5C0, n_heads_q * head_dim, 0.3);
    let k_f32 = seeded_f32(0x5C1, n_tokens * n_heads_kv * head_dim, 0.3);
    let v_f32 = seeded_f32(0x5C2, n_tokens * n_heads_kv * head_dim, 0.3);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = upload_f16(&dev, &q_f16);
    let d_k = upload_f16(&dev, &k_f16);
    let d_v = upload_f16(&dev, &v_f16);
    let out_n = n_heads_q * head_dim;
    let d_out_single = dev.alloc(out_n * 2).unwrap();
    let d_out_split = dev.alloc(out_n * 2).unwrap();

    let chunk_size = 128usize;
    let n_chunks = n_tokens.div_ceil(chunk_size);
    let d_part_m = dev.alloc(n_heads_q * n_chunks * 4).unwrap();
    let d_part_s = dev.alloc(n_heads_q * n_chunks * 4).unwrap();
    let d_part_o = dev.alloc(n_heads_q * n_chunks * head_dim * 4).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();
    attention_decode_f16(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out_single,
        n_heads_q, n_heads_kv, head_dim, n_tokens, scale, window,
    )
    .unwrap();
    attention_decode_f16_splitk(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out_split,
        d_part_m, d_part_s, d_part_o,
        n_heads_q, n_heads_kv, head_dim, n_tokens, chunk_size, scale, window,
    )
    .unwrap();
    dev.default_stream().synchronize().unwrap();
    let single = download_f16(&dev, d_out_single, out_n);
    let split = download_f16(&dev, d_out_split, out_n);
    let mut max_abs = 0.0f32;
    for (a, b) in single.iter().zip(split.iter()) {
        let d = (a.to_f32() - b.to_f32()).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(max_abs < 5e-3, "splitk SWA vs single-pass SWA max-abs-diff {max_abs} too high");

    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2).unwrap();
        dev.dealloc(d_k, k_f16.len() * 2).unwrap();
        dev.dealloc(d_v, v_f16.len() * 2).unwrap();
        dev.dealloc(d_out_single, out_n * 2).unwrap();
        dev.dealloc(d_out_split, out_n * 2).unwrap();
        dev.dealloc(d_part_m, n_heads_q * n_chunks * 4).unwrap();
        dev.dealloc(d_part_s, n_heads_q * n_chunks * 4).unwrap();
        dev.dealloc(d_part_o, n_heads_q * n_chunks * head_dim * 4).unwrap();
    }
}

#[test]
fn swa_prefill_window_3_of_8() {
    // Per-query SWA: each q at position p attends only to [max(0, p-w+1), p].
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let head_dim = 64usize;
    let n_heads_q = 2usize;
    let n_heads_kv = 1usize;
    let n_q = 3usize;        // < 4 so we definitely hit the oracle prefill path
    let n_k = 8usize;
    let q_offset = 5usize;   // q's at global positions 5, 6, 7
    let window = 3i32;       // each q attends to last 3 keys ≤ its position

    let q_f32 = seeded_f32(0xDEAD0, n_q * n_heads_q * head_dim, 0.5);
    let k_f32 = seeded_f32(0xDEAD1, n_k * n_heads_kv * head_dim, 0.5);
    let v_f32 = seeded_f32(0xDEAD2, n_k * n_heads_kv * head_dim, 0.5);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = upload_f16(&dev, &q_f16);
    let d_k = upload_f16(&dev, &k_f16);
    let d_v = upload_f16(&dev, &v_f16);
    let out_n = n_q * n_heads_q * head_dim;
    let d_out = dev.alloc(out_n * 2).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();
    attention_prefill_f16(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out,
        n_q, n_heads_q, n_heads_kv, head_dim, n_k, q_offset, scale, window,
    )
    .unwrap();
    dev.default_stream().synchronize().unwrap();
    let got_f16 = download_f16(&dev, d_out, out_n);
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();

    let q_ref: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let k_ref: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
    let v_ref: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; out_n];
    for q_idx in 0..n_q {
        let qpos = q_offset + q_idx;
        // SWA per-query range, intersected with causal limit qpos+1.
        let t_start = (qpos + 1).saturating_sub(window as usize).min(n_k);
        let t_end = (qpos + 1).min(n_k);
        let q_slice = &q_ref[q_idx * n_heads_q * head_dim..(q_idx + 1) * n_heads_q * head_dim];
        let per_q = cpu_decode_attn_ref(
            q_slice, &k_ref, &v_ref, n_heads_q, n_heads_kv, head_dim,
            t_start, t_end, scale,
        );
        reference[q_idx * n_heads_q * head_dim..(q_idx + 1) * n_heads_q * head_dim]
            .copy_from_slice(&per_q);
    }
    let err = max_abs_diff(&got, &reference);
    assert!(err < 5e-3, "SWA prefill max-abs-diff {err} too high");

    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2).unwrap();
        dev.dealloc(d_k, k_f16.len() * 2).unwrap();
        dev.dealloc(d_v, v_f16.len() * 2).unwrap();
        dev.dealloc(d_out, out_n * 2).unwrap();
    }
}

#[test]
fn softcap_f32_parity() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let cap = 30.0f32; // Gemma4 default
    let n = 1024usize;
    let x = seeded_f32(0x50FC, n, 60.0); // span ±60 so |x/cap| spans ±2

    let d_x = upload_f32(&dev, &x);
    let d_y = dev.alloc(n * 4).unwrap();

    apply_softcap_f32(&reg, dev.default_stream(), d_x, d_y, n, cap).unwrap();
    dev.default_stream().synchronize().unwrap();
    let got = download_f32(&dev, d_y, n);

    let expected: Vec<f32> = x.iter().map(|v| (v / cap).tanh() * cap).collect();
    let err = max_abs_diff(&got, &expected);
    assert!(err < 1e-5, "softcap max-abs-diff {err} too high");

    unsafe {
        dev.dealloc(d_x, n * 4).unwrap();
        dev.dealloc(d_y, n * 4).unwrap();
    }
}

/// ggml tanh-GELU (`ggml-cpu/vec.h:986`).
fn cpu_gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEF_A: f32 = 0.044_715;
    let t = SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x);
    0.5 * x * (1.0 + t.tanh())
}

#[test]
fn gelu_f32_to_f16_parity() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let n = 1024usize;
    let a = seeded_f32(0x6E10, n, 3.0); // span ±3 — GELU is interesting around 0
    let b = seeded_f32(0x6E11, n, 1.5);
    let d_a = upload_f32(&dev, &a);
    let d_b = upload_f32(&dev, &b);
    let d_y = dev.alloc(n * 2).unwrap();

    gelu_f32_to_f16(&reg, dev.default_stream(), d_a, d_b, d_y, n).unwrap();
    dev.default_stream().synchronize().unwrap();
    let got_f16 = download_f16(&dev, d_y, n);
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
    let expected: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f16::from_f32(cpu_gelu(x) * y).to_f32())
        .collect();
    let err = max_abs_diff(&got, &expected);
    assert!(err < 5e-3, "gelu_f32_to_f16 max-abs-diff {err} too high");

    unsafe {
        dev.dealloc(d_a, n * 4).unwrap();
        dev.dealloc(d_b, n * 4).unwrap();
        dev.dealloc(d_y, n * 2).unwrap();
    }
}

#[test]
fn gelu_mul_f32_parity() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let n = 1024usize;
    let a = seeded_f32(0x6E20, n, 3.0);
    let b = seeded_f32(0x6E21, n, 1.5);
    let d_a = upload_f32(&dev, &a);
    let d_b = upload_f32(&dev, &b);
    let d_y = dev.alloc(n * 4).unwrap();

    gelu_mul_f32(&reg, dev.default_stream(), d_a, d_b, d_y, n).unwrap();
    dev.default_stream().synchronize().unwrap();
    let got = download_f32(&dev, d_y, n);
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| cpu_gelu(x) * y).collect();
    let err = max_abs_diff(&got, &expected);
    assert!(err < 1e-5, "gelu_mul_f32 max-abs-diff {err} too high");

    unsafe {
        dev.dealloc(d_a, n * 4).unwrap();
        dev.dealloc(d_b, n * 4).unwrap();
        dev.dealloc(d_y, n * 4).unwrap();
    }
}

/// SWA prefill at n_q ≥ 4 — exercises the flash_tile kernel path
/// (previously routed to oracle when window > 0).
#[test]
fn swa_prefill_flash_tile_window_4() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let head_dim = 64usize;
    let n_heads_q = 2usize;
    let n_heads_kv = 1usize;
    let n_q = 8usize;          // ≥ 4 → flash_tile path
    let n_k = 16usize;
    let q_offset = 4usize;     // q's at global positions 4..11
    let window = 4i32;

    let q_f32 = seeded_f32(0xF11A, n_q * n_heads_q * head_dim, 0.5);
    let k_f32 = seeded_f32(0xF11B, n_k * n_heads_kv * head_dim, 0.5);
    let v_f32 = seeded_f32(0xF11C, n_k * n_heads_kv * head_dim, 0.5);
    let q_f16: Vec<f16> = q_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q = upload_f16(&dev, &q_f16);
    let d_k = upload_f16(&dev, &k_f16);
    let d_v = upload_f16(&dev, &v_f16);
    let out_n = n_q * n_heads_q * head_dim;
    let d_out = dev.alloc(out_n * 2).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();
    attention_prefill_f16(
        &reg, dev.default_stream(), d_q, d_k, d_v, d_out,
        n_q, n_heads_q, n_heads_kv, head_dim, n_k, q_offset, scale, window,
    )
    .unwrap();
    dev.default_stream().synchronize().unwrap();
    let got_f16 = download_f16(&dev, d_out, out_n);
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();

    let q_ref: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
    let k_ref: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
    let v_ref: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();
    let mut reference = vec![0.0f32; out_n];
    for q_idx in 0..n_q {
        let qpos = q_offset + q_idx;
        let t_start = (qpos + 1).saturating_sub(window as usize).min(n_k);
        let t_end = (qpos + 1).min(n_k);
        let q_slice = &q_ref[q_idx * n_heads_q * head_dim..(q_idx + 1) * n_heads_q * head_dim];
        let per_q = cpu_decode_attn_ref(
            q_slice, &k_ref, &v_ref, n_heads_q, n_heads_kv, head_dim,
            t_start, t_end, scale,
        );
        reference[q_idx * n_heads_q * head_dim..(q_idx + 1) * n_heads_q * head_dim]
            .copy_from_slice(&per_q);
    }
    let err = max_abs_diff(&got, &reference);
    assert!(err < 5e-3, "flash_tile SWA n_q=8 window=4 max-abs-diff {err}");

    unsafe {
        dev.dealloc(d_q, q_f16.len() * 2).unwrap();
        dev.dealloc(d_k, k_f16.len() * 2).unwrap();
        dev.dealloc(d_v, v_f16.len() * 2).unwrap();
        dev.dealloc(d_out, out_n * 2).unwrap();
    }
}

/// flash_tile prefill with window=0 must produce output bit-equivalent
/// to the pre-#15 baseline (regression guard for the kernel change).
#[test]
fn flash_tile_window_zero_matches_oracle() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let head_dim = 64usize;
    let n_heads_q = 2usize;
    let n_heads_kv = 1usize;
    let n_q_short = 3usize;    // oracle path
    let n_q_long = 8usize;     // flash_tile path
    let n_k_long = 16usize;
    let q_offset = 0usize;

    let q_long_f32 = seeded_f32(0xF120, n_q_long * n_heads_q * head_dim, 0.5);
    let k_f32 = seeded_f32(0xF121, n_k_long * n_heads_kv * head_dim, 0.5);
    let v_f32 = seeded_f32(0xF122, n_k_long * n_heads_kv * head_dim, 0.5);
    let q_long: Vec<f16> = q_long_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let k: Vec<f16> = k_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let v: Vec<f16> = v_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let d_q_long = upload_f16(&dev, &q_long);
    let d_k = upload_f16(&dev, &k);
    let d_v = upload_f16(&dev, &v);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // flash_tile path (n_q=8, window=0).
    let out_long = n_q_long * n_heads_q * head_dim;
    let d_out_long = dev.alloc(out_long * 2).unwrap();
    attention_prefill_f16(
        &reg, dev.default_stream(), d_q_long, d_k, d_v, d_out_long,
        n_q_long, n_heads_q, n_heads_kv, head_dim, n_k_long, q_offset, scale, 0,
    )
    .unwrap();

    // Oracle path: chunk into 3 separate small calls (n_q ≤ 3) so each
    // hits the oracle kernel. Concatenate outputs.
    let mut out_oracle = vec![f16::from_f32(0.0); out_long];
    let mut q_off_cur = q_offset;
    let mut q_cursor = 0usize;
    while q_cursor < n_q_long {
        let chunk = (n_q_long - q_cursor).min(n_q_short);
        let q_chunk_host: Vec<f16> = q_long
            [q_cursor * n_heads_q * head_dim..(q_cursor + chunk) * n_heads_q * head_dim]
            .to_vec();
        let d_q_chunk = upload_f16(&dev, &q_chunk_host);
        let chunk_out = chunk * n_heads_q * head_dim;
        let d_out_chunk = dev.alloc(chunk_out * 2).unwrap();
        attention_prefill_f16(
            &reg, dev.default_stream(), d_q_chunk, d_k, d_v, d_out_chunk,
            chunk, n_heads_q, n_heads_kv, head_dim, n_k_long, q_off_cur, scale, 0,
        )
        .unwrap();
        dev.default_stream().synchronize().unwrap();
        let chunk_host = download_f16(&dev, d_out_chunk, chunk_out);
        out_oracle[q_cursor * n_heads_q * head_dim..(q_cursor + chunk) * n_heads_q * head_dim]
            .copy_from_slice(&chunk_host);
        unsafe {
            dev.dealloc(d_q_chunk, q_chunk_host.len() * 2).unwrap();
            dev.dealloc(d_out_chunk, chunk_out * 2).unwrap();
        }
        q_off_cur += chunk;
        q_cursor += chunk;
    }
    dev.default_stream().synchronize().unwrap();

    let got_flash = download_f16(&dev, d_out_long, out_long);
    let got_f: Vec<f32> = got_flash.iter().map(|v| v.to_f32()).collect();
    let got_o: Vec<f32> = out_oracle.iter().map(|v| v.to_f32()).collect();
    let err = max_abs_diff(&got_f, &got_o);
    // F16 accumulation order differs between oracle and flash_tile; allow tolerance.
    assert!(err < 5e-3, "flash_tile window=0 vs oracle max-abs-diff {err}");

    unsafe {
        dev.dealloc(d_q_long, q_long.len() * 2).unwrap();
        dev.dealloc(d_k, k.len() * 2).unwrap();
        dev.dealloc(d_v, v.len() * 2).unwrap();
        dev.dealloc(d_out_long, out_long * 2).unwrap();
    }
}

#[test]
fn softcap_f32_inplace() {
    let Some(dev) = dev_or_skip() else { return; };
    let reg = OpsRegistry::new(&dev).unwrap();

    let cap = 30.0f32;
    let n = 256usize;
    let x = seeded_f32(0x50FC2, n, 60.0);

    let d_x = upload_f32(&dev, &x);
    apply_softcap_f32(&reg, dev.default_stream(), d_x, d_x, n, cap).unwrap();
    dev.default_stream().synchronize().unwrap();
    let got = download_f32(&dev, d_x, n);

    let expected: Vec<f32> = x.iter().map(|v| (v / cap).tanh() * cap).collect();
    let err = max_abs_diff(&got, &expected);
    assert!(err < 1e-5, "softcap in-place max-abs-diff {err} too high");

    unsafe {
        dev.dealloc(d_x, n * 4).unwrap();
    }
}
