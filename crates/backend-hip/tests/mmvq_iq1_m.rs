//! End-to-end correctness for `flambeau_mmvq_iq1_m_q8_1` on MI50.
//! No per-block `d` in the wire bytes — must be reassembled from 4 nibbles
//! spread across the 4 u16 scale-words. We construct the scale bytes such
//! that the reassembled d_bits resolves to a bounded fp16 value.

#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture")]
#![expect(clippy::cast_possible_wrap, reason = "test fixture")]

use flambeau_backend_hip::{device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;
use flambeau_quant::{BlockIq1M, BlockQ8_1, QK8_0, QK_K};
use half::f16;

const QK8: usize = QK8_0;
const BLOCK_BYTES: usize = 56;

fn maybe_skip() -> bool { matches!(device_count(), Ok(n) if n >= 1) }
fn seeded_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 24) as u8 }).collect()
}
fn seeded_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); let u = (s >> 32) as u32; (u as f32 / u32::MAX as f32) * 2.0 - 1.0 }).collect()
}
fn random_block(seed: u64, idx: usize) -> BlockIq1M {
    let bytes = seeded_bytes(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)), BLOCK_BYTES);
    let mut qs = [0u8; QK_K / 8]; qs.copy_from_slice(&bytes[0..QK_K / 8]);
    let mut qh = [0u8; QK_K / 16]; qh.copy_from_slice(&bytes[QK_K / 8..QK_K / 8 + QK_K / 16]);
    let mut scales = [0u8; QK_K / 32]; scales.copy_from_slice(&bytes[QK_K / 8 + QK_K / 16..]);
    // Pin the reassembled d_bits to a small positive fp16. d_bits =
    // (sc[0]>>12) | ((sc[1]>>8) & 0xF0) | ((sc[2]>>4) & 0xF00) | (sc[3] & 0xF000)
    // Tiny d (~1e-3..1e-4) keeps F32 dot precision at k=2048.
    // Choose target d_bits = 0x0850 (= 0.001..ish fp16). Distribute its
    // 4 nibbles across the top of each u16 scale-word:
    //   sc[0] top nib = 0x0 (bits 0..3 of d_bits)
    //   sc[1] top nib = 0x5 (bits 4..7)
    //   sc[2] top nib = 0x8 (bits 8..11)
    //   sc[3] top nib = 0x0 (bits 12..15)
    // We write the scale bytes as little-endian u16 with the chosen top nibble.
    let pin_nib = [0x0u16, 0x5, 0x8, 0x0];
    for i in 0..4 {
        let lo = u16::from_le_bytes([scales[2*i], scales[2*i+1]]);
        let new = (lo & 0x0FFF) | (pin_nib[i] << 12);
        scales[2*i..2*i+2].copy_from_slice(&new.to_le_bytes());
    }
    BlockIq1M { qs, qh, scales }
}
fn q8_1_rt(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for i in 0..xs.len()/QK8 {
        let block = &xs[i*QK8..(i+1)*QK8];
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0; let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        for (j, &v) in block.iter().enumerate() { let q = (v * id).round().clamp(-127.0, 127.0) as i32; out[i*QK8 + j] = (q as f32) * d; }
    } out
}
fn ref_mm(w: &[f32], y: &[f32], rows: usize, k: usize) -> Vec<f32> {
    (0..rows).map(|r| (0..k).map(|j| (w[r*k + j] * y[j]) as f64).sum::<f64>() as f32).collect()
}
fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe { dev.memcpy_async(dev.default_stream(), CopyDirection::HostToDevice, d, DevicePtr(data.as_ptr() as usize), bytes).unwrap(); }
    dev.default_stream().synchronize().unwrap(); d
}
fn run(rows: usize, k: usize, seed: u64, stem: &'static str, entry: &'static str, rpb: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(k % QK_K, 0); let nsb = k / QK_K;
    let dev = HipDevice::new(0).unwrap(); dev.bind().unwrap();
    let qm = HipModule::load(0, kernels::hsaco("quantize_q8_1").unwrap()).unwrap();
    let mm = HipModule::load(0, kernels::hsaco(stem).unwrap()).unwrap();
    let kq: HipKernel<'_> = qm.kernel("flambeau_quantize_row_q8_1").unwrap();
    let km: HipKernel<'_> = mm.kernel(entry).unwrap();
    let blocks: Vec<BlockIq1M> = (0..rows*nsb).map(|i| random_block(seed, i)).collect();
    let mut deq = vec![0.0f32; rows*k];
    flambeau_quant::dequantize_into(flambeau_quant::GgmlDType::Iq1M, bytemuck::cast_slice(&blocks), &mut deq).unwrap();
    let y = seeded_f32(seed.wrapping_add(31), k);
    let d_x = upload(&dev, &blocks); let d_y = upload(&dev, &y);
    let yb = k / QK8;
    let d_yq = dev.alloc(yb * std::mem::size_of::<BlockQ8_1>()).unwrap();
    let d_d = dev.alloc(rows * 4).unwrap();
    { let mut a = KernelArgs::new(); let p1: u64 = d_y.as_usize() as u64; let p2: u64 = d_yq.as_usize() as u64; let n = k as i32; a.push(&p1); a.push(&p2); a.push(&n);
      unsafe { kq.launch(dev.default_stream(), LaunchCfg::one_d(yb as u32, QK8 as u32), a).unwrap(); } dev.default_stream().synchronize().unwrap(); }
    { let mut a = KernelArgs::new(); let p1: u64 = d_x.as_usize() as u64; let p2: u64 = d_yq.as_usize() as u64; let p3: u64 = d_d.as_usize() as u64; let nr = rows as i32; let ns = nsb as i32; a.push(&p1); a.push(&p2); a.push(&p3); a.push(&nr); a.push(&ns);
      unsafe { km.launch(dev.default_stream(), LaunchCfg::one_d(rows.div_ceil(rpb) as u32, 64), a).unwrap(); } dev.default_stream().synchronize().unwrap(); }
    let mut dst = vec![0.0f32; rows];
    unsafe { dev.memcpy_async(dev.default_stream(), CopyDirection::DeviceToHost, DevicePtr(dst.as_mut_ptr() as usize), d_d, rows*4).unwrap(); }
    dev.default_stream().synchronize().unwrap();
    unsafe { dev.dealloc(d_x, blocks.len() * std::mem::size_of::<BlockIq1M>()).unwrap(); dev.dealloc(d_y, y.len() * 4).unwrap();
             dev.dealloc(d_yq, yb * std::mem::size_of::<BlockQ8_1>()).unwrap(); dev.dealloc(d_d, rows * 4).unwrap(); }
    (dst, ref_mm(&deq, &q8_1_rt(&y), rows, k))
}
fn err(g: &[f32], r: &[f32]) -> f32 { g.iter().zip(r).map(|(a,b)| (a-b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max) }
fn tol(k: usize) -> f32 { 1e-2 * (k as f32 / 128.0).sqrt() }

#[test] fn iq1_m_small() { if !maybe_skip() { return; } let (g,r) = run(4, QK_K, 0xC0FFEE, "mmvq_iq1_m", "flambeau_mmvq_iq1_m_q8_1", 1); let e = err(&g,&r); eprintln!("[iq1_m 4x{QK_K}] err={e:.3e}"); assert!(e <= tol(QK_K)); }
#[test] fn iq1_m_2048() { if !maybe_skip() { return; } let (g,r) = run(8, 2048, 0xFEEDFACE, "mmvq_iq1_m", "flambeau_mmvq_iq1_m_q8_1", 1); let e = err(&g,&r); eprintln!("[iq1_m 8x2048] err={e:.3e}"); assert!(e <= tol(2048)); }
#[test] fn iq1_m_r2_small() { if !maybe_skip() { return; } let (g,r) = run(4, QK_K, 0xC0FFEE, "mmvq_iq1_m_r2", "flambeau_mmvq_iq1_m_r2_q8_1", 2); let e = err(&g,&r); eprintln!("[iq1_m_r2 4x{QK_K}] err={e:.3e}"); assert!(e <= tol(QK_K)); }
#[test] fn iq1_m_r2_2048() { if !maybe_skip() { return; } let (g,r) = run(8, 2048, 0xFEEDFACE, "mmvq_iq1_m_r2", "flambeau_mmvq_iq1_m_r2_q8_1", 2); let e = err(&g,&r); eprintln!("[iq1_m_r2 8x2048] err={e:.3e}"); assert!(e <= tol(2048)); }
#[test] fn iq1_m_r2_odd() { if !maybe_skip() { return; } let (g,r) = run(7, 1024, 0xDEADBEEF, "mmvq_iq1_m_r2", "flambeau_mmvq_iq1_m_r2_q8_1", 2); let e = err(&g,&r); eprintln!("[iq1_m_r2 7x1024] err={e:.3e}"); assert!(e <= tol(1024)); }
