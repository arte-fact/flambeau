//! /d — parity test for the fused `swiglu_f32_to_q8_1` kernel.
//! Compares the fused kernel's Q8_1 output against the unfused chain
//! (`swiglu_f32` → `quantize_q8_1`) on the same F32 inputs. Every Q8_1
//! block (32 elements) must match exactly: same `d` (F16), same `s`
//! (F16), same 32 int8 quants. The reductions are bit-identical
//! (`__shfl_xor` over 32 lanes, same accumulation tree), so we expect
//! zero drift.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel launch \
              over host/device buffers that live for the bounded synchronize \
              that follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::OpsRegistry;
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping swiglu_f32_to_q8_1_parity");
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

fn download_bytes(dev: &HipDevice, src: DevicePtr, n_bytes: usize) -> Vec<u8> {
    let mut host = vec![0u8; n_bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n_bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

#[test]
fn swiglu_f32_to_q8_1_matches_unfused_chain() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;
    // Cover the ssm_out / shared-expert intermediate sizes we actually
    // hit in production (Coder-Next d_inner=4096, Qwen3.6-35B
    // d_inner=4096, shared-expert intermediate range 2048..8192).
    let test_sizes = [256usize, 1024, 4096, 8192];
    for &n in &test_sizes {
        assert_eq!(n % 32, 0);
        // Mixed-magnitude inputs so amax / scale path exercises corner cases.
        let a: Vec<f32> = (0..n).map(|i| 0.07 * (i as f32 - n as f32 / 2.0)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.13 * ((i as f32) * 0.31).sin()).collect();
        let d_a = upload(&dev, &a);
        let d_b = upload(&dev, &b);

        let block_bytes = 4 + 4 + 32; // F16 d, F16 s, 32 i8 (block_q8_1 layout)
        let n_blocks = n / 32;
        let total_bytes = n_blocks * block_bytes;

        // === Reference: unfused chain ===
        let d_gated = dev.alloc(n * 4)?;
        let d_q8_ref = dev.alloc(total_bytes)?;
        flambeau_ops::hip::mlp::swiglu_f32(&reg, dev.default_stream(), d_a, d_b, d_gated, n)?;
        flambeau_ops::hip::norm::quantize_q8_1(&reg, dev.default_stream(), d_gated, d_q8_ref, n)?;
        dev.default_stream().synchronize()?;
        let ref_bytes = download_bytes(&dev, d_q8_ref, total_bytes);

        // === Fused ===
        let d_q8_fused = dev.alloc(total_bytes)?;
        flambeau_ops::hip::mlp::swiglu_f32_to_q8_1(
            &reg,
            dev.default_stream(),
            d_a,
            d_b,
            d_q8_fused,
            n,
        )?;
        dev.default_stream().synchronize()?;
        let fused_bytes = download_bytes(&dev, d_q8_fused, total_bytes);

        // Byte-exact comparison.
        let mut mismatches = 0usize;
        let mut first_mismatch: Option<(usize, u8, u8)> = None;
        for (i, (r, f)) in ref_bytes.iter().zip(fused_bytes.iter()).enumerate() {
            if r != f {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, *r, *f));
                }
            }
        }
        if mismatches != 0 {
            // Decode + report the first mismatched block for debug.
            let (off, r, f) = first_mismatch.unwrap();
            let block_idx = off / block_bytes;
            let block_off = off % block_bytes;
            let label = if block_off < 2 {
                "d-byte"
            } else if block_off < 4 {
                "s-byte"
            } else {
                "qs[byte]"
            };
            eprintln!(
                "n={n}: {mismatches} byte mismatches; first at byte {off} \
                 (block {block_idx}, {label} offset {block_off}): \
                 ref=0x{r:02x} fused=0x{f:02x}"
            );
            // Decode block's d/s for both for easier debugging.
            let parse_block = |buf: &[u8], idx: usize| -> (f16, f16, [i8; 32]) {
                let base = idx * block_bytes;
                let d = f16::from_le_bytes([buf[base], buf[base + 1]]);
                let s = f16::from_le_bytes([buf[base + 2], buf[base + 3]]);
                let mut q = [0i8; 32];
                for (j, qj) in q.iter_mut().enumerate() {
                    *qj = buf[base + 4 + j] as i8;
                }
                (d, s, q)
            };
            let (rd, rs, rq) = parse_block(&ref_bytes, block_idx);
            let (fd, fs, fq) = parse_block(&fused_bytes, block_idx);
            eprintln!("  ref   block {block_idx}: d={rd:?} s={rs:?} q={rq:?}");
            eprintln!("  fused block {block_idx}: d={fd:?} s={fs:?} q={fq:?}");
        }
        assert_eq!(mismatches, 0, "n={n}: fused output disagrees with unfused chain");

        unsafe {
            dev.dealloc(d_a, n * 4)?;
            dev.dealloc(d_b, n * 4)?;
            dev.dealloc(d_gated, n * 4)?;
            dev.dealloc(d_q8_ref, total_bytes)?;
            dev.dealloc(d_q8_fused, total_bytes)?;
        }
    }
    Ok(())
}
