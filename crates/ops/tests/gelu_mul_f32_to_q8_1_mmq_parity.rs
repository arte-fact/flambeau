//! Parity test for the fused `gelu_mul_f32_to_q8_1_mmq` kernel (Gemma-4 dense
//! FFN prefill). Compares the fused kernel's DS4 144-B MMQ output against the
//! unfused chain (`gelu_mul_f32` → `quantize_q8_1_mmq`) on the same F32
//! inputs. Both stay in F32 up to the quantize, the GELU math is byte-identical
//! to `gelu_mul_f32`, and the amax/sum reductions are bit-identical
//! (`__shfl_xor` over 32 lanes), so every output byte must match exactly.

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

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping gelu_mul_f32_to_q8_1_mmq_parity");
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
fn gelu_mul_f32_to_q8_1_mmq_matches_unfused_chain() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;
    // (total_b tokens, ncols intermediate); ncols % 128 == 0.
    let shapes = [(1usize, 128usize), (4, 256), (3, 384), (2, 512)];
    for (total_b, ncols) in shapes {
        let n = total_b * ncols;
        let a: Vec<f32> = (0..n).map(|i| 0.07 * (i as f32 - n as f32 / 2.0)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.13 * ((i as f32) * 0.31).sin()).collect();
        let d_a = upload(&dev, &a);
        let d_b = upload(&dev, &b);

        let n_big = ncols / 128;
        let out_bytes = n_big * total_b * 144;

        // === Reference: unfused chain (gelu_mul_f32 [F32] → quantize_q8_1_mmq) ===
        let d_gated = dev.alloc(n * 4)?;
        let d_ref = dev.alloc(out_bytes)?;
        flambeau_ops::hip::mlp::gelu_mul_f32(&reg, dev.default_stream(), d_a, d_b, d_gated, n)?;
        flambeau_ops::hip::norm::quantize_q8_1_mmq(
            &reg,
            dev.default_stream(),
            d_gated,
            d_ref,
            ncols,
            total_b,
        )?;
        dev.default_stream().synchronize()?;
        let ref_bytes = download_bytes(&dev, d_ref, out_bytes);

        // === Fused ===
        let d_fused = dev.alloc(out_bytes)?;
        flambeau_ops::hip::mlp::gelu_mul_f32_to_q8_1_mmq(
            &reg,
            dev.default_stream(),
            d_a,
            d_b,
            d_fused,
            ncols,
            total_b,
        )?;
        dev.default_stream().synchronize()?;
        let fused_bytes = download_bytes(&dev, d_fused, out_bytes);

        let mut mismatches = 0usize;
        let mut first: Option<(usize, u8, u8)> = None;
        for (i, (r, f)) in ref_bytes.iter().zip(fused_bytes.iter()).enumerate() {
            if r != f {
                mismatches += 1;
                if first.is_none() {
                    first = Some((i, *r, *f));
                }
            }
        }
        if mismatches != 0 {
            let (off, r, f) = first.unwrap();
            let block = off / 144;
            let boff = off % 144;
            eprintln!(
                "total_b={total_b} ncols={ncols}: {mismatches} byte mismatches; \
                 first at byte {off} (block {block}, offset {boff}): \
                 ref=0x{r:02x} fused=0x{f:02x}"
            );
        }
        assert_eq!(
            mismatches, 0,
            "total_b={total_b} ncols={ncols}: fused disagrees with unfused chain"
        );

        unsafe {
            dev.dealloc(d_a, n * 4)?;
            dev.dealloc(d_b, n * 4)?;
            dev.dealloc(d_gated, n * 4)?;
            dev.dealloc(d_ref, out_bytes)?;
            dev.dealloc(d_fused, out_bytes)?;
        }
    }
    Ok(())
}
