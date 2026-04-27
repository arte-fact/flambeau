//! V2.31.b diagnostic — compare `flambeau_mmvq_q8_0_r4_dp4a_q8_1` output
//! vs `flambeau_mmvq_q8_0_dp4a_vdr2_q8_1` output on a tiny matrix.
//!
//! The r4 kernel produces different last_ids on 27B decode (per
//! V2.31.b initial run). This is a minimal reproducer to find the
//! divergence point.

#![cfg(feature = "hip")]

#![expect(clippy::undocumented_unsafe_blocks, reason = "test fixture")]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice, HipKernel, KernelArgs, LaunchCfg};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
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

// Mirror of flambeau_block_q8_0.
#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ8_0 {
    d: u16,            // f16 bits
    qs: [i8; 32],
}

// Mirror of flambeau_block_q8_1.
#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ8_1 {
    d: u16,            // f16 bits
    s: u16,            // f16 bits (sum * d)
    qs: [i8; 32],
}

#[test]
fn mmvq_q8_0_r4_matches_vdr2() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        eprintln!("no HIP device — skip");
        return Ok(());
    };
    let reg = flambeau_ops::OpsRegistry::new(&dev)?;

    // V2.31.b realistic 27B attn_q fused Q|gate shape: n_rows=12288,
    // k=5120 = 160 blocks. Start there — bug may only appear at
    // production-sized shapes.
    let n_rows: usize = std::env::var("TEST_N_ROWS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(12288);
    let n_blocks: usize = std::env::var("TEST_N_BLOCKS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(160);
    eprintln!("test config: n_rows={n_rows} n_blocks={n_blocks}");

    // Deterministic weights: block[r, b].qs[i] = (r * 32 + b * 8 + i) % 37 - 18
    let mut weights: Vec<BlockQ8_0> = Vec::with_capacity(n_rows * n_blocks);
    for r in 0..n_rows {
        for b in 0..n_blocks {
            let mut qs = [0i8; 32];
            for i in 0..32 {
                qs[i] = ((r * 32 + b * 8 + i) as i32 % 37 - 18) as i8;
            }
            let d = f16::from_f32(0.02 * (r as f32 + 1.0) * (b as f32 + 0.5));
            weights.push(BlockQ8_0 { d: d.to_bits(), qs });
        }
    }

    // Activation: qs[i] = (i * 7) % 31 - 15, d = 0.03
    let mut acts: Vec<BlockQ8_1> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = ((b * 8 + i) as i32 * 7 % 31 - 15) as i8;
        }
        let d = f16::from_f32(0.03);
        let s = f16::from_f32(0.0);
        acts.push(BlockQ8_1 {
            d: d.to_bits(),
            s: s.to_bits(),
            qs,
        });
    }

    let d_w = upload(&dev, &weights);
    let d_y = upload(&dev, &acts);
    let d_vdr2 = dev.alloc(n_rows * 4)?;
    let d_r4 = dev.alloc(n_rows * 4)?;

    // Launch vdr2.
    {
        let module = reg.module("mmvq_q8_0_dp4a_vdr2").expect("module");
        let kernel: HipKernel<'_> = module.kernel("flambeau_mmvq_q8_0_dp4a_vdr2_q8_1")?;
        let w: u64 = d_w.as_usize() as u64;
        let y: u64 = d_y.as_usize() as u64;
        let o: u64 = d_vdr2.as_usize() as u64;
        let nr = n_rows as i32;
        let nb = n_blocks as i32;
        let mut args = KernelArgs::new();
        args.push(&w);
        args.push(&y);
        args.push(&o);
        args.push(&nr);
        args.push(&nb);
        let cfg = LaunchCfg::one_d(n_rows as u32, 256);
        unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
        dev.default_stream().synchronize()?;
    }

    // Launch r4.
    {
        let module = reg.module("mmvq_q8_0_r4_dp4a").expect("module");
        let kernel: HipKernel<'_> = module.kernel("flambeau_mmvq_q8_0_r4_dp4a_q8_1")?;
        let w: u64 = d_w.as_usize() as u64;
        let y: u64 = d_y.as_usize() as u64;
        let o: u64 = d_r4.as_usize() as u64;
        let nr = n_rows as i32;
        let nb = n_blocks as i32;
        let mut args = KernelArgs::new();
        args.push(&w);
        args.push(&y);
        args.push(&o);
        args.push(&nr);
        args.push(&nb);
        let grid = ((n_rows + 3) / 4) as u32;
        let cfg = LaunchCfg::one_d(grid, 64);
        unsafe { kernel.launch(dev.default_stream(), cfg, args)? };
        dev.default_stream().synchronize()?;
    }

    // Download.
    let mut out_vdr2 = vec![0f32; n_rows];
    let mut out_r4 = vec![0f32; n_rows];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_vdr2.as_mut_ptr() as usize),
            d_vdr2,
            n_rows * 4,
        )?;
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out_r4.as_mut_ptr() as usize),
            d_r4,
            n_rows * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    // Print first 16 rows + any divergent rows.
    let mut div_rows: Vec<usize> = (0..n_rows)
        .filter(|&r| (out_r4[r] - out_vdr2[r]).abs() > 1e-3)
        .collect();
    div_rows.truncate(16);
    eprintln!("row : vdr2 (ref)    r4 (got)       diff");
    for &r in &div_rows {
        let diff = out_r4[r] - out_vdr2[r];
        eprintln!("  {r}: {:>14.6}  {:>14.6}  {:>+10.6}", out_vdr2[r], out_r4[r], diff);
    }
    if div_rows.is_empty() {
        for r in 0..(n_rows.min(8)) {
            let diff = out_r4[r] - out_vdr2[r];
            eprintln!("  {r}: {:>14.6}  {:>14.6}  {:>+10.6}", out_vdr2[r], out_r4[r], diff);
        }
    }
    let n_div = (0..n_rows).filter(|&r| (out_r4[r] - out_vdr2[r]).abs() > 1e-3).count();
    eprintln!("divergent rows: {n_div} / {n_rows}");

    let mut max_diff = 0f32;
    for r in 0..n_rows {
        max_diff = max_diff.max((out_r4[r] - out_vdr2[r]).abs());
    }
    eprintln!("max |diff| = {max_diff}");

    // Allow minor F32 drift but not wholesale wrong.
    assert!(
        max_diff < 1e-3,
        "r4 output diverges from vdr2 by {max_diff}"
    );

    unsafe {
        dev.dealloc(d_w, weights.len() * std::mem::size_of::<BlockQ8_0>())?;
        dev.dealloc(d_y, acts.len() * std::mem::size_of::<BlockQ8_1>())?;
        dev.dealloc(d_vdr2, n_rows * 4)?;
        dev.dealloc(d_r4, n_rows * 4)?;
    }
    Ok(())
}
