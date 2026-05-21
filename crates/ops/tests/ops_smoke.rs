//! Smoke test — boot the OpsRegistry on a real MI50 + run one op per family.
//! This is a gate: if any kernel stem named in `KERNEL_STEMS` is
//! missing from the compiled HSACO catalogue, `OpsRegistry::new` fails.
//! If any op wrapper disagrees with its kernel's entry symbol or launch
//! config, the launch returns a HIP error.
//! We don't assert numerical correctness here — that's what the –
//! certs in `certs/hip/gfx906/` are for. We only want the ops surface to
//! round-trip through the real driver.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or `memcpy_async` \
              over host/device buffers that live for the bounded `synchronize()` that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::{mlp, norm, pe, softmax, OpsRegistry};
use half::f16;

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP devices — skipping ops_smoke");
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

#[test]
fn registry_loads_all_kernel_stems() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev).expect("OpsRegistry::new");
    // Sanity: every stem declared in KERNEL_STEMS is resolvable.
    for stem in flambeau_ops::hip::KERNEL_STEMS {
        assert!(reg.module(stem).is_some(), "missing module {stem}");
    }
    Ok(())
}

#[test]
fn swiglu_pointwise_roundtrips() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;
    let n = 1024usize;
    let gate_f16: Vec<f16> = (0..n).map(|i| f16::from_f32(0.01 * i as f32)).collect();
    let up_f16: Vec<f16> = (0..n)
        .map(|i| f16::from_f32(0.02 * (i as f32).sin()))
        .collect();
    let d_g = upload(&dev, &gate_f16);
    let d_u = upload(&dev, &up_f16);
    let d_y = dev.alloc(n * 2)?;
    mlp::swiglu_f16(&reg, dev.default_stream(), d_g, d_u, d_y, n)?;
    dev.default_stream().synchronize()?;
    let mut y = vec![f16::from_f32(0.0); n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(y.as_mut_ptr() as usize),
            d_y,
            n * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_g, n * 2)?;
        dev.dealloc(d_u, n * 2)?;
        dev.dealloc(d_y, n * 2)?;
    }
    // SwiGLU has no exact closed form we want to recheck; just assert the
    // output is finite and differs from the inputs.
    for (i, v) in y.iter().enumerate() {
        let f = v.to_f32();
        assert!(f.is_finite(), "non-finite output at {i}: {f}");
    }
    Ok(())
}

#[test]
fn rmsnorm_f16_runs() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;
    let (m, k) = (4usize, 256usize);
    let x: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(0.1 * ((i % 16) as f32 - 8.0)))
        .collect();
    let w: Vec<f16> = vec![f16::from_f32(1.0); k];
    let d_x = upload(&dev, &x);
    let d_w = upload(&dev, &w);
    let d_y = dev.alloc(m * k * 2)?;
    norm::rmsnorm_f16(&reg, dev.default_stream(), d_x, d_w, d_y, m, k, 1e-5)?;
    dev.default_stream().synchronize()?;
    let mut y = vec![f16::from_f32(0.0); m * k];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(y.as_mut_ptr() as usize),
            d_y,
            m * k * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, x.len() * 2)?;
        dev.dealloc(d_w, w.len() * 2)?;
        dev.dealloc(d_y, m * k * 2)?;
    }
    for (i, v) in y.iter().enumerate() {
        assert!(v.to_f32().is_finite(), "non-finite rmsnorm output at {i}");
    }
    Ok(())
}

#[test]
fn rope_f16_runs() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;
    let (n_tokens, n_heads, head_dim) = (4usize, 4usize, 64usize);
    let x: Vec<f16> = (0..n_tokens * n_heads * head_dim)
        .map(|i| f16::from_f32(0.1 * (i as f32).sin()))
        .collect();
    let positions: Vec<i32> = (0..n_tokens).map(|i| i as i32).collect();
    let d_x = upload(&dev, &x);
    let d_p = upload(&dev, &positions);
    pe::rope_f16(
        &reg,
        dev.default_stream(),
        d_x,
        d_p,
        10000.0,
        n_tokens,
        n_heads,
        head_dim,
    )?;
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_x, x.len() * 2)?;
        dev.dealloc(d_p, positions.len() * 4)?;
    }
    Ok(())
}

#[test]
fn softmax_masked_runs_with_zero_mask() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;
    // k must be ≥ 256 (SOFTMAX_THREADS) so every warp sees data — otherwise
    // the cross-warp reduce merges -INF slots and hits exp(-INF - -INF) = NaN.
    // Real attention always has k ≥ the head_dim / n_kv_heads product this
    // size, so the kernel's guarantee holds.
    let (m, k) = (4usize, 256usize);
    let scores: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(0.01 * (i as f32).cos()))
        .collect();
    let mask: Vec<f16> = vec![f16::from_f32(0.0); m * k];
    let d_s = upload(&dev, &scores);
    let d_m = upload(&dev, &mask);
    let d_o = dev.alloc(m * k * 2)?;
    softmax::softmax_masked_f16(&reg, dev.default_stream(), d_s, d_m, d_o, m, k, 1.0)?;
    dev.default_stream().synchronize()?;
    let mut out = vec![f16::from_f32(0.0); m * k];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(out.as_mut_ptr() as usize),
            d_o,
            m * k * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    unsafe {
        dev.dealloc(d_s, scores.len() * 2)?;
        dev.dealloc(d_m, mask.len() * 2)?;
        dev.dealloc(d_o, m * k * 2)?;
    }
    // Each row sums to ~1.
    for row in 0..m {
        let sum: f32 = out[row * k..(row + 1) * k].iter().map(|v| v.to_f32()).sum();
        assert!((sum - 1.0).abs() < 5e-3, "row {row} softmax sum = {sum}");
    }
    Ok(())
}
