//! V1.7.3-e2 smoke test — token embedding gather (host dequant + upload).
//!
//! Validates that `forward_embed_decode_host` returns the correct
//! embedding row from an F16 `token_embd.weight`. Each vocab row is
//! seeded with a distinct arithmetic pattern so the host can verify the
//! gathered row matches by value.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async \
              over host/device buffers that live for the bounded synchronize that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::HipDevice;
use flambeau_quant::GgmlDType;
use flambeau_qwen3_moe::forward::forward_embed_decode_host;
use flambeau_qwen3_moe::weights::DeviceTensor;
use half::f16;
use std::sync::Arc;

fn hip_device() -> Option<HipDevice> {
    let n = flambeau_backend_hip::device_count().ok()?;
    if n < 1 {
        return None;
    }
    HipDevice::new(0).ok()
}

#[test]
fn forward_embed_decode_host_picks_correct_row() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_embed_decode_host_picks_correct_row");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let vocab = 8usize;
    let hidden = 64usize;

    // Build an F16 token_embd where row[v, i] = 0.1 * v + 0.001 * i. That
    // distinct pattern per vocab row lets the smoke test verify by value.
    let mut host = vec![f16::from_f32(0.0); vocab * hidden];
    for v in 0..vocab {
        for i in 0..hidden {
            host[v * hidden + i] =
                f16::from_f32(0.1 * v as f32 + 0.001 * i as f32);
        }
    }
    let bytes = host.len() * 2;
    let ptr = device.alloc(bytes)?;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    stream.synchronize()?;

    let token_embd = DeviceTensor {
        ptr,
        dtype: GgmlDType::F16,
        dims: vec![vocab as u64, hidden as u64],
        bytes,
        name: Arc::from("token_embd.weight"),
    };

    let out = device.alloc(hidden * 2)?;
    for tok in [0u32, 3, 7] {
        forward_embed_decode_host(&device, stream, &token_embd, tok, out, hidden)?;
        let mut got = vec![f16::from_f32(0.0); hidden];
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(got.as_mut_ptr() as usize),
                out,
                hidden * 2,
            )?;
        }
        stream.synchronize()?;
        for (i, v) in got.iter().enumerate() {
            let want = f16::from_f32(0.1 * tok as f32 + 0.001 * i as f32);
            assert_eq!(
                v.to_bits(),
                want.to_bits(),
                "token {tok} lane {i}: got {} want {}",
                v.to_f32(),
                want.to_f32()
            );
        }
    }

    // Out-of-range token_id must error, not silently read garbage.
    let err = forward_embed_decode_host(&device, stream, &token_embd, vocab as u32, out, hidden)
        .expect_err("token_id >= vocab should fail");
    let s = format!("{err}");
    assert!(
        s.contains("token_id") && s.contains("vocab"),
        "unexpected error message: {s}"
    );

    unsafe {
        device.dealloc(ptr, bytes)?;
        device.dealloc(out, hidden * 2)?;
    }
    Ok(())
}
