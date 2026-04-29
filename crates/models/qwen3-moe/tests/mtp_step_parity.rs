//! MTP-3.5 — parity test: run `forward_mtp_step` on the same fixed-seed
//! input the Python reference used (`tools/mtp_reference.py`), compare
//! `h_final` against `tests/data/mtp_ref/expected_h_final.bin`.
//!
//! Tolerance band: this is F16 hidden state pushed through 8 sequential
//! Q8_1-quantized × Q8_0-weight matmuls vs F32 PyTorch reference. The
//! Q8_1 ACTIVATION quantize alone adds ~1.5% per encode (5 encodes:
//! fc_in, h0n, gated, h1n, swiglu); the 8 matmuls compound that as
//! ~5-8% mean drift in the final h_final. Real structural bugs (sigmoid
//! → silu, transposed weight, wrong head-dim split) would show 10×
//! larger drift, so we set:
//!     max_abs_diff < 0.5   (catches structural; ignores Q8 outliers)
//!     mean_abs_diff < 0.1  (matches the ~5-8% noise floor)
//!
//! Reference h_final std ≈ 1.29, so 0.1 mean ≈ 7-8% S/N — same order
//! as the predicted noise floor. Tighter would catch noise; looser
//! would hide bugs.
//!
//! Skipped when:
//!   - no HIP device, OR
//!   - `/artefact/models/Qwen3.6-27B-mtp.gguf` missing, OR
//!   - reference vectors missing.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a memcpy or kernel \
              launch over host/device buffers that live for the bounded \
              synchronize that follows."
)]

use anyhow::{bail, Result};
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::mtp::{forward_mtp_step, load_mtp_head, MtpForwardScratch};
use flambeau_qwen3_moe::Qwen3MoEConfig;
use half::f16;
use std::path::PathBuf;

const HIDDEN: usize = 5120;
// Resolve relative to the package's manifest dir so the test runs from
// any cargo-invocation working directory.
const REF_DIR_REL: &str = "tests/data/mtp_ref";
// Match the Python ref's MTP_TEST_POSITION env var.
fn test_position() -> usize {
    std::env::var("MTP_TEST_POSITION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
const MAX_ABS_TOL: f32 = 0.5;
const MEAN_ABS_TOL: f32 = 0.1;

fn ref_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(REF_DIR_REL)
        .join(name)
}

fn read_f32_bin(name: &str) -> Result<Vec<f32>> {
    let path = ref_path(name);
    let bytes = std::fs::read(&path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{}: not a multiple of 4 bytes", path.display());
    }
    let n = bytes.len() / 4;
    let mut out = vec![0f32; n];
    // SAFETY: dst has n*4 bytes, src has n*4 bytes (we just checked).
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * 4,
        );
    }
    Ok(out)
}

#[test]
fn mtp_step_parity_position_zero() -> Result<()> {
    if device_count().unwrap_or(0) < 1 {
        eprintln!("skip: no HIP device");
        return Ok(());
    }
    let mtp_path = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    if !mtp_path.exists() {
        eprintln!("skip: {} not present", mtp_path.display());
        return Ok(());
    }
    // Pair with Q4_0 base — we just need its config (num heads, hidden,
    // rope params, etc.) — not the layer weights.
    let base_path = PathBuf::from("/artefact/models/Qwen3.6-27B-Q4_0.gguf");
    if !base_path.exists() {
        eprintln!("skip: {} not present", base_path.display());
        return Ok(());
    }

    // ── Load reference vectors
    let h_t_f32 = read_f32_bin("h_t.bin")?;
    let e_token_f32 = read_f32_bin("e_token.bin")?;
    let expected_f32 = read_f32_bin("expected_h_final.bin")?;
    assert_eq!(h_t_f32.len(), HIDDEN, "h_t.bin shape");
    assert_eq!(e_token_f32.len(), HIDDEN, "e_token.bin shape");
    assert_eq!(expected_f32.len(), HIDDEN, "expected_h_final.bin shape");

    // ── Load model config + MTP weights
    let device = HipDevice::new(0)?;
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    assert_eq!(cfg.hidden_size, HIDDEN);
    assert_eq!(cfg.num_heads, 24);
    assert_eq!(cfg.num_kv_heads, 4);
    assert_eq!(cfg.head_dim, 256);

    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, &device)?;

    // ── Upload inputs as F16 (cast on host)
    let h_t_f16: Vec<f16> = h_t_f32.iter().map(|x| f16::from_f32(*x)).collect();
    let e_token_f16: Vec<f16> = e_token_f32.iter().map(|x| f16::from_f32(*x)).collect();
    let h_t_dev = device.alloc(HIDDEN * 2)?;
    let e_token_dev = device.alloc(HIDDEN * 2)?;
    let h_final_dev = device.alloc(HIDDEN * 2)?;
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            h_t_dev,
            DevicePtr(h_t_f16.as_ptr() as usize),
            HIDDEN * 2,
        )?;
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            e_token_dev,
            DevicePtr(e_token_f16.as_ptr() as usize),
            HIDDEN * 2,
        )?;
    }
    device.default_stream().synchronize()?;

    // ── Forward
    let ops = OpsRegistry::new(&device)?;
    let scratch = MtpForwardScratch::new(&device, &cfg)?;
    forward_mtp_step(
        &ops,
        device.default_stream(),
        &device,
        &cfg,
        &mtp,
        &scratch,
        h_t_dev,
        e_token_dev,
        test_position(),
        h_final_dev,
        None,
    )?;

    // ── Download h_final, cast F16 → F32 for compare
    let mut h_final_f16 = vec![f16::from_f32(0.0); HIDDEN];
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(h_final_f16.as_mut_ptr() as usize),
            h_final_dev,
            HIDDEN * 2,
        )?;
    }
    device.default_stream().synchronize()?;
    let got_f32: Vec<f32> = h_final_f16.iter().map(|x| x.to_f32()).collect();

    // ── Compare
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut argmax_idx = 0usize;
    for (i, (g, e)) in got_f32.iter().zip(expected_f32.iter()).enumerate() {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
            argmax_idx = i;
        }
        sum_abs += d as f64;
    }
    let mean_abs = (sum_abs / HIDDEN as f64) as f32;

    eprintln!("h_final compare:");
    eprintln!("  ref [0..8]: {:?}", &expected_f32[..8]);
    eprintln!("  got [0..8]: {:?}", &got_f32[..8]);
    eprintln!("  max_abs_diff  = {max_abs:.6e}  at index {argmax_idx} (ref={:.4} got={:.4})",
        expected_f32[argmax_idx], got_f32[argmax_idx]);
    eprintln!("  mean_abs_diff = {mean_abs:.6e}");
    eprintln!("  tol max={MAX_ABS_TOL} mean={MEAN_ABS_TOL}");

    // Free
    unsafe {
        device.dealloc(h_t_dev, HIDDEN * 2)?;
        device.dealloc(e_token_dev, HIDDEN * 2)?;
        device.dealloc(h_final_dev, HIDDEN * 2)?;
    }
    scratch.dispose(&device)?;

    assert!(max_abs < MAX_ABS_TOL, "max_abs {max_abs} >= tol {MAX_ABS_TOL}");
    assert!(mean_abs < MEAN_ABS_TOL, "mean_abs {mean_abs} >= tol {MEAN_ABS_TOL}");
    Ok(())
}
