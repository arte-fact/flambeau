//! MTP-INV-1 — stage-by-stage activation bisect vs Python ref.
//!
//! After the canonical parity (`mtp_step_parity.rs`) was confirmed
//! to fail by max=10/mean=1.4 against fresh ref data, we need to
//! find which stage of `forward_mtp_step` first diverges from the
//! reference computation.
//!
//! Stages compared (from `tools/mtp_reference.py`):
//!   * fc_in           — concat([norm_e, norm_h]) before fc matmul         [10240]
//!   * h0              — fc output                                         [5120]
//!   * attn_pre_gate   — attention output before sigmoid gate              [6144]
//!   * attn_post_gate  — after sigmoid_mul                                 [6144]
//!   * h1              — h0 + attn_proj (post first residual)              [5120]
//!   * h2              — h1 + down_proj (post second residual)             [5120]
//!   * h_final         — final norm output                                 [5120]
//!
//! For each stage we report max_abs_diff and the first 8 elements of
//! ref and got side-by-side. The first stage with max_abs > a small
//! threshold is the bug locus.

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
const HEAD_DIM: usize = 256;
const NUM_Q_HEADS: usize = 24;
const REF_DIR_REL: &str = "tests/data/mtp_ref";

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
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * 4,
        );
    }
    Ok(out)
}

fn dl_f16_to_f32(dev: &HipDevice, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut h = vec![f16::from_f32(0.0); n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(h.as_mut_ptr() as usize),
            ptr,
            n * 2,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(h.iter().map(|v| v.to_f32()).collect())
}

fn dl_f32(dev: &HipDevice, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut h = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(h.as_mut_ptr() as usize),
            ptr,
            n * 4,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(h)
}

fn compare(stage: &str, got: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), expected.len(), "{stage}: shape mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut argmax_idx = 0usize;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
            argmax_idx = i;
        }
        sum_abs += d as f64;
    }
    let mean_abs = (sum_abs / got.len() as f64) as f32;

    eprintln!("── stage {stage} ({} elems)", got.len());
    eprintln!("    ref [0..8]: {:?}", &expected[..8.min(expected.len())]);
    eprintln!("    got [0..8]: {:?}", &got[..8.min(got.len())]);
    eprintln!("    max_abs    = {max_abs:.4e} at idx {argmax_idx} (ref={:.4} got={:.4})",
        expected[argmax_idx], got[argmax_idx]);
    eprintln!("    mean_abs   = {mean_abs:.4e}");
    (max_abs, mean_abs)
}

#[test]
fn mtp_step_bisect_position_zero() -> Result<()> {
    if device_count().unwrap_or(0) < 1 {
        eprintln!("skip: no HIP device");
        return Ok(());
    }
    let mtp_path = PathBuf::from("/artefact/models/Qwen3.6-27B-mtp.gguf");
    let base_path = PathBuf::from("/artefact/models/Qwen3.6-27B-Q4_0.gguf");
    if !mtp_path.exists() || !base_path.exists() {
        eprintln!("skip: required GGUFs not present");
        return Ok(());
    }

    // Reference vectors (regenerated from tools/mtp_reference.py).
    let h_t_f32 = read_f32_bin("h_t.bin")?;
    let e_token_f32 = read_f32_bin("e_token.bin")?;
    let exp_fc_in = read_f32_bin("expected_fc_in.bin")?;
    let exp_h0 = read_f32_bin("expected_h0.bin")?;
    let exp_attn_pre = read_f32_bin("expected_attn_pre_gate.bin")?;
    let exp_attn_post = read_f32_bin("expected_attn_post_gate.bin")?;
    let exp_h1 = read_f32_bin("expected_h1.bin")?;
    let exp_h2 = read_f32_bin("expected_h2.bin")?;
    let exp_h_final = read_f32_bin("expected_h_final.bin")?;

    // Load model + run forward.
    let device = HipDevice::new(0)?;
    let base_file = GgufFile::open(&base_path)?;
    let cfg = Qwen3MoEConfig::from_gguf(&base_file)?;
    assert_eq!(cfg.hidden_size, HIDDEN);
    let mtp_file = GgufFile::open(&mtp_path)?;
    let mtp = load_mtp_head(&mtp_file, &device)?;

    let h_t_f16: Vec<f16> = h_t_f32.iter().map(|x| f16::from_f32(*x)).collect();
    let e_token_f16: Vec<f16> = e_token_f32.iter().map(|x| f16::from_f32(*x)).collect();
    let h_t_dev = device.alloc(HIDDEN * 2)?;
    let e_token_dev = device.alloc(HIDDEN * 2)?;
    let h_final_dev = device.alloc(HIDDEN * 2)?;
    unsafe {
        device.memcpy_async(device.default_stream(), CopyDirection::HostToDevice,
            h_t_dev, DevicePtr(h_t_f16.as_ptr() as usize), HIDDEN * 2)?;
        device.memcpy_async(device.default_stream(), CopyDirection::HostToDevice,
            e_token_dev, DevicePtr(e_token_f16.as_ptr() as usize), HIDDEN * 2)?;
    }
    device.default_stream().synchronize()?;

    let ops = OpsRegistry::new(&device)?;
    let scratch = MtpForwardScratch::new(&device, &cfg)?;

    forward_mtp_step(
        &ops, device.default_stream(), &device, &cfg, &mtp, &scratch,
        h_t_dev, e_token_dev, 0, h_final_dev, None,
    )?;

    // Download every stage. Note: h0 buffer in scratch is F16 (h0_f16
    // is the cast of the F32 mmvq output). We compare h0_f16 since
    // that's what the next op consumes, and it matches the Python
    // ref's h0 which is the same scalar values.
    let got_fc_in    = dl_f16_to_f32(&device, scratch.fc_in_f16, 2 * HIDDEN)?;
    let got_h0       = dl_f32(&device, scratch.h0_f32, HIDDEN)?;
    let got_attn_pre = dl_f16_to_f32(&device, scratch.attn_out_f16, NUM_Q_HEADS * HEAD_DIM)?;
    let got_attn_post= dl_f16_to_f32(&device, scratch.gated_out_f16, NUM_Q_HEADS * HEAD_DIM)?;
    // h1 = h0_f32 + attn_proj_f32 written into attn_proj_f32 in F32 —
    // but the F32 sum is then cast to F16 (h1_f16) before
    // post-attn-norm. The Python ref h1 is from BEFORE any cast.
    let got_h1_f32   = dl_f32(&device, scratch.attn_proj_f32, HIDDEN)?;
    let got_h2_f32   = dl_f32(&device, scratch.down_f32, HIDDEN)?;
    let got_h_final  = dl_f16_to_f32(&device, h_final_dev, HIDDEN)?;

    // Stage-by-stage compare.
    eprintln!("\n=== MTP-INV-1 stage bisect (position=0) ===\n");
    compare("fc_in",          &got_fc_in,    &exp_fc_in);
    compare("h0",             &got_h0,       &exp_h0);
    compare("attn_pre_gate",  &got_attn_pre, &exp_attn_pre);
    compare("attn_post_gate", &got_attn_post,&exp_attn_post);
    compare("h1",             &got_h1_f32,   &exp_h1);
    compare("h2",             &got_h2_f32,   &exp_h2);
    compare("h_final",        &got_h_final,  &exp_h_final);

    // ── Head-by-head diff for attn_pre_gate to verify Q split layout ──
    eprintln!("\n── per-head max_abs(attn_pre_gate) ──");
    for h in 0..NUM_Q_HEADS {
        let lo = h * HEAD_DIM;
        let hi = lo + HEAD_DIM;
        let mut m = 0.0f32;
        for i in lo..hi {
            let d = (got_attn_pre[i] - exp_attn_pre[i]).abs();
            if d > m { m = d; }
        }
        eprintln!("    head {h:2}: max_abs = {m:.4e}");
    }

    // ── Q/gate split layout probe: dump q_full_f32 (raw q_proj output)
    //    and check both layouts against the post_gate divergence.
    let q_full = dl_f32(&device, scratch.q_full_f32, 2 * NUM_Q_HEADS * HEAD_DIM)?;
    let exp_pre = &exp_attn_pre; // Python's "Q half" == first 6144 (layout A)

    // For element 0 of post_gate: ref says sigmoid(gate_ref)*attn = -1.381 / -1.877.
    // sigmoid(gate_ref) ≈ 0.735 → gate_ref ≈ +1.02.
    //
    // Layout A says gate[0] = q_full[6144]. Layout B says gate[0] = q_full[256].
    // Print both to disambiguate.
    eprintln!("\n── q_proj raw output layout probe ──");
    eprintln!("    q_full[0..8]      (head-0 Q for both layouts):");
    eprintln!("      {:?}", &q_full[..8]);
    eprintln!("    q_full[256..264]  (layout-B head-0 gate / layout-A head-1 Q):");
    eprintln!("      {:?}", &q_full[256..264]);
    eprintln!("    q_full[6144..6152](layout-A all-heads gate start):");
    eprintln!("      {:?}", &q_full[6144..6152]);
    eprintln!("    Expected gate from post_gate = -1.381 / attn={:.4} → gate≈{:.4}",
        exp_pre[0], (-1.381f32 / exp_pre[0]).ln() * -1.0); // not quite right but informative
    let _ = exp_pre;

    // Cleanup.
    unsafe {
        device.dealloc(h_t_dev, HIDDEN * 2)?;
        device.dealloc(e_token_dev, HIDDEN * 2)?;
        device.dealloc(h_final_dev, HIDDEN * 2)?;
    }
    scratch.dispose(&device)?;

    // Always returns Ok — this is a measurement test, not a gate.
    Ok(())
}
