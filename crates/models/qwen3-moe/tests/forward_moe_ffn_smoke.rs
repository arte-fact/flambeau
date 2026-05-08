//! d1 smoke test — routed MoE FFN decode step on real HIP with zero
//! Q4_K weights + hardcoded routing. Asserts the composition (quant →
//! indexed gate+up → swiglu → cast → quant → indexed down → cast → combine)
//! produces finite output and that `moe_combine_f16` with zero expert
//! outputs returns the residual unchanged (hidden F16 lanes preserved).

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async \
              over host/device buffers that live for the bounded synchronize that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, OpsRegistry};
use flambeau_quant::{BlockQ4K, GgmlDType, QK_K};
use flambeau_qwen3_moe::forward::{forward_moe_ffn_decode, MoeScratch};
use flambeau_qwen3_moe::weights::{DeviceTensor, FfnWeights};
use flambeau_qwen3_moe::{AttentionFamily, GdnDims, Qwen3MoEConfig, RopeSpec};
use half::f16;
use std::sync::Arc;

fn hip_device() -> Option<HipDevice> {
    let n = flambeau_backend_hip::device_count().ok()?;
    if n < 1 {
        return None;
    }
    HipDevice::new(0).ok()
}

fn tiny_moe_cfg() -> Qwen3MoEConfig {
    // Tight fixture: hidden = moe_intermediate_size = QK_K = 256 so every
    // Q4_K matmul has `n_sb_per_row = 1`.
    Qwen3MoEConfig {
        arch: "qwen35moe".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: QK_K,
        vocab_size: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        context_length: 16,
        rms_norm_eps: 1e-6,
        rope: RopeSpec {
            freq_base: 10_000.0,
            rotated_dims: 32,
            sections: None,
        },
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: QK_K,
        shared_expert_intermediate_size: None,
        full_attention_interval: Some(4),
        gdn: Some(GdnDims {
            d_inner: 64,
            head_k_dim: 128,
            num_k_heads: 1,
            num_v_heads: 2,
            conv_kernel: 4,
        }),
        tied_lm_head: true,
            pooling_type: None,
    }
}

/// Allocate a zero-filled Q4_K expert tensor. Shape: `[n_experts, n_rows, k]`.
fn alloc_q4k_experts(
    device: &HipDevice,
    name: &str,
    n_experts: usize,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert_eq!(k % QK_K, 0, "Q4_K requires k % {QK_K} == 0");
    let n_blocks = n_experts * n_rows * (k / QK_K);
    let bytes = n_blocks * std::mem::size_of::<BlockQ4K>();
    let ptr = device.alloc(bytes)?;
    let zeros = vec![0u8; bytes];
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(zeros.as_ptr() as usize),
            bytes,
        )?;
    }
    device.default_stream().synchronize()?;
    drop(zeros);
    Ok(DeviceTensor {
        ptr,
        dtype: GgmlDType::Q4K,
        // GGUF outermost-first: [n_experts, n_rows, k].
        dims: vec![n_experts as u64, n_rows as u64, k as u64],
        bytes,
        name: Arc::from(name),
    })
}

fn alloc_f32_vec(device: &HipDevice, name: &str, host: &[f32]) -> Result<DeviceTensor> {
    let bytes = std::mem::size_of_val(host);
    let ptr = device.alloc(bytes)?;
    unsafe {
        device.memcpy_async(
            device.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    device.default_stream().synchronize()?;
    Ok(DeviceTensor {
        ptr,
        dtype: GgmlDType::F32,
        dims: vec![host.len() as u64],
        bytes,
        name: Arc::from(name),
    })
}

fn free_tensor(device: &HipDevice, t: DeviceTensor) -> Result<()> {
    unsafe {
        device.dealloc(t.ptr, t.bytes)?;
    }
    Ok(())
}

#[test]
fn forward_moe_ffn_decode_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_moe_ffn_decode_smoke");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let cfg = tiny_moe_cfg();
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;

    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let top_k = cfg.num_experts_per_tok;
    let n_experts = cfg.num_experts;

    // Dummy expert weights: all zero Q4_K → expert matmuls produce zero →
    // swiglu produces zero → combine falls back to pure residual.
    let gate_w = alloc_q4k_experts(
        &device,
        "blk.0.ffn_gate_exps.weight",
        n_experts,
        inter,
        hidden,
    )?;
    let up_w = alloc_q4k_experts(
        &device,
        "blk.0.ffn_up_exps.weight",
        n_experts,
        inter,
        hidden,
    )?;
    let down_w = alloc_q4k_experts(
        &device,
        "blk.0.ffn_down_exps.weight",
        n_experts,
        hidden,
        inter,
    )?;

    // ffn_gate_inp is unused in d1 (router lands in d3) — we still
    // populate the DeviceTensor field with a placeholder so FfnWeights is
    // complete.
    let gate_inp_host = vec![0.0f32; n_experts * hidden];
    let gate_inp = alloc_f32_vec(&device, "blk.0.ffn_gate_inp.weight", &gate_inp_host)?;

    let ffn = FfnWeights {
        ffn_gate_inp: Some(gate_inp.clone()),
        ffn_gate_exps: Some(gate_w.clone()),
        ffn_up_exps: Some(up_w.clone()),
        ffn_down_exps: Some(down_w.clone()),
        shared: None,
        dense: None,
    };

    let mut scratch = MoeScratch::new(&cfg, &device)?;

    // Hardcoded routing: expert ids {0, 1}, weights that sum to 1.0 so the
    // "weighted routed sum" is well-defined even though the contributions
    // will be zero. Feed into the scratch buffers the forward reads from.
    let expert_ids_host: Vec<i32> = (0..top_k as i32).collect();
    let expert_weights_host: Vec<f32> = vec![1.0 / top_k as f32; top_k];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.expert_ids,
            DevicePtr(expert_ids_host.as_ptr() as usize),
            top_k * 4,
        )?;
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            scratch.expert_weights,
            DevicePtr(expert_weights_host.as_ptr() as usize),
            top_k * 4,
        )?;
    }
    stream.synchronize()?;

    // Input + residual + out — fill x_norm and residual with distinct
    // patterns so we can verify the combine's residual-pass-through.
    let x_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32((i + 1) as f32 * 0.01))
        .collect();
    let residual_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32(-0.5 + (i as f32) * 0.002))
        .collect();
    let bytes = hidden * 2;
    let d_x = device.alloc(bytes)?;
    let d_residual = device.alloc(bytes)?;
    let d_out = device.alloc(bytes)?;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(x_host.as_ptr() as usize),
            bytes,
        )?;
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            d_residual,
            DevicePtr(residual_host.as_ptr() as usize),
            bytes,
        )?;
    }
    stream.synchronize()?;

    forward_moe_ffn_decode(
        &ops,
        stream,
        &cfg,
        &ffn,
        &mut scratch,
        d_x,
        d_residual,
        None,
        d_out,
    )?;

    let mut out_host = vec![f16::from_f32(0.0); hidden];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out_host.as_mut_ptr() as usize),
            d_out,
            bytes,
        )?;
    }
    stream.synchronize()?;

    // Zero expert outputs → out == residual, bit-exact.
    for (i, (got, want)) in out_host.iter().zip(&residual_host).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "lane {i}: combine with zero experts must equal residual; got {}, want {}",
            got.to_f32(),
            want.to_f32()
        );
    }

    // Teardown.
    unsafe {
        device.dealloc(d_x, bytes)?;
        device.dealloc(d_residual, bytes)?;
        device.dealloc(d_out, bytes)?;
    }
    scratch.dispose(&device)?;
    for t in [gate_w, up_w, down_w, gate_inp] {
        free_tensor(&device, t)?;
    }
    Ok(())
}
