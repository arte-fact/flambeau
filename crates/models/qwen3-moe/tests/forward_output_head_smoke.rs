//! e3 smoke test — output norm + LM head + argmax sampling. Uses
//! zero Q8_0 LM head weight so logits are all zero, then verifies that
//! `argmax_token_host` returns 0 (first lane wins on ties).

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
use flambeau_quant::{BlockQ8_0, GgmlDType};
use flambeau_qwen3_moe::forward::{
    argmax_token_host, forward_output_head_decode, OutputHeadScratch,
};
use flambeau_qwen3_moe::weights::DeviceTensor;
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

fn tiny_cfg(hidden: usize, vocab: usize) -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        arch: "qwen35moe".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: hidden,
        vocab_size: vocab,
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
        moe_intermediate_size: 256,
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

fn alloc_q8_0_zero(
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert_eq!(k % 32, 0);
    let n_blocks = n_rows * (k / 32);
    let bytes = n_blocks * std::mem::size_of::<BlockQ8_0>();
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
        dtype: GgmlDType::Q8_0,
        dims: vec![n_rows as u64, k as u64],
        bytes,
        name: Arc::from(name),
    })
}

fn alloc_f16_ones(device: &HipDevice, name: &str, n: usize) -> Result<DeviceTensor> {
    let host: Vec<f16> = vec![f16::from_f32(1.0); n];
    let bytes = n * 2;
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
    drop(host);
    Ok(DeviceTensor {
        ptr,
        dtype: GgmlDType::F16,
        dims: vec![n as u64],
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
fn forward_output_head_zero_lm_head_argmax_zero() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_output_head_zero_lm_head_argmax_zero");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let hidden = 128usize;
    let vocab = 64usize;
    let cfg = tiny_cfg(hidden, vocab);
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;

    let output_norm = alloc_f16_ones(&device, "output_norm.weight", hidden)?;
    let lm_head = alloc_q8_0_zero(&device, "output.weight", vocab, hidden)?;
    let mut scratch = OutputHeadScratch::new(&cfg, &device)?;

    let x_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32((i + 1) as f32 * 0.01))
        .collect();
    let x = device.alloc(hidden * 2)?;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            x,
            DevicePtr(x_host.as_ptr() as usize),
            hidden * 2,
        )?;
    }
    stream.synchronize()?;

    forward_output_head_decode(&ops, stream, &cfg, &output_norm, &lm_head, &mut scratch, x)?;
    let token = argmax_token_host(&device, stream, scratch.logits_f32, vocab)?;

    // Zero LM head weights → every logit is exactly 0.0 → argmax picks
    // the first lane (index 0).
    assert_eq!(token, 0, "expected argmax over zeros to return 0");

    // Also verify every logit is finite + zero.
    let mut host = vec![0.0f32; vocab];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            scratch.logits_f32,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    for (i, v) in host.iter().enumerate() {
        assert!(
            v.is_finite() && v.abs() < 1e-3,
            "logit {i} = {v}; expected near-zero"
        );
    }

    unsafe {
        device.dealloc(x, hidden * 2)?;
    }
    scratch.dispose(&device)?;
    for t in [output_norm, lm_head] {
        free_tensor(&device, t)?;
    }
    Ok(())
}
