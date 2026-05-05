//! V1.7.3-f1 smoke test — full-attention prefill over L = 4 tokens on real
//! HIP with zero Q8_0 projection weights. Verifies that the prefill
//! function runs to completion, appends 4 tokens to the KV cache, and
//! writes a finite near-zero delta for every input row.

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
use flambeau_qwen3_moe::forward::{forward_full_attn_prefill, FullAttnPrefillScratch};
use flambeau_qwen3_moe::weights::{DeviceTensor, FullAttnWeights};
use flambeau_qwen3_moe::{AttentionFamily, GdnDims, Qwen3MoEConfig, RopeSpec};
use flambeau_runtime::{F16Contig, KvCache};
use half::f16;
use std::sync::Arc;

fn hip_device() -> Option<HipDevice> {
    let n = flambeau_backend_hip::device_count().ok()?;
    if n < 1 {
        return None;
    }
    HipDevice::new(0).ok()
}

fn tiny_cfg() -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        arch: "qwen35moe".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: 256,
        vocab_size: 128,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 128,
        context_length: 16,
        rms_norm_eps: 1e-6,
        rope: RopeSpec {
            freq_base: 10_000.0,
            rotated_dims: 64,
            sections: None,
        },
        num_experts: 0,
        num_experts_per_tok: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: None,
        full_attention_interval: Some(1),
        gdn: Some(GdnDims {
            d_inner: 64,
            head_k_dim: 32,
            num_k_heads: 2,
            num_v_heads: 2,
            conv_kernel: 4,
        }),
        tied_lm_head: true,
            pooling_type: None,
    }
}

fn alloc_q8_0(
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
fn forward_full_attn_prefill_l4_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_full_attn_prefill_l4_smoke");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let cfg = tiny_cfg();
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;

    let hidden = cfg.hidden_size;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let l = 8usize;

    let attn_norm = alloc_f16_ones(&device, "blk.0.attn_norm.weight", hidden)?;
    let attn_q_norm = alloc_f16_ones(&device, "blk.0.attn_q_norm.weight", head_dim)?;
    let attn_k_norm = alloc_f16_ones(&device, "blk.0.attn_k_norm.weight", head_dim)?;
    let attn_q = alloc_q8_0(&device, "blk.0.attn_q.weight", 2 * n_heads * head_dim, hidden)?;
    let attn_k = alloc_q8_0(&device, "blk.0.attn_k.weight", n_kv_heads * head_dim, hidden)?;
    let attn_v = alloc_q8_0(&device, "blk.0.attn_v.weight", n_kv_heads * head_dim, hidden)?;
    let attn_output =
        alloc_q8_0(&device, "blk.0.attn_output.weight", hidden, n_heads * head_dim)?;
    let weights = FullAttnWeights {
        attn_q: attn_q.clone(),
        attn_k: attn_k.clone(),
        attn_v: attn_v.clone(),
        attn_output: attn_output.clone(),
        attn_q_norm: attn_q_norm.clone(),
        attn_k_norm: attn_k_norm.clone(),
    };

    let mut kv: KvCache<F16Contig, HipDevice> =
        KvCache::new(&device, n_kv_heads, head_dim, cfg.context_length)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut scratch = FullAttnPrefillScratch::new(&cfg, &device, l)?;

    // Input: L rows of non-trivial F16 data.
    let x_host: Vec<f16> = (0..l * hidden)
        .map(|i| f16::from_f32((i + 1) as f32 * 0.001))
        .collect();
    let bytes = l * hidden * 2;
    let x_in = device.alloc(bytes)?;
    let delta_out = device.alloc(bytes)?;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            x_in,
            DevicePtr(x_host.as_ptr() as usize),
            bytes,
        )?;
    }
    stream.synchronize()?;

    forward_full_attn_prefill(
        &ops,
        stream,
        &device,
        &cfg,
        &attn_norm,
        &weights,
        &mut kv,
        &mut scratch,
        x_in,
        delta_out,
        l,
        0,
        None,
    )?;


    // KV cache should now hold L tokens of history.
    assert_eq!(
        kv.current_tokens(),
        l,
        "KV cache should have {l} tokens after prefill; got {}",
        kv.current_tokens()
    );

    // All L × hidden output lanes must be finite + near-zero (zero weights).
    let mut out_host = vec![f16::from_f32(0.0); l * hidden];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out_host.as_mut_ptr() as usize),
            delta_out,
            bytes,
        )?;
    }
    stream.synchronize()?;
    let bad: Vec<(usize, f32)> = out_host
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let f = v.to_f32();
            (!f.is_finite() || f.abs() > 1e-3).then_some((i, f))
        })
        .collect();
    assert!(
        bad.is_empty(),
        "prefill delta must be finite + near-zero with zero weights; bad: {:?}",
        &bad[..bad.len().min(4)]
    );

    unsafe {
        device.dealloc(x_in, bytes)?;
        device.dealloc(delta_out, bytes)?;
    }
    scratch.dispose(&device)?;
    kv.dispose(&device).map_err(|e| anyhow::anyhow!("{e}"))?;
    for t in [
        attn_norm,
        attn_q_norm,
        attn_k_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
    ] {
        free_tensor(&device, t)?;
    }
    Ok(())
}
