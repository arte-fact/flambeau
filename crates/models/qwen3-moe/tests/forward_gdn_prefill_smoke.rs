//! V1.7.3-f2 smoke test — Gated-Delta-Net prefill over L = 4 tokens on real
//! HIP. Zero Q8_0 projection weights (attn_qkv/attn_gate/ssm_alpha/ssm_beta/
//! ssm_out) + ones-pattern F32 ssm_conv1d / ssm_norm / ssm_a / ssm_dt_bias
//! (same setup as the decode smoke, just with L > 1). Expects a finite
//! near-zero delta across all L × hidden output lanes.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or `memcpy_async` \
              over host/device buffers that live for the bounded `synchronize()` that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, OpsRegistry};
use flambeau_quant::{BlockQ8_0, GgmlDType};
use flambeau_qwen3_moe::forward::{forward_gdn_prefill, GdnPrefillScratch};
use flambeau_qwen3_moe::weights::{DeviceTensor, GdnWeights};
use flambeau_qwen3_moe::{AttentionFamily, GdnDims, Qwen3MoEConfig, RopeSpec};
use flambeau_qwen3_moe::{GdnLayerState, LayerCache, Qwen3MoESession};
use half::f16;
use std::sync::Arc;

fn hip_device() -> Option<HipDevice> {
    let n = flambeau_backend_hip::device_count().ok()?;
    if n < 1 {
        return None;
    }
    HipDevice::new(0).ok()
}

fn tiny_gdn_cfg() -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        arch: "qwen35moe".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: 128,
        vocab_size: 128,
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
        num_experts: 0,
        num_experts_per_tok: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: None,
        full_attention_interval: Some(4),
        gdn: Some(GdnDims {
            d_inner: 256,
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

fn alloc_f32(device: &HipDevice, name: &str, host: &[f32]) -> Result<DeviceTensor> {
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

fn alloc_f32_2d(
    device: &HipDevice,
    name: &str,
    host: &[f32],
    n_rows: u64,
    k: u64,
) -> Result<DeviceTensor> {
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
        dims: vec![n_rows, k],
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
fn forward_gdn_prefill_l4_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_gdn_prefill_l4_smoke");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let cfg = tiny_gdn_cfg();
    let gdn = cfg.gdn.as_ref().unwrap();
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;

    let hidden = cfg.hidden_size;
    let d_inner = gdn.d_inner;
    let num_v_heads = gdn.num_v_heads;
    let conv_channels = gdn.conv_channels();
    let conv_kernel = gdn.conv_kernel;
    let head_k_dim = gdn.head_k_dim;
    let l = 8usize;

    let attn_norm_val = 1.0f32;
    let _attn_norm = alloc_f32(
        &device,
        "blk.0.attn_norm.weight",
        &vec![attn_norm_val; hidden],
    )?;
    let attn_norm = DeviceTensor {
        // Override: rmsnorm_quant_q8_1 takes F16 weight.
        ptr: {
            let host: Vec<f16> = vec![f16::from_f32(1.0); hidden];
            let bytes = hidden * 2;
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
            ptr
        },
        dtype: GgmlDType::F16,
        dims: vec![hidden as u64],
        bytes: hidden * 2,
        name: Arc::from("blk.0.attn_norm.weight"),
    };
    // (previous step replaced the attn_norm F32 fill with the F16 alloc above;
    // `attn_norm_val` is a Copy scalar so no explicit drop is needed.)
    let _ = attn_norm_val;

    let attn_qkv = alloc_q8_0_zero(&device, "blk.0.attn_qkv.weight", conv_channels, hidden)?;
    let attn_gate = alloc_q8_0_zero(&device, "blk.0.attn_gate.weight", d_inner, hidden)?;
    let ssm_alpha = alloc_q8_0_zero(&device, "blk.0.ssm_alpha.weight", num_v_heads, hidden)?;
    let ssm_beta = alloc_q8_0_zero(&device, "blk.0.ssm_beta.weight", num_v_heads, hidden)?;
    let ssm_out = alloc_q8_0_zero(&device, "blk.0.ssm_out.weight", hidden, d_inner)?;

    let ssm_a_host: Vec<f32> = (0..num_v_heads).map(|i| -(0.1 + 0.05 * i as f32)).collect();
    let ssm_dt_host: Vec<f32> = (0..num_v_heads).map(|i| 0.01 * i as f32).collect();
    let ssm_conv1d_host: Vec<f32> = vec![1.0 / conv_kernel as f32; conv_channels * conv_kernel];
    let ssm_norm_host: Vec<f32> = vec![1.0; head_k_dim];

    let ssm_a = alloc_f32(&device, "blk.0.ssm_a", &ssm_a_host)?;
    let ssm_dt_bias = alloc_f32(&device, "blk.0.ssm_dt.bias", &ssm_dt_host)?;
    let ssm_conv1d = alloc_f32_2d(
        &device,
        "blk.0.ssm_conv1d.weight",
        &ssm_conv1d_host,
        conv_channels as u64,
        conv_kernel as u64,
    )?;
    let ssm_norm = alloc_f32(&device, "blk.0.ssm_norm.weight", &ssm_norm_host)?;

    let weights = GdnWeights {
        attn_qkv: attn_qkv.clone(),
        attn_gate: attn_gate.clone(),
        ssm_alpha: Some(ssm_alpha.clone()),
        ssm_beta: Some(ssm_beta.clone()),
        ssm_ba: None,
        ssm_a: ssm_a.clone(),
        ssm_dt_bias: ssm_dt_bias.clone(),
        ssm_conv1d: ssm_conv1d.clone(),
        ssm_norm: ssm_norm.clone(),
        ssm_out: ssm_out.clone(),
    };

    // Session hands us the per-layer state + conv_history. Copy out into a
    // `GdnLayerState` we can mutate (same pattern as the decode smoke).
    let mut session = Qwen3MoESession::new(&cfg, &device, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let mut state_take = match &session.layers_mut()[0] {
        LayerCache::Gdn(s) => GdnLayerState {
            state: s.state,
            state_bytes: s.state_bytes,
            conv_history: s.conv_history,
            conv_history_bytes: s.conv_history_bytes,
            num_v_heads: s.num_v_heads,
            head_k_dim: s.head_k_dim,
            head_v_dim: s.head_v_dim,
            conv_kernel: s.conv_kernel,
            conv_channels: s.conv_channels,
            snapshot_state: None,
            snapshot_conv_history: None,
        },
        LayerCache::FullAttn(_) | LayerCache::FullAttnQ8(_) => {
            anyhow::bail!("layer 0 should be GDN")
        }
    };

    let mut scratch = GdnPrefillScratch::new(&cfg, &device, l)?;

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

    forward_gdn_prefill(
        &ops,
        stream,
        &device,
        &cfg,
        &attn_norm,
        &weights,
        &mut state_take,
        &mut scratch,
        x_in,
        delta_out,
        l,
        None,
    )?;

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
        "GDN prefill must produce finite near-zero output with zero ssm_out weights; \
         bad lanes ({} total): {:?}",
        bad.len(),
        &bad[..bad.len().min(4)]
    );

    unsafe {
        device.dealloc(x_in, bytes)?;
        device.dealloc(delta_out, bytes)?;
    }
    scratch.dispose(&device)?;
    session.dispose(&device)?;
    for t in [
        attn_norm, attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out, ssm_a, ssm_dt_bias,
        ssm_conv1d, ssm_norm,
    ] {
        free_tensor(&device, t)?;
    }
    Ok(())
}
