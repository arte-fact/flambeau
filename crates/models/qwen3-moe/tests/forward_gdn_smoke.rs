//! V1.7.3-c2 smoke test — run one decode step of `forward_gdn_decode` on
//! real HIP hardware with dummy weights and assert the 17-step kernel chain
//! produces finite output. Fixture uses S_v=128 (the only instantiation of
//! the GDN state-step kernel today).
//!
//! Not a correctness cert; that lands in V1.7.4. This just validates the
//! kernel composition + pointer plumbing + conv-history rotation +
//! alpha/beta host compute + state-alias handling.

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
use flambeau_qwen3_moe::forward::{forward_gdn_decode, GdnScratch};
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

/// Smallest GDN fixture that respects the kernel constraints:
/// - `head_k_dim == head_v_dim == 128` (V1.7.2.F kernel specialisation)
/// - `hidden % 32 == 0` (Q8_1 quant)
/// - `d_inner % 32 == 0` (ssm_out Q8_1 quant)
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
        // With full_attention_interval=4, is_recurrent(il) = (il+1) % 4 != 0
        // → layer 0 is GDN (matches Qwen3.6: 30 GDN + 10 full-attn).
        full_attention_interval: Some(4),
        gdn: Some(GdnDims {
            d_inner: 256, // 2 × head_v_dim × num_v_heads = 2 × 128
            head_k_dim: 128,
            num_k_heads: 1,
            num_v_heads: 2,
            conv_kernel: 4,
        }),
        tied_lm_head: true,
            pooling_type: None,
    }
}

fn alloc_q8_0_zero_weight(
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert!(k % 32 == 0, "Q8_0 requires k % 32 == 0");
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
        // GGUF outermost-first: dims = [n_rows, k].
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
    k: u64,
    n_rows: u64,
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
        dims: vec![k, n_rows],
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
fn forward_gdn_decode_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_gdn_decode_smoke");
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
    let num_k_heads = gdn.num_k_heads;
    let conv_channels = gdn.conv_channels();
    let conv_kernel = gdn.conv_kernel;
    let head_k_dim = gdn.head_k_dim;

    // Weights: all matmul weights are Q8_0 zeros; norm weights F16 ones;
    // ssm_a and ssm_dt_bias small non-trivial F32 values so the
    // softplus/sigmoid path doesn't degenerate to zeros.
    let attn_norm = alloc_f16_ones(&device, "blk.0.attn_norm.weight", hidden)?;
    let attn_qkv = alloc_q8_0_zero_weight(
        &device,
        "blk.0.attn_qkv.weight",
        conv_channels,
        hidden,
    )?;
    let attn_gate = alloc_q8_0_zero_weight(&device, "blk.0.attn_gate.weight", d_inner, hidden)?;
    let ssm_alpha = alloc_q8_0_zero_weight(&device, "blk.0.ssm_alpha.weight", num_v_heads, hidden)?;
    let ssm_beta = alloc_q8_0_zero_weight(&device, "blk.0.ssm_beta.weight", num_v_heads, hidden)?;
    let ssm_out = alloc_q8_0_zero_weight(&device, "blk.0.ssm_out.weight", hidden, d_inner)?;

    let ssm_a_host: Vec<f32> = (0..num_v_heads).map(|i| -(0.1 + 0.05 * i as f32)).collect();
    let ssm_dt_host: Vec<f32> = (0..num_v_heads).map(|i| 0.01 * i as f32).collect();
    let ssm_conv1d_host: Vec<f32> = vec![1.0 / conv_kernel as f32; conv_kernel * conv_channels];
    let ssm_norm_host: Vec<f32> = vec![1.0; head_k_dim];

    let ssm_a = alloc_f32(&device, "blk.0.ssm_a", &ssm_a_host)?;
    let ssm_dt_bias = alloc_f32(&device, "blk.0.ssm_dt.bias", &ssm_dt_host)?;
    // ssm_conv1d.weight shape [conv_kernel, conv_channels] — GGUF convention
    // puts inner (channels) first in dims[], with rows (kernel taps) second.
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

    // Session for the single GDN layer: allocates state + conv_history.
    let mut session = Qwen3MoESession::new(&cfg, &device, flambeau_qwen3_moe::session::KvLayout::F16)?;
    let state = match &mut session.layers_mut()[0] {
        LayerCache::Gdn(s) => s,
        LayerCache::FullAttn(_) | LayerCache::FullAttnQ8(_) => {
            anyhow::bail!("layer 0 should be GDN")
        }
    };
    // Re-pack so we can pass a mutable reference without fighting the borrow
    // checker across the helper call below — take it out, use, put back.
    let mut state_take = GdnLayerState {
        state: state.state,
        state_bytes: state.state_bytes,
        conv_history: state.conv_history,
        conv_history_bytes: state.conv_history_bytes,
        num_v_heads: state.num_v_heads,
        head_k_dim: state.head_k_dim,
        head_v_dim: state.head_v_dim,
        conv_kernel: state.conv_kernel,
        conv_channels: state.conv_channels,
        snapshot_state: None,
        snapshot_conv_history: None,
    };

    let mut scratch = GdnScratch::new(&cfg, &device)?;

    let x_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32((i + 1) as f32 * 0.01))
        .collect();
    let x_bytes = hidden * 2;
    let x_in = device.alloc(x_bytes)?;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            x_in,
            DevicePtr(x_host.as_ptr() as usize),
            x_bytes,
        )?;
    }
    stream.synchronize()?;
    let delta_out = device.alloc(x_bytes)?;

    forward_gdn_decode(
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
    )?;

    // Read back and verify every lane is finite.
    let mut out_host = vec![f16::from_f32(0.0); hidden];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(out_host.as_mut_ptr() as usize),
            delta_out,
            x_bytes,
        )?;
    }
    stream.synchronize()?;

    let bad: Vec<(usize, f32)> = out_host
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let f = v.to_f32();
            (!f.is_finite()).then_some((i, f))
        })
        .collect();
    assert!(
        bad.is_empty(),
        "GDN forward produced non-finite output at {} lanes: {:?}",
        bad.len(),
        &bad[..bad.len().min(4)]
    );
    // Zero Q8_0 ssm_out weights → output is exactly zero.
    assert!(
        out_host.iter().all(|v| v.to_f32().abs() < 1e-3),
        "expected near-zero output with zero ssm_out weights, got first lane = {}",
        out_host[0].to_f32()
    );

    // `num_k_heads` and `num_v_heads` must match the fixture — quick
    // sanity to make sure the scratch sizes line up with our fixture.
    assert_eq!(num_k_heads, 1);
    assert_eq!(num_v_heads, 2);

    // Teardown.
    unsafe {
        device.dealloc(x_in, x_bytes)?;
        device.dealloc(delta_out, x_bytes)?;
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
