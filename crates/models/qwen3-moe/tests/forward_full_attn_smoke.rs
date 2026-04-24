//! V1.7.3-b smoke test — run one decode step of `forward_full_attn_decode`
//! on real HIP hardware with dummy weights + inputs. Verifies the whole
//! kernel chain (rmsnorm → mmvq → cast → split_q_gate → per-head rmsnorm
//! → RoPE → kv append → attention_decode → swiglu → mmvq → cast) wires up
//! without panicking and writes a finite (non-NaN, non-Inf) output.
//!
//! This is not a correctness cert — it's the "no crash, no NaN" gate.
//! Per-layer parity vs llama.cpp lands in V1.7.4.

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
use flambeau_qwen3_moe::forward::{forward_full_attn_decode, FullAttnScratch};
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
    // Tiny shape to keep the smoke test cheap. Matches Qwen3.x structure:
    // GQA (n_heads != n_kv_heads), gated-full-attn (Q|gate fused), partial
    // RoPE. hidden % 32 must be 0 for Q8_1 quant.
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
        full_attention_interval: Some(1),
        gdn: Some(GdnDims {
            d_inner: 64,
            head_k_dim: 32,
            num_k_heads: 2,
            num_v_heads: 2,
            conv_kernel: 4,
        }),
        tied_lm_head: true,
    }
}

/// Build a Q8_0 weight tensor of shape `[n_rows, k]` on device with every
/// block zeroed. Enough to drive MMVQ without NaN (0 × anything = 0).
fn alloc_q8_0_weight(
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert!(k % 32 == 0, "Q8_0 requires k % 32 == 0");
    let n_blocks = n_rows * (k / 32);
    let bytes = n_blocks * std::mem::size_of::<BlockQ8_0>();
    let ptr = device.alloc(bytes)?;
    // Zero-fill via a small host staging buffer.
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
        // GGUF convention after `gguf.rs::dims.reverse()`: outermost-first.
        // For a 2D weight `[n_rows, k]` we store dims = [n_rows, k].
        dims: vec![n_rows as u64, k as u64],
        bytes,
        name: Arc::from(name),
    })
}

/// Build a host `[u8]` of F16 ones, allocate on device, upload, wrap as
/// `DeviceTensor`. Used for norm weights so the layer has a sane bias.
fn alloc_f16_ones(device: &HipDevice, name: &str, n_elems: usize) -> Result<DeviceTensor> {
    let host: Vec<f16> = vec![f16::from_f32(1.0); n_elems];
    let bytes = n_elems * 2;
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
        dims: vec![n_elems as u64],
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
fn forward_full_attn_decode_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_full_attn_decode_smoke");
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

    // Dummy weights. attn_q projects hidden → 2*n_heads*head_dim (fused
    // Q|gate); attn_k/v project hidden → n_kv_heads*head_dim; attn_output
    // projects n_heads*head_dim → hidden.
    let attn_norm = alloc_f16_ones(&device, "blk.0.attn_norm.weight", hidden)?;
    let attn_q_norm = alloc_f16_ones(&device, "blk.0.attn_q_norm.weight", head_dim)?;
    let attn_k_norm = alloc_f16_ones(&device, "blk.0.attn_k_norm.weight", head_dim)?;
    let attn_q =
        alloc_q8_0_weight(&device, "blk.0.attn_q.weight", 2 * n_heads * head_dim, hidden)?;
    let attn_k = alloc_q8_0_weight(&device, "blk.0.attn_k.weight", n_kv_heads * head_dim, hidden)?;
    let attn_v = alloc_q8_0_weight(&device, "blk.0.attn_v.weight", n_kv_heads * head_dim, hidden)?;
    let attn_output = alloc_q8_0_weight(
        &device,
        "blk.0.attn_output.weight",
        hidden,
        n_heads * head_dim,
    )?;
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
    let mut scratch = FullAttnScratch::new(&cfg, &device)?;

    // Input + output buffers. Feed a simple [0.01, 0.02, ...] pattern so
    // rmsnorm sees real non-zero variance.
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

    // One decode step at position 0.
    forward_full_attn_decode(
        &ops,
        stream,
        &device,
        &cfg,
        &alloc_f16_ones(&device, "blk.0.attn_norm.weight", hidden)?,
        None,
        &weights,
        &mut kv,
        &mut scratch,
        x_in,
        delta_out,
        0,
        None,
    )?;

    // Read back delta_out and confirm every lane is finite.
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
        "forward produced non-finite output at {} lanes: {:?}",
        bad.len(),
        &bad[..bad.len().min(4)]
    );
    // With zero Q8_0 weights the output should itself be zero.
    assert!(
        out_host.iter().all(|v| v.to_f32().abs() < 1e-3),
        "expected near-zero output with zero weights, got first lane = {}",
        out_host[0].to_f32()
    );

    // Teardown — explicit, matches the dispose pattern in V1.7.3-a.
    unsafe {
        device.dealloc(x_in, x_bytes)?;
        device.dealloc(delta_out, x_bytes)?;
    }
    scratch.dispose(&device)?;
    kv.dispose(&device).map_err(|e| anyhow::anyhow!("{e}"))?;
    // Drop all weight tensors.
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
