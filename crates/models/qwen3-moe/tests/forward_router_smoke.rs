//! V1.7.3-d3 smoke test — router (`forward_router_decode`) end-to-end into
//! `forward_moe_ffn_decode`. Replaces the hardcoded routing used by the
//! V1.7.3-d1 smoke with real top-k + softmax on logits produced from
//! F32×F16 dense GEMV.
//!
//! Invariants:
//! - Router weight is F32, shape `[n_experts, hidden]` (outermost-first).
//! - With zero expert matmul weights the routed contribution is still zero,
//!   so `out == residual` exactly — same assertion as V1.7.3-d1 but now
//!   driven by a real router pick rather than hand-chosen ids.
//! - Asserts `expert_ids` and `expert_weights` device buffers are populated
//!   with finite top-k values (read back to host and sanity-checked).

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, OpsRegistry};
use flambeau_quant::{BlockQ4K, GgmlDType, QK_K};
use flambeau_qwen3_moe::forward::{
    forward_moe_ffn_decode, forward_router_decode, MoeScratch,
};
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

fn tiny_router_cfg() -> Qwen3MoEConfig {
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
    }
}

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
        dims: vec![n_experts as u64, n_rows as u64, k as u64],
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
fn forward_router_then_moe_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_router_then_moe_smoke");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let cfg = tiny_router_cfg();
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;

    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let n_experts = cfg.num_experts;
    let top_k = cfg.num_experts_per_tok;

    // Router weight: non-trivial F32 pattern so every expert's logit is
    // different. Expert 2 gets the largest pattern so topk should pick it
    // first; expert 0 the smallest.
    let gate_inp_host: Vec<f32> = (0..n_experts)
        .flat_map(|e| (0..hidden).map(move |i| 0.01 * e as f32 * (i as f32 + 1.0)))
        .collect();
    let gate_inp = alloc_f32_2d(
        &device,
        "blk.0.ffn_gate_inp.weight",
        &gate_inp_host,
        n_experts as u64,
        hidden as u64,
    )?;
    let gate_exps = alloc_q4k_experts(
        &device,
        "blk.0.ffn_gate_exps.weight",
        n_experts,
        inter,
        hidden,
    )?;
    let up_exps =
        alloc_q4k_experts(&device, "blk.0.ffn_up_exps.weight", n_experts, inter, hidden)?;
    let down_exps =
        alloc_q4k_experts(&device, "blk.0.ffn_down_exps.weight", n_experts, hidden, inter)?;

    let ffn = FfnWeights {
        ffn_gate_inp: gate_inp.clone(),
        ffn_gate_exps: gate_exps.clone(),
        ffn_up_exps: up_exps.clone(),
        ffn_down_exps: down_exps.clone(),
        shared: None,
    };

    let mut scratch = MoeScratch::new(&cfg, &device)?;

    // Input — non-constant so router gets a real signal.
    let x_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32((i + 1) as f32 * 0.01))
        .collect();
    let residual_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32(-0.3 + (i as f32) * 0.001))
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

    // 1. Router.
    forward_router_decode(&ops, stream, &cfg, &ffn.ffn_gate_inp, &mut scratch, d_x)?;

    // Read back expert_ids + weights and sanity check.
    let mut ids_host = vec![0i32; top_k];
    let mut weights_host = vec![0.0f32; top_k];
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(ids_host.as_mut_ptr() as usize),
            scratch.expert_ids,
            top_k * 4,
        )?;
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(weights_host.as_mut_ptr() as usize),
            scratch.expert_weights,
            top_k * 4,
        )?;
    }
    stream.synchronize()?;
    eprintln!("router ids = {ids_host:?}");
    eprintln!("router weights = {weights_host:?}");

    for (i, &id) in ids_host.iter().enumerate() {
        assert!(
            (0..n_experts as i32).contains(&id),
            "slot {i} expert_id {id} out of [0, {n_experts}); router likely wrote garbage"
        );
    }
    // expert_ids are distinct (no duplicate expert in top-k).
    assert!(
        ids_host[0] != ids_host[1],
        "top-2 router returned duplicate expert {:?}",
        ids_host
    );
    // Weights are a softmax over the top-k, so they're in [0, 1] and sum to 1.
    let sum: f32 = weights_host.iter().sum();
    assert!(
        weights_host.iter().all(|w| w.is_finite() && (0.0..=1.0).contains(w)),
        "router weights out of [0,1]: {weights_host:?}"
    );
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "router weights should sum to 1.0, got {sum}"
    );

    // 2. MoE FFN — with zero expert weights this should return residual.
    forward_moe_ffn_decode(&ops, stream, &cfg, &ffn, &mut scratch, d_x, d_residual, d_out)?;

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

    for (i, (got, want)) in out_host.iter().zip(&residual_host).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "lane {i}: zero-expert MoE with router pick must equal residual; got {}, want {}",
            got.to_f32(),
            want.to_f32()
        );
    }

    unsafe {
        device.dealloc(d_x, bytes)?;
        device.dealloc(d_residual, bytes)?;
        device.dealloc(d_out, bytes)?;
    }
    scratch.dispose(&device)?;
    for t in [gate_inp, gate_exps, up_exps, down_exps] {
        free_tensor(&device, t)?;
    }
    Ok(())
}
