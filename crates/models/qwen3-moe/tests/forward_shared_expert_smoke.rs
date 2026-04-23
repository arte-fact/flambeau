//! V1.7.3-d2 smoke test — shared expert decode step on real HIP with zero
//! Q4_K weights. Asserts the 8-op chain (quant → 2× dense gate/up mmvq →
//! swiglu_f32 → cast → quant → down mmvq → cast x_norm → shared_expert_scale →
//! cast) produces a finite, all-zero F16 output (zero matmul outputs stay
//! zero under sigmoid(0)·0 = 0).

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
use flambeau_qwen3_moe::forward::{forward_shared_expert_decode, SharedExpertScratch};
use flambeau_qwen3_moe::weights::{DeviceTensor, SharedExpertWeights};
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

fn tiny_shexp_cfg() -> Qwen3MoEConfig {
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
        shared_expert_intermediate_size: Some(QK_K),
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

fn alloc_q4k_2d(
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert_eq!(k % QK_K, 0, "Q4_K requires k % {QK_K} == 0");
    let n_blocks = n_rows * (k / QK_K);
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
        // GGUF outermost-first: [n_rows, k].
        dims: vec![n_rows as u64, k as u64],
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
fn forward_shared_expert_decode_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_shared_expert_decode_smoke");
        return Ok(());
    };
    device.bind()?;
    let stream = device.default_stream();

    let cfg = tiny_shexp_cfg();
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;

    let hidden = cfg.hidden_size;
    let inter = cfg.shared_expert_intermediate_size.unwrap();

    // All projections zero → dense FFN output is zero → scaled by any
    // sigmoid gate, still zero → F16 delta is zero on every lane.
    let gate_w = alloc_q4k_2d(&device, "blk.0.ffn_gate_shexp.weight", inter, hidden)?;
    let up_w = alloc_q4k_2d(&device, "blk.0.ffn_up_shexp.weight", inter, hidden)?;
    let down_w = alloc_q4k_2d(&device, "blk.0.ffn_down_shexp.weight", hidden, inter)?;
    // ffn_gate_inp_shexp: pattern of 0.1s so the gate-scale kernel's
    // dot-product is a real (non-zero) operation — still produces 0.5 via
    // sigmoid and 0.5 × 0 = 0, but the scale kernel runs end-to-end.
    let gate_inp_host: Vec<f32> = vec![0.1; hidden];
    let gate_inp =
        alloc_f32_vec(&device, "blk.0.ffn_gate_inp_shexp.weight", &gate_inp_host)?;

    let shared = SharedExpertWeights {
        ffn_gate_inp_shexp: gate_inp.clone(),
        ffn_gate_shexp: gate_w.clone(),
        ffn_up_shexp: up_w.clone(),
        ffn_down_shexp: down_w.clone(),
    };

    let mut scratch = SharedExpertScratch::new(&cfg, &device)?;

    // Non-trivial x_norm so the gate-scale kernel's dot product is non-zero
    // (exercises the sigmoid SFU path).
    let x_host: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32(0.01 * (i + 1) as f32))
        .collect();
    let bytes = hidden * 2;
    let d_x = device.alloc(bytes)?;
    let d_out = device.alloc(bytes)?;
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            d_x,
            DevicePtr(x_host.as_ptr() as usize),
            bytes,
        )?;
    }
    stream.synchronize()?;

    forward_shared_expert_decode(
        &ops, stream, &cfg, &shared, &mut scratch, d_x, d_out,
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

    for (i, v) in out_host.iter().enumerate() {
        let f = v.to_f32();
        assert!(
            f.is_finite(),
            "lane {i} produced non-finite output: {f}"
        );
        assert!(
            f.abs() < 1e-3,
            "zero-weight shared expert must produce zero delta; lane {i} = {f}"
        );
    }

    unsafe {
        device.dealloc(d_x, bytes)?;
        device.dealloc(d_out, bytes)?;
    }
    scratch.dispose(&device)?;
    for t in [gate_w, up_w, down_w, gate_inp] {
        free_tensor(&device, t)?;
    }
    Ok(())
}
