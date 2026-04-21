//! V1.7.3-f4 capstone smoke test — `forward_prefill` end-to-end for an L=4
//! prompt through a 2-layer synthetic hybrid model (layer 0 = GDN, layer 1
//! = full-attn). Exercises every prefill path (embed → GDN prefill →
//! full-attn prefill → MoE prefill + shared prefill + router prefill →
//! output head → argmax). Zero Q4_K / Q8_0 matmul weights → zero logits
//! → argmax picks token 0.

#![cfg(feature = "hip")]

use anyhow::Result;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, OpsRegistry};
use flambeau_quant::{BlockQ4K, BlockQ8_0, GgmlDType, QK_K};
use flambeau_qwen3_moe::forward::{forward_prefill, ForwardPrefillScratch};
use flambeau_qwen3_moe::weights::{
    AttnWeights, DeviceTensor, FfnWeights, FullAttnWeights, GdnWeights, LayerWeights,
    ModelWeights, SharedExpertWeights,
};
use flambeau_qwen3_moe::{AttentionFamily, GdnDims, Qwen3MoEConfig, Qwen3MoESession, RopeSpec};
use half::f16;
use std::sync::Arc;

fn hip_device() -> Option<HipDevice> {
    let n = flambeau_backend_hip::device_count().ok()?;
    if n < 1 {
        return None;
    }
    HipDevice::new(0).ok()
}

fn f4_cfg() -> Qwen3MoEConfig {
    // Same shape as the V1.7.3-e5 end-to-end decode capstone — ensures the
    // prefill path is exercised against the same fixture shape.
    Qwen3MoEConfig {
        arch: "qwen35moe".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: QK_K, // 256
        vocab_size: 32,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 128,
        context_length: 8,
        rms_norm_eps: 1e-6,
        rope: RopeSpec {
            freq_base: 10_000.0,
            rotated_dims: 64,
            sections: None,
        },
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: QK_K,
        shared_expert_intermediate_size: Some(QK_K),
        full_attention_interval: Some(2),
        gdn: Some(GdnDims {
            d_inner: 256,
            head_k_dim: 128,
            num_k_heads: 1,
            num_v_heads: 2,
            conv_kernel: 4,
        }),
        tied_lm_head: true,
    }
}

// Cookie-cutter alloc helpers — same as V1.7.3-e5 smoke, duplicated to
// keep the test file self-contained.

fn alloc_q4k_2d(device: &HipDevice, name: &str, n_rows: usize, k: usize) -> Result<DeviceTensor> {
    assert_eq!(k % QK_K, 0);
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
        dims: vec![n_rows as u64, k as u64],
        bytes,
        name: Arc::from(name),
    })
}

fn alloc_q4k_3d(
    device: &HipDevice,
    name: &str,
    n_experts: usize,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert_eq!(k % QK_K, 0);
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

fn alloc_q8_0_2d(device: &HipDevice, name: &str, n_rows: usize, k: usize) -> Result<DeviceTensor> {
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

fn alloc_f16_pattern_2d(
    device: &HipDevice,
    name: &str,
    vocab: usize,
    hidden: usize,
) -> Result<DeviceTensor> {
    let mut host = vec![f16::from_f32(0.0); vocab * hidden];
    for v in 0..vocab {
        for i in 0..hidden {
            host[v * hidden + i] = f16::from_f32(0.01 * v as f32 + 0.001 * i as f32);
        }
    }
    let bytes = host.len() * 2;
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
        dims: vec![vocab as u64, hidden as u64],
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

fn build_gdn_weights(device: &HipDevice, cfg: &Qwen3MoEConfig, il: usize) -> Result<GdnWeights> {
    let hidden = cfg.hidden_size;
    let gdn = cfg.gdn.as_ref().unwrap();
    let conv_channels = gdn.conv_channels();
    Ok(GdnWeights {
        attn_qkv: alloc_q8_0_2d(device, &format!("blk.{il}.attn_qkv.weight"), conv_channels, hidden)?,
        attn_gate: alloc_q8_0_2d(device, &format!("blk.{il}.attn_gate.weight"), gdn.d_inner, hidden)?,
        ssm_alpha: Some(alloc_q8_0_2d(device, &format!("blk.{il}.ssm_alpha.weight"), gdn.num_v_heads, hidden)?),
        ssm_beta: Some(alloc_q8_0_2d(device, &format!("blk.{il}.ssm_beta.weight"), gdn.num_v_heads, hidden)?),
        ssm_ba: None,
        ssm_a: alloc_f32_vec(device, &format!("blk.{il}.ssm_a"),
            &(0..gdn.num_v_heads).map(|i| -(0.05 + 0.01 * i as f32)).collect::<Vec<_>>())?,
        ssm_dt_bias: alloc_f32_vec(device, &format!("blk.{il}.ssm_dt.bias"), &vec![0.0; gdn.num_v_heads])?,
        ssm_conv1d: alloc_f32_2d(device, &format!("blk.{il}.ssm_conv1d.weight"),
            &vec![1.0 / gdn.conv_kernel as f32; conv_channels * gdn.conv_kernel],
            conv_channels as u64, gdn.conv_kernel as u64)?,
        ssm_norm: alloc_f32_vec(device, &format!("blk.{il}.ssm_norm.weight"), &vec![1.0f32; gdn.head_v_dim()])?,
        ssm_out: alloc_q8_0_2d(device, &format!("blk.{il}.ssm_out.weight"), hidden, gdn.d_inner)?,
    })
}

fn build_full_attn_weights(
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    il: usize,
) -> Result<FullAttnWeights> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    Ok(FullAttnWeights {
        attn_q: alloc_q8_0_2d(device, &format!("blk.{il}.attn_q.weight"), 2 * n_heads * head_dim, hidden)?,
        attn_k: alloc_q8_0_2d(device, &format!("blk.{il}.attn_k.weight"), n_kv_heads * head_dim, hidden)?,
        attn_v: alloc_q8_0_2d(device, &format!("blk.{il}.attn_v.weight"), n_kv_heads * head_dim, hidden)?,
        attn_output: alloc_q8_0_2d(device, &format!("blk.{il}.attn_output.weight"), hidden, n_heads * head_dim)?,
        attn_q_norm: alloc_f16_ones(device, &format!("blk.{il}.attn_q_norm.weight"), head_dim)?,
        attn_k_norm: alloc_f16_ones(device, &format!("blk.{il}.attn_k_norm.weight"), head_dim)?,
    })
}

fn build_ffn_weights(device: &HipDevice, cfg: &Qwen3MoEConfig, il: usize) -> Result<FfnWeights> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size.unwrap();
    let n_experts = cfg.num_experts;

    let gate_inp_host: Vec<f32> = (0..n_experts)
        .flat_map(|e| (0..hidden).map(move |i| 0.01 * e as f32 * (i as f32 + 1.0)))
        .collect();
    Ok(FfnWeights {
        ffn_gate_inp: alloc_f32_2d(device, &format!("blk.{il}.ffn_gate_inp.weight"),
            &gate_inp_host, n_experts as u64, hidden as u64)?,
        ffn_gate_exps: alloc_q4k_3d(device, &format!("blk.{il}.ffn_gate_exps.weight"), n_experts, inter, hidden)?,
        ffn_up_exps: alloc_q4k_3d(device, &format!("blk.{il}.ffn_up_exps.weight"), n_experts, inter, hidden)?,
        ffn_down_exps: alloc_q4k_3d(device, &format!("blk.{il}.ffn_down_exps.weight"), n_experts, hidden, inter)?,
        shared: Some(SharedExpertWeights {
            ffn_gate_inp_shexp: alloc_f32_vec(device, &format!("blk.{il}.ffn_gate_inp_shexp.weight"),
                &vec![0.05f32; hidden])?,
            ffn_gate_shexp: alloc_q4k_2d(device, &format!("blk.{il}.ffn_gate_shexp.weight"), shared_inter, hidden)?,
            ffn_up_shexp: alloc_q4k_2d(device, &format!("blk.{il}.ffn_up_shexp.weight"), shared_inter, hidden)?,
            ffn_down_shexp: alloc_q4k_2d(device, &format!("blk.{il}.ffn_down_shexp.weight"), hidden, shared_inter)?,
        }),
    })
}

fn build_synthetic_weights(device: &HipDevice, cfg: &Qwen3MoEConfig) -> Result<ModelWeights> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let token_embd = alloc_f16_pattern_2d(device, "token_embd.weight", vocab, hidden)?;
    let output_norm = alloc_f16_ones(device, "output_norm.weight", hidden)?;
    let output = alloc_q8_0_2d(device, "output.weight", vocab, hidden)?;

    let mut layers = Vec::new();
    for il in 0..cfg.num_layers {
        let attn_norm = alloc_f16_ones(device, &format!("blk.{il}.attn_norm.weight"), hidden)?;
        let post_attention_norm = Some(alloc_f16_ones(
            device,
            &format!("blk.{il}.post_attention_norm.weight"),
            hidden,
        )?);
        let attn = if cfg.is_recurrent(il) {
            AttnWeights::Gdn(build_gdn_weights(device, cfg, il)?)
        } else {
            AttnWeights::FullAttn(build_full_attn_weights(device, cfg, il)?)
        };
        let ffn = build_ffn_weights(device, cfg, il)?;
        layers.push(LayerWeights {
            layer_idx: il,
            attn_norm,
            post_attention_norm,
            ffn_norm: None,
            attn,
            ffn,
        });
    }

    Ok(ModelWeights::from_parts(
        device.id(),
        token_embd,
        output_norm,
        Some(output),
        layers,
    ))
}

#[test]
fn forward_prefill_l4_end_to_end_smoke() -> Result<()> {
    let Some(device) = hip_device() else {
        eprintln!("no HIP device — skipping forward_prefill_l4_end_to_end_smoke");
        return Ok(());
    };
    device.bind()?;

    let cfg = f4_cfg();
    let ops = OpsRegistry::new(&device)
        .map_err(|e| anyhow::anyhow!("OpsRegistry: {e}"))?;
    let weights = build_synthetic_weights(&device, &cfg)?;
    let mut session = Qwen3MoESession::new(&cfg, &device)?;
    let mut scratch = ForwardPrefillScratch::new(&cfg, &device, 4)?;

    let tokens = [3u32, 7, 1, 15];
    let next = forward_prefill(
        &ops,
        device.default_stream(),
        &device,
        &cfg,
        &weights,
        &mut session,
        &mut scratch,
        &tokens,
        0,
    )?;

    // Zero Q4_K / Q8_0 matmul weights → every per-layer delta is zero →
    // residual chain carries embeddings unchanged → LM head mmvq with zero
    // weights → zero logits → argmax lane 0.
    assert_eq!(
        next, 0,
        "expected argmax(zero logits) == 0 for L=4 prefill; got {next}"
    );

    scratch.dispose(&device)?;
    session.dispose(&device)?;
    weights.dispose(&device)?;
    Ok(())
}
