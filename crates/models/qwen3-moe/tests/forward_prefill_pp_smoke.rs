//! smoke test — pipeline-parallel prefill on a tiny synthetic
//! 2-layer hybrid model sharded across 2 HIP devices.
//! Same fixture as `forward_one_token_pp_smoke` but runs an L=4 prefill:
//! layer 0 (GDN) on rank 0, layer 1 (full-attn) on rank 1, embed on rank
//! 0, output head on rank 1. Zero Q4_K / Q8_0 matmul weights → every
//! delta zero → LM head matmul is zero × zero → argmax lane 0.
//! Skips when `device_count < 2`. Real-weight PP prefill parity is
//! covered by `forward_prefill_pp_real` and the wider chunked-prefill
//! test suite.

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async \
              over host/device buffers that live for the bounded synchronize that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, OpsRegistry};
use flambeau_quant::{BlockQ4K, BlockQ8_0, GgmlDType, QK_K};
use flambeau_qwen3_moe::forward::{forward_prefill_pp, ShardedForwardPrefillScratch};
use flambeau_qwen3_moe::weights::{
    AttnWeights, DeviceTensor, FfnWeights, FullAttnWeights, GdnWeights, LayerWeights,
    SharedExpertWeights,
};
use flambeau_qwen3_moe::{
    AttentionFamily, GdnDims, ModelLayout, Qwen3MoEConfig, Qwen3MoEShardedModel,
    Qwen3MoEShardedSession, RopeSpec,
};
use flambeau_runtime::{LayerAssignment, RankId};
use half::f16;
use std::sync::Arc;

fn cfg() -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        arch: "qwen35moe".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: QK_K,
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
        tied_lm_head: false,
            pooling_type: None,
    }
}

fn alloc_q4k_2d(dev: &HipDevice, name: &str, n_rows: usize, k: usize) -> Result<DeviceTensor> {
    assert_eq!(k % QK_K, 0);
    let bytes = n_rows * (k / QK_K) * std::mem::size_of::<BlockQ4K>();
    alloc_zero(dev, name, bytes, GgmlDType::Q4K, vec![n_rows as u64, k as u64])
}

fn alloc_q4k_3d(
    dev: &HipDevice,
    name: &str,
    n_experts: usize,
    n_rows: usize,
    k: usize,
) -> Result<DeviceTensor> {
    assert_eq!(k % QK_K, 0);
    let bytes = n_experts * n_rows * (k / QK_K) * std::mem::size_of::<BlockQ4K>();
    alloc_zero(
        dev,
        name,
        bytes,
        GgmlDType::Q4K,
        vec![n_experts as u64, n_rows as u64, k as u64],
    )
}

fn alloc_q8_0_2d(dev: &HipDevice, name: &str, n_rows: usize, k: usize) -> Result<DeviceTensor> {
    assert_eq!(k % 32, 0);
    let bytes = n_rows * (k / 32) * std::mem::size_of::<BlockQ8_0>();
    alloc_zero(dev, name, bytes, GgmlDType::Q8_0, vec![n_rows as u64, k as u64])
}

fn alloc_zero(
    dev: &HipDevice,
    name: &str,
    bytes: usize,
    dtype: GgmlDType,
    dims: Vec<u64>,
) -> Result<DeviceTensor> {
    dev.bind()?;
    let ptr = dev.alloc(bytes)?;
    let zeros = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(zeros.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    drop(zeros);
    Ok(DeviceTensor {
        ptr,
        dtype,
        dims,
        bytes,
        name: Arc::from(name),
    })
}

fn alloc_f16_ones(dev: &HipDevice, name: &str, n: usize) -> Result<DeviceTensor> {
    dev.bind()?;
    let host: Vec<f16> = vec![f16::from_f32(1.0); n];
    let bytes = n * 2;
    let ptr = dev.alloc(bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
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
    dev: &HipDevice,
    name: &str,
    vocab: usize,
    hidden: usize,
) -> Result<DeviceTensor> {
    dev.bind()?;
    let mut host = vec![f16::from_f32(0.0); vocab * hidden];
    for v in 0..vocab {
        for i in 0..hidden {
            host[v * hidden + i] = f16::from_f32(0.01 * v as f32 + 0.001 * i as f32);
        }
    }
    let bytes = host.len() * 2;
    let ptr = dev.alloc(bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    drop(host);
    Ok(DeviceTensor {
        ptr,
        dtype: GgmlDType::F16,
        dims: vec![vocab as u64, hidden as u64],
        bytes,
        name: Arc::from(name),
    })
}

fn alloc_f32(dev: &HipDevice, name: &str, host: &[f32]) -> Result<DeviceTensor> {
    dev.bind()?;
    let bytes = std::mem::size_of_val(host);
    let ptr = dev.alloc(bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(DeviceTensor {
        ptr,
        dtype: GgmlDType::F32,
        dims: vec![host.len() as u64],
        bytes,
        name: Arc::from(name),
    })
}

fn alloc_f32_2d(
    dev: &HipDevice,
    name: &str,
    host: &[f32],
    n_rows: u64,
    k: u64,
) -> Result<DeviceTensor> {
    dev.bind()?;
    let bytes = std::mem::size_of_val(host);
    let ptr = dev.alloc(bytes)?;
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )?;
    }
    dev.default_stream().synchronize()?;
    Ok(DeviceTensor {
        ptr,
        dtype: GgmlDType::F32,
        dims: vec![n_rows, k],
        bytes,
        name: Arc::from(name),
    })
}

fn build_gdn(dev: &HipDevice, cfg: &Qwen3MoEConfig, il: usize) -> Result<GdnWeights> {
    let hidden = cfg.hidden_size;
    let gdn = cfg.gdn.as_ref().unwrap();
    let conv_channels = gdn.conv_channels();
    Ok(GdnWeights {
        attn_qkv: alloc_q8_0_2d(dev, &format!("blk.{il}.attn_qkv.weight"), conv_channels, hidden)?,
        attn_gate: alloc_q8_0_2d(dev, &format!("blk.{il}.attn_gate.weight"), gdn.d_inner, hidden)?,
        ssm_alpha: Some(alloc_q8_0_2d(dev, &format!("blk.{il}.ssm_alpha.weight"), gdn.num_v_heads, hidden)?),
        ssm_beta: Some(alloc_q8_0_2d(dev, &format!("blk.{il}.ssm_beta.weight"), gdn.num_v_heads, hidden)?),
        ssm_ba: None,
        ssm_a: alloc_f32(dev, &format!("blk.{il}.ssm_a"),
            &(0..gdn.num_v_heads).map(|i| -(0.05 + 0.01 * i as f32)).collect::<Vec<_>>())?,
        ssm_dt_bias: alloc_f32(dev, &format!("blk.{il}.ssm_dt.bias"), &vec![0.0; gdn.num_v_heads])?,
        ssm_conv1d: alloc_f32_2d(dev, &format!("blk.{il}.ssm_conv1d.weight"),
            &vec![1.0 / gdn.conv_kernel as f32; conv_channels * gdn.conv_kernel],
            conv_channels as u64, gdn.conv_kernel as u64)?,
        ssm_norm: alloc_f32(dev, &format!("blk.{il}.ssm_norm.weight"), &vec![1.0f32; gdn.head_v_dim()])?,
        ssm_out: alloc_q8_0_2d(dev, &format!("blk.{il}.ssm_out.weight"), hidden, gdn.d_inner)?,
    })
}

fn build_full_attn(dev: &HipDevice, cfg: &Qwen3MoEConfig, il: usize) -> Result<FullAttnWeights> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.num_heads;
    let n_kv_heads = cfg.num_kv_heads;
    Ok(FullAttnWeights {
        attn_q: alloc_q8_0_2d(dev, &format!("blk.{il}.attn_q.weight"), 2 * n_heads * head_dim, hidden)?,
        attn_k: alloc_q8_0_2d(dev, &format!("blk.{il}.attn_k.weight"), n_kv_heads * head_dim, hidden)?,
        attn_v: alloc_q8_0_2d(dev, &format!("blk.{il}.attn_v.weight"), n_kv_heads * head_dim, hidden)?,
        attn_output: alloc_q8_0_2d(dev, &format!("blk.{il}.attn_output.weight"), hidden, n_heads * head_dim)?,
        attn_q_norm: alloc_f16_ones(dev, &format!("blk.{il}.attn_q_norm.weight"), head_dim)?,
        attn_k_norm: alloc_f16_ones(dev, &format!("blk.{il}.attn_k_norm.weight"), head_dim)?,
    })
}

fn build_ffn(dev: &HipDevice, cfg: &Qwen3MoEConfig, il: usize) -> Result<FfnWeights> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size.unwrap();
    let n_experts = cfg.num_experts;
    let gate_inp_host: Vec<f32> = (0..n_experts)
        .flat_map(|e| (0..hidden).map(move |i| 0.01 * e as f32 * (i as f32 + 1.0)))
        .collect();
    Ok(FfnWeights {
        ffn_gate_inp: Some(alloc_f32_2d(dev, &format!("blk.{il}.ffn_gate_inp.weight"),
            &gate_inp_host, n_experts as u64, hidden as u64)?),
        ffn_gate_exps: Some(alloc_q4k_3d(dev, &format!("blk.{il}.ffn_gate_exps.weight"), n_experts, inter, hidden)?),
        ffn_up_exps: Some(alloc_q4k_3d(dev, &format!("blk.{il}.ffn_up_exps.weight"), n_experts, inter, hidden)?),
        ffn_down_exps: Some(alloc_q4k_3d(dev, &format!("blk.{il}.ffn_down_exps.weight"), n_experts, hidden, inter)?),
        shared: Some(SharedExpertWeights {
            ffn_gate_inp_shexp: alloc_f32(dev, &format!("blk.{il}.ffn_gate_inp_shexp.weight"),
                &vec![0.05f32; hidden])?,
            ffn_gate_shexp: alloc_q4k_2d(dev, &format!("blk.{il}.ffn_gate_shexp.weight"), shared_inter, hidden)?,
            ffn_up_shexp: alloc_q4k_2d(dev, &format!("blk.{il}.ffn_up_shexp.weight"), shared_inter, hidden)?,
            ffn_down_shexp: alloc_q4k_2d(dev, &format!("blk.{il}.ffn_down_shexp.weight"), hidden, shared_inter)?,
        }),
        dense: None,
    })
}

fn build_layer(dev: &HipDevice, cfg: &Qwen3MoEConfig, il: usize) -> Result<LayerWeights> {
    let hidden = cfg.hidden_size;
    let attn = if cfg.is_recurrent(il) {
        AttnWeights::Gdn(build_gdn(dev, cfg, il)?)
    } else {
        AttnWeights::FullAttn(build_full_attn(dev, cfg, il)?)
    };
    Ok(LayerWeights {
        layer_idx: il,
        attn_norm: alloc_f16_ones(dev, &format!("blk.{il}.attn_norm.weight"), hidden)?,
        post_attention_norm: Some(alloc_f16_ones(dev, &format!("blk.{il}.post_attention_norm.weight"), hidden)?),
        ffn_norm: None,
        attn,
        ffn: build_ffn(dev, cfg, il)?,
    })
}

#[test]
fn forward_prefill_pp_synthetic_2rank_l4() -> Result<()> {
    run_prefill_pp_smoke(/*scratch_max_tokens=*/ 4)
}

/// chunking smoke: scratch sized below L forces
/// `forward_prefill_pp` to internally loop over ubatches of
/// `max_tokens` tokens. Same fixture, same expected argmax.
#[test]
fn forward_prefill_pp_synthetic_2rank_l4_chunked() -> Result<()> {
    run_prefill_pp_smoke(/*scratch_max_tokens=*/ 2)
}

fn run_prefill_pp_smoke(scratch_max_tokens: usize) -> Result<()> {
    let n = device_count().unwrap_or(0);
    if n < 2 {
        eprintln!("need ≥ 2 HIP devices for PP prefill smoke — got {n}, skipping");
        return Ok(());
    }

    let cfg = cfg();
    let cluster = HipCluster::new(&[0, 1])?;
    let assignment = LayerAssignment::contiguous(cfg.num_layers, cluster.ranks() as u32);
    assert_eq!(assignment.rank_for(0), RankId(0));
    assert_eq!(assignment.rank_for(1), RankId(1));

    let layout = ModelLayout {
        token_embd: flambeau_qwen3_moe::layout::ResolvedTensor {
            name: "token_embd.weight".to_string(),
            dtype: GgmlDType::F16,
            dims: vec![cfg.vocab_size as u64, cfg.hidden_size as u64],
            size_bytes: (cfg.vocab_size * cfg.hidden_size * 2) as u64,
        },
        output_norm: flambeau_qwen3_moe::layout::ResolvedTensor {
            name: "output_norm.weight".to_string(),
            dtype: GgmlDType::F16,
            dims: vec![cfg.hidden_size as u64],
            size_bytes: (cfg.hidden_size * 2) as u64,
        },
        output: None,
        layers: Vec::new(),
    };

    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let dev0 = cluster.device(0);
    let dev1 = cluster.device(1);

    let token_embd = alloc_f16_pattern_2d(dev0, "token_embd.weight", vocab, hidden)?;
    let output_norm = alloc_f16_ones(dev1, "output_norm.weight", hidden)?;
    let output = alloc_q8_0_2d(dev1, "output.weight", vocab, hidden)?;

    let layer0 = build_layer(dev0, &cfg, 0)?;
    let layer1 = build_layer(dev1, &cfg, 1)?;

    let ops0 = OpsRegistry::new(dev0).map_err(|e| anyhow::anyhow!("OpsRegistry rank 0: {e}"))?;
    let ops1 = OpsRegistry::new(dev1).map_err(|e| anyhow::anyhow!("OpsRegistry rank 1: {e}"))?;

    let shard0 = Qwen3MoEShardedModel::new_shard(
        RankId(0), dev0.id(), ops0,
        Some(token_embd), None, None, vec![layer0],
    );
    let shard1 = Qwen3MoEShardedModel::new_shard(
        RankId(1), dev1.id(), ops1,
        None, Some(output_norm), Some(output), vec![layer1],
    );

    let model = Qwen3MoEShardedModel::from_parts(cfg.clone(), layout, assignment, vec![shard0, shard1]);
    let mut session = Qwen3MoEShardedSession::new(&model, &cluster, flambeau_qwen3_moe::session::KvLayout::F16)?;
    // L=4 prefill; scratch may be sized smaller (=> internal chunking).
    let mut scratch =
        ShardedForwardPrefillScratch::new(&model, &cluster, scratch_max_tokens)?;

    let tokens = vec![3u32, 5, 7, 11]; // any in-vocab ids
    let next = forward_prefill_pp(
        &model,
        &mut session,
        &cluster,
        &mut scratch,
        &tokens,
        /*start_position=*/ 0,
    )?;

    // Zero Q4_K / Q8_0 matmul weights → zero logits → argmax lane 0.
    assert_eq!(next, 0, "expected argmax(zero logits) == 0, got {next}");

    scratch.dispose(&cluster)?;
    session.dispose(&cluster)?;
    model.dispose(&cluster)?;
    cluster.dispose()?;
    Ok(())
}
