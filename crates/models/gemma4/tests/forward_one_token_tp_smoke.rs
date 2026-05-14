//! S9-A — tensor-parallel decode smoke. Synthetic 2-rank fixture,
//! 2 layers, all full-attn dense FFN, sharded Q/K/V/FFN. Verifies:
//! 1. `Gemma4TpStage::validate_shardable` accepts a model that divides
//!    cleanly across n_ranks=2 and rejects one that doesn't.
//! 2. `Gemma4TpDriver` builds, embeds replicated, runs the per-layer
//!    TP composition with AR-sum across ranks after attn output proj
//!    and FFN down, and produces a finite argmax on the head rank.
//! 3. Multi-step decode grows per-rank head-shard KV caches.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use std::sync::Arc;

use flambeau_backend_hip::{device_count, HipCluster, HipDevice};
use flambeau_blocks::WeightHandle;
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_gemma4::{
    DeviceTensor, Gemma4Config, Gemma4LayerWeights, Gemma4TpDriver, Gemma4TpStage,
    Gemma4Variant, ModelLayout,
};
use flambeau_quant::GgmlDType;
use half::f16;

const HIDDEN: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FF_LEN: usize = 128;
const VOCAB: usize = 64;
const N_LAYERS: usize = 2;
const N_RANKS: usize = 2;

fn cluster_or_skip() -> Option<HipCluster> {
    let n = device_count().ok()?;
    if (n as usize) < N_RANKS {
        eprintln!(
            "skipping TP smoke — need {N_RANKS} HIP devices but only {n} available"
        );
        return None;
    }
    let ids: Vec<i32> = (0..N_RANKS as i32).collect();
    let c = HipCluster::new(&ids).ok()?;
    // BarP2pAllReduce requires fully-connected peer access; skip when not.
    if !c.peer_access_full() {
        eprintln!("skipping TP smoke — cluster peer access not full");
        return None;
    }
    Some(c)
}

fn upload<T: Copy>(dev: &HipDevice, data: &[T]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn alloc_zeroed(dev: &HipDevice, bytes: usize) -> DevicePtr {
    let d = dev.alloc(bytes).unwrap();
    let z = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(z.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn alloc_q8_0_zero(dev: &HipDevice, n_rows: usize, k: usize) -> DevicePtr {
    assert_eq!(k % 32, 0);
    alloc_zeroed(dev, n_rows * (k / 32) * 34)
}

fn alloc_f16_ones(dev: &HipDevice, n: usize) -> DevicePtr {
    upload(dev, &vec![f16::from_f32(1.0); n])
}

fn make_q8_0(dev: &HipDevice, n_rows: usize, k: usize) -> (WeightHandle, DeviceTensor) {
    let bytes = n_rows * (k / 32) * 34;
    let ptr = alloc_q8_0_zero(dev, n_rows, k);
    let wh = WeightHandle { ptr, dtype: QDtype::Q8_0, dims: [n_rows, k] };
    let dt = DeviceTensor { ptr, dtype: GgmlDType::Q8_0, bytes };
    (wh, dt)
}

fn make_f16_ones(dev: &HipDevice, n: usize) -> (DevicePtr, DeviceTensor) {
    let bytes = n * 2;
    let ptr = alloc_f16_ones(dev, n);
    (ptr, DeviceTensor { ptr, dtype: GgmlDType::F16, bytes })
}

/// Build a sharded layer for one rank. `q_rows_local = q_rows_global /
/// N_RANKS`; same shard rule for K/V/gate/up. Output and down are
/// row-parallel: weight cols = (q or ff) local; weight rows = hidden.
fn make_layer_shard(dev: &HipDevice) -> Gemma4LayerWeights {
    let q_rows_local = (N_HEADS / N_RANKS) * HEAD_DIM;
    let kv_rows_local = (N_KV_HEADS / N_RANKS) * HEAD_DIM;
    let ff_local = FF_LEN / N_RANKS;

    let (attn_q, _) = make_q8_0(dev, q_rows_local, HIDDEN);
    let (attn_k, _) = make_q8_0(dev, kv_rows_local, HIDDEN);
    let (attn_v, _) = make_q8_0(dev, kv_rows_local, HIDDEN);
    let (attn_output, _) = make_q8_0(dev, HIDDEN, q_rows_local);
    let (ffn_gate, _) = make_q8_0(dev, ff_local, HIDDEN);
    let (ffn_up, _) = make_q8_0(dev, ff_local, HIDDEN);
    let (ffn_down, _) = make_q8_0(dev, HIDDEN, ff_local);
    let (attn_norm, _) = make_f16_ones(dev, HIDDEN);
    let (attn_q_norm, _) = make_f16_ones(dev, HEAD_DIM);
    let (attn_k_norm, _) = make_f16_ones(dev, HEAD_DIM);
    let (post_attention_norm, _) = make_f16_ones(dev, HIDDEN);
    let (ffn_norm, _) = make_f16_ones(dev, HIDDEN);
    let (post_ffw_norm, _) = make_f16_ones(dev, HIDDEN);

    Gemma4LayerWeights {
        attn_norm,
        attn_q,
        attn_k: Some(attn_k),
        attn_v: Some(attn_v),
        attn_output,
        attn_q_norm,
        attn_k_norm: Some(attn_k_norm),
        post_attention_norm,
        layer_output_scale: None,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm,
        per_layer_embed: None,
        moe: None,
    }
}

fn synthetic_cfg() -> Gemma4Config {
    Gemma4Config {
        arch: "gemma4".into(),
        variant: Gemma4Variant::E4B,
        hidden_size: HIDDEN,
        vocab_size: VOCAB,
        num_layers: N_LAYERS,
        num_heads: N_HEADS,
        num_kv_heads: vec![N_KV_HEADS; N_LAYERS],
        head_dim: HEAD_DIM,
        head_dim_swa: HEAD_DIM,
        context_length: 16,
        rms_norm_eps: 1e-6,
        feed_forward_length: FF_LEN,
        rope_freq_base: 10_000.0,
        rope_freq_base_swa: 10_000.0,
        rope_dim: HEAD_DIM,
        rope_dim_swa: HEAD_DIM,
        swa_layers: vec![false; N_LAYERS],
        sliding_window: 0,
        shared_kv_layers: 0,
        moe: None,
        per_layer_embed: None,
        final_logit_softcap: 30.0,
        tied_lm_head: true,
    }
}

fn build_synthetic_tp_driver(cluster: HipCluster) -> Gemma4TpDriver {
    let cfg = synthetic_cfg();
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let cluster_arc = Arc::new(cluster);
    let head_rank = 0;
    let mut stages = Vec::with_capacity(N_RANKS);
    for rank in 0..N_RANKS {
        let dev = cluster_arc.device(rank);
        dev.bind().unwrap();
        let layer_weights: Vec<Gemma4LayerWeights> =
            (0..N_LAYERS).map(|_| make_layer_shard(dev)).collect();
        // Replicate token_embd / output_norm / lm_head per rank.
        let (_, tok_embd_dt) = make_q8_0(dev, VOCAB, HIDDEN);
        let (_, output_norm_dt) = make_f16_ones(dev, HIDDEN);
        let (_, lm_head_dt) = make_q8_0(dev, VOCAB, HIDDEN);
        let is_head_rank = rank == head_rank;
        let stage = Gemma4TpStage::from_pieces(
            dev,
            rank,
            &cfg,
            &layout,
            N_RANKS,
            layer_weights,
            tok_embd_dt,
            [VOCAB, HIDDEN],
            output_norm_dt,
            Some(lm_head_dt),
            is_head_rank,
            /*max_tokens=*/ 16,
            flambeau_blocks::RawAllocTracker::new(),
        )
        .expect("stage");
        stages.push(stage);
    }
    Gemma4TpDriver::from_pieces(cluster_arc, cfg, layout, stages, head_rank).expect("driver")
}

#[test]
fn validate_shardable_rejects_unshardable() {
    let mut cfg = synthetic_cfg();
    // Make n_heads not divisible by n_ranks=2.
    cfg.num_heads = 3;
    let err = Gemma4TpStage::validate_shardable(&cfg, 2).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("n_heads"), "unexpected: {msg}");
}

#[test]
fn forward_one_token_tp_smoke() {
    let Some(cluster) = cluster_or_skip() else { return; };
    let mut driver = build_synthetic_tp_driver(cluster);

    let mut tok = 0u32;
    for pos in 0..4 {
        tok = driver.forward_one_token(tok, pos).expect("forward");
        assert!((tok as usize) < driver.cfg.vocab_size, "argmax oob: {tok}");
    }
    // Per-rank KV caches grew to 4 tokens each.
    for r in 0..N_RANKS {
        let kv = driver.stages[r].kv_caches[0]
            .as_ref()
            .expect("layer 0 has KV");
        assert_eq!(kv.current_tokens(), 4, "rank {r} layer 0 kv count");
    }

    driver.dispose().expect("dispose");
}
