//! S10-A — pp2tp2 hybrid decode smoke. Synthetic 4-rank fixture
//! (2 stages × 2 ranks), 4 layers (2 per stage, sharded across the
//! stage's 2 ranks). Verifies:
//! 1. Sub-clusters built BEFORE the global cluster (MEMORY.md
//!    `hybrid_cluster_order`).
//! 2. `Gemma4HybridDriver` partitions layers across stages and runs
//!    the per-layer TP composition + AR within each stage.
//! 3. Inter-stage handoff carries the residual via the global
//!    cluster's `peer_copy_via_host`.
//! 4. Output head runs on `(head_stage, head_rank)`; argmax in range.
//!
//! Skipped when fewer than 4 HIP devices or peer access isn't full.

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
    partition_layers_pp, DeviceTensor, Gemma4Config, Gemma4HybridDriver, Gemma4HybridStage,
    Gemma4LayerWeights, Gemma4Variant, HybridRankState, ModelLayout,
};
use flambeau_quant::GgmlDType;
use half::f16;

const HIDDEN: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FF_LEN: usize = 128;
const VOCAB: usize = 64;
const N_LAYERS: usize = 4;
const PP_SIZE: usize = 2;
const TP_SIZE: usize = 2;
const TOTAL_RANKS: usize = PP_SIZE * TP_SIZE;

/// Build sub-clusters BEFORE the global cluster (MEMORY.md
/// `hybrid_cluster_order`). Returns `(sub_clusters[pp], global)` or
/// `None` when fewer than 4 devices / peer access incomplete.
///
/// Mesh: `hip:0,2,1,3` per MEMORY.md `never_tp4_use_pp2tp2`.
/// Stage 0 = hip:0,2 (global ranks 0,1).
/// Stage 1 = hip:1,3 (global ranks 2,3).
fn clusters_or_skip() -> Option<(Vec<Arc<HipCluster>>, Arc<HipCluster>)> {
    let n = device_count().ok()?;
    if (n as usize) < TOTAL_RANKS {
        eprintln!(
            "skipping hybrid smoke — need {TOTAL_RANKS} HIP devices but only {n}"
        );
        return None;
    }
    // Synthetic uses first 4 devices straight through; real-rig
    // production code uses `hip:0,2,1,3` to avoid the {2,3} link
    // fault. The synthetic test only requires peer access to be full,
    // not the avoidance pattern.
    let mesh = [0i32, 1, 2, 3];
    let stage_0_ids = [mesh[0], mesh[1]];
    let stage_1_ids = [mesh[2], mesh[3]];

    let sub0 = Arc::new(HipCluster::new(&stage_0_ids).ok()?);
    let sub1 = Arc::new(HipCluster::new(&stage_1_ids).ok()?);
    if !sub0.peer_access_full() || !sub1.peer_access_full() {
        eprintln!("skipping hybrid smoke — sub-cluster peer access incomplete");
        return None;
    }
    let global = Arc::new(HipCluster::new(&mesh).ok()?);
    if !global.peer_access_full() {
        eprintln!("skipping hybrid smoke — global cluster peer access incomplete");
        return None;
    }
    Some((vec![sub0, sub1], global))
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

fn mq(dev: &HipDevice, n_rows: usize, k: usize) -> (WeightHandle, DeviceTensor) {
    let bytes = n_rows * (k / 32) * 34;
    let ptr = alloc_q8_0_zero(dev, n_rows, k);
    (
        WeightHandle { ptr, dtype: QDtype::Q8_0, dims: [n_rows, k] },
        DeviceTensor { ptr, dtype: GgmlDType::Q8_0, bytes },
    )
}

fn mfo(dev: &HipDevice, n: usize) -> (DevicePtr, DeviceTensor) {
    let bytes = n * 2;
    let ptr = alloc_f16_ones(dev, n);
    (ptr, DeviceTensor { ptr, dtype: GgmlDType::F16, bytes })
}

fn make_layer_shard(dev: &HipDevice) -> Gemma4LayerWeights {
    let q_local = (N_HEADS / TP_SIZE) * HEAD_DIM;
    let kv_local = (N_KV_HEADS / TP_SIZE) * HEAD_DIM;
    let ff_local = FF_LEN / TP_SIZE;
    let (attn_q, _) = mq(dev, q_local, HIDDEN);
    let (attn_k, _) = mq(dev, kv_local, HIDDEN);
    let (attn_v, _) = mq(dev, kv_local, HIDDEN);
    let (attn_output, _) = mq(dev, HIDDEN, q_local);
    let (ffn_gate, _) = mq(dev, ff_local, HIDDEN);
    let (ffn_up, _) = mq(dev, ff_local, HIDDEN);
    let (ffn_down, _) = mq(dev, HIDDEN, ff_local);
    let (attn_norm, _) = mfo(dev, HIDDEN);
    let (attn_q_norm, _) = mfo(dev, HEAD_DIM);
    let (attn_k_norm, _) = mfo(dev, HEAD_DIM);
    let (post_attention_norm, _) = mfo(dev, HIDDEN);
    let (ffn_norm, _) = mfo(dev, HIDDEN);
    let (post_ffw_norm, _) = mfo(dev, HIDDEN);
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

fn build_driver(sub_clusters: Vec<Arc<HipCluster>>, global: Arc<HipCluster>) -> Gemma4HybridDriver {
    let cfg = synthetic_cfg();
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    let layer_to_stage = partition_layers_pp(PP_SIZE, &layout).expect("partition_pp");
    assert_eq!(layer_to_stage, vec![0, 0, 1, 1]);

    let head_stage_idx = PP_SIZE - 1;
    let head_rank_in_head_stage_idx = 0;

    let mut stages = Vec::with_capacity(PP_SIZE);
    for stage_idx in 0..PP_SIZE {
        let layers_global: Vec<usize> = (0..N_LAYERS)
            .filter(|&i| layer_to_stage[i] == stage_idx)
            .collect();
        let sub = sub_clusters[stage_idx].clone();

        let mut rank_state = Vec::with_capacity(TP_SIZE);
        for r in 0..TP_SIZE {
            let dev = sub.device(r);
            dev.bind().unwrap();
            let layer_weights: Vec<Gemma4LayerWeights> =
                layers_global.iter().map(|_| make_layer_shard(dev)).collect();
            // token_embd: only on stage 0 ranks (replicated within
            // stage so every rank can embed independently).
            let (tok_embd, tok_dims) = if stage_idx == 0 {
                let (_, dt) = mq(dev, VOCAB, HIDDEN);
                (Some(dt), Some([VOCAB, HIDDEN]))
            } else {
                (None, None)
            };
            // output_norm + lm_head: only on head stage.
            let (out_norm, lm_head, lm_dims) = if stage_idx == head_stage_idx {
                let (_, on) = mfo(dev, HIDDEN);
                let (_, lm) = mq(dev, VOCAB, HIDDEN);
                (Some(on), Some(lm), Some([VOCAB, HIDDEN]))
            } else {
                (None, None, None)
            };
            let is_head_rank =
                stage_idx == head_stage_idx && r == head_rank_in_head_stage_idx;
            let rs = HybridRankState::from_pieces(
                dev,
                r,
                &cfg,
                &layout,
                &layers_global,
                TP_SIZE,
                layer_weights,
                tok_embd,
                tok_dims,
                out_norm,
                lm_head,
                lm_dims,
                is_head_rank,
                /*max_tokens=*/ 16,
            )
            .expect("rank state");
            rank_state.push(rs);
        }
        let stage = Gemma4HybridStage::new(stage_idx, sub, layers_global, rank_state)
            .expect("stage");
        stages.push(stage);
    }

    Gemma4HybridDriver::from_pieces(
        global,
        cfg,
        layout,
        stages,
        layer_to_stage,
        TP_SIZE,
        head_stage_idx,
        head_rank_in_head_stage_idx,
    )
    .expect("driver")
}

#[test]
fn forward_one_token_hybrid_pp2tp2_smoke() {
    let Some((sub_clusters, global)) = clusters_or_skip() else { return; };
    let mut driver = build_driver(sub_clusters, global);

    let mut tok = 0u32;
    for pos in 0..3 {
        tok = driver.forward_one_token(tok, pos).expect("forward");
        assert!((tok as usize) < driver.cfg.vocab_size, "argmax oob: {tok}");
    }
    // Each rank's KV caches grew by 3 (within each stage's TP shard).
    for s in 0..PP_SIZE {
        for r in 0..TP_SIZE {
            let rs = &driver.stages[s].rank_state[r];
            for (il_local, kv) in rs.kv_caches.iter().enumerate() {
                let kv = kv.as_ref().expect("layer has KV");
                assert_eq!(
                    kv.current_tokens(),
                    3,
                    "stage {s} rank {r} local-layer {il_local} kv count"
                );
            }
        }
    }
    driver.dispose().expect("dispose");
}

#[test]
fn partition_pp_rejects_invalid_shared_kv_split() {
    let mut cfg = synthetic_cfg();
    cfg.num_layers = 4;
    cfg.num_kv_heads = vec![N_KV_HEADS; 4];
    cfg.swa_layers = vec![false; 4];
    cfg.shared_kv_layers = 2; // tail layers 2,3 borrow from earlier
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    // 4 stages → layer 2 on stage 2, kv_share_src on stage 1 → cross-stage.
    let err = partition_layers_pp(4, &layout).expect_err("must reject");
    assert!(format!("{err}").contains("shared-KV tail"));
}
