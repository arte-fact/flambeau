//! S8-A — pipeline-parallel decode smoke. Synthetic 2-rank fixture,
//! 4 layers (2 per rank), all full-attn dense FFN. Verifies:
//! 1. `partition_layers` distributes layers without violating
//!    shared-KV constraints (none here — every layer owns KV).
//! 2. `Gemma4PpDriver` builds, embeds, runs the per-layer loop across
//!    rank stages with peer-copy between, and emits an argmax.
//! 3. Multi-step decode grows the per-rank KV caches and stays finite.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use flambeau_backend_hip::{device_count, HipCluster, HipDevice};
use flambeau_blocks::WeightHandle;
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_gemma4::{
    partition_layers, DeviceTensor, Gemma4Config, Gemma4LayerWeights, Gemma4PpDriver,
    Gemma4PpStage, Gemma4Variant, ModelLayout,
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
const N_RANKS: usize = 2;

fn cluster_or_skip() -> Option<HipCluster> {
    let n = device_count().ok()?;
    if (n as usize) < N_RANKS {
        eprintln!(
            "skipping PP smoke — need {N_RANKS} HIP devices but only {n} available"
        );
        return None;
    }
    let ids: Vec<i32> = (0..N_RANKS as i32).collect();
    HipCluster::new(&ids).ok()
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

fn alloc_zeroed_n(dev: &HipDevice, bytes: usize) -> DevicePtr {
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
    alloc_zeroed_n(dev, n_rows * (k / 32) * 34)
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

fn make_layer(dev: &HipDevice) -> Gemma4LayerWeights {
    let q_rows = N_HEADS * HEAD_DIM;
    let kv_rows = N_KV_HEADS * HEAD_DIM;
    let (attn_q, _) = make_q8_0(dev, q_rows, HIDDEN);
    let (attn_k, _) = make_q8_0(dev, kv_rows, HIDDEN);
    let (attn_v, _) = make_q8_0(dev, kv_rows, HIDDEN);
    let (attn_output, _) = make_q8_0(dev, HIDDEN, q_rows);
    let (ffn_gate, _) = make_q8_0(dev, FF_LEN, HIDDEN);
    let (ffn_up, _) = make_q8_0(dev, FF_LEN, HIDDEN);
    let (ffn_down, _) = make_q8_0(dev, HIDDEN, FF_LEN);
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
        post_attention_norm_f32: None,
        layer_output_scale: None,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm,
        per_layer_embed: None,
        moe: None,
        tp_moe: None,
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

fn build_synthetic_pp_driver(cluster: HipCluster) -> Gemma4PpDriver {
    let cfg = synthetic_cfg();
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();

    let layer_to_rank = partition_layers(N_RANKS, &layout).expect("partition");
    // With 4 layers / 2 ranks: layers 0,1 → rank 0; layers 2,3 → rank 1.
    assert_eq!(layer_to_rank, vec![0, 0, 1, 1]);

    let mut stages = Vec::with_capacity(N_RANKS);
    for rank in 0..N_RANKS {
        let dev = cluster.device(rank);
        dev.bind().unwrap();

        let layer_weights: Vec<Gemma4LayerWeights> = (0..N_LAYERS)
            .filter(|&i| layer_to_rank[i] == rank)
            .map(|_| make_layer(dev))
            .collect();

        let (token_embd, token_embd_dims) = if rank == 0 {
            let (_, dt) = make_q8_0(dev, VOCAB, HIDDEN);
            (Some(dt), Some([VOCAB, HIDDEN]))
        } else {
            (None, None)
        };
        let (output_norm, output) = if rank == N_RANKS - 1 {
            let (_, on) = make_f16_ones(dev, HIDDEN);
            // For tied LM head we also need the LM-head weight on the
            // last rank. Reuse the same Q8_0 zero pattern.
            let (_, lm) = make_q8_0(dev, VOCAB, HIDDEN);
            (Some(on), Some(lm))
        } else {
            (None, None)
        };

        let stage = Gemma4PpStage::from_pieces(
            dev,
            rank,
            &cfg,
            &layout,
            &layer_to_rank,
            layer_weights,
            token_embd,
            token_embd_dims,
            output_norm,
            // gemma4 ties LM head — last-rank `output` holds the
            // (zero-filled) Q8_0 weight that the LM-head matmul reads.
            // In real-GGUF integration this would be a replica of
            // rank-0's token_embd; for the dummy fixture a fresh
            // allocation is fine (also zero ⇒ identical math).
            output,
            /*max_tokens=*/ 16,
        )
        .expect("stage");
        // Override token_embd_dims on last rank so output_head can
        // compute the LM-head dims from a tied/untied tensor uniformly.
        let mut stage = stage;
        if rank == N_RANKS - 1 && stage.token_embd_dims.is_none() {
            stage.token_embd_dims = Some([VOCAB, HIDDEN]);
        }
        stages.push(stage);
    }

    Gemma4PpDriver::from_pieces(cluster, cfg, layout, layer_to_rank, stages).expect("driver")
}

#[test]
fn forward_one_token_pp_smoke() {
    let Some(cluster) = cluster_or_skip() else { return; };
    let mut driver = build_synthetic_pp_driver(cluster);

    // Decode a sequence of tokens through the pipeline.
    let mut tok = 0u32;
    for pos in 0..4 {
        tok = driver.forward_one_token(tok, pos).expect("forward");
        assert!((tok as usize) < driver.cfg.vocab_size, "argmax oob: {tok}");
    }
    // Per-rank KV caches grew.
    let r0_layer0 = driver.stages[0].kv_caches[0]
        .as_ref()
        .expect("rank 0 layer 0 has KV")
        .current_tokens();
    assert_eq!(r0_layer0, 4, "rank 0 layer 0 should have 4 tokens");
    let r1_layer3_local = driver.stages[1].kv_caches[1]
        .as_ref()
        .expect("rank 1 has KV for its layers")
        .current_tokens();
    assert_eq!(r1_layer3_local, 4, "rank 1 last layer kv");

    driver.dispose().expect("dispose");
}

#[test]
fn partition_rejects_invalid_shared_kv_split() {
    // Construct a layout where shared-KV would cross stage boundaries.
    let mut cfg = synthetic_cfg();
    cfg.num_layers = 4;
    cfg.num_kv_heads = vec![N_KV_HEADS; 4];
    cfg.swa_layers = vec![false; 4];
    cfg.shared_kv_layers = 2; // last 2 layers (2 + 3) reuse from earlier
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    // resolve_kv_sharing makes layer 2 borrow from layer 1 (most recent owner).
    // Split 4 layers across 4 ranks → layer 2 on rank 2, layer 1 on rank 1 → cross-stage.
    let err = partition_layers(4, &layout).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("shared-KV tail") && msg.contains("kv_share_src"),
        "unexpected error: {msg}"
    );
}
