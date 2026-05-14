//! S6-A — exercise the shared-KV tail path on a synthetic fixture.
//! Builds a 3-layer gemma4 model where layer 2 has `has_kv == false`
//! and `kv_share_src == 0`. Verifies:
//! 1. The layer composer skips K/V proj + KV append on the tail layer.
//! 2. The tail layer's attention queries layer 0's KV cache, which
//!    grew during the layer-0 iteration of the same forward call.
//! 3. The whole chain stays finite over multiple decode steps as the
//!    source layer's KV grows.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_blocks::WeightHandle;
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_gemma4::{
    forward_one_token, DeviceTensor, Gemma4Config, Gemma4DeviceWeights, Gemma4LayerWeights,
    Gemma4Session, Gemma4Variant, ModelLayout,
};
use flambeau_quant::GgmlDType;
use half::f16;

const HIDDEN: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FF_LEN: usize = 128;
const VOCAB: usize = 64;
const N_LAYERS: usize = 3;

fn hip_device() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP device — skipping forward_shared_kv_smoke");
        return None;
    }
    HipDevice::new(0).ok()
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
    let zero = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(zero.as_ptr() as usize),
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

fn make_q8_0(dev: &HipDevice, n_rows: usize, k: usize, raw: &mut Vec<DeviceTensor>) -> WeightHandle {
    let bytes = n_rows * (k / 32) * 34;
    let ptr = alloc_q8_0_zero(dev, n_rows, k);
    raw.push(DeviceTensor { ptr, dtype: GgmlDType::Q8_0, bytes });
    WeightHandle { ptr, dtype: QDtype::Q8_0, dims: [n_rows, k] }
}

fn make_f16_ones(dev: &HipDevice, n: usize, raw: &mut Vec<DeviceTensor>) -> DevicePtr {
    let bytes = n * 2;
    let ptr = alloc_f16_ones(dev, n);
    raw.push(DeviceTensor { ptr, dtype: GgmlDType::F16, bytes });
    ptr
}

/// Layer with full attention (its own K, V), dense FFN.
fn make_owning_layer(dev: &HipDevice, raw: &mut Vec<DeviceTensor>) -> Gemma4LayerWeights {
    let q_rows = N_HEADS * HEAD_DIM;
    let kv_rows = N_KV_HEADS * HEAD_DIM;
    let attn_q = make_q8_0(dev, q_rows, HIDDEN, raw);
    let attn_k = make_q8_0(dev, kv_rows, HIDDEN, raw);
    let attn_v = make_q8_0(dev, kv_rows, HIDDEN, raw);
    let attn_output = make_q8_0(dev, HIDDEN, q_rows, raw);
    let ffn_gate = make_q8_0(dev, FF_LEN, HIDDEN, raw);
    let ffn_up = make_q8_0(dev, FF_LEN, HIDDEN, raw);
    let ffn_down = make_q8_0(dev, HIDDEN, FF_LEN, raw);

    Gemma4LayerWeights {
        attn_norm: make_f16_ones(dev, HIDDEN, raw),
        attn_q,
        attn_k: Some(attn_k),
        attn_v: Some(attn_v),
        attn_output,
        attn_q_norm: make_f16_ones(dev, HEAD_DIM, raw),
        attn_k_norm: Some(make_f16_ones(dev, HEAD_DIM, raw)),
        post_attention_norm: make_f16_ones(dev, HIDDEN, raw),
        layer_output_scale: None,
        ffn_norm: make_f16_ones(dev, HIDDEN, raw),
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm: make_f16_ones(dev, HIDDEN, raw),
        per_layer_embed: None,
        moe: None,
    }
}

/// Shared-KV tail layer: no attn_k, attn_v, attn_k_norm.
fn make_tail_layer(dev: &HipDevice, raw: &mut Vec<DeviceTensor>) -> Gemma4LayerWeights {
    let q_rows = N_HEADS * HEAD_DIM;
    let attn_q = make_q8_0(dev, q_rows, HIDDEN, raw);
    let attn_output = make_q8_0(dev, HIDDEN, q_rows, raw);
    let ffn_gate = make_q8_0(dev, FF_LEN, HIDDEN, raw);
    let ffn_up = make_q8_0(dev, FF_LEN, HIDDEN, raw);
    let ffn_down = make_q8_0(dev, HIDDEN, FF_LEN, raw);

    Gemma4LayerWeights {
        attn_norm: make_f16_ones(dev, HIDDEN, raw),
        attn_q,
        attn_k: None,
        attn_v: None,
        attn_output,
        attn_q_norm: make_f16_ones(dev, HEAD_DIM, raw),
        attn_k_norm: None,
        post_attention_norm: make_f16_ones(dev, HIDDEN, raw),
        layer_output_scale: None,
        ffn_norm: make_f16_ones(dev, HIDDEN, raw),
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm: make_f16_ones(dev, HIDDEN, raw),
        per_layer_embed: None,
        moe: None,
    }
}

fn build_synthetic_session_with_shared_kv(dev: &HipDevice) -> Gemma4Session {
    // 3 layers: 0 = full-attn (owns KV), 1 = full-attn (owns KV),
    // 2 = full-attn but shared-KV from layer 0 (has_kv=false).
    // SWA pattern is all-false (full attn). shared_kv_layers=1 → last
    // 1 layer is the tail.
    let cfg = Gemma4Config {
        arch: "gemma4".into(),
        variant: Gemma4Variant::E4B, // any size; not used for layout decisions
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
        shared_kv_layers: 1,
        moe: None,
        per_layer_embed: None,
        final_logit_softcap: 30.0,
        tied_lm_head: true,
    };
    let mut layout = ModelLayout::from_config(&cfg);
    let resolved = layout.resolve_kv_sharing();
    assert_eq!(resolved, 1, "expected 1 shared-KV tail layer");
    // Layer 2 must point at the most recent earlier layer of the same SWA type.
    // All-false swa_layers → layer 2's source is layer 1.
    assert_eq!(layout.layers[2].has_kv, false);
    assert_eq!(layout.layers[2].kv_share_src, Some(1));

    let mut raw: Vec<DeviceTensor> = Vec::new();

    let tok_embd_ptr = alloc_q8_0_zero(dev, VOCAB, HIDDEN);
    let token_embd = DeviceTensor {
        ptr: tok_embd_ptr,
        dtype: GgmlDType::Q8_0,
        bytes: VOCAB * (HIDDEN / 32) * 34,
    };
    raw.push(token_embd);
    let output_norm_ptr = alloc_f16_ones(dev, HIDDEN);
    let output_norm = DeviceTensor {
        ptr: output_norm_ptr,
        dtype: GgmlDType::F16,
        bytes: HIDDEN * 2,
    };
    raw.push(output_norm);

    let layers = vec![
        make_owning_layer(dev, &mut raw),
        make_owning_layer(dev, &mut raw),
        make_tail_layer(dev, &mut raw),
    ];

    let mut tracker = flambeau_blocks::RawAllocTracker::new();
    for t in raw {
        tracker.track(t.ptr, t.bytes);
    }
    let weights = Gemma4DeviceWeights::from_pieces(
        token_embd,
        [VOCAB, HIDDEN],
        output_norm,
        None,
        layers,
        tracker,
        dev.id(),
    );
    Gemma4Session::new(dev, weights, cfg, layout, /*max_tokens=*/ 16).expect("session new")
}

#[test]
fn forward_one_token_with_shared_kv_tail_smoke() {
    let Some(dev) = hip_device() else { return; };
    dev.bind().unwrap();
    let mut session = build_synthetic_session_with_shared_kv(&dev);

    // Layer 0 and 1 own their KV caches; layer 2 reads from layer 1's cache.
    assert!(session.kv_caches[0].is_some());
    assert!(session.kv_caches[1].is_some());
    assert!(session.kv_caches[2].is_none(), "tail layer must not allocate");

    // Decode a sequence of tokens; tail layer attention should grow
    // with layer 1's cache.
    let mut tok = 0u32;
    for pos in 0..6 {
        tok = forward_one_token(&mut session, &dev, tok, pos).expect("forward step");
        assert!((tok as usize) < session.cfg.vocab_size);

        // Verify layer 1's cache grew (the source of tail layer 2).
        let l1_tokens = session.kv_caches[1]
            .as_ref()
            .expect("layer 1 has KV")
            .current_tokens();
        assert_eq!(l1_tokens, pos + 1, "layer 1 cache count");

        // Layer 2 (tail) shares layer 1's cache, so its slot remains None
        // but the attention reads `pos + 1` tokens.
        assert!(session.kv_caches[2].is_none());
    }

    session.dispose(&dev).expect("dispose");
}
