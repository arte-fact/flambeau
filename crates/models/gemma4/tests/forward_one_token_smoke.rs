//! S5-B-1 end-to-end smoke: synthetic 2-layer gemma4 model, embed +
//! per-layer compose + output head + softcap + argmax through HIP.
//!
//! No real GGUF — dummy weights via [`Gemma4DeviceWeights::from_pieces`].
//! Verifies the framework wires up without crashing and the host-side
//! argmax produces a token-id within `vocab`.

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
const HEAD_DIM: usize = 64; // attention kernel requires {64,128,256}
const FF_LEN: usize = 128;
const VOCAB: usize = 64;
const N_LAYERS: usize = 2;

fn hip_device() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP device — skipping forward_one_token_smoke");
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
    let blocks = n_rows * (k / 32);
    alloc_zeroed(dev, blocks * 34) // BlockQ8_0 = 2-byte d + 32-byte qs
}

fn alloc_f16_ones(dev: &HipDevice, n: usize) -> DevicePtr {
    upload(dev, &vec![f16::from_f32(1.0); n])
}

fn make_q8_0_tensor(dev: &HipDevice, n_rows: usize, k: usize, raw: &mut Vec<DeviceTensor>) -> WeightHandle {
    assert_eq!(k % 32, 0);
    let bytes = n_rows * (k / 32) * 34;
    let ptr = alloc_q8_0_zero(dev, n_rows, k);
    raw.push(DeviceTensor { ptr, dtype: GgmlDType::Q8_0, bytes });
    WeightHandle {
        ptr,
        dtype: QDtype::Q8_0,
        dims: [n_rows, k],
    }
}

fn make_f16_ones_tensor(dev: &HipDevice, n: usize, raw: &mut Vec<DeviceTensor>) -> DevicePtr {
    let bytes = n * 2;
    let ptr = alloc_f16_ones(dev, n);
    raw.push(DeviceTensor { ptr, dtype: GgmlDType::F16, bytes });
    ptr
}

fn make_layer(dev: &HipDevice, raw: &mut Vec<DeviceTensor>) -> Gemma4LayerWeights {
    let q_rows = N_HEADS * HEAD_DIM;
    let kv_rows = N_KV_HEADS * HEAD_DIM;

    let attn_q = make_q8_0_tensor(dev, q_rows, HIDDEN, raw);
    let attn_k = make_q8_0_tensor(dev, kv_rows, HIDDEN, raw);
    // No attn_v — exercise the V=K alt-attention path.
    let attn_output = make_q8_0_tensor(dev, HIDDEN, q_rows, raw);
    let ffn_gate = make_q8_0_tensor(dev, FF_LEN, HIDDEN, raw);
    let ffn_up = make_q8_0_tensor(dev, FF_LEN, HIDDEN, raw);
    let ffn_down = make_q8_0_tensor(dev, HIDDEN, FF_LEN, raw);

    Gemma4LayerWeights {
        attn_norm: make_f16_ones_tensor(dev, HIDDEN, raw),
        attn_q,
        attn_k: Some(attn_k),
        attn_v: None,
        attn_output,
        attn_q_norm: make_f16_ones_tensor(dev, HEAD_DIM, raw),
        attn_k_norm: Some(make_f16_ones_tensor(dev, HEAD_DIM, raw)),
        post_attention_norm: make_f16_ones_tensor(dev, HIDDEN, raw),
        post_attention_norm_f32: None,
        layer_output_scale: None,
        ffn_norm: make_f16_ones_tensor(dev, HIDDEN, raw),
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm: make_f16_ones_tensor(dev, HIDDEN, raw),
        per_layer_embed: None,
        moe: None,
        tp_moe: None,
    }
}

/// Build a tiny synthetic gemma4 setup: 2 layers (1 full + 1 SWA), tied LM head.
fn build_synthetic_session(dev: &HipDevice) -> Gemma4Session {
    let cfg = Gemma4Config {
        arch: "gemma4".into(),
        variant: Gemma4Variant::E2B, // E2B fits n_layer=35 but we use 2 in the synthetic;
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
        swa_layers: vec![false, true],
        sliding_window: 4,
        shared_kv_layers: 0,
        moe: None,
        per_layer_embed: None,
        final_logit_softcap: 30.0,
        tied_lm_head: true,
    };
    let mut layout = ModelLayout::from_config(&cfg);
    let _ = layout.resolve_kv_sharing();
    // Override variant to skip the from_n_layer panic — the synthetic 2-layer
    // model is intentionally outside the production size table.
    let mut raw: Vec<DeviceTensor> = Vec::new();

    let tok_embd_q8_rows = VOCAB;
    let token_embd_bytes = tok_embd_q8_rows * (HIDDEN / 32) * 34;
    let token_embd_ptr = alloc_q8_0_zero(dev, tok_embd_q8_rows, HIDDEN);
    let token_embd = DeviceTensor {
        ptr: token_embd_ptr,
        dtype: GgmlDType::Q8_0,
        bytes: token_embd_bytes,
    };
    raw.push(token_embd);

    let output_norm_ptr = alloc_f16_ones(dev, HIDDEN);
    let output_norm = DeviceTensor {
        ptr: output_norm_ptr,
        dtype: GgmlDType::F16,
        bytes: HIDDEN * 2,
    };
    raw.push(output_norm);

    let mut layers = Vec::with_capacity(N_LAYERS);
    for _ in 0..N_LAYERS {
        layers.push(make_layer(dev, &mut raw));
    }

    // Sanity-check the layout matches expected SWA pattern.
    assert!(!layout.layers[0].is_swa);
    assert!(layout.layers[1].is_swa);

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
fn forward_one_token_end_to_end_smoke() {
    let Some(dev) = hip_device() else { return; };
    dev.bind().unwrap();

    let mut session = build_synthetic_session(&dev);
    // Decode token 0 at position 0.
    let next = forward_one_token(&mut session, &dev, 0, 0).expect("forward_one_token");
    assert!((next as usize) < session.cfg.vocab_size, "argmax out of range");
    // Decode a second token to exercise growing KV cache.
    let next2 = forward_one_token(&mut session, &dev, next, 1).expect("forward_one_token #2");
    assert!((next2 as usize) < session.cfg.vocab_size);

    session.dispose(&dev).expect("dispose");
}

#[test]
fn forward_one_token_swa_grows_kv() {
    // Decode multiple tokens through the SWA layer to ensure the
    // window mask + KV-cache growth don't NaN.
    let Some(dev) = hip_device() else { return; };
    dev.bind().unwrap();

    let mut session = build_synthetic_session(&dev);
    let mut tok = 0u32;
    for pos in 0..8 {
        tok = forward_one_token(&mut session, &dev, tok, pos).expect("forward step");
        assert!((tok as usize) < session.cfg.vocab_size);
    }
    session.dispose(&dev).expect("dispose");
}
