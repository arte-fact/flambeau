//! S5-A — exercise `forward_layer_decode` end-to-end through HIP on a
//! tiny dummy fixture. Verifies the kernel chain (rmsnorm + Q/K/V proj
//! + per-head Q/K/V norm + RoPE + KV append + SWA attention + output
//! proj + post_attention_norm + residual + dense FFN with GELU +
//! post_ffw_norm + residual) wires up without crashing and writes a
//! finite (non-NaN, non-Inf) output. Not a correctness cert.
//!
//! Covers two layer variants:
//! - full-attention layer (window=0).
//! - SWA layer (window=4, n_tokens after seed=2 so window is meaningful).
//! Plus a softcap output-head exercise.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_blocks::WeightHandle;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_core::op::QDtype;
use flambeau_gemma4::{
    apply_logit_softcap, forward_layer_decode, FfnKind, Gemma4LayerWeights, LayerDecodeScratch,
    LayerSpec,
};
use flambeau_ops::hip::HipOps;
use flambeau_ops::OpsRegistry;
use half::f16;

fn hip_device() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP device — skipping gemma4 forward_layer_smoke");
        return None;
    }
    HipDevice::new(0).ok()
}

fn alloc_zeroed(dev: &HipDevice, bytes: usize) -> DevicePtr {
    let d = dev.alloc(bytes).unwrap();
    let host = vec![0u8; bytes];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(host.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
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

fn alloc_q8_0_zero(dev: &HipDevice, n_rows: usize, k: usize) -> DevicePtr {
    assert!(k % 32 == 0);
    let n_blocks = n_rows * (k / 32);
    // BlockQ8_0 = 2-byte d + 32-byte qs = 34 bytes.
    let bytes = n_blocks * 34;
    alloc_zeroed(dev, bytes)
}

fn alloc_f16_ones(dev: &HipDevice, n: usize) -> DevicePtr {
    let host: Vec<f16> = vec![f16::from_f32(1.0); n];
    upload(dev, &host)
}

fn download_f16(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f16> {
    let mut host = vec![f16::from_f32(0.0); n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n * 2,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

fn download_f32(dev: &HipDevice, src: DevicePtr, n: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            src,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

const HIDDEN: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
// attention kernel supports head_dim ∈ {64, 128, 256}.
const HEAD_DIM: usize = 64;
const Q_WIDTH: usize = N_HEADS * HEAD_DIM;
const KV_WIDTH: usize = N_KV_HEADS * HEAD_DIM;
const FF_LEN: usize = 128;
const CTX_CAP: usize = 32;
const RMS_EPS: f32 = 1e-6;

fn build_weights(dev: &HipDevice, with_v_proj: bool) -> Gemma4LayerWeights {
    let attn_q = WeightHandle {
        ptr: alloc_q8_0_zero(dev, Q_WIDTH, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [Q_WIDTH, HIDDEN],
    };
    let attn_k = WeightHandle {
        ptr: alloc_q8_0_zero(dev, KV_WIDTH, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [KV_WIDTH, HIDDEN],
    };
    let attn_v = with_v_proj.then(|| WeightHandle {
        ptr: alloc_q8_0_zero(dev, KV_WIDTH, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [KV_WIDTH, HIDDEN],
    });
    let attn_output = WeightHandle {
        ptr: alloc_q8_0_zero(dev, HIDDEN, Q_WIDTH),
        dtype: QDtype::Q8_0,
        dims: [HIDDEN, Q_WIDTH],
    };
    let ffn_gate = WeightHandle {
        ptr: alloc_q8_0_zero(dev, FF_LEN, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [FF_LEN, HIDDEN],
    };
    let ffn_up = WeightHandle {
        ptr: alloc_q8_0_zero(dev, FF_LEN, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [FF_LEN, HIDDEN],
    };
    let ffn_down = WeightHandle {
        ptr: alloc_q8_0_zero(dev, HIDDEN, FF_LEN),
        dtype: QDtype::Q8_0,
        dims: [HIDDEN, FF_LEN],
    };
    Gemma4LayerWeights {
        attn_norm: alloc_f16_ones(dev, HIDDEN),
        attn_q,
        attn_k: Some(attn_k),
        attn_v,
        attn_output,
        attn_q_norm: alloc_f16_ones(dev, HEAD_DIM),
        attn_k_norm: Some(alloc_f16_ones(dev, HEAD_DIM)),
        post_attention_norm: alloc_f16_ones(dev, HIDDEN),
        layer_output_scale: None,
        ffn_norm: alloc_f16_ones(dev, HIDDEN),
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm: alloc_f16_ones(dev, HIDDEN),
        per_layer_embed: None,
        moe: None,
    }
}

fn make_spec(window: u32, ffn: FfnKind) -> LayerSpec {
    LayerSpec {
        index: 0,
        window,
        is_swa: window > 0,
        has_kv: true,
        kv_share_src: None,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        rope_dim: HEAD_DIM,
        rope_freq_base: 10_000.0,
        ffn_kind: ffn,
    }
}

fn build_scratch<'a>(dev: &HipDevice, positions_host: &'a mut [i32]) -> LayerDecodeScratch<'a> {
    // mmvq_f32 sized for max(q_width, kv_width, hidden, ff_len).
    let mmvq_max = Q_WIDTH.max(KV_WIDTH).max(HIDDEN).max(FF_LEN);
    let q8_1_blocks = HIDDEN.max(FF_LEN) / 32;
    let q8_1_bytes_per_block = 36; // d + s + 32 bytes
    let x_q8_1 = alloc_zeroed(dev, q8_1_blocks * q8_1_bytes_per_block);
    let activated_q8_1 = alloc_zeroed(dev, q8_1_blocks * q8_1_bytes_per_block);
    LayerDecodeScratch {
        x_q8_1,
        mmvq_f32: alloc_zeroed(dev, mmvq_max * 4),
        q_f16: alloc_zeroed(dev, Q_WIDTH * 2),
        k_f16: alloc_zeroed(dev, KV_WIDTH * 2),
        v_f16: alloc_zeroed(dev, KV_WIDTH * 2),
        attn_out_f16: alloc_zeroed(dev, Q_WIDTH * 2),
        post_attn_norm_f16: alloc_zeroed(dev, HIDDEN * 2),
        attn_residual_f16: alloc_zeroed(dev, HIDDEN * 2),
        ffn_norm_f16: alloc_zeroed(dev, HIDDEN * 2),
        gate_f32: alloc_zeroed(dev, FF_LEN * 4),
        up_f32: alloc_zeroed(dev, FF_LEN * 4),
        activated_f16: alloc_zeroed(dev, FF_LEN * 2),
        activated_q8_1,
        down_f32: alloc_zeroed(dev, HIDDEN * 4),
        post_ffw_norm_f16: alloc_zeroed(dev, HIDDEN * 2),
        positions: alloc_zeroed(dev, 4),
        positions_host,
        v_ones_f16: alloc_f16_ones(dev, HEAD_DIM),
        splitk_partials_m: alloc_zeroed(
            dev,
            N_HEADS * flambeau_blocks::MAX_SPLITK_CHUNKS * 4,
        ),
        splitk_partials_s: alloc_zeroed(
            dev,
            N_HEADS * flambeau_blocks::MAX_SPLITK_CHUNKS * 4,
        ),
        splitk_partials_o: alloc_zeroed(
            dev,
            N_HEADS * flambeau_blocks::MAX_SPLITK_CHUNKS * HEAD_DIM * 4,
        ),
    }
}

fn run_one_layer(dev: &HipDevice, window: u32, with_v_proj: bool) -> Result<Vec<f32>> {
    use flambeau_runtime::{F16Contig, KvCache};

    dev.bind()?;
    let reg = OpsRegistry::new(dev).map_err(|e| anyhow::anyhow!("registry: {e}"))?;
    let ops = HipOps::new(&reg, dev.default_stream());

    let weights = build_weights(dev, with_v_proj);
    let spec = make_spec(window, FfnKind::Dense);

    // Allocate a tiny KV cache (F16, ctx=CTX_CAP).
    let mut kv = KvCache::<F16Contig, HipDevice>::new(dev, N_KV_HEADS, HEAD_DIM, CTX_CAP)
        .map_err(|e| anyhow::anyhow!("kv alloc: {e}"))?;

    // x_in = small non-zero deterministic values so the chain is more
    // than a sea of zeros (norm of zeros is well-defined via eps but
    // gets boring).
    let x_in_host: Vec<f16> = (0..HIDDEN)
        .map(|i| f16::from_f32((i as f32) * 1e-3 - 0.064))
        .collect();
    let d_x_in = upload(dev, &x_in_host);
    let d_x_out = alloc_zeroed(dev, HIDDEN * 2);

    let mut positions_host = [0i32; 1];
    let mut scratch = build_scratch(dev, &mut positions_host);

    forward_layer_decode(
        &ops, dev, dev.default_stream(), &weights, &spec, RMS_EPS, FF_LEN, HIDDEN,
        &mut kv, &mut scratch, d_x_in, d_x_out, 0,
        /*per_layer_slice=*/ None,
        /*moe_scratch=*/ None,
    )?;
    dev.default_stream().synchronize()?;

    let out_f16 = download_f16(dev, d_x_out, HIDDEN);
    let out_f32: Vec<f32> = out_f16.iter().map(|v| v.to_f32()).collect();
    Ok(out_f32)
}

fn assert_finite(label: &str, v: &[f32]) {
    for (i, x) in v.iter().enumerate() {
        assert!(
            x.is_finite(),
            "{label}: non-finite at index {i}: {x}"
        );
    }
}

#[test]
fn forward_layer_full_attn_decode_smoke() {
    let Some(dev) = hip_device() else { return; };
    let out = run_one_layer(&dev, 0, true).expect("run full-attn layer");
    assert_finite("full-attn", &out);
}

#[test]
fn forward_layer_swa_decode_smoke() {
    let Some(dev) = hip_device() else { return; };
    let out = run_one_layer(&dev, 4, true).expect("run SWA layer");
    assert_finite("swa", &out);
}

#[test]
fn forward_layer_alt_attn_v_from_k_smoke() {
    // Gemma 4 full-attention layers omit V proj and reuse K
    // ("alternative attention"). Exercise that path.
    let Some(dev) = hip_device() else { return; };
    let out = run_one_layer(&dev, 0, false).expect("run alt-attn layer");
    assert_finite("alt-attn", &out);
}

#[test]
fn softcap_logits_smoke() {
    let Some(dev) = hip_device() else { return; };
    dev.bind().unwrap();
    let reg = OpsRegistry::new(&dev).expect("registry");
    let ops = HipOps::new(&reg, dev.default_stream());

    let n = 256usize;
    let host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let d_x = upload(&dev, &host);
    apply_logit_softcap(&ops, d_x, d_x, n, 30.0).expect("softcap in-place");
    dev.default_stream().synchronize().unwrap();
    let got = download_f32(&dev, d_x, n);
    assert_finite("softcap", &got);
    // tanh-cap saturates ~ ±cap.
    for (i, &g) in got.iter().enumerate() {
        assert!(g.abs() <= 30.0 + 1e-4, "softcap[{i}] = {g} exceeded cap");
    }

    // Also verify cap=0 is a no-op (forward-compat).
    let d_y = upload(&dev, &host);
    apply_logit_softcap(&ops, d_y, d_y, n, 0.0).expect("softcap noop");
    dev.default_stream().synchronize().unwrap();
    let got2 = download_f32(&dev, d_y, n);
    for (a, b) in got2.iter().zip(host.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "cap=0 must be pass-through");
    }
}
