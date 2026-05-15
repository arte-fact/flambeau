//! S6-B-A — gemma4 MoE FFN smoke. Synthetic 1-layer fixture with
//! shared dense MLP + 4-expert routed branch, run through the
//! `forward_ffn_moe` composer. Verifies:
//! 1. blocks::MoeExperts builds with `Activation::Gelu` and a
//!    gemma4-shaped RouterPolicy.
//! 2. The composer's two parallel branches + combine + final
//!    rmsnorm + residual stay finite on a dummy attn_residual input.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_blocks::{
    Activation, MoeExperts, MoeExpertsDecodeScratch, RouterInput, RouterNormalize, RouterPolicy,
    WeightHandle,
};
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_gemma4::{
    forward_ffn_moe, Gemma4LayerWeights, Gemma4MoeFfnWeights, Gemma4MoeScratch,
};
use flambeau_ops::hip::HipOps;
use flambeau_ops::OpsRegistry;
use half::f16;

const HIDDEN: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FF_LEN: usize = 128;             // shared MLP dim
const N_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const INTERMEDIATE: usize = 64;        // routed expert ff dim
const RMS_EPS: f32 = 1e-6;

fn hip_device() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP device — skipping forward_moe_smoke");
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

fn alloc_f32_const(dev: &HipDevice, n: usize, value: f32) -> DevicePtr {
    upload(dev, &vec![value; n])
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

fn build_layer_weights(dev: &HipDevice) -> Gemma4LayerWeights {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;
    Gemma4LayerWeights {
        attn_norm: alloc_f16_ones(dev, HIDDEN),
        attn_q: WeightHandle {
            ptr: alloc_q8_0_zero(dev, q_width, HIDDEN),
            dtype: QDtype::Q8_0,
            dims: [q_width, HIDDEN],
        },
        attn_k: Some(WeightHandle {
            ptr: alloc_q8_0_zero(dev, kv_width, HIDDEN),
            dtype: QDtype::Q8_0,
            dims: [kv_width, HIDDEN],
        }),
        attn_v: None,
        attn_output: WeightHandle {
            ptr: alloc_q8_0_zero(dev, HIDDEN, q_width),
            dtype: QDtype::Q8_0,
            dims: [HIDDEN, q_width],
        },
        attn_q_norm: alloc_f16_ones(dev, HEAD_DIM),
        attn_k_norm: Some(alloc_f16_ones(dev, HEAD_DIM)),
        post_attention_norm: alloc_f16_ones(dev, HIDDEN),
        post_attention_norm_f32: None,
        layer_output_scale: None,
        ffn_norm: alloc_f16_ones(dev, HIDDEN),
        ffn_gate: WeightHandle {
            ptr: alloc_q8_0_zero(dev, FF_LEN, HIDDEN),
            dtype: QDtype::Q8_0,
            dims: [FF_LEN, HIDDEN],
        },
        ffn_up: WeightHandle {
            ptr: alloc_q8_0_zero(dev, FF_LEN, HIDDEN),
            dtype: QDtype::Q8_0,
            dims: [FF_LEN, HIDDEN],
        },
        ffn_down: WeightHandle {
            ptr: alloc_q8_0_zero(dev, HIDDEN, FF_LEN),
            dtype: QDtype::Q8_0,
            dims: [HIDDEN, FF_LEN],
        },
        post_ffw_norm: alloc_f16_ones(dev, HIDDEN),
        per_layer_embed: None,
        moe: None,
        tp_moe: None,
    }
}

fn build_moe_weights(dev: &HipDevice) -> Gemma4MoeFfnWeights {
    // Router weight: F32 [n_experts, hidden].
    let router_w = WeightHandle {
        ptr: alloc_f32_const(dev, N_EXPERTS * HIDDEN, 0.01),
        dtype: QDtype::F32,
        dims: [N_EXPERTS, HIDDEN],
    };
    let gate_exps = WeightHandle {
        ptr: alloc_q8_0_zero(dev, N_EXPERTS * INTERMEDIATE, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [N_EXPERTS * INTERMEDIATE, HIDDEN],
    };
    let up_exps = WeightHandle {
        ptr: alloc_q8_0_zero(dev, N_EXPERTS * INTERMEDIATE, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [N_EXPERTS * INTERMEDIATE, HIDDEN],
    };
    let down_exps = WeightHandle {
        ptr: alloc_q8_0_zero(dev, N_EXPERTS * HIDDEN, INTERMEDIATE),
        dtype: QDtype::Q8_0,
        dims: [N_EXPERTS * HIDDEN, INTERMEDIATE],
    };
    let moe = MoeExperts::new(router_w, gate_exps, up_exps, down_exps, HIDDEN, INTERMEDIATE, N_EXPERTS, TOP_K)
        .expect("MoeExperts::new")
        .with_activation(Activation::Gelu)
        .with_router_policy(RouterPolicy {
            input: RouterInput::AttnOut,
            // Pre-scale + pre-scalar folded into pre_router_weight_f16
            // at upload time — declare them None here so the block
            // doesn't try to apply them again.
            pre_scale: None,
            pre_scalar: 1.0,
            normalize: RouterNormalize::TopkRenorm,
        });
    // Pre-router weight bakes `(1/sqrt(hidden)) * ffn_gate_inp_s` into a
    // single F16 [hidden] vector. For the dummy fixture we use a small
    // constant.
    let pre_router_scale = (1.0 / (HIDDEN as f32).sqrt()) * 0.1;
    let pre_router_host: Vec<f16> = vec![f16::from_f32(pre_router_scale); HIDDEN];
    let pre_router_weight_f16 = upload(dev, &pre_router_host);
    Gemma4MoeFfnWeights {
        moe,
        pre_router_weight_f16,
        pre_ffw_norm_2: alloc_f16_ones(dev, HIDDEN),
        post_ffw_norm_1: alloc_f16_ones(dev, HIDDEN),
        post_ffw_norm_2: alloc_f16_ones(dev, HIDDEN),
    }
}

fn build_moe_scratch(dev: &HipDevice) -> Gemma4MoeScratch {
    // Sized for our dummy shapes.
    let q8_1_blocks_hidden = HIDDEN / 32;
    let q8_1_blocks_inter = INTERMEDIATE / 32;
    let q8_1_bytes_per_block = 36;
    Gemma4MoeScratch {
        router_input_f16: alloc_zeroed(dev, HIDDEN * 2),
        cur_mlp_f16: alloc_zeroed(dev, HIDDEN * 2),
        cur_moe_f16: alloc_zeroed(dev, HIDDEN * 2),
        cur_combined_f16: alloc_zeroed(dev, HIDDEN * 2),
        zero_hidden_f16: alloc_zeroed(dev, HIDDEN * 2),
        moe_scratch: MoeExpertsDecodeScratch {
            x_q8_1: alloc_zeroed(dev, q8_1_blocks_hidden * q8_1_bytes_per_block),
            router_logits: alloc_zeroed(dev, N_EXPERTS * 4),
            expert_ids: alloc_zeroed(dev, TOP_K * 4),
            expert_weights: alloc_zeroed(dev, TOP_K * 4),
            gate_out_f32: alloc_zeroed(dev, TOP_K * INTERMEDIATE * 4),
            up_out_f32: alloc_zeroed(dev, TOP_K * INTERMEDIATE * 4),
            activated_f16: alloc_zeroed(dev, TOP_K * INTERMEDIATE * 2),
            activated_q8_1: alloc_zeroed(dev, TOP_K * q8_1_blocks_inter * q8_1_bytes_per_block),
            down_f32: alloc_zeroed(dev, TOP_K * HIDDEN * 4),
            down_f16: alloc_zeroed(dev, TOP_K * HIDDEN * 2),
        },
    }
}

#[test]
fn forward_ffn_moe_smoke() -> Result<()> {
    let Some(dev) = hip_device() else { return Ok(()); };
    dev.bind()?;
    let stream = dev.default_stream();
    let reg = OpsRegistry::new(&dev).map_err(|e| anyhow::anyhow!("registry: {e}"))?;
    let ops = HipOps::new(&reg, stream);

    let layer = build_layer_weights(&dev);
    let moe = build_moe_weights(&dev);
    let scratch_moe = build_moe_scratch(&dev);

    // Layer-level scratches (subset needed by forward_ffn_moe).
    let q8_1_blocks = HIDDEN.max(FF_LEN) / 32;
    let q8_1_bytes_per_block = 36;
    let x_q8_1 = alloc_zeroed(&dev, q8_1_blocks * q8_1_bytes_per_block);
    let mmvq_f32 = alloc_zeroed(&dev, FF_LEN.max(HIDDEN) * 4);
    let gate_f32 = alloc_zeroed(&dev, FF_LEN * 4);
    let up_f32 = alloc_zeroed(&dev, FF_LEN * 4);
    let activated_f16 = alloc_zeroed(&dev, FF_LEN * 2);
    let activated_q8_1 = alloc_zeroed(&dev, FF_LEN.div_ceil(32) * q8_1_bytes_per_block);
    let down_f32 = alloc_zeroed(&dev, HIDDEN * 4);

    // Synthetic attn_residual (small non-zero values).
    let attn_residual_host: Vec<f16> = (0..HIDDEN)
        .map(|i| f16::from_f32((i as f32) * 1e-3 - 0.064))
        .collect();
    let attn_residual = upload(&dev, &attn_residual_host);
    let x_out = alloc_zeroed(&dev, HIDDEN * 2);

    forward_ffn_moe(
        &ops,
        &layer,
        &moe,
        x_q8_1,
        mmvq_f32,
        gate_f32,
        up_f32,
        activated_f16,
        activated_q8_1,
        down_f32,
        &scratch_moe,
        attn_residual,
        x_out,
        FF_LEN,
        HIDDEN,
        RMS_EPS,
    )?;
    stream.synchronize()?;

    let out = download_f16(&dev, x_out, HIDDEN);
    for (i, v) in out.iter().enumerate() {
        let f = v.to_f32();
        assert!(f.is_finite(), "out[{i}] = {f} not finite");
    }
    Ok(())
}
