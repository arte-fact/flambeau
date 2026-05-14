//! S5-B-2 / #17 — per-layer side-channel embedding + layer_output_scale.
//! Exercise the layer composer's optional per-layer-embd post-block on
//! a synthetic E4B-like single-layer fixture. Verifies:
//! 1. `build_inp_per_layer_table` (host-side) runs without panic and
//!    produces `pe * n_layer` finite F32 values.
//! 2. `forward_per_layer_post_block` chains the GELU + mul + proj +
//!    norm + residual-add through HIP without NaN.
//! 3. `forward_layer_decode` with per-layer-slice + layer_output_scale
//!    set produces a finite output on a tiny fixture.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async over \
              host/device buffers that live for the bounded synchronize that follows."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_blocks::WeightHandle;
use flambeau_core::op::QDtype;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_gemma4::{
    build_inp_per_layer_table, forward_layer_decode, table_slice_ptr,
    upload_inp_per_layer_table, FfnKind, Gemma4LayerWeights, LayerDecodeScratch, LayerSpec,
    PerLayerEmbedLayerWeights,
};
use flambeau_ops::hip::HipOps;
use flambeau_ops::OpsRegistry;
use flambeau_quant::GgmlDType;
use half::f16;

const HIDDEN: usize = 256;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const FF_LEN: usize = 128;
const PE: usize = 64;
const N_LAYERS: usize = 1;
const CTX_CAP: usize = 16;
const RMS_EPS: f32 = 1e-6;

fn hip_device() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        eprintln!("no HIP device — skipping per_layer_embd smoke");
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

fn build_weights(dev: &HipDevice) -> Gemma4LayerWeights {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;
    let attn_q = WeightHandle {
        ptr: alloc_q8_0_zero(dev, q_width, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [q_width, HIDDEN],
    };
    let attn_k = WeightHandle {
        ptr: alloc_q8_0_zero(dev, kv_width, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [kv_width, HIDDEN],
    };
    let attn_v = WeightHandle {
        ptr: alloc_q8_0_zero(dev, kv_width, HIDDEN),
        dtype: QDtype::Q8_0,
        dims: [kv_width, HIDDEN],
    };
    let attn_output = WeightHandle {
        ptr: alloc_q8_0_zero(dev, HIDDEN, q_width),
        dtype: QDtype::Q8_0,
        dims: [HIDDEN, q_width],
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

    // Per-layer-embd weights: inp_gate F32 [pe, hidden], proj F32 [hidden, pe],
    // post_norm F16 [hidden] (cast at load in real path; F16 ones here).
    let pe_w = PerLayerEmbedLayerWeights {
        inp_gate: alloc_f32_const(dev, PE * HIDDEN, 0.01),
        proj: alloc_f32_const(dev, HIDDEN * PE, 0.01),
        post_norm_f16: alloc_f16_ones(dev, HIDDEN),
    };

    Gemma4LayerWeights {
        attn_norm: alloc_f16_ones(dev, HIDDEN),
        attn_q,
        attn_k: Some(attn_k),
        attn_v: Some(attn_v),
        attn_output,
        attn_q_norm: alloc_f16_ones(dev, HEAD_DIM),
        attn_k_norm: Some(alloc_f16_ones(dev, HEAD_DIM)),
        post_attention_norm: alloc_f16_ones(dev, HIDDEN),
        // Exercise the layer_output_scale path with a non-1.0 value.
        layer_output_scale: Some(0.5),
        ffn_norm: alloc_f16_ones(dev, HIDDEN),
        ffn_gate,
        ffn_up,
        ffn_down,
        post_ffw_norm: alloc_f16_ones(dev, HIDDEN),
        per_layer_embed: Some(pe_w),
        moe: None,
    }
}

fn make_spec() -> LayerSpec {
    LayerSpec {
        index: 0,
        window: 0,
        is_swa: false,
        has_kv: true,
        kv_share_src: None,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        rope_dim: HEAD_DIM,
        rope_freq_base: 10_000.0,
        ffn_kind: FfnKind::Dense,
    }
}

fn build_scratch<'a>(dev: &HipDevice, positions_host: &'a mut [i32]) -> LayerDecodeScratch<'a> {
    let q_width = N_HEADS * HEAD_DIM;
    let kv_width = N_KV_HEADS * HEAD_DIM;
    let mmvq_max = q_width.max(kv_width).max(HIDDEN).max(FF_LEN);
    let q8_1_blocks = HIDDEN.max(FF_LEN) / 32;
    let q8_1_bytes_per_block = 36;
    let x_q8_1 = alloc_zeroed(dev, q8_1_blocks * q8_1_bytes_per_block);
    let activated_q8_1 = alloc_zeroed(dev, q8_1_blocks * q8_1_bytes_per_block);
    LayerDecodeScratch {
        x_q8_1,
        mmvq_f32: alloc_zeroed(dev, mmvq_max * 4),
        q_f16: alloc_zeroed(dev, q_width * 2),
        k_f16: alloc_zeroed(dev, kv_width * 2),
        v_f16: alloc_zeroed(dev, kv_width * 2),
        attn_out_f16: alloc_zeroed(dev, q_width * 2),
        post_attn_norm_f16: alloc_zeroed(dev, HIDDEN * 2),
        attn_residual_f16: alloc_zeroed(dev, HIDDEN * 2),
        ffn_norm_f16: alloc_zeroed(dev, HIDDEN * 2),
        gate_f32: alloc_zeroed(dev, FF_LEN.max(PE) * 4),
        up_f32: alloc_zeroed(dev, FF_LEN.max(PE) * 4),
        activated_f16: alloc_zeroed(dev, FF_LEN.max(PE) * 2),
        activated_q8_1,
        down_f32: alloc_zeroed(dev, HIDDEN * 4),
        post_ffw_norm_f16: alloc_zeroed(dev, HIDDEN * 2),
        positions: alloc_zeroed(dev, 4),
        positions_host,
        v_ones_f16: alloc_f16_ones(dev, HEAD_DIM),
    }
}

#[test]
fn build_inp_per_layer_table_finite() {
    // Pure host test — no HIP required. Verifies the build function
    // produces finite values on a synthetic fixture.
    let pe = PE;
    let n_layer = 4;
    let hidden = HIDDEN;

    // Synthetic tok-embd row: F32 [pe * n_layer], small values.
    let tok_row_f32: Vec<f32> = (0..pe * n_layer).map(|i| (i as f32) * 1e-4).collect();
    let tok_row_bytes: &[u8] = bytemuck::cast_slice(&tok_row_f32);

    // Synthetic model_proj: F32 [pe*n_layer, hidden], small values.
    let proj_f32: Vec<f32> = (0..pe * n_layer * hidden)
        .map(|i| ((i as f32 * 0.7).sin()) * 1e-3)
        .collect();
    let proj_bytes: &[u8] = bytemuck::cast_slice(&proj_f32);

    // Synthetic proj_norm: F32 [pe], ones.
    let norm: Vec<f32> = vec![1.0; pe];
    let norm_bytes: &[u8] = bytemuck::cast_slice(&norm);

    let inp_batch: Vec<f16> = (0..hidden)
        .map(|i| f16::from_f32((i as f32) * 1e-3 - 0.064))
        .collect();

    let out = build_inp_per_layer_table(
        tok_row_bytes,
        GgmlDType::F32,
        proj_bytes,
        GgmlDType::F32,
        norm_bytes,
        &inp_batch,
        pe,
        n_layer,
        hidden,
        RMS_EPS,
    )
    .expect("build table");
    assert_eq!(out.len(), pe * n_layer);
    for (i, v) in out.iter().enumerate() {
        assert!(v.is_finite(), "table[{i}] = {v} not finite");
    }
}

#[test]
fn forward_layer_decode_with_per_layer_embd_smoke() -> Result<()> {
    let Some(dev) = hip_device() else { return Ok(()); };
    dev.bind()?;
    let stream = dev.default_stream();
    let reg = OpsRegistry::new(&dev).map_err(|e| anyhow::anyhow!("registry: {e}"))?;
    let ops = HipOps::new(&reg, stream);

    let weights = build_weights(&dev);
    let spec = make_spec();

    use flambeau_runtime::{F16Contig, KvCache};
    let mut kv = KvCache::<F16Contig, HipDevice>::new(&dev, N_KV_HEADS, HEAD_DIM, CTX_CAP)
        .map_err(|e| anyhow::anyhow!("kv: {e}"))?;

    // Build a per-layer table for 1 layer × PE.
    let tok_row_f32: Vec<f32> = (0..PE * N_LAYERS).map(|i| (i as f32) * 1e-4).collect();
    let proj_f32: Vec<f32> = (0..PE * N_LAYERS * HIDDEN)
        .map(|i| ((i as f32 * 0.7).sin()) * 1e-3)
        .collect();
    let norm: Vec<f32> = vec![1.0; PE];
    let inp_batch_host: Vec<f16> = (0..HIDDEN)
        .map(|i| f16::from_f32((i as f32) * 1e-3 - 0.064))
        .collect();
    let table = build_inp_per_layer_table(
        bytemuck::cast_slice(&tok_row_f32),
        GgmlDType::F32,
        bytemuck::cast_slice(&proj_f32),
        GgmlDType::F32,
        bytemuck::cast_slice(&norm),
        &inp_batch_host,
        PE,
        N_LAYERS,
        HIDDEN,
        RMS_EPS,
    )?;
    let table_base = alloc_zeroed(&dev, table.len() * 4);
    upload_inp_per_layer_table(&dev, stream, &table, table_base)?;
    let slice_l0 = table_slice_ptr(table_base, 0, PE);

    let d_x_in = upload(&dev, &inp_batch_host);
    let d_x_out = alloc_zeroed(&dev, HIDDEN * 2);

    let mut positions_host = [0i32; 1];
    let mut scratch = build_scratch(&dev, &mut positions_host);

    forward_layer_decode(
        &ops, &dev, stream, &weights, &spec, RMS_EPS, FF_LEN, HIDDEN,
        &mut kv, &mut scratch, d_x_in, d_x_out, 0,
        Some((slice_l0, PE)),
        /*moe_scratch=*/ None,
    )?;
    stream.synchronize()?;

    let out_f16 = download_f16(&dev, d_x_out, HIDDEN);
    for (i, v) in out_f16.iter().enumerate() {
        let f = v.to_f32();
        assert!(f.is_finite(), "out[{i}] = {f} not finite");
    }
    Ok(())
}
