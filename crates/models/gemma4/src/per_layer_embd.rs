//! Per-layer side-channel embedding (E2B / E4B only).
//!
//! Two phases per forward step:
//!
//! 1. **Pre-loop projection** — once per token, host-side: dequant
//!    `per_layer_token_embd[token]` (Q5_K → F32) scaled by
//!    `sqrt(per_layer_embd)`, plus `per_layer_model_proj @ inp_batch`
//!    (BF16 × F16 → F32) scaled by `1/sqrt(n_embd)`, RMSNorm'd via
//!    `per_layer_proj_norm`, then `(table + proj) * (1/sqrt(2))`.
//!    Output: F32 `[n_layer × pe]`, uploaded once per token, sliced
//!    per layer at decode time. Mirrors llama.cpp PR #21612 which
//!    moves this projection out of the layer loop.
//!
//! 2. **Per-layer post-block apply** — at the end of each layer:
//!    `pe_in = cur` (F16 hidden), then GELU(`inp_gate @ pe_in`) *
//!    `table_slice` → cast → `proj @ activated` → cast → `rmsnorm`
//!    with `post_norm` → `add pe_in`.

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, HipStream};
use flambeau_ops::Ops;
use flambeau_quant::{GgmlDType, TensorInfo};
use half::f16;

use crate::weights_hip::DeviceTensor;

/// Per-layer-embd weights for one layer. All F32 on disk; norm gets
/// cast to F16 at upload to match `rmsnorm_f16`'s contract.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedLayerWeights {
    /// F32 `[pe, hidden]`.
    pub inp_gate: DevicePtr,
    /// F32 `[hidden, pe]`.
    pub proj: DevicePtr,
    /// F16 `[hidden]` (cast from on-disk F32).
    pub post_norm_f16: DevicePtr,
}

/// Per-layer-embd globals (E2B / E4B only). For host-side build steps
/// we read directly from the GGUF mmap; the device-resident copy is
/// kept here for completeness and future on-device acceleration.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedGlobals {
    pub per_layer_token_embd_t: DeviceTensor,
    pub per_layer_model_proj_t: DeviceTensor,
    pub per_layer_proj_norm_t: DeviceTensor,
}

/// CPU dequant + projection. Mirrors gemma4-iswa.cpp:264-322 for
/// n_tokens = 1 (decode path). Output: `Vec<f32>` of length
/// `pe * n_layer`, laid out as `[layer][pe]` (layer-major).
#[allow(clippy::too_many_arguments)]
pub fn build_inp_per_layer_table(
    tok_embd_row_raw: &[u8],
    tok_embd_dtype: GgmlDType,
    model_proj_raw: &[u8],
    model_proj_dtype: GgmlDType,
    proj_norm_raw: &[u8],
    inp_batch_f16: &[f16],
    pe: usize,
    n_layer: usize,
    hidden: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>> {
    if inp_batch_f16.len() != hidden {
        bail!(
            "build_inp_per_layer_table: inp_batch len {} != hidden {hidden}",
            inp_batch_f16.len()
        );
    }
    let total = pe * n_layer;

    // 1. Dequant the per_layer_token_embd row → F32 [pe*n_layer], scale by sqrt(pe).
    let table = if tok_embd_dtype == GgmlDType::F32 {
        let bytes_needed = total * 4;
        if tok_embd_row_raw.len() < bytes_needed {
            bail!(
                "tok_embd row {} < expected {}",
                tok_embd_row_raw.len(),
                bytes_needed
            );
        }
        bytemuck::cast_slice::<u8, f32>(&tok_embd_row_raw[..bytes_needed]).to_vec()
    } else {
        flambeau_quant::dequantize_to_vec(tok_embd_dtype, tok_embd_row_raw, total)
            .map_err(|e| anyhow!("dequant per_layer_token_embd row: {e}"))?
    };
    let pe_sqrt = (pe as f32).sqrt();
    let mut table: Vec<f32> = table.iter().map(|v| *v * pe_sqrt).collect();

    // 2. proj = per_layer_model_proj @ inp_batch_f16 → F32 [pe*n_layer].
    let mut proj = vec![0.0f32; total];
    match model_proj_dtype {
        GgmlDType::F32 => {
            let need = total * hidden * 4;
            if model_proj_raw.len() < need {
                bail!(
                    "per_layer_model_proj {} < expected {}",
                    model_proj_raw.len(),
                    need
                );
            }
            let w: &[f32] = bytemuck::cast_slice(&model_proj_raw[..need]);
            for row in 0..total {
                let mut acc = 0.0f64;
                for col in 0..hidden {
                    acc += (w[row * hidden + col] * inp_batch_f16[col].to_f32()) as f64;
                }
                proj[row] = acc as f32;
            }
        }
        GgmlDType::BF16 => {
            let need = total * hidden * 2;
            if model_proj_raw.len() < need {
                bail!(
                    "per_layer_model_proj {} < expected {}",
                    model_proj_raw.len(),
                    need
                );
            }
            let w: &[u16] = bytemuck::cast_slice(&model_proj_raw[..need]);
            for row in 0..total {
                let mut acc = 0.0f64;
                for col in 0..hidden {
                    let bf = w[row * hidden + col];
                    let bits = (bf as u32) << 16;
                    let wv = f32::from_bits(bits);
                    acc += (wv * inp_batch_f16[col].to_f32()) as f64;
                }
                proj[row] = acc as f32;
            }
        }
        other => bail!(
            "per_layer_model_proj dtype {other:?} not supported (expected F32 or BF16)"
        ),
    }

    // 3. proj *= 1/sqrt(n_embd).
    let inv_sqrt_n = 1.0 / (hidden as f32).sqrt();
    for v in proj.iter_mut() {
        *v *= inv_sqrt_n;
    }

    // 4. proj = rmsnorm(proj, per_layer_proj_norm). Treats proj as
    //    [n_layer, pe] — n_rows = n_layer, each row k = pe.
    if proj_norm_raw.len() < pe * 4 {
        bail!(
            "per_layer_proj_norm {} < pe*4={}",
            proj_norm_raw.len(),
            pe * 4
        );
    }
    let proj_norm: &[f32] = bytemuck::cast_slice(&proj_norm_raw[..pe * 4]);
    for il in 0..n_layer {
        let row = &mut proj[il * pe..(il + 1) * pe];
        let mut ss = 0.0f64;
        for &v in row.iter() {
            ss += (v as f64) * (v as f64);
        }
        let inv_rms = 1.0 / ((ss / pe as f64).sqrt() + rms_norm_eps as f64);
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v as f64) * inv_rms * proj_norm[i] as f64) as f32;
        }
    }

    // 5. inp_per_layer = (table + proj) * (1/sqrt(2)).
    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
    for (t, p) in table.iter_mut().zip(proj.iter()) {
        *t = (*t + *p) * inv_sqrt_2;
    }
    Ok(table)
}

/// Upload an `[n_layer × pe]` F32 table to a pre-allocated device
/// buffer.
pub fn upload_inp_per_layer_table(
    device: &HipDevice,
    stream: &HipStream,
    table_f32: &[f32],
    dst: DevicePtr,
) -> Result<()> {
    let bytes = std::mem::size_of_val(table_f32);
    // SAFETY: dst sized for `bytes`; `table_f32` outlives the bounded sync.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            dst,
            DevicePtr(table_f32.as_ptr() as usize),
            bytes,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Per-layer post-block side-channel apply. Writes the new residual
/// to `x_out`. Caller pre-allocates the seven scratch buffers.
#[allow(clippy::too_many_arguments)]
pub fn forward_per_layer_post_block<O: Ops>(
    ops: &O,
    weights: PerLayerEmbedLayerWeights,
    pe_in: DevicePtr,
    table_slice: DevicePtr,
    gate_out_f32: DevicePtr,
    activated_f32: DevicePtr,
    activated_f16: DevicePtr,
    proj_out_f32: DevicePtr,
    proj_out_f16: DevicePtr,
    normed_f16: DevicePtr,
    x_out: DevicePtr,
    pe: usize,
    hidden: usize,
    rms_norm_eps: f32,
) -> Result<()> {
    ops.dense_gemv_f32_f16(weights.inp_gate, pe_in, gate_out_f32, pe, hidden)
        .context("per_layer_embd inp_gate")?;
    ops.gelu_mul_f32(gate_out_f32, table_slice, activated_f32, pe)
        .context("per_layer_embd gelu_mul")?;
    ops.cast_f32_to_f16(activated_f32, activated_f16, pe)
        .context("per_layer_embd cast activated → f16")?;
    ops.dense_gemv_f32_f16(weights.proj, activated_f16, proj_out_f32, hidden, pe)
        .context("per_layer_embd proj")?;
    ops.cast_f32_to_f16(proj_out_f32, proj_out_f16, hidden)
        .context("per_layer_embd cast proj → f16")?;
    ops.rmsnorm_f16(
        proj_out_f16,
        weights.post_norm_f16,
        normed_f16,
        1,
        hidden,
        rms_norm_eps,
    )
    .context("per_layer_embd post_norm")?;
    ops.add_f16(pe_in, normed_f16, x_out, hidden)
        .context("per_layer_embd residual add")?;
    Ok(())
}

/// Compute the device pointer for `inp_per_layer_table[il]`.
pub fn table_slice_ptr(table_base: DevicePtr, il: usize, pe: usize) -> DevicePtr {
    table_base.offset_bytes(il * pe * 4)
}

/// Helper: raw byte width of one `per_layer_token_embd` row.
pub fn per_layer_token_embd_row_bytes(t: &TensorInfo) -> Result<usize> {
    let bs = t.dtype.block_size() as usize;
    let ts = t.dtype.type_size() as usize;
    let row_elems = t.dims.get(1).copied().unwrap_or(0) as usize;
    if row_elems == 0 || row_elems % bs != 0 {
        bail!(
            "per_layer_token_embd row width {row_elems} % block_size {bs} != 0 for {:?}",
            t.dtype
        );
    }
    Ok((row_elems / bs) * ts)
}
