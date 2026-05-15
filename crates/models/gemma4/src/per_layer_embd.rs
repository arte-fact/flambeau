//! Per-layer side-channel embedding (E2B / E4B only) — gemma4-specific
//! host-side build + GGUF coupling. The per-layer apply (block-shaped
//! kernel sequence) lives in `flambeau-blocks::per_layer_embd`; this
//! module re-exports the block types and adds the gemma4-specific
//! tensor names + table build.
//!
//! Two phases per forward step:
//!
//! 1. **Pre-loop projection** — once per token, host-side: dequant
//!    `per_layer_token_embd[token]` (Q5_K → F32) scaled by
//!    `sqrt(per_layer_embd)`, plus `per_layer_model_proj @ inp_batch`
//!    (BF16 × F16 → F32) scaled by `1/sqrt(n_embd)`, RMSNorm'd via
//!    `per_layer_proj_norm`, then `(table + proj) * (1/sqrt(2))`.
//!    Output: F32 `[n_layer × pe]`, uploaded once per token, sliced
//!    per layer at decode time.
//!
//! 2. **Per-layer post-block apply** — runs the
//!    [`flambeau_blocks::PerLayerEmbedBlock`] forward sequence
//!    (GELU(gate × pe_in) * table_slice → proj → norm → residual_add).

#![cfg(feature = "hip")]

use anyhow::{anyhow, bail, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, HipStream};
use flambeau_quant::{GgmlDType, TensorInfo};
use half::f16;

pub use flambeau_blocks::{
    per_layer_table_slice_ptr as table_slice_ptr, PerLayerEmbedBlock,
    PerLayerEmbedDecodeScratch, PerLayerEmbedLayerWeights,
};

use crate::weights_hip::DeviceTensor;

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
