//! Per-layer side-channel embedding block.
//!
//! Gemma 4 E2B / E4B applies a token-derived F32 side-channel table
//! to each layer's output as a residual delta:
//!
//! ```text
//! gate_f32      = inp_gate @ pe_in                 (dense_gemv_f32_f16)
//! activated_f32 = gelu(gate_f32) * table_slice     (gelu_mul_f32)
//! activated_f16 = cast(activated_f32)              (cast_f32_to_f16)
//! proj_f32      = proj @ activated_f16             (dense_gemv_f32_f16)
//! proj_f16      = cast(proj_f32)                   (cast_f32_to_f16)
//! normed_f16    = rmsnorm_f16(proj_f16, post_norm) (rmsnorm_f16)
//! x_out         = pe_in + normed_f16               (add_f16)
//! ```
//!
//! The per-token build of `inp_per_layer_table[]` (Q5_K dequant +
//! BF16 matmul + rmsnorm + add) is the host-side function
//! [`build_inp_per_layer_table`] below — model crates pass the GGUF
//! mmap slices for `per_layer_token_embd`, `per_layer_model_proj`,
//! `per_layer_proj_norm` plus the current token's F16 main embedding
//! and get back a `[n_layer * pe]` F32 table to upload once per token.

use anyhow::{anyhow, bail, Context, Result};
use flambeau_core::DevicePtr;
use flambeau_ops::Ops;
use flambeau_quant::GgmlDType;
use half::f16;

/// Per-layer weights for the side-channel apply. All F32 on disk;
/// `post_norm_f16` gets cast to F16 at upload time to match
/// `rmsnorm_f16`'s contract.
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedLayerWeights {
    /// F32 `[pe, hidden]`.
    pub inp_gate: DevicePtr,
    /// F32 `[hidden, pe]`.
    pub proj: DevicePtr,
    /// F16 `[hidden]` (cast from on-disk F32).
    pub post_norm_f16: DevicePtr,
}

/// Per-call scratch buffers — caller-allocated, reused per layer.
/// Sized so `gate_out_f32`, `activated_f32`, `activated_f16` cover
/// `pe` elements; `proj_out_f32`, `proj_out_f16`, `normed_f16` cover
/// `hidden`. The `_f32` / `_f16` sibling slots may overlap with the
/// layer's FFN scratch when the FFN intermediate buffers happen to be
/// large enough (the gemma4 PP path reuses `scratch.gate_f32` etc.).
#[derive(Debug, Clone, Copy)]
pub struct PerLayerEmbedDecodeScratch {
    pub gate_out_f32: DevicePtr,
    pub activated_f32: DevicePtr,
    pub activated_f16: DevicePtr,
    pub proj_out_f32: DevicePtr,
    pub proj_out_f16: DevicePtr,
    pub normed_f16: DevicePtr,
}

/// Side-channel embedding block. Holds the per-layer weights plus the
/// shape config; constructed once per layer at upload time and reused
/// across forward calls.
pub struct PerLayerEmbedBlock {
    pub weights: PerLayerEmbedLayerWeights,
    pub pe: usize,
    pub hidden: usize,
    pub rms_norm_eps: f32,
}

impl PerLayerEmbedBlock {
    pub fn new(
        weights: PerLayerEmbedLayerWeights,
        pe: usize,
        hidden: usize,
        rms_norm_eps: f32,
    ) -> Self {
        Self {
            weights,
            pe,
            hidden,
            rms_norm_eps,
        }
    }

    /// Apply the side-channel post-block. Reads `pe_in` (F16 `[hidden]`,
    /// the layer's output residual) plus `table_slice` (F32 `[pe]`, this
    /// layer's slice of the prebuilt per-layer table) and writes the
    /// updated residual to `x_out` (F16 `[hidden]`). `pe_in` and `x_out`
    /// may alias for in-place update.
    pub fn forward_decode<O: Ops>(
        &self,
        ops: &O,
        pe_in: DevicePtr,
        table_slice: DevicePtr,
        scratch: PerLayerEmbedDecodeScratch,
        x_out: DevicePtr,
    ) -> Result<()> {
        self.forward_n_tokens(ops, pe_in, table_slice, scratch, x_out, 1)
    }

    /// Generalised forward: each token in `pe_in` (F16
    /// `[n_tokens, hidden]`) gets the side-channel residual applied
    /// against `table_slice` (F32 `[n_tokens, pe]`, this layer's
    /// contiguous slice of the prebuilt layer-major
    /// `[n_layer, n_tokens, pe]` table). `pe_in` and `x_out` may alias.
    pub fn forward_n_tokens<O: Ops>(
        &self,
        ops: &O,
        pe_in: DevicePtr,
        table_slice: DevicePtr,
        scratch: PerLayerEmbedDecodeScratch,
        x_out: DevicePtr,
        n_tokens: usize,
    ) -> Result<()> {
        let PerLayerEmbedDecodeScratch {
            gate_out_f32,
            activated_f32,
            activated_f16,
            proj_out_f32,
            proj_out_f16,
            normed_f16,
        } = scratch;
        if n_tokens == 1 {
            ops.dense_gemv_f32_f16(
                self.weights.inp_gate,
                pe_in,
                gate_out_f32,
                self.pe,
                self.hidden,
            )
            .context("per_layer_embd inp_gate")?;
        } else {
            ops.dense_gemv_f32_f16_batched(
                self.weights.inp_gate,
                pe_in,
                gate_out_f32,
                self.pe,
                self.hidden,
                n_tokens,
            )
            .context("per_layer_embd inp_gate batched")?;
        }
        ops.gelu_mul_f32(gate_out_f32, table_slice, activated_f32, n_tokens * self.pe)
            .context("per_layer_embd gelu_mul")?;
        ops.cast_f32_to_f16(activated_f32, activated_f16, n_tokens * self.pe)
            .context("per_layer_embd cast activated → f16")?;
        if n_tokens == 1 {
            ops.dense_gemv_f32_f16(
                self.weights.proj,
                activated_f16,
                proj_out_f32,
                self.hidden,
                self.pe,
            )
            .context("per_layer_embd proj")?;
        } else {
            ops.dense_gemv_f32_f16_batched(
                self.weights.proj,
                activated_f16,
                proj_out_f32,
                self.hidden,
                self.pe,
                n_tokens,
            )
            .context("per_layer_embd proj batched")?;
        }
        // Fused F32→F16 rmsnorm + add residual: replaces
        // (cast_f32_to_f16 → rmsnorm_f16 → add_f16). One kernel
        // launch instead of three per layer per token.
        let _ = proj_out_f16;
        let _ = normed_f16;
        ops.rmsnorm_f32_to_f16_add_residual(
            proj_out_f32,
            self.weights.post_norm_f16,
            pe_in,
            x_out,
            n_tokens,
            self.hidden,
            self.rms_norm_eps,
        )
        .context("per_layer_embd fused post_norm + residual")?;
        Ok(())
    }
}

/// Compute the device pointer for `inp_per_layer_table[il]` — `il * pe`
/// F32 elements into `table_base`. Helper kept on the block module for
/// callers that compute per-layer slices outside the forward call.
pub fn table_slice_ptr(table_base: DevicePtr, il: usize, pe: usize) -> DevicePtr {
    table_base.offset_bytes(il * pe * 4)
}

/// Host-side build with the `per_layer_model_proj @ inp_batch` matmul
/// PRECOMPUTED, generalised to `n_tokens >= 1`. The caller supplies
/// `proj_matmul_f32` of length `n_tokens * pe * n_layer` in TOKEN-MAJOR
/// layout `[n_tokens, n_layer * pe]` (the natural output of
/// `dense_gemv_f16_f16_batched`).
///
/// `tok_embd_rows_raw` is `n_tokens` consecutive `per_layer_token_embd`
/// rows (`row_bytes` each). The output is laid out LAYER-MAJOR
/// `[n_layer, n_tokens, pe]` so the per-layer apply can slice
/// `&table[il * n_tokens * pe..]` as a contiguous `[n_tokens, pe]`
/// block for `dense_gemv_*_batched` + `gelu_mul_f32` consumers.
///
/// Math per (token, layer) matches the single-token build:
/// 1. table = dequant(per_layer_token_embd[token, layer, :]) * sqrt(pe)
/// 2. proj  = proj_matmul_f32[token, layer, :] * (1 / sqrt(hidden))
/// 3. proj  = rmsnorm(proj, per_layer_proj_norm)
/// 4. out[layer, token, :] = (table + proj) * (1 / sqrt(2))
pub fn build_inp_per_layer_table_with_proj(
    tok_embd_rows_raw: &[u8],
    tok_embd_dtype: GgmlDType,
    tok_embd_row_bytes: usize,
    proj_matmul_f32: &[f32],
    proj_norm_raw: &[u8],
    pe: usize,
    n_layer: usize,
    n_tokens: usize,
    hidden: usize,
    rms_norm_eps: f32,
) -> Result<Vec<f32>> {
    let per_token = pe * n_layer;
    let total = n_tokens * per_token;
    if proj_matmul_f32.len() != total {
        bail!(
            "build_inp_per_layer_table_with_proj: proj len {} != n_tokens*pe*n_layer {}",
            proj_matmul_f32.len(),
            total
        );
    }
    if tok_embd_rows_raw.len() < n_tokens * tok_embd_row_bytes {
        bail!(
            "tok_embd rows len {} < expected {} (n_tokens={n_tokens} * row_bytes={tok_embd_row_bytes})",
            tok_embd_rows_raw.len(),
            n_tokens * tok_embd_row_bytes
        );
    }
    if proj_norm_raw.len() < pe * 4 {
        bail!(
            "per_layer_proj_norm {} < pe*4={}",
            proj_norm_raw.len(),
            pe * 4
        );
    }
    let proj_norm: &[f32] = bytemuck::cast_slice(&proj_norm_raw[..pe * 4]);
    let pe_sqrt = (pe as f32).sqrt();
    let inv_sqrt_hidden = 1.0 / (hidden as f32).sqrt();
    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();

    let mut out = vec![0.0f32; total];
    for t in 0..n_tokens {
        let row_off = t * tok_embd_row_bytes;
        let row_raw = &tok_embd_rows_raw[row_off..row_off + tok_embd_row_bytes];
        let row_table = if tok_embd_dtype == GgmlDType::F32 {
            bytemuck::cast_slice::<u8, f32>(&row_raw[..per_token * 4]).to_vec()
        } else {
            flambeau_quant::dequantize_to_vec(tok_embd_dtype, row_raw, per_token)
                .map_err(|e| anyhow!("dequant per_layer_token_embd row {t}: {e}"))?
        };
        let proj_token = &proj_matmul_f32[t * per_token..(t + 1) * per_token];
        for il in 0..n_layer {
            let mut proj_row = [0.0f32; 256];
            let proj_slice = &proj_token[il * pe..(il + 1) * pe];
            if pe > proj_row.len() {
                bail!(
                    "build_inp_per_layer_table_with_proj: pe {pe} > scratch {}",
                    proj_row.len()
                );
            }
            for i in 0..pe {
                proj_row[i] = proj_slice[i] * inv_sqrt_hidden;
            }
            let mut ss = 0.0f64;
            for v in &proj_row[..pe] {
                ss += (*v as f64) * (*v as f64);
            }
            let inv_rms = 1.0 / ((ss / pe as f64).sqrt() + rms_norm_eps as f64);
            let dst_base = il * n_tokens * pe + t * pe;
            for i in 0..pe {
                let table_v = row_table[il * pe + i] * pe_sqrt;
                let proj_v = (proj_row[i] as f64 * inv_rms * proj_norm[i] as f64) as f32;
                out[dst_base + i] = (table_v + proj_v) * inv_sqrt_2;
            }
        }
    }
    Ok(out)
}

/// Host-side build of `inp_per_layer_table` for the current token.
///
/// Mirrors llama.cpp's `project_per_layer_inputs` for n_tokens = 1:
///
/// ```text
/// table = dequant(per_layer_token_embd[token]) * sqrt(pe)
/// proj  = per_layer_model_proj @ inp_batch_f16 * (1 / sqrt(hidden))
/// proj  = rmsnorm(proj_view[layer, pe], per_layer_proj_norm)   per layer
/// out   = (table + proj) * (1 / sqrt(2))
/// ```
///
/// `tok_embd_row_raw` is the slice for a single token row of
/// `per_layer_token_embd` (`pe * n_layer` elements after dequant).
/// `model_proj_raw` covers the full `[pe * n_layer, hidden]` matrix
/// (BF16 or F32 on disk). `proj_norm_raw` is the `[pe]` F32 norm
/// weight. Output layout: layer-major, `out[il * pe + i]`.
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
        other => bail!("per_layer_model_proj dtype {other:?} not supported (expected F32 or BF16)"),
    }

    let inv_sqrt_n = 1.0 / (hidden as f32).sqrt();
    for v in proj.iter_mut() {
        *v *= inv_sqrt_n;
    }

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

    let inv_sqrt_2 = 1.0 / 2.0f32.sqrt();
    for (t, p) in table.iter_mut().zip(proj.iter()) {
        *t = (*t + *p) * inv_sqrt_2;
    }
    Ok(table)
}
