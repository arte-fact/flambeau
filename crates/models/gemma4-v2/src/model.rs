//! Gemma-4 forward. Every layer is FullAttn. FFN dispatches dense vs
//! MoE based on `model.config.moe`. The MoE path uses the 5-norm F32
//! cascade in `flambeau-forward::core::composites::moe_ffn` (gemma4
//! branch), which is activated when `MoeWeights::pre_router_weight_f16`
//! is `Some`.
//!
//! E2B / E4B (`config.per_layer_embd.is_some()`): a per-layer
//! side-channel embedding is mixed into the residual after the FFN
//! residual add. The side-channel table is built once per forward call
//! over all n_tokens (prefill or decode) and laid out
//! `[n_layer, n_tokens, pe]` so each layer's apply reads a contiguous
//! `[n_tokens, pe]` slice.

use anyhow::{anyhow, bail, Context, Result};
use flambeau_forward::ctx::ForwardCtx;
use flambeau_quant::GgmlDType;
use half::f16;

use crate::loader::{Gemma4V2Model, PerLayerEmbdGlobals};

pub fn forward<C: ForwardCtx>(
    model: &Gemma4V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
) -> Result<()> {
    let n = tokens.len();
    let is_moe = model.config.moe.is_some();
    let mut resid = ctx.embed(&model.embedding, tokens)?;

    // Build the side-channel table once for all n_tokens. The rows are
    // assembled per-token in the order of `tokens`; the GPU matmul +
    // host finishing produces a layer-major
    // `[n_layer, n_tokens, pe]` table at `globals.table_dev`.
    //
    // Every PP rank does its own build: PP rank > 0's `resid` (from
    // `ctx.embed` → `peer_recv`) is the post-prior-layers hidden, not
    // the post-embed hidden the side-channel matmul needs. Building
    // host-side from `globals.main_embd_token_embd_raw` keeps every
    // rank's per_layer table coherent.
    let mut tok_rows_buf: Vec<u8> = Vec::new();
    if let Some(globals) = model.per_layer_embd_globals.as_ref() {
        tok_rows_buf.reserve_exact(n * globals.tok_embd_row_bytes);
        for &t in tokens {
            let token = t as usize;
            let row_off = token * globals.tok_embd_row_bytes;
            let row_end = row_off + globals.tok_embd_row_bytes;
            if row_end > globals.tok_embd_raw.len() {
                bail!(
                    "per_layer_token_embd row OOB at token {token}: {row_end} > {}",
                    globals.tok_embd_raw.len()
                );
            }
            tok_rows_buf.extend_from_slice(&globals.tok_embd_raw[row_off..row_end]);
        }
        let main_embd_host = build_main_embd_host_f16(globals, tokens)?;
        ctx.per_layer_embd_build_table(
            flambeau_forward::per_layer_embd::PerLayerBuildTableSpec {
                main_embd_host_f16: &main_embd_host,
                main_embd_scratch_dev: globals.main_embd_scratch_dev,
                tok_embd_rows_raw: &tok_rows_buf,
                tok_embd_dtype: globals.tok_embd_dtype,
                tok_embd_row_bytes: globals.tok_embd_row_bytes,
                model_proj_f16_dev: globals.model_proj_f16_dev,
                proj_matmul_f32_dev: globals.proj_matmul_f32_dev,
                proj_norm_raw: &globals.proj_norm_raw,
                table_dev: globals.table_dev,
                pe: globals.pe,
                n_layer: model.config.num_layers,
                hidden: model.config.hidden,
                rms_eps: model.config.rms_eps,
            },
        )?;
    }

    // Diagnostic: bail after layer N (env FLAMBEAU_GEMMA4_BAIL_AFTER_LAYER=N).
    // Used to localize which layer first produces degenerate output during
    // the 31B regression hunt. Tail-call output_head on the partial resid.
    let bail_after: Option<usize> = std::env::var("FLAMBEAU_GEMMA4_BAIL_AFTER_LAYER")
        .ok()
        .and_then(|s| s.parse().ok());

    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = model.attn[li]
            .as_ref()
            .expect("attn weights missing for owned layer (PP slice mismatch)");

        let delta = ctx.standard_attn(&resid, attn_w, li, positions, slot_ids, None)?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let delta = if is_moe {
            let moe_w = model.moe[li]
                .as_ref()
                .expect("moe weights missing for owned MoE layer (PP slice mismatch)");
            ctx.moe_ffn(&resid, moe_w, n, None)?
        } else {
            let ffn_w = model.ffn[li]
                .as_ref()
                .expect("ffn weights missing for owned dense layer (PP slice mismatch)");
            ctx.dense_ffn(&resid, ffn_w, n, None)?
        };
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        // Per-layer side-channel embedding apply (E2B / E4B). Sits
        // between the FFN residual add and the `layer_output_scale`.
        if let (Some(globals), Some(pe_w)) = (
            model.per_layer_embd_globals.as_ref(),
            model.per_layer_embd.get(li).and_then(|o| o.as_ref()),
        ) {
            ctx.per_layer_embd_apply(
                &mut resid,
                pe_w,
                flambeau_forward::per_layer_embd::PerLayerApplySpec {
                    table_dev: globals.table_dev,
                    layer_idx: li,
                    pe: globals.pe,
                    n_tokens: n,
                    n_tokens_total: n,
                    rms_eps: model.config.rms_eps,
                },
            )?;
        }

        if let Some(scale) = model.layer_output_scale[li] {
            resid = ctx.scale_inplace_f16(resid, scale, n)?;
        }

        if Some(li) == bail_after {
            ctx.output_head(&resid, &model.lm_head, slot_ids)?;
            return Ok(());
        }
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}

/// Sarathi-Serve mixed-batch forward. Rows `[0..prefill_rows)` are
/// a prefill chunk for `slot_ids[0]` at contiguous positions; rows
/// `[prefill_rows..n)` are batched decodes across N distinct slots.
/// Routes attention through `standard_attn_mixed`; embed / FFN /
/// residual_add / per_layer_embd_apply / scale_inplace all run at
/// `n = K + N` (none of them depend on row identity beyond what's
/// already in `tokens`/`positions`). Output head emits `N + 1` rows:
/// the (K-1)-th prefill row + the N decode rows.
pub fn forward_mixed<C: ForwardCtx>(
    model: &Gemma4V2Model,
    ctx: &mut C,
    tokens: &[u32],
    positions: &[usize],
    slot_ids: &[usize],
    prefill_rows: usize,
) -> Result<()> {
    let n = tokens.len();
    if prefill_rows == 0 || prefill_rows >= n {
        bail!(
            "forward_mixed: prefill_rows must satisfy 0 < K < n (got K={prefill_rows}, n={n})"
        );
    }
    let is_moe = model.config.moe.is_some();
    let mut resid = ctx.embed(&model.embedding, tokens)?;

    let mut tok_rows_buf: Vec<u8> = Vec::new();
    if let Some(globals) = model.per_layer_embd_globals.as_ref() {
        tok_rows_buf.reserve_exact(n * globals.tok_embd_row_bytes);
        for &t in tokens {
            let token = t as usize;
            let row_off = token * globals.tok_embd_row_bytes;
            let row_end = row_off + globals.tok_embd_row_bytes;
            if row_end > globals.tok_embd_raw.len() {
                bail!(
                    "per_layer_token_embd row OOB at token {token}: {row_end} > {}",
                    globals.tok_embd_raw.len()
                );
            }
            tok_rows_buf.extend_from_slice(&globals.tok_embd_raw[row_off..row_end]);
        }
        let main_embd_host = build_main_embd_host_f16(globals, tokens)?;
        ctx.per_layer_embd_build_table(
            flambeau_forward::per_layer_embd::PerLayerBuildTableSpec {
                main_embd_host_f16: &main_embd_host,
                main_embd_scratch_dev: globals.main_embd_scratch_dev,
                tok_embd_rows_raw: &tok_rows_buf,
                tok_embd_dtype: globals.tok_embd_dtype,
                tok_embd_row_bytes: globals.tok_embd_row_bytes,
                model_proj_f16_dev: globals.model_proj_f16_dev,
                proj_matmul_f32_dev: globals.proj_matmul_f32_dev,
                proj_norm_raw: &globals.proj_norm_raw,
                table_dev: globals.table_dev,
                pe: globals.pe,
                n_layer: model.config.num_layers,
                hidden: model.config.hidden,
                rms_eps: model.config.rms_eps,
            },
        )?;
    }

    let layers: Vec<usize> = ctx.layer_range(&model.layout).collect();
    for li in layers {
        let attn_w = model.attn[li]
            .as_ref()
            .expect("attn weights missing for owned layer (PP slice mismatch)");

        let delta = ctx.standard_attn_mixed(
            &resid,
            attn_w,
            li,
            flambeau_forward::core::MixedBatch { positions, slot_ids, prefill_rows },
            None,
        )?;
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        let delta = if is_moe {
            let moe_w = model.moe[li]
                .as_ref()
                .expect("moe weights missing for owned MoE layer (PP slice mismatch)");
            ctx.moe_ffn(&resid, moe_w, n, None)?
        } else {
            let ffn_w = model.ffn[li]
                .as_ref()
                .expect("ffn weights missing for owned dense layer (PP slice mismatch)");
            ctx.dense_ffn(&resid, ffn_w, n, None)?
        };
        if let Some(d) = delta {
            resid = ctx.residual_add(resid, d, n)?;
        }

        if let (Some(globals), Some(pe_w)) = (
            model.per_layer_embd_globals.as_ref(),
            model.per_layer_embd.get(li).and_then(|o| o.as_ref()),
        ) {
            ctx.per_layer_embd_apply(
                &mut resid,
                pe_w,
                flambeau_forward::per_layer_embd::PerLayerApplySpec {
                    table_dev: globals.table_dev,
                    layer_idx: li,
                    pe: globals.pe,
                    n_tokens: n,
                    n_tokens_total: n,
                    rms_eps: model.config.rms_eps,
                },
            )?;
        }

        if let Some(scale) = model.layer_output_scale[li] {
            resid = ctx.scale_inplace_f16(resid, scale, n)?;
        }
    }

    // Output head emits the (K-1)-th prefill row + N decode rows
    // via a sliced residual view + sliced slot_ids. Mirrors
    // `qwen35-v2::forward_mixed`.
    let hidden = model.layout.hidden;
    let n_emit = n - prefill_rows + 1;
    let emit_ptr = resid.ptr.offset_bytes((prefill_rows - 1) * hidden * 2);
    let emit_input = unsafe {
        flambeau_model_ops::Tensor::<flambeau_model_ops::F16>::from_raw(emit_ptr, n_emit * hidden)
    };
    let mut emit_slots: Vec<usize> = Vec::with_capacity(n_emit);
    emit_slots.push(slot_ids[0]);
    emit_slots.extend_from_slice(&slot_ids[prefill_rows..]);
    ctx.output_head(&emit_input, &model.lm_head, &emit_slots)?;
    Ok(())
}

fn build_main_embd_host_f16(
    globals: &PerLayerEmbdGlobals,
    tokens: &[u32],
) -> Result<Vec<f16>> {
    let hidden = globals.main_embd_hidden;
    let dtype = globals.main_embd_token_embd_dtype;
    let raw = &globals.main_embd_token_embd_raw;
    let bs = dtype.block_size();
    let ts = dtype.type_size();
    if hidden % bs != 0 {
        bail!("token_embd hidden {hidden} % block_size {bs} != 0 for {dtype:?}");
    }
    let row_bytes = (hidden / bs) * ts;
    let scale = globals.main_embd_post_scale.unwrap_or(1.0);
    let n = tokens.len();
    let mut out: Vec<f16> = Vec::with_capacity(n * hidden);
    let mut row_f32 = vec![0.0f32; hidden];
    for &t in tokens {
        let token = t as usize;
        let row_off = token * row_bytes;
        let row_end = row_off + row_bytes;
        if row_end > raw.len() {
            bail!(
                "token_embd row OOB at token {token}: {row_end} > {}",
                raw.len()
            );
        }
        if dtype == GgmlDType::F32 {
            let src: &[f32] = bytemuck::cast_slice(&raw[row_off..row_end]);
            for (i, &v) in src.iter().enumerate() {
                row_f32[i] = v;
            }
        } else if dtype == GgmlDType::F16 {
            let src: &[f16] = bytemuck::cast_slice(&raw[row_off..row_end]);
            for (i, &v) in src.iter().enumerate() {
                row_f32[i] = v.to_f32();
            }
        } else {
            flambeau_quant::dequantize_into(dtype, &raw[row_off..row_end], &mut row_f32)
                .map_err(|e| anyhow!("dequant token_embd row {token}: {e}"))
                .context("token_embd host dequant")?;
        }
        for &v in &row_f32 {
            out.push(f16::from_f32(v * scale));
        }
    }
    Ok(out)
}
