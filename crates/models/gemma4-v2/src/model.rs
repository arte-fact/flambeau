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

use anyhow::{bail, Result};
use flambeau_forward::ctx::ForwardCtx;

use crate::loader::Gemma4V2Model;

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
        ctx.per_layer_embd_build_table(
            &resid,
            &tok_rows_buf,
            globals.tok_embd_dtype,
            globals.tok_embd_row_bytes,
            globals.model_proj_f16_dev,
            globals.proj_matmul_f32_dev,
            &globals.proj_norm_raw,
            globals.table_dev,
            globals.pe,
            model.config.num_layers,
            model.config.hidden,
            model.config.rms_eps,
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
                globals.table_dev,
                li,
                globals.pe,
                n,
                n,
                model.config.rms_eps,
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
