//! Gemma-4 forward. Every layer is FullAttn. FFN dispatches dense vs
//! MoE based on `model.config.moe`. The MoE path uses the 5-norm F32
//! cascade in `flambeau-forward::core::composites::moe_ffn` (gemma4
//! branch), which is activated when `MoeWeights::pre_router_weight_f16`
//! is `Some`.
//!
//! E2B / E4B (`config.per_layer_embd.is_some()`): a per-layer
//! side-channel embedding is mixed into the residual after the FFN
//! residual add. The side-channel table is rebuilt host-side once per
//! token, then sliced per layer.

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
    let has_per_layer_embd = model.config.per_layer_embd.is_some();
    // First-ship E2B / E4B prefill: loop the whole forward token-by-
    // token. The per-token table build + per-layer apply require n=1;
    // batching `n > 1` is a follow-up (n-token table build + an apply
    // generalised to read `inp_per_layer[token, layer, :]` per row).
    if has_per_layer_embd && n > 1 {
        for i in 0..n {
            forward(model, ctx, &tokens[i..i + 1], &positions[i..i + 1], &slot_ids[i..i + 1])?;
        }
        return Ok(());
    }
    let mut resid = ctx.embed(&model.embedding, tokens)?;

    // gemma 4n / E2B / E4B: rebuild the per-token side-channel table.
    // `model.embedding` for n=1 has been written into `resid` with the
    // sqrt(n_embd) scale already applied — that's the same `inp_batch`
    // llama.cpp's `project_per_layer_inputs` consumes.
    if let Some(globals) = model.per_layer_embd_globals.as_ref() {
        let token = tokens[0] as usize;
        let row_off = token * globals.tok_embd_row_bytes;
        let row_end = row_off + globals.tok_embd_row_bytes;
        if row_end > globals.tok_embd_raw.len() {
            bail!(
                "per_layer_token_embd row OOB at token {token}: {row_end} > {}",
                globals.tok_embd_raw.len()
            );
        }
        let row = &globals.tok_embd_raw[row_off..row_end];
        ctx.per_layer_embd_build_table(
            &resid,
            row,
            globals.tok_embd_dtype,
            &globals.model_proj_raw,
            globals.model_proj_dtype,
            &globals.proj_norm_raw,
            globals.table_dev,
            globals.pe,
            model.config.num_layers,
            model.config.hidden,
            model.config.rms_eps,
        )?;
    }

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
                model.config.rms_eps,
            )?;
        }

        if let Some(scale) = model.layer_output_scale[li] {
            resid = ctx.scale_inplace_f16(resid, scale, n)?;
        }
    }
    ctx.output_head(&resid, &model.lm_head, slot_ids)?;
    Ok(())
}
