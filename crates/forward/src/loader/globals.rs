//! Embedding + LM-head: the book-end "global" layer weights, always
//! replicated across ranks.

use anyhow::Result;
use flambeau_core::{Device, DevicePtr};
use flambeau_quant::GgufFile;

use crate::ctx::{EmbeddingWeights, LmHeadWeights};

use super::primitives::upload_dequant_to_f16;
use super::shard::upload_quant_weight;

pub struct EmbeddingSpec<'a> {
    pub token_embd_name: &'a str,
    pub vocab_size: usize,
    pub hidden: usize,
    /// Gemma4 sets `Some(sqrt(hidden))`; qwen leaves `None`.
    pub post_scale: Option<f32>,
}

pub fn load_embedding(
    file: &GgufFile,
    device: &impl Device,
    spec: &EmbeddingSpec,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<EmbeddingWeights> {
    let token_embd = upload_dequant_to_f16(
        file,
        device,
        spec.token_embd_name,
        spec.vocab_size * spec.hidden,
        allocs,
    )?;
    Ok(EmbeddingWeights {
        token_embd,
        vocab_size: spec.vocab_size,
        hidden: spec.hidden,
        post_scale: spec.post_scale,
    })
}

pub struct LmHeadSpec<'a> {
    pub output_norm_name: &'a str,
    /// For tied LM heads, point this at `token_embd.weight`.
    pub lm_head_name: &'a str,
    pub vocab_size: usize,
    pub hidden: usize,
    pub rms_eps: f32,
    /// Gemma4: `Some(30.0)`. Qwen / others: `None`.
    pub final_logit_softcap: Option<f32>,
}

pub fn load_lm_head(
    file: &GgufFile,
    device: &impl Device,
    spec: &LmHeadSpec,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<LmHeadWeights> {
    let output_norm =
        upload_dequant_to_f16(file, device, spec.output_norm_name, spec.hidden, allocs)?;
    let lm_head = upload_quant_weight(
        file,
        device,
        spec.lm_head_name,
        spec.vocab_size * spec.hidden,
        allocs,
    )?;
    Ok(LmHeadWeights {
        output_norm,
        lm_head,
        final_logit_softcap: spec.final_logit_softcap,
        vocab_size: spec.vocab_size,
        hidden: spec.hidden,
        rms_eps: spec.rms_eps,
    })
}
