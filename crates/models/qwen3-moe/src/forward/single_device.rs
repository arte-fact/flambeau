//! Mesh&lt;1&gt; (single-device) forward entry points — decode and prefill.
//!
//! The pipeline-parallel (Mesh&lt;N&gt; for N > 1) counterparts live in
//! `forward::pp`. Both sides compose the same per-layer helpers from
//! `forward::layer`; the difference is only in the outer loop (single
//! device walks every layer, PP walks a rank-local layer range and hands
//! the hidden state off to the next rank via `HipCluster`).

#![cfg(feature = "hip")]

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "forward-path composition — every unsafe block is a kernel.launch or \
              memcpy_async over DevicePtrs owned by the session's scratch / weights / \
              KV cache. Buffers live for the whole session; sync is driven by the top- \
              level forward_*_decode/prefill caller."
)]

use anyhow::{bail, Context, Result};
use flambeau_core::{Device, DevicePtr};
use flambeau_ops::hip::{HipDevice, HipStream, OpsRegistry};

use super::{
    argmax_token_host, forward_embed_decode_host, forward_layer_decode, forward_layer_prefill,
    forward_output_head_decode, LayerForwardScratch, LayerPrefillScratch, OutputHeadScratch,
};
use crate::config::Qwen3MoEConfig;

// ---------------------------------------------------------------------------
// V1.7.3-e5 — forward_one_token end-to-end.
// ---------------------------------------------------------------------------

/// Complete per-token scratch: two hidden-state buffers (ping/pong for
/// the per-layer loop) + the per-layer composition scratches + the output
/// head scratch. Allocates all once per session.
pub struct ForwardOneTokenScratch {
    pub hidden_a: DevicePtr,    // F16 [hidden]
    pub hidden_b: DevicePtr,    // F16 [hidden]
    pub layer: Option<LayerForwardScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl ForwardOneTokenScratch {
    pub fn new(cfg: &Qwen3MoEConfig, device: &HipDevice) -> Result<Self> {
        let hidden_bytes = cfg.hidden_size * 2;
        let hidden_a = device.alloc(hidden_bytes)?;
        let hidden_b = device.alloc(hidden_bytes)?;
        let layer = Some(LayerForwardScratch::new(cfg, device)?);
        let output_head = Some(OutputHeadScratch::new(cfg, device)?);
        Ok(Self {
            hidden_a,
            hidden_b,
            layer,
            output_head,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for ForwardOneTokenScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "ForwardOneTokenScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// End-to-end single-token decode. Reads the weights + layout from the
/// model, updates the per-sequence session (KV caches, GDN state, conv
/// history), and returns the argmax-sampled next token id.
///
/// Flow:
/// 1. Embed `token_id` → hidden_a [hidden] F16.
/// 2. For il in 0..num_layers: `forward_layer_decode(il, hidden_{a,b}, ...)` then swap.
/// 3. `forward_output_head_decode(hidden, output_norm, lm_head)` → logits.
/// 4. `argmax_token_host(logits)` → next token id.
///
/// `lm_head`: if `cfg.tied_lm_head` is true, pass `&weights.token_embd` —
/// otherwise the untied `&weights.output`. Wiring this pick is the
/// model-struct's job; we keep the forward path dtype-agnostic.
pub fn forward_one_token(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    weights: &crate::weights::ModelWeights,
    session: &mut crate::session::Qwen3MoESession,
    scratch: &mut ForwardOneTokenScratch,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let hidden = cfg.hidden_size;

    // 1. Gather the input embedding row. Writes hidden_a.
    forward_embed_decode_host(
        device,
        stream,
        &weights.token_embd,
        token_id,
        scratch.hidden_a,
        hidden,
    )?;

    // 2. Per-layer loop. Ping-pong between hidden_a and hidden_b so each
    // layer reads the previous output without a stream-stalling copy.
    let layer_scratch = scratch
        .layer
        .as_mut()
        .context("ForwardOneTokenScratch.layer missing")?;
    let (mut x_in, mut x_out) = (scratch.hidden_a, scratch.hidden_b);
    for (il, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut session.layers_mut()[il];
        forward_layer_decode(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            layer_scratch,
            x_in,
            x_out,
            position,
        )?;
        std::mem::swap(&mut x_in, &mut x_out);
    }
    // After the swap in the last iter, `x_in` holds the final output.
    let x_final = x_in;

    // 3. Output head: rmsnorm + LM head → logits.
    let lm_head = weights
        .output
        .as_ref()
        .unwrap_or(&weights.token_embd);
    let output_head_scratch = scratch
        .output_head
        .as_mut()
        .context("ForwardOneTokenScratch.output_head missing")?;
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &weights.output_norm,
        lm_head,
        output_head_scratch,
        x_final,
    )?;

    // 4. Sample.
    argmax_token_host(device, stream, output_head_scratch.logits_f32, cfg.vocab_size)
}



pub struct ForwardPrefillScratch {
    pub max_tokens: usize,
    pub hidden_a: DevicePtr,
    pub hidden_b: DevicePtr,
    pub layer: Option<LayerPrefillScratch>,
    pub output_head: Option<OutputHeadScratch>,
    hidden_bytes: usize,
    disposed: bool,
}

impl ForwardPrefillScratch {
    pub fn new(
        cfg: &Qwen3MoEConfig,
        device: &HipDevice,
        max_tokens: usize,
    ) -> Result<Self> {
        let hidden_bytes = max_tokens * cfg.hidden_size * 2;
        let hidden_a = device.alloc(hidden_bytes)?;
        let hidden_b = device.alloc(hidden_bytes)?;
        let layer = Some(LayerPrefillScratch::new(cfg, device, max_tokens)?);
        let output_head = Some(OutputHeadScratch::new(cfg, device)?);
        Ok(Self {
            max_tokens,
            hidden_a,
            hidden_b,
            layer,
            output_head,
            hidden_bytes,
            disposed: false,
        })
    }

    pub fn dispose(mut self, device: &HipDevice) -> Result<()> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        unsafe {
            device.dealloc(self.hidden_a, self.hidden_bytes)?;
            device.dealloc(self.hidden_b, self.hidden_bytes)?;
        }
        if let Some(s) = self.layer.take() {
            s.dispose(device)?;
        }
        if let Some(s) = self.output_head.take() {
            s.dispose(device)?;
        }
        Ok(())
    }
}

impl Drop for ForwardPrefillScratch {
    fn drop(&mut self) {
        if !self.disposed {
            tracing::warn!(
                target: "flambeau_qwen3_moe::forward",
                "ForwardPrefillScratch dropped without dispose(device); device buffers leaked"
            );
        }
    }
}

/// Prefill a chunk of L tokens end-to-end. Feeds the whole chunk through
/// every layer, then runs the output head on the LAST token's hidden and
/// returns the argmax-sampled next token id.
///
/// Semantics:
/// - Each token's embedding is gathered on host and uploaded into
///   `hidden_a[t]` (one F16 row per token).
/// - `forward_layer_prefill` runs over all L tokens per layer.
/// - KV cache / GDN state / conv history are updated with L tokens of
///   history before returning.
/// - Output head runs on the last token's final hidden (F16 `[hidden]`
///   slice at offset `(L-1) * hidden`). Argmax on host.
///
/// Caller is responsible for chunking a long prompt if `L > scratch.max_tokens`.
pub fn forward_prefill(
    ops: &OpsRegistry,
    stream: &HipStream,
    device: &HipDevice,
    cfg: &Qwen3MoEConfig,
    weights: &crate::weights::ModelWeights,
    session: &mut crate::session::Qwen3MoESession,
    scratch: &mut ForwardPrefillScratch,
    tokens: &[u32],
    start_position: usize,
) -> Result<u32> {
    let l = tokens.len();
    if l == 0 {
        bail!("forward_prefill called with empty tokens");
    }
    if l > scratch.max_tokens {
        bail!(
            "forward_prefill: L={l} > scratch.max_tokens={}",
            scratch.max_tokens
        );
    }

    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    // 1. Gather embeddings for all L tokens into hidden_a, row-by-row.
    for (t, &token_id) in tokens.iter().enumerate() {
        forward_embed_decode_host(
            device,
            stream,
            &weights.token_embd,
            token_id,
            scratch.hidden_a.offset_bytes(t * row_bytes),
            hidden,
        )?;
    }

    // 2. Per-layer loop with ping-pong hidden state.
    let layer_scratch = scratch
        .layer
        .as_mut()
        .context("ForwardPrefillScratch.layer missing")?;
    let (mut x_in, mut x_out) = (scratch.hidden_a, scratch.hidden_b);
    for (il, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut session.layers_mut()[il];
        forward_layer_prefill(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            layer_cache,
            layer_scratch,
            x_in,
            x_out,
            l,
            start_position,
        )?;
        std::mem::swap(&mut x_in, &mut x_out);
    }
    // `x_in` now holds the final hidden `[L, hidden]` F16.

    // 3. Output head on the LAST token only — argmax logits for that token
    //    are what the sampler needs. Upstream prefill users also typically
    //    care only about the final position; if a use case wants
    //    per-position logits, a variant could return them all.
    let last_token_hidden = x_in.offset_bytes((l - 1) * row_bytes);
    let lm_head = weights
        .output
        .as_ref()
        .unwrap_or(&weights.token_embd);
    let output_head_scratch = scratch
        .output_head
        .as_mut()
        .context("ForwardPrefillScratch.output_head missing")?;
    forward_output_head_decode(
        ops,
        stream,
        cfg,
        &weights.output_norm,
        lm_head,
        output_head_scratch,
        last_token_hidden,
    )?;

    argmax_token_host(device, stream, output_head_scratch.logits_f32, cfg.vocab_size)
}

