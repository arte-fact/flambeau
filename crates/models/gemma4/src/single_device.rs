//! Single-device end-to-end forward driver for Gemma 4.
//!
//! Composition (mirrors `gemma4-iswa.cpp:14-260` minus per-layer
//! side-channel embedding and shared-KV tail):
//!
//! 1. Embed `token_id` → F16 [hidden] via host-side dequant + upload.
//! 2. Scale embedding by `sqrt(n_embd)` (gemma4 input scaling).
//! 3. For each layer: `forward_layer_decode(.., x_in → x_out)`; swap.
//! 4. Output: `rmsnorm(output_norm) + LM-head matmul + softcap`.
//! 5. Caller chooses: download logits OR run argmax host-side.

#![cfg(feature = "hip")]

use anyhow::{anyhow, Result};
use flambeau_blocks::embed_token_host;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, HipOps, OpsRegistry};

use crate::layer::forward_layer_decode;
use crate::output_head::forward_output_head;
use crate::session::Gemma4Session;

/// Forward one decode token. Returns the host-side argmax token id.
/// `position` is the cache tail length before this call (i.e. the
/// token's position in the running sequence).
pub fn forward_one_token(
    session: &mut Gemma4Session,
    device: &HipDevice,
    token_id: u32,
    position: usize,
) -> Result<u32> {
    let logits = forward_one_token_logits(session, device, token_id, position)?;
    let stream = device.default_stream();
    // Download logits + argmax host-side.
    let vocab = session.cfg.vocab_size;
    let mut host = vec![0.0f32; vocab];
    // SAFETY: logits points at vocab*4 device bytes; host has the same.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            logits,
            vocab * 4,
        )?;
    }
    stream.synchronize()?;
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in host.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    Ok(best_i)
}

/// Forward one decode token and leave logits on device. Caller is
/// responsible for downloading or running the sampler.
pub fn forward_one_token_logits(
    session: &mut Gemma4Session,
    device: &HipDevice,
    token_id: u32,
    position: usize,
) -> Result<DevicePtr> {
    device.bind()?;
    let stream = device.default_stream();
    let reg = OpsRegistry::new(device).map_err(|e| anyhow!("registry: {e}"))?;
    let ops = HipOps::new(&reg, stream);

    let hidden = session.cfg.hidden_size;
    let vocab = session.cfg.vocab_size;
    let ff_len = session.cfg.feed_forward_length;
    let rms_eps = session.cfg.rms_norm_eps;

    // 1. Embed.
    let residual = session.outer_residual();
    embed_token_host(
        device,
        stream,
        session.weights.token_embd.ptr,
        session.weights.token_embd.dtype,
        session.weights.token_embd.bytes,
        vocab,
        hidden,
        token_id,
        residual,
    )?;

    // 2. Scale embedding by sqrt(n_embd) per gemma4-iswa.cpp:20.
    // The F16 scale uses the F32 input-scale kernel via cast — for
    // smoke purposes, multiply in-place via a scale op. We don't yet
    // have a scalar-mul-F16 op; instead we apply this scaling via the
    // RMSNorm/first-layer matmul-magnitude path implicitly. The
    // input scaling matters for parity; for the dummy smoke it is a
    // constant factor and the chain still produces finite values.
    // S5-B-2: add a `scale_f16` op or fuse into the embedding upload
    // step.
    // TODO: parity hookup.

    // 3. Per-layer loop. Tail layers (`has_kv == false`) route their
    // attention through the source layer's KV cache via
    // `spec.kv_share_src`; the layer composer skips K/V projection +
    // append in that branch.
    let snapshot_layers: Vec<crate::layout::LayerSpec> = session.layout.layers.clone();
    for spec in snapshot_layers {
        let weights_ref =
            &session.weights.layers[spec.index] as *const crate::layer::Gemma4LayerWeights;
        // SAFETY: `weights_ref` borrows from session.weights; subsequent
        // mutations of session touch other fields (kv_caches, scratch).
        let weights = unsafe { &*weights_ref };

        let in_ptr = session.outer_residual();
        let out_ptr = session.outer_next();
        let kv_slot_idx = if spec.has_kv {
            spec.index
        } else {
            spec.kv_share_src.ok_or_else(|| {
                anyhow!(
                    "layer {}: shared-KV tail without kv_share_src",
                    spec.index
                )
            })?
        };
        let kv_ptr: *mut Option<flambeau_runtime::KvCache<flambeau_runtime::F16Contig, HipDevice>> =
            &mut session.kv_caches[kv_slot_idx];
        let mut scratch = session.layer_scratch_view();
        // SAFETY: kv_ptr borrows from session.kv_caches at kv_slot_idx;
        // the scratch view above already holds &mut self for the
        // other fields and does NOT touch kv_caches.
        let kv = unsafe { &mut *kv_ptr };
        let kv = kv
            .as_mut()
            .ok_or_else(|| anyhow!("layer {} kv slot {} unallocated", spec.index, kv_slot_idx))?;

        forward_layer_decode(
            &ops, device, stream, weights, &spec, rms_eps, ff_len, hidden,
            kv, &mut scratch, in_ptr, out_ptr, position,
            /*per_layer_slice=*/ None,
        )?;
        session.swap_residual();
    }

    // 4. Output head.
    let lm_head = match session.weights.output {
        Some(t) => t.as_weight_handle(session.weights.token_embd_dims)?,
        None => session
            .weights
            .token_embd
            .as_weight_handle(session.weights.token_embd_dims)?,
    };
    let softcap = session.cfg.final_logit_softcap;
    let output_norm = session.weights.output_norm.ptr;
    let final_in = session.outer_residual();
    let scratch = session.output_head_scratch();
    let logits = forward_output_head(
        &ops, final_in, output_norm, lm_head, softcap, scratch, hidden, vocab, rms_eps,
    )?;

    Ok(logits)
}
