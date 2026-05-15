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

use anyhow::{anyhow, bail, Context, Result};
use flambeau_blocks::embed_token_host;
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::hip::{HipDevice, HipOps, OpsRegistry};
use flambeau_ops::Ops;

use crate::layer::forward_layer_decode;
use crate::output_head::forward_output_head;
use crate::per_layer_embd::{
    build_inp_per_layer_table, per_layer_token_embd_row_bytes, table_slice_ptr,
    upload_inp_per_layer_table,
};
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

    // 2. Scale embedding by sqrt(n_embd) (gemma4-iswa.cpp:20
    // `inpL = scale(inpL, sqrtf(n_embd))`).
    ops.scale_f16(residual, residual, hidden, (hidden as f32).sqrt())
        .context("embedding sqrt(n_embd) scale")?;

    // 2b. Per-layer-embd table build (E2B/E4B only). Reads the input
    // token's `per_layer_token_embd` row from the mmap, dequants +
    // projects + RMSNorms host-side, uploads to
    // `session.inp_per_layer_table_buf`.
    let pe_table_base = build_per_layer_table_for_token(session, token_id, device, stream)?;

    // 3. Per-layer loop. Tail layers (`has_kv == false`) route their
    // attention through the source layer's KV cache via
    // `spec.kv_share_src`; the layer composer skips K/V projection +
    // append in that branch.
    let snapshot_layers: Vec<crate::layout::LayerSpec> = session.layout.layers.clone();
    let pe_n_embd_per_layer = session.cfg.per_layer_embed.as_ref().map(|p| p.n_embd_per_layer);
    // moe_scratch_view returns Option<&Gemma4MoeScratch>; capture by
    // raw ptr so the per-layer scratch view (which borrows session
    // disjointly) doesn't conflict.
    let moe_scratch_ptr: Option<*const crate::moe::Gemma4MoeScratch> =
        session.moe_scratch_view().map(|s| s as *const _);
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

        // Per-layer slice for E2B/E4B side-channel embedding. `None`
        // for variants without per-layer-embd.
        let per_layer_slice = pe_table_base
            .zip(pe_n_embd_per_layer)
            .map(|(base, pe)| (table_slice_ptr(base, spec.index, pe), pe));
        // SAFETY: moe_scratch lives on session and is disjoint from
        // the per-layer scratch view's borrows.
        let moe_scratch_ref = moe_scratch_ptr.map(|p| unsafe { &*p });
        forward_layer_decode(
            &ops, device, stream, weights, &spec, rms_eps, ff_len, hidden,
            kv, &mut scratch, in_ptr, out_ptr, position,
            per_layer_slice,
            moe_scratch_ref,
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

/// Build the `inp_per_layer_table` for a single decode token and
/// upload it to `session.inp_per_layer_table_buf`. Returns the device
/// base pointer of the table, or `None` when the variant has no
/// per-layer-embd (table_buf is None).
///
/// Reads three tensors from the session's `Arc<GgufFile>`:
/// - `per_layer_token_embd[token_id, :]` row (Q5_K on E4B, dequant
///   host-side to F32)
/// - `per_layer_model_proj` (BF16 on E4B; matmul on host)
/// - `per_layer_proj_norm` (F32, RMSNorm scale)
///
/// Plus the F16 input embedding (which we read back from device — the
/// `residual` buffer has just had `sqrt(n_embd)` applied; we use it
/// post-scale to match `gemma4-iswa.cpp:264-322`).
fn build_per_layer_table_for_token(
    session: &mut Gemma4Session,
    token_id: u32,
    device: &HipDevice,
    stream: &flambeau_backend_hip::HipStream,
) -> Result<Option<DevicePtr>> {
    let Some(pe_cfg) = session.cfg.per_layer_embed else {
        return Ok(None);
    };
    let pe = pe_cfg.n_embd_per_layer;
    let n_layer = session.cfg.num_layers;
    let hidden = session.cfg.hidden_size;
    let rms_eps = session.cfg.rms_norm_eps;
    let gguf = session
        .gguf
        .as_ref()
        .ok_or_else(|| anyhow!(
            "build_per_layer_table_for_token: session has no GgufFile (use \
             Gemma4Session::new_with_gguf for E2B/E4B variants)"
        ))?
        .clone();
    let globals = session
        .weights
        .per_layer_embd_globals
        .ok_or_else(|| anyhow!("per_layer_embd_globals missing despite cfg.per_layer_embed.is_some()"))?;
    let table_buf = session
        .inp_per_layer_table_buf
        .ok_or_else(|| anyhow!("inp_per_layer_table_buf missing despite cfg.per_layer_embed.is_some()"))?;

    // Tensor infos from the GGUF.
    let g_names = crate::names::GlobalNames::default_names();
    let tokembd_info = gguf
        .tensors
        .get(&g_names.per_layer_token_embd)
        .ok_or_else(|| anyhow!("per_layer_token_embd missing in GGUF"))?;
    let modelproj_info = gguf
        .tensors
        .get(&g_names.per_layer_model_proj)
        .ok_or_else(|| anyhow!("per_layer_model_proj missing in GGUF"))?;
    let projnorm_info = gguf
        .tensors
        .get(&g_names.per_layer_proj_norm)
        .ok_or_else(|| anyhow!("per_layer_proj_norm missing in GGUF"))?;

    // 1. Read the token's per_layer_token_embd row from mmap.
    let row_bytes = per_layer_token_embd_row_bytes(tokembd_info)?;
    let tokembd_raw = gguf
        .tensor_raw(&tokembd_info.name)
        .with_context(|| format!("tensor_raw `{}`", tokembd_info.name))?;
    let vocab = tokembd_info.dims[0] as usize;
    if (token_id as usize) >= vocab {
        bail!(
            "per_layer build: token_id {token_id} >= per_layer_token_embd vocab {vocab}"
        );
    }
    let offset = (token_id as usize) * row_bytes;
    if offset + row_bytes > tokembd_raw.len() {
        bail!("per_layer_token_embd row OOB at token {token_id}");
    }
    let row = &tokembd_raw[offset..offset + row_bytes];

    // 2. Full model_proj + proj_norm slabs.
    let modelproj_raw = gguf
        .tensor_raw(&modelproj_info.name)
        .with_context(|| format!("tensor_raw `{}`", modelproj_info.name))?;
    let projnorm_raw = gguf
        .tensor_raw(&projnorm_info.name)
        .with_context(|| format!("tensor_raw `{}`", projnorm_info.name))?;

    // 3. Read back the post-scale F16 embedding from device.
    let mut inp_batch_f16 = vec![half::f16::from_f32(0.0); hidden];
    // SAFETY: outer_residual holds hidden*2 F16 bytes; host buf same.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(inp_batch_f16.as_mut_ptr() as usize),
            session.outer_residual(),
            hidden * 2,
        )?;
    }
    stream.synchronize()?;

    // 4. Host-side build.
    let table = build_inp_per_layer_table(
        row,
        tokembd_info.dtype,
        modelproj_raw,
        modelproj_info.dtype,
        projnorm_raw,
        &inp_batch_f16,
        pe,
        n_layer,
        hidden,
        rms_eps,
    )?;
    let _ = globals; // on-device copies kept for future on-device build path

    // 5. Upload to the session's table buffer.
    upload_inp_per_layer_table(device, stream, &table, table_buf)?;
    Ok(Some(table_buf))
}
