//! **P2.9b-i2** — batched decode driver: run N concurrent slots through
//! one PP forward pass with shared per-rank scratch.
//! ## What batches and what doesn't
//! For each layer, this driver dispatches:
//! - **Full-attn layers** → [`super::attn::forward_full_attn_layer_decode_batched`]
//! (i2-A1). RMSNorm + Q|gate / K / V projection + RoPE batch as single
//! kernel launches over `[N, *]`; per-slot KV-append + per-slot
//! `attention_decode_f16_slots` because each slot owns its own KV cache
//! and query history.
//! - **GDN layers** → per-slot loop calling
//! [`super::gdn::forward_gdn_layer_decode`]. GDN is recurrent (one
//! `GdnLayerState` evolves through one new token per slot); not
//! batchable across slots without a kernel rewrite. Reuses one
//! shared `GdnScratch` (`rank_scratch.gdn_decode`) sequentially.
//! - **Post-attn add + RMSNorm** → batched at `n_tokens=N` via the
//! existing `add_f16` / `rmsnorm_f16` kernels.
//! - **FFN/MoE** → batched at `n_tokens=N` via the prefill kernels
//! (`forward_dense_ffn_prefill` / `forward_router_prefill` /
//! `forward_moe_ffn_prefill` / `forward_shared_expert_prefill`),
//! which all accept arbitrary `n_tokens`. This is the biggest single
//! throughput lever on dense+MoE archs.
//! - **Output head** → per-slot loop on the last rank
//! (one `forward_output_head_decode` call per slot, per-slot logits
//! row downloaded between calls).
//! ## Lifetime / scratch
//! The driver consumes a single shared `ShardedForwardPrefillScratch`
//! sized for `max_tokens >= N` slots — typically allocated once at
//! server boot and reused per dispatch. Each `Qwen3MoEShardedSession`
//! contributes its per-layer `LayerCache` and its per-rank GDN state
//! through the layer dispatch; sessions don't share scratch.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_ops::hip::{
    mlp::add_f16,
    norm::rmsnorm_f16,
    OpsRegistry,
};

use super::attn::forward_full_attn_layer_decode_batched;
use super::dense_ffn::forward_dense_ffn_prefill;
use super::gdn::forward_gdn_layer_decode;
use super::moe::{
    forward_moe_ffn_prefill, forward_router_prefill, forward_shared_expert_prefill,
};

/// One queued slot in a batched decode dispatch.
#[derive(Debug, Clone, Copy)]
pub struct BatchSlot {
    /// Index into the caller's `sessions` parallel array.
    pub idx: usize,
    /// Token to decode this step.
    pub token_id: u32,
    /// Cache position to decode at (`= current_tokens` of the slot's
    /// KV cache before this token is appended).
    pub position: usize,
}

/// **P2.9b-i2-A1-wire** — drive `slots.len()` concurrent decode steps
/// through the PP topology with real per-layer batching, returning
/// per-slot `[vocab]` F32 logits.
/// `sessions[s.idx]` is the per-slot session for `BatchSlot` `s`. The
/// caller must hold the i1 slot-pool guards for the lifetime of this
/// call.
/// `scratch` is the shared per-rank batched workspace (size sufficient
/// for `>= slots.len()` tokens; typically allocated once at server boot
/// at `FLAMBEAU_INFLIGHT_SLOTS`).
/// `logits_out[s.idx]` is resized to `vocab_size` and overwritten with
/// the slot's logits row.
pub fn forward_decode_batched_pp(
    model: &crate::sharded::Qwen3MoEShardedModel,
    sessions: &mut [&mut crate::sharded::Qwen3MoEShardedSession],
    cluster: &flambeau_backend_hip::HipCluster,
    scratch: &mut super::pp::ShardedForwardPrefillScratch,
    slots: &[BatchSlot],
    logits_out: &mut [&mut Vec<f32>],
) -> Result<()> {
    let n = slots.len();
    if n == 0 {
        bail!("forward_decode_batched_pp: empty slot list");
    }
    if sessions.len() != logits_out.len() {
        bail!(
            "forward_decode_batched_pp: sessions({}) != logits_out({})",
            sessions.len(),
            logits_out.len(),
        );
    }
    for s in slots {
        if s.idx >= sessions.len() {
            bail!(
                "forward_decode_batched_pp: BatchSlot.idx {} OOB (n={})",
                s.idx,
                sessions.len()
            );
        }
    }
    let n_ranks = model.shards.len();
    if n_ranks == 0 {
        bail!("forward_decode_batched_pp: zero-rank cluster");
    }
    let cfg = &model.config;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;
    let chunk_bytes = n * row_bytes;

    // Sanity: per-rank scratch must be sized for >= n tokens.
    for (rank_idx, rs) in scratch.per_rank.iter().enumerate() {
        if rs.max_tokens < n {
            bail!(
                "forward_decode_batched_pp: rank {rank_idx} scratch.max_tokens={} < n={n}",
                rs.max_tokens
            );
        }
    }

    // 1. Embed N tokens at rank 0. Each slot's token row goes into
    // rank-0 hidden_a at offset s * row_bytes — same pattern as
    // forward_prefill_pp's host-loop embed.
    {
        let rank0 = cluster.device(0);
        rank0.bind()?;
        let shard0 = &model.shards[0];
        let scratch0 = &mut scratch.per_rank[0];
        let token_embd = shard0
            .token_embd
            .as_ref()
            .context("rank 0 shard missing token_embd")?;
        for (s_pos, slot) in slots.iter().enumerate() {
            super::io::forward_embed_decode_host(
                rank0,
                rank0.default_stream(),
                token_embd,
                slot.token_id,
                scratch0.hidden_a.offset_bytes(s_pos * row_bytes),
                hidden,
            )?;
        }
    }

    // Build a per-rank slot-position vec once (positions are the same
    // across all ranks since each slot's per-layer KV cache shares the
    // same tail across the rank's local layers).
    let slot_positions: Vec<usize> = slots.iter().map(|s| s.position).collect();

    // 2. Per-rank layer loop with stage-boundary peer_copy_via_host of
    // [N, hidden] F16.
    for rank_idx in 0..n_ranks {
        let device = cluster.device(rank_idx);

        if rank_idx > 0 {
            // SAFETY: both hidden_a buffers are sized for max_tokens * row_bytes
            // ≥ chunk_bytes; no other stream touches them here.
            unsafe {
                cluster.peer_copy_via_host(
                    scratch.per_rank[rank_idx].hidden_a,
                    rank_idx,
                    scratch.per_rank[rank_idx - 1].hidden_a,
                    rank_idx - 1,
                    chunk_bytes,
                )?;
            }
        }
        device.bind()?;

        let shard = &model.shards[rank_idx];
        let rank_scratch = &mut scratch.per_rank[rank_idx];
        let layer_scratch = rank_scratch
            .layer
            .as_mut()
            .context("RankForwardPrefillScratch.layer missing")?;

        // Walk the rank's local layers, ping-pong through hidden_a/hidden_b.
        let (mut x_in, mut x_out) = (rank_scratch.hidden_a, rank_scratch.hidden_b);
        // We'll need to access each session's per-rank LayerCache vec for
        // each layer; gather mutable references slot-by-slot, layer-by-layer.
        for (local_idx, layer_weights) in shard.layers.iter().enumerate() {
            forward_layer_decode_batched(
                &shard.ops,
                device.default_stream(),
                device,
                cfg,
                layer_weights,
                sessions,
                rank_idx,
                local_idx,
                layer_scratch,
                rank_scratch.gdn_decode.as_mut(),
                x_in,
                x_out,
                &slot_positions,
            )
            .with_context(|| {
                format!(
                    "batched-decode rank {} layer {} ({})",
                    rank_idx,
                    layer_weights.layer_idx,
                    if cfg.is_recurrent(layer_weights.layer_idx) {
                        "gdn"
                    } else {
                        "full_attn"
                    },
                )
            })?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Land final hidden in hidden_a for the next peer-copy / output head.
        if x_in != rank_scratch.hidden_a {
            // SAFETY: both pointers live for this scope; size = chunk_bytes.
            unsafe {
                device.memcpy_async(
                    device.default_stream(),
                    CopyDirection::DeviceToDevice,
                    rank_scratch.hidden_a,
                    x_in,
                    chunk_bytes,
                )?;
            }
        }
    }

    // 3. Output head on the last rank — per-slot loop, since
    // OutputHeadScratch is sized for one F32 logit row at a time.
    // Each slot reads its row of hidden_a, runs the output head,
    // downloads logits to host, then the next slot reuses the same
    // scratch.
    let last_idx = n_ranks - 1;
    let last_device = cluster.device(last_idx);
    last_device.bind()?;
    let last_shard = &model.shards[last_idx];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("last rank missing output_norm")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .or(last_shard.token_embd.as_ref())
        .context("last rank missing both output.weight and tied token_embd")?;
    let last_scratch = &mut scratch.per_rank[last_idx];
    let output_head_scratch = last_scratch
        .output_head
        .as_mut()
        .context("last rank missing output_head scratch")?;

    for (s_pos, slot) in slots.iter().enumerate() {
        let x_final_row = last_scratch.hidden_a.offset_bytes(s_pos * row_bytes);
        super::io::forward_output_head_decode(
            &last_shard.ops,
            last_device.default_stream(),
            cfg,
            output_norm,
            lm_head,
            output_head_scratch,
            x_final_row,
        )
        .with_context(|| format!("batched-decode output head slot {}", slot.idx))?;
        super::io::download_logits_host(
            last_device,
            last_device.default_stream(),
            output_head_scratch.logits_f32,
            cfg.vocab_size,
            logits_out[slot.idx],
        )
        .with_context(|| format!("batched-decode logits download slot {}", slot.idx))?;
    }

    Ok(())
}

/// Per-layer batched dispatcher used by [`forward_decode_batched_pp`].
/// Walks one local layer (`local_idx` within the rank's `shard.layers`),
/// dispatching:
/// - full-attn → [`forward_full_attn_layer_decode_batched`]
/// - GDN → per-slot loop with shared `gdn_decode_scratch`
/// then runs post-attn add + rmsnorm and FFN/MoE batched at `n_tokens=N`.
#[allow(clippy::too_many_arguments)]
fn forward_layer_decode_batched(
    ops: &OpsRegistry,
    stream: &flambeau_backend_hip::HipStream,
    device: &flambeau_backend_hip::HipDevice,
    cfg: &crate::config::Qwen3MoEConfig,
    layer_weights: &crate::weights::LayerWeights,
    sessions: &mut [&mut crate::sharded::Qwen3MoEShardedSession],
    rank_idx: usize,
    local_idx: usize,
    scratch: &mut super::LayerPrefillScratch,
    gdn_decode_scratch: Option<&mut super::GdnScratch>,
    x_in: DevicePtr,
    x_out: DevicePtr,
    slot_positions: &[usize],
) -> Result<()> {
    let n = slot_positions.len();
    let il = layer_weights.layer_idx;
    let hidden = cfg.hidden_size;
    let row_bytes = hidden * 2;

    // Step 1 — attention: full-attn batched, GDN per-slot loop.
    if cfg.is_recurrent(il) {
        // Per-slot GDN. Each slot's row in x_in → that slot's GDN state
        // mutated; output written to mid_f16 row.
        let gdn = gdn_decode_scratch.context(
            "forward_layer_decode_batched: GDN layer requires rank gdn_decode scratch",
        )?;
        for s in 0..n {
            let slot_x_in = x_in.offset_bytes(s * row_bytes);
            let slot_delta = scratch.mid_f16.offset_bytes(s * row_bytes);
            let session = &mut *sessions[s];
            let layer_cache = &mut session.per_rank[rank_idx].caches[local_idx];
            forward_gdn_layer_decode(
                ops,
                stream,
                device,
                cfg,
                layer_weights,
                layer_cache,
                gdn,
                slot_x_in,
                slot_delta,
            )
            .with_context(|| format!("GDN slot {s} layer {il}"))?;
        }
    } else {
        // Gather per-slot LayerCache mutable refs for THIS layer.
        // SAFETY: each session is unique in `sessions[]`; we form one
        // disjoint `&mut LayerCache` per session. They live in different
        // session structs so the borrow checker accepts the gather via
        // an unsafe split-borrow over the slice. Each cache is only
        // touched once per call.
        let mut slot_caches: Vec<&mut crate::session::LayerCache> = Vec::with_capacity(n);
        // Use raw pointer split to satisfy the borrow checker — sessions
        // is &mut [&mut Session]; reborrowing each slot independently
        // would require splitting the slice.
        let sessions_ptr = sessions.as_mut_ptr();
        for s in 0..n {
            // SAFETY: indices 0..n are distinct and within bounds; each
            // `*sessions[s]` is a unique session, and we extract one cache
            // from each. The borrows formed below alias only their
            // respective sessions' `caches[local_idx]` and don't overlap.
            unsafe {
                let session_ref: &mut crate::sharded::Qwen3MoEShardedSession =
                    &mut **sessions_ptr.add(s);
                let cache_ref: &mut crate::session::LayerCache =
                    &mut session_ref.per_rank[rank_idx].caches[local_idx];
                slot_caches.push(cache_ref);
            }
        }
        forward_full_attn_layer_decode_batched(
            ops,
            stream,
            device,
            cfg,
            layer_weights,
            &mut slot_caches,
            scratch
                .full_attn
                .as_mut()
                .context("LayerPrefillScratch.full_attn missing")?,
            x_in,
            scratch.mid_f16,
            slot_positions,
        )?;
    }

    // Step 2 — residual: mid = x_in + attn_delta (in place on mid_f16).
    add_f16(
        ops,
        stream,
        x_in,
        scratch.mid_f16,
        scratch.mid_f16,
        n * hidden,
    )
    .context("batched-decode layer residual: x_in + attn_delta")?;

    // Step 3 — post-attn / ffn norm.
    let post_norm = layer_weights
        .post_attention_norm
        .as_ref()
        .or(layer_weights.ffn_norm.as_ref())
        .context("layer missing both post_attention_norm and ffn_norm")?;
    rmsnorm_f16(
        ops,
        stream,
        scratch.mid_f16,
        post_norm.ptr,
        scratch.mid_norm_f16,
        n,
        hidden,
        cfg.rms_norm_eps,
    )
    .context("batched-decode post-attn rmsnorm")?;

    // Step 4 — FFN. Dense (qwen35) or MoE + optional shared expert.
    if cfg.is_dense_ffn() {
        let dense_w = layer_weights
            .ffn
            .dense
            .as_ref()
            .context("dense FFN: layer.ffn.dense missing")?;
        let dense_scratch = scratch
            .dense_ffn
            .as_mut()
            .context("LayerPrefillScratch.dense_ffn missing")?;
        forward_dense_ffn_prefill(
            ops,
            stream,
            cfg,
            dense_w,
            dense_scratch,
            scratch.mid_norm_f16,
            scratch.mid_f16,
            x_out,
            n,
        )?;
        return Ok(());
    }

    // MoE path.
    let moe_residual = if let (Some(shared_w), Some(shared_scratch)) =
        (layer_weights.ffn.shared.as_ref(), scratch.shared.as_mut())
    {
        forward_shared_expert_prefill(
            ops,
            stream,
            cfg,
            shared_w,
            shared_scratch,
            scratch.mid_norm_f16,
            scratch.shared_delta_f16,
            n,
        )?;
        add_f16(
            ops,
            stream,
            scratch.mid_f16,
            scratch.shared_delta_f16,
            scratch.moe_residual_f16,
            n * hidden,
        )
        .context("batched-decode moe residual: mid + shared_delta")?;
        scratch.moe_residual_f16
    } else {
        scratch.mid_f16
    };

    let moe = scratch
        .moe
        .as_mut()
        .context("LayerPrefillScratch.moe missing")?;
    let ffn_gate_inp = layer_weights
        .ffn
        .ffn_gate_inp
        .as_ref()
        .context("MoE branch: ffn.ffn_gate_inp missing")?;
    forward_router_prefill(
        ops, stream, cfg, ffn_gate_inp, moe, scratch.mid_norm_f16, n,
    )?;
    forward_moe_ffn_prefill(
        ops,
        stream,
        cfg,
        &layer_weights.ffn,
        moe,
        scratch.mid_norm_f16,
        moe_residual,
        x_out,
        n,
    )?;

    Ok(())
}
