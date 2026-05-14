//! server-side LoadedModel + Inflight session abstractions.
//! Both the PP and TP topologies expose a same-shape surface to the
//! request handlers: `Inflight::new(...)` allocates per-request
//! session + scratch, `prefill_logits(...)` ingests the prompt, and
//! `decode_logits(...)` advances by one token at a time. The handlers
//! call those helpers without caring which topology the cluster is
//! configured for; the dispatch lives here.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster};
use flambeau_qwen3_moe::forward::{
    forward_one_token_hybrid_keep_logits_on_device, forward_one_token_hybrid_logits,
    forward_one_token_pp_keep_logits_on_device, forward_one_token_pp_logits,
    forward_one_token_tp_keep_logits_on_device, forward_one_token_tp_logits,
    forward_prefill_hybrid_logits, forward_prefill_pp, forward_prefill_pp_logits,
    forward_prefill_tp_logits, forward_prefill_tp_logits_pooled,
    ShardedForwardOneTokenScratch, ShardedForwardOneTokenScratchHybrid,
    ShardedForwardOneTokenScratchTp, ShardedForwardPrefillScratch,
    ShardedForwardPrefillScratchTp,
};
use flambeau_qwen3_moe::session::LayerCacheSnapshot;
use flambeau_qwen3_moe::{
    Qwen3MoEHybridModel, Qwen3MoEHybridSession, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
    Qwen3MoETpModel, Qwen3MoETpSession,
};

/// Pipeline-parallel sharded model. One whole layer per rank stage;
/// cross-stage hand-off via host-bounce peer copy.
pub struct PpHipModel {
    pub model: Qwen3MoEShardedModel,
}

/// Tensor-parallel sharded model. Every rank holds every layer
/// (sliced); intra-layer Megatron splits + BAR1 P2P AllReduce.
///
/// `tp: TpCluster` bundles the `Arc<HipCluster>` + `BarP2pAllReduce`
/// pair so callers thread one handle through forward APIs instead of
/// the previous `(&HipCluster, &BarP2pAllReduce)` pair. Use
/// `model.tp.cluster()` for device lookups and `model.tp.ar()` for
/// AllReduce calls. The struct-field name preserves the `model.tp`
/// access pattern used by routes.rs / model_handle.rs.
pub struct TpHipModel {
    pub model: Qwen3MoETpModel,
    pub tp: flambeau_blocks::TpCluster,
}

impl TpHipModel {
    /// Backwards-compatible accessor for sites that read `&model.ar`
    /// directly. Prefer threading `&model.tp` instead.
    pub fn ar(&self) -> &BarP2pAllReduce {
        self.tp.ar()
    }
}

/// Hybrid PP-of-TP. `pp_size` contiguous layer stages, each owning a
/// `tp_size`-rank TP subgroup. `hc: HybridCluster` carries the per-
/// stage sub-clusters + per-stage ARs + the global cluster. The
/// `HybridCluster::new` constructor enforces the sub-cluster-before-
/// global construction order as a type invariant (the gotcha captured
/// in `project_hybrid_cluster_order`: building global first disables
/// BAR1 on the sub-cluster off-diagonal).
pub struct HybridHipModel {
    pub model: Qwen3MoEHybridModel,
    pub hc: flambeau_blocks::HybridCluster,
}

impl HybridHipModel {
    /// Backwards-compatible accessor returning per-stage AR borrows in
    /// stage-index order. Callers that took `&[BarP2pAllReduce]`
    /// receive `&[&BarP2pAllReduce]` — slice-of-refs since
    /// `BarP2pAllReduce` is not `Clone`.
    pub fn stage_ars(&self) -> Vec<&BarP2pAllReduce> {
        self.hc.stages().iter().map(|s| &s.ar).collect()
    }
}

/// Loaded weights + per-topology auxiliary state. `Arc<dyn HipModel>`
/// so per-request sessions can hold a cheap back-reference.
pub type LoadedModel = std::sync::Arc<dyn crate::model_handle::HipModel>;

pub struct PpHipSession {
    pub session: Qwen3MoEShardedSession,
    pub prefill: ShardedForwardPrefillScratch,
    pub decode: ShardedForwardOneTokenScratch,
}

pub struct TpHipSession {
    pub session: Qwen3MoETpSession,
    pub decode: ShardedForwardOneTokenScratchTp,
}

pub struct HybridHipSession {
    pub session: Qwen3MoEHybridSession,
    pub decode: ShardedForwardOneTokenScratchHybrid,
}

/// Per-request session + scratch. Variant must match the
/// [`LoadedModel`] variant used to construct it; mismatches bail at
/// dispatch.
pub enum Inflight {
    Pp(PpHipSession),
    Tp(TpHipSession),
    Hybrid(HybridHipSession),
}

impl Inflight {
    pub fn as_pp(&self) -> Option<&PpHipSession> {
        if let Inflight::Pp(s) = self { Some(s) } else { None }
    }
    pub fn as_pp_mut(&mut self) -> Option<&mut PpHipSession> {
        if let Inflight::Pp(s) = self { Some(s) } else { None }
    }
    pub fn as_tp(&self) -> Option<&TpHipSession> {
        if let Inflight::Tp(s) = self { Some(s) } else { None }
    }
    pub fn as_tp_mut(&mut self) -> Option<&mut TpHipSession> {
        if let Inflight::Tp(s) = self { Some(s) } else { None }
    }
    pub fn as_hybrid(&self) -> Option<&HybridHipSession> {
        if let Inflight::Hybrid(s) = self { Some(s) } else { None }
    }
    pub fn as_hybrid_mut(&mut self) -> Option<&mut HybridHipSession> {
        if let Inflight::Hybrid(s) = self { Some(s) } else { None }
    }
}

impl Inflight {
    /// Allocate per-rank caches + scratches for a single request.
    /// `prompt_len` sizes the PP prefill scratch; ignored for TP (TP
    /// loops `forward_one_token_tp` over the prompt instead).
    pub fn new(
        model: &LoadedModel,
        cluster: &HipCluster,
        prefill_ubatch: usize,
        kv_layout: flambeau_qwen3_moe::session::KvLayout,
    ) -> Result<Self> {
        // **chunked prefill** (Phase B / task #237). The per-layer
        // attention scratch is O(heads × L²) and OOMs past ~5k tokens
        // on a 16 GB MI50. forward_prefill_pp recursively chunks when
        // L > scratch.max_tokens.
        // **Critical:** chunk size MUST stay in the same MMQ-dispatch
        // bucket as the model expects. On gfx906 / Q4_1 the boundary
        // is m=128: m<128 routes to MMVQ, m>=128 routes to MMQ-4warp;
        // mixing them across calls writes numerically-different K
        // bytes for the same input token (Phase A parity test
        // verified bit-equality only when chunks stay in one bucket).
        // 512 (the default `--prefill-ubatch`) gives generous headroom
        // past the 128 boundary; floor enforced at 128.
        let scratch_tokens = prefill_ubatch.max(1);

        // **leak fix** — every alloc step here that ships a fresh GPU
        // resource must roll back the prior steps' GPU resources on
        // failure. Without this, an OOM in a downstream alloc (the
        // common case under long-context prompts) drops the
        // already-built earlier resource via its `Drop` impl, which
        // only WARNS and leaves the device buffers pinned until
        // process exit. One failed request used to lose ~3 GB of
        // VRAM on each rank.
        if let Some(p) = model.as_pp() {
            let m = &p.model;
            let session =
                Qwen3MoEShardedSession::new(m, cluster, kv_layout).context("create PP session")?;
            let prefill = match ShardedForwardPrefillScratch::new(m, cluster, scratch_tokens) {
                Ok(s) => s,
                Err(e) => {
                    let _ = session.dispose(cluster);
                    return Err(e).context("PP prefill scratch");
                }
            };
            let decode = match ShardedForwardOneTokenScratch::new(m, cluster) {
                Ok(s) => s,
                Err(e) => {
                    let _ = prefill.dispose(cluster);
                    let _ = session.dispose(cluster);
                    return Err(e).context("PP decode scratch");
                }
            };
            Ok(Inflight::Pp(PpHipSession {
                session,
                prefill,
                decode,
            }))
        } else if let Some(t) = model.as_tp() {
            let m = &t.model;
            let session =
                Qwen3MoETpSession::new(m, cluster, kv_layout).context("create TP session")?;
            let decode = match ShardedForwardOneTokenScratchTp::new(&m.config, cluster) {
                Ok(s) => s,
                Err(e) => {
                    let _ = session.dispose(cluster);
                    return Err(e).context("TP decode scratch");
                }
            };
            Ok(Inflight::Tp(TpHipSession { session, decode }))
        } else if let Some(h) = model.as_hybrid() {
            // `cluster` here is the server's global cluster; the
            // hybrid session/scratch are sized per-stage against each
            // stage's owning sub-cluster.
            let m = &h.model;
            let session =
                Qwen3MoEHybridSession::new(m, kv_layout).context("create hybrid session")?;
            let decode = match ShardedForwardOneTokenScratchHybrid::new(m) {
                Ok(s) => s,
                Err(e) => {
                    let _ = session.dispose(m);
                    return Err(e).context("hybrid decode scratch");
                }
            };
            Ok(Inflight::Hybrid(HybridHipSession { session, decode }))
        } else {
            bail!("Inflight::new: model topology has no registered HipModel impl")
        }
    }

    /// Free per-rank caches + scratches. The hybrid variant also takes
    /// the parent [`LoadedModel`] (since each per-stage sub-cluster
    /// lives inside the model, not on the server-owned global
    /// cluster); pass `model` from the same registration.
    pub fn dispose(self, cluster: &HipCluster, model: &LoadedModel) -> Result<()> {
        match self {
            Inflight::Pp(PpHipSession {
                session,
                prefill,
                decode,
            }) => {
                decode
                    .dispose(cluster)
                    .context("dispose PP decode scratch")?;
                prefill
                    .dispose(cluster)
                    .context("dispose PP prefill scratch")?;
                session.dispose(cluster).context("dispose PP session")?;
                Ok(())
            }
            Inflight::Tp(TpHipSession { session, decode }) => {
                decode
                    .dispose(cluster)
                    .context("dispose TP decode scratch")?;
                session.dispose(cluster).context("dispose TP session")?;
                Ok(())
            }
            Inflight::Hybrid(HybridHipSession { session, decode }) => {
                let h = model.as_hybrid().context(
                    "Inflight::Hybrid::dispose: paired LoadedModel is not a hybrid topology",
                )?;
                decode
                    .dispose(&h.model)
                    .context("dispose hybrid decode scratch")?;
                session
                    .dispose(&h.model)
                    .context("dispose hybrid session")?;
                Ok(())
            }
        }
    }

    /// **P2.9a (slot pool)** — reset KV state for the next request
    /// without freeing scratch / KV buffers. Cheap O(num_layers)
    /// walk + a single sync per device. Lets the slot pool reuse the
    /// same Inflight across requests instead of paying the
    /// alloc/dispose cycle every time.
    pub fn reset_for_next_request(
        &mut self,
        cluster: &HipCluster,
        model: &LoadedModel,
    ) -> Result<()> {
        match self {
            Inflight::Pp(PpHipSession { session, .. }) if model.as_pp().is_some() => session
                .reset_for_next_request(cluster)
                .context("reset PP session"),
            Inflight::Tp(TpHipSession { session, .. }) if model.as_tp().is_some() => session
                .reset_for_next_request(cluster)
                .context("reset TP session"),
            Inflight::Hybrid(HybridHipSession { session, .. }) => {
                let h = model.as_hybrid().context(
                    "reset_for_next_request: paired LoadedModel is not a hybrid topology",
                )?;
                session
                    .reset_for_next_request(&h.model)
                    .context("reset Hybrid session")
            }
            _ => bail!("Inflight / LoadedModel variant mismatch in reset_for_next_request"),
        }
    }
}

/// **#229 GDN-at-chunk-boundary** — callback fired after each
/// internal chunk completes during chunked prefill. Receives a
/// host-side KV+GDN snapshot of the session at that boundary and the
/// absolute token-count that the snapshot covers (= `start_position +
/// chunk_end`, always a `chunk_tokens`-aligned multiple). Used by the
/// prefix cache to populate intermediate (prefix-only, no-logits)
/// entries during a fresh prefill so future requests with shared
/// prefixes can restore at any chunk boundary. Callback is NOT fired
/// for the final chunk (which terminates at the prompt end and may
/// be partial-tail-aligned); the caller's full-prompt capture handles
/// that case via `prefix_cache_try_capture_full`. Hybrid topology
/// doesn't fire the callback — its per-stage per-rank snapshot
/// shape is V2 work.
pub type BoundaryCallback<'a> =
    &'a mut dyn FnMut(Vec<Vec<LayerCacheSnapshot>>, usize) -> Result<()>;

/// Capture a host-side KV/GDN snapshot of every rank in a PP session.
/// Mirrors the per-rank loop inside `capture_kv_from_inflight`'s PP
/// arm but takes the session directly (so it can be called from
/// inside `prefill_logits`'s match arm where the inflight is
/// destructured).
fn snapshot_pp_session(
    session: &Qwen3MoEShardedSession,
    cluster: &HipCluster,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    use flambeau_qwen3_moe::session::snapshot_layer_caches_to_host;
    let mut out = Vec::with_capacity(session.per_rank.len());
    for (rank_idx, rank) in session.per_rank.iter().enumerate() {
        let device = cluster.device(rank_idx);
        let s = snapshot_layer_caches_to_host(&rank.caches, device)
            .with_context(|| format!("PP snapshot rank {rank_idx}"))?;
        out.push(s);
    }
    Ok(out)
}

/// Capture a host-side KV/GDN snapshot of every rank in a TP session.
fn snapshot_tp_caches(
    caches: &[Vec<flambeau_qwen3_moe::session::LayerCache>],
    cluster: &HipCluster,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    use flambeau_qwen3_moe::session::snapshot_layer_caches_to_host;
    let mut out = Vec::with_capacity(caches.len());
    for (rank_idx, c) in caches.iter().enumerate() {
        let device = cluster.device(rank_idx);
        let s = snapshot_layer_caches_to_host(c, device)
            .with_context(|| format!("TP snapshot rank {rank_idx}"))?;
        out.push(s);
    }
    Ok(out)
}

/// **#229 Hybrid** — capture a host-side KV/GDN snapshot of every
/// (stage, tp_rank) pair in a Hybrid (PP+TP) session. Returned as a
/// flat `Vec<Vec<LayerCacheSnapshot>>` indexed by global rank
/// `g = stage_idx * tp_size + tp_rank`. Each stage's
/// `caches[tp_rank]` holds only the layers in that stage's
/// `layer_range`, so the inner `Vec<LayerCacheSnapshot>` lengths
/// vary across global-rank entries (one entry's inner length equals
/// the owning stage's layer_range size). The cache's TopologyTag
/// carries `pp_size`/`tp_size`, so the consumer can unpack the flat
/// vector unambiguously.
fn snapshot_hybrid_session(
    session: &flambeau_qwen3_moe::Qwen3MoEHybridSession,
    model: &flambeau_qwen3_moe::Qwen3MoEHybridModel,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    use flambeau_qwen3_moe::session::snapshot_layer_caches_to_host;
    let mut out: Vec<Vec<LayerCacheSnapshot>> = Vec::with_capacity(
        session
            .stages
            .iter()
            .map(|s| s.caches.len())
            .sum::<usize>(),
    );
    for (stage_idx, stage_session) in session.stages.iter().enumerate() {
        let stage_model = model
            .stages
            .get(stage_idx)
            .ok_or_else(|| anyhow::anyhow!("hybrid snapshot: stage {stage_idx} missing in model"))?;
        for (tp_rank, layer_caches) in stage_session.caches.iter().enumerate() {
            let device = stage_model.sub_cluster.device(tp_rank);
            let s = snapshot_layer_caches_to_host(layer_caches, device)
                .with_context(|| format!("Hybrid snapshot stage {stage_idx} rank {tp_rank}"))?;
            out.push(s);
        }
    }
    Ok(out)
}


/// Ingest the full prompt and write the logits row for the **last**
/// prompt position into `logits_out`. Each topology dispatches through
/// its own `forward_prefill_*_logits` entry point; the TP path is a
/// per-token loop today () and gets batched-across-L kernels
/// in /c.
/// `tp_pool_prefill`: when `Some` and topology is TP, the pooled
/// `forward_prefill_tp_logits_pooled` is used, skipping per-call
/// alloc/dispose. Callers must already hold `prefill_serialiser`
/// (the TP/Hybrid chat handlers do for #321) — the scratch isn't
/// safe for parallel use. `None` falls back to alloc-per-call.
/// **#229** — `start_position` is the position-offset of the first
/// token in `prompt_ids` within the *original* full prompt. `0` means
/// the entire prompt is being prefilled from scratch (today's behaviour
/// for first turn / cache miss). `> 0` means the caller restored a
/// prefix-cache snapshot covering `[0..start_position)` and is now
/// prefilling only the tail; the per-layer `current_tokens` is already
/// set to `start_position` by the restore step.
/// **#229 GDN-boundary** — `on_boundary`, when set, is invoked after
/// every internal chunk completes (PP/TP only — Hybrid ignores). The
/// callback receives a host-side snapshot at that boundary and the
/// absolute token count covered. Use this to populate prefix-cache
/// entries at every chunk boundary, enabling prefix-match hits on
/// future requests that share an extending prefix.
pub fn prefill_logits(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    prompt_ids: &[u32],
    start_position: usize,
    logits_out: &mut Vec<f32>,
    tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
    mut on_boundary: Option<BoundaryCallback<'_>>,
    prefill_ubatch: usize,
) -> Result<()> {
    if prompt_ids.is_empty() {
        bail!("prefill_logits: empty prompt");
    }
    if let (Some(p), Some(pp_s)) = (model.as_pp(), inflight.as_pp_mut()) {
        let m = &p.model;
        let session = &mut pp_s.session;
        let prefill = &mut pp_s.prefill;
        {
            // **#229 GDN-boundary** — explicit per-chunk loop (mirrors
            // TP/Hybrid below). Earlier this arm relied on
            // `forward_prefill_pp`'s internal recursion for non-final
            // chunks, but that hides chunk boundaries from the caller.
            // Iterating explicitly lets us fire `on_boundary` after
            // each non-final chunk so the prefix cache can record
            // intermediate KV+GDN snapshots. KV/GDN parity at chunk
            // boundaries is verified in Phase A.
            let chunk = prefill.per_rank[0].max_tokens;
            let l = prompt_ids.len();
            if l <= chunk {
                forward_prefill_pp_logits(
                    m, session, cluster, prefill, prompt_ids, start_position, logits_out,
                )
                .context("PP prefill_logits")?;
            } else {
                tracing::debug!(
                    target: "server.prefill",
                    prompt_len = l,
                    chunk,
                    start_position,
                    "chunked PP prefill (explicit per-chunk)"
                );
                let mut start = 0usize;
                while start < l {
                    let end = (start + chunk).min(l);
                    let is_last = end == l;
                    if is_last {
                        forward_prefill_pp_logits(
                            m,
                            session,
                            cluster,
                            prefill,
                            &prompt_ids[start..end],
                            start_position + start,
                            logits_out,
                        )
                        .with_context(|| {
                            format!("PP prefill_logits chunk [{start}..{end}) (final)")
                        })?;
                    } else {
                        forward_prefill_pp(
                            m,
                            session,
                            cluster,
                            prefill,
                            &prompt_ids[start..end],
                            start_position + start,
                        )
                        .with_context(|| {
                            format!("PP prefill chunk [{start}..{end})")
                        })?;
                        if let Some(cb) = on_boundary.as_mut() {
                            let snap = snapshot_pp_session(session, cluster)
                                .context("PP boundary snapshot")?;
                            cb(snap, start_position + end)?;
                        }
                    }
                    start = end;
                }
            }
            Ok(())
        }
    } else if let (Some(t), Some(tp_s)) = (model.as_tp(), inflight.as_tp_mut()) {
        let model = &t.model;
        let ar = t.ar();
        let session = &mut tp_s.session;
        let decode = &mut tp_s.decode;
        {
            // Chunked TP prefill (Phase B3a-TP). Phase A2-TP parity
            // test verified bit-exact KV at L=4096 chunk=512 (8 chunks),
            // so chunking is safe at chunk>=128. **#324** — when
            // `tp_pool_prefill` is `Some`, the caller (chat handler
            // holding `ServerState::prefill_serialiser`) hands us a
            // shared pre-allocated scratch and we skip per-call alloc.
            // When `None`, fall back to the legacy alloc-per-call
            // path inside `forward_prefill_tp_logits`.
            let chunk = prefill_ubatch.max(128);
            let l = prompt_ids.len();
            // We can't reuse `&mut tp_pool_prefill` across loop iterations
            // because the pooled call holds a reborrow; instead, take()
            // a local Option and re-store at end. But for the simple
            // single-chunk path we can just pass the &mut directly.
            if let Some(pool) = tp_pool_prefill {
                if l <= chunk {
                    forward_prefill_tp_logits_pooled(
                        model, decode, pool, cluster, ar, &mut session.caches,
                        prompt_ids, start_position, logits_out,
                    )
                    .context("TP prefill_logits (pooled)")?;
                } else {
                    tracing::debug!(
                        target: "server.prefill",
                        prompt_len = l,
                        chunk,
                        start_position,
                        "chunked TP prefill (pooled)"
                    );
                    let mut start = 0usize;
                    let mut sink: Vec<f32> = Vec::new();
                    while start < l {
                        let end = (start + chunk).min(l);
                        let is_last = end == l;
                        let dst: &mut Vec<f32> =
                            if is_last { &mut *logits_out } else { &mut sink };
                        forward_prefill_tp_logits_pooled(
                            model, decode, pool, cluster, ar, &mut session.caches,
                            &prompt_ids[start..end], start_position + start, dst,
                        )
                        .with_context(|| {
                            format!("TP prefill_logits chunk [{start}..{end}) (pooled)")
                        })?;
                        if !is_last {
                            if let Some(cb) = on_boundary.as_mut() {
                                let snap = snapshot_tp_caches(&session.caches, cluster)
                                    .context("TP boundary snapshot (pooled)")?;
                                cb(snap, start_position + end)?;
                            }
                        }
                        start = end;
                    }
                }
            } else if l <= chunk {
                forward_prefill_tp_logits(
                    model, decode, cluster, ar, &mut session.caches,
                    prompt_ids, start_position, logits_out,
                )
                .context("TP prefill_logits")?;
            } else {
                tracing::debug!(
                    target: "server.prefill",
                    prompt_len = l,
                    chunk,
                    start_position,
                    "chunked TP prefill"
                );
                let mut start = 0usize;
                let mut sink: Vec<f32> = Vec::new();
                while start < l {
                    let end = (start + chunk).min(l);
                    let is_last = end == l;
                    let dst: &mut Vec<f32> =
                        if is_last { &mut *logits_out } else { &mut sink };
                    forward_prefill_tp_logits(
                        model, decode, cluster, ar, &mut session.caches,
                        &prompt_ids[start..end], start_position + start, dst,
                    )
                    .with_context(|| format!("TP prefill_logits chunk [{start}..{end})"))?;
                    if !is_last {
                        if let Some(cb) = on_boundary.as_mut() {
                            let snap = snapshot_tp_caches(&session.caches, cluster)
                                .context("TP boundary snapshot")?;
                            cb(snap, start_position + end)?;
                        }
                    }
                    start = end;
                }
            }
            Ok(())
        }
    } else if let (Some(h), Some(hyb_s)) = (model.as_hybrid(), inflight.as_hybrid_mut()) {
        let hmodel = &h.model;
        let stage_ars = &h.stage_ars();
        let session = &mut hyb_s.session;
        let decode = &mut hyb_s.decode;
        {
            // Chunked Hybrid prefill (Phase B4a-Hybrid). Parity
            // verified bit-exact at L=4096 chunk=512 (8 chunks).
            let chunk = prefill_ubatch.max(128);
            let l = prompt_ids.len();
            if l <= chunk {
                forward_prefill_hybrid_logits(
                    hmodel, decode, cluster, stage_ars, session, prompt_ids, start_position, logits_out,
                )
                .context("hybrid prefill_logits")?;
            } else {
                tracing::debug!(
                    target: "server.prefill",
                    prompt_len = l,
                    chunk,
                    start_position,
                    "chunked hybrid prefill"
                );
                let mut start = 0usize;
                let mut sink: Vec<f32> = Vec::new();
                while start < l {
                    let end = (start + chunk).min(l);
                    let is_last = end == l;
                    let dst: &mut Vec<f32> =
                        if is_last { &mut *logits_out } else { &mut sink };
                    forward_prefill_hybrid_logits(
                        hmodel, decode, cluster, stage_ars, session,
                        &prompt_ids[start..end], start_position + start, dst,
                    )
                    .with_context(|| {
                        format!("hybrid prefill_logits chunk [{start}..{end})")
                    })?;
                    if !is_last {
                        if let Some(cb) = on_boundary.as_mut() {
                            let snap = snapshot_hybrid_session(session, hmodel)
                                .context("Hybrid boundary snapshot")?;
                            cb(snap, start_position + end)?;
                        }
                    }
                    start = end;
                }
            }
            Ok(())
        }
    } else {
        bail!("LoadedModel/Inflight variant mismatch")
    }
}

/// Advance one token; write that position's logits into `logits_out`.
pub fn decode_logits(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    token: u32,
    position: usize,
    logits_out: &mut Vec<f32>,
) -> Result<()> {
    // **Cleanup (post-i2-B-wire)** — the old i2-A1 env-flag dispatch
    // here was a stepping stone that engaged the new batched code
    // path for PP at N=1 before the scheduler shipped. Now that the
    // scheduler-aware handler in `routes.rs::run_completion_scheduler_pp_blocking`
    // exists for PP / TP / Hybrid, the env flag belongs entirely at
    // handler entry (`scheduler_can_engage`). `decode_logits` itself
    // is the legacy fallback path — invariant: takes a held mutex
    // guard and runs the single-slot forward.
    if let (Some(p), Some(s)) = (model.as_pp(), inflight.as_pp_mut()) {
        forward_one_token_pp_logits(
            &p.model,
            &mut s.session,
            cluster,
            &mut s.decode,
            token,
            position,
            logits_out,
        )
        .context("PP decode_logits")
    } else if let (Some(t), Some(s)) = (model.as_tp(), inflight.as_tp_mut()) {
        forward_one_token_tp_logits(
            &t.model,
            &mut s.decode,
            cluster,
            t.ar(),
            &mut s.session.caches,
            token,
            position,
            logits_out,
        )
        .context("TP decode_logits")
    } else if let (Some(h), Some(s)) = (model.as_hybrid(), inflight.as_hybrid_mut()) {
        forward_one_token_hybrid_logits(
            &h.model,
            &mut s.decode,
            cluster,
            &h.stage_ars(),
            &mut s.session,
            token,
            position,
            logits_out,
        )
        .context("hybrid decode_logits")
    } else {
        bail!("LoadedModel/Inflight variant mismatch")
    }
}

/// Like [`decode_logits`] but skips the F32 logits DtoH; the row stays
/// in the head rank's `output_head.logits_f32` for the GPU sampler.
pub fn decode_keep_logits_on_device(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    token: u32,
    position: usize,
) -> Result<()> {
    if let (Some(p), Some(s)) = (model.as_pp(), inflight.as_pp_mut()) {
        forward_one_token_pp_keep_logits_on_device(
            &p.model,
            &mut s.session,
            cluster,
            &mut s.decode,
            token,
            position,
        )
        .context("PP decode_keep_logits_on_device")
    } else if let (Some(t), Some(s)) = (model.as_tp(), inflight.as_tp_mut()) {
        forward_one_token_tp_keep_logits_on_device(
            &t.model,
            &mut s.decode,
            cluster,
            t.ar(),
            &mut s.session.caches,
            token,
            position,
        )
        .context("TP decode_keep_logits_on_device")
    } else if let (Some(h), Some(s)) = (model.as_hybrid(), inflight.as_hybrid_mut()) {
        forward_one_token_hybrid_keep_logits_on_device(
            &h.model,
            &mut s.decode,
            cluster,
            &h.stage_ars(),
            &mut s.session,
            token,
            position,
        )
        .context("Hybrid decode_keep_logits_on_device")
    } else {
        bail!("decode_keep_logits_on_device: model topology has no GPU sampler wiring")
    }
}

/// **#229 P2.10c** — capture the active inflight session's KV state
/// into a host-side snapshot, sized for the current `current_tokens`
/// of every layer.
/// Layout: `result[r]` covers rank `r`'s full layer set. PP and TP
/// supported; Hybrid bails (V2 follow-up — see `restore_kv_into_inflight`).
/// **Cost**: D→H copy of `total_bytes()` per rank. On 27B/TP2/ctx=4096
/// that's ~1 GB across 2 ranks, ~150 ms over PCIe 3.0 x16. On hit the
/// inverse H→D pays the same — still a net win vs the ~2-3 s prefill
/// it replaces.
pub fn capture_kv_from_inflight(
    inflight: &Inflight,
    cluster: &HipCluster,
    model: &LoadedModel,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    use crate::model_extensions::KvSnapshot;
    if let (Inflight::Pp(s), Some(_)) = (inflight, model.as_pp()) {
        return s.capture(cluster);
    }
    if let (Inflight::Tp(s), Some(_)) = (inflight, model.as_tp()) {
        return s.capture(cluster);
    }
    if let (Inflight::Hybrid(s), Some(h)) = (inflight, model.as_hybrid()) {
        return s.capture_with_model(h);
    }
    bail!("capture_kv_from_inflight: model/inflight variant mismatch")
}

/// **#229** — total host-RAM bytes a snapshot occupies. Used by the
/// LRU's VRAM budget accounting (despite the name, the backing store
/// is host RAM in V1; #229 reuses the same field for host-RAM
/// accounting and renames are V2).
pub fn snapshot_bytes(snapshot: &[Vec<LayerCacheSnapshot>]) -> usize {
    snapshot
        .iter()
        .flat_map(|rank| rank.iter())
        .map(|s| match s {
            LayerCacheSnapshot::FullAttn { k, v, .. } => k.len() + v.len(),
            LayerCacheSnapshot::Gdn { state, conv_history } => {
                state.len() + conv_history.len()
            }
        })
        .sum()
}

/// **#229** — true if any layer in the snapshot is a `Gdn` recurrent
/// state. GDN state is a single-step recurrent matrix (not
/// position-indexed), so it cannot be cleanly truncated to a chunk
/// boundary the way KV slabs can. Callers gate the prefix cache off
/// for snapshots containing GDN layers in V1; chunk-boundary GDN
/// snapshotting is V2 work.
pub fn snapshot_has_gdn(snapshot: &[Vec<LayerCacheSnapshot>]) -> bool {
    snapshot
        .iter()
        .flat_map(|rank| rank.iter())
        .any(|s| matches!(s, LayerCacheSnapshot::Gdn { .. }))
}

/// **#229** — truncate a freshly-captured snapshot down to
/// `n_target_tokens` of FullAttn KV state. Used at insert time to
/// store only the chunk-boundary prefix (the post-prefill snapshot
/// covers the full prompt, but the cache key chain identifies a
/// shorter prefix).
/// Each FullAttn layer's K/V byte buffer is truncated to
/// `n_target_tokens * bytes_per_token` (where `bytes_per_token =
/// existing_bytes / current_tokens`). `current_tokens` is updated to
/// `n_target_tokens`. GDN layers cannot be truncated and are passed
/// through unchanged — callers should bail before reaching here when
/// any layer is GDN. (`snapshot_has_gdn` is the gate.)
pub fn truncate_snapshot_to_tokens(
    snapshot: &mut [Vec<LayerCacheSnapshot>],
    n_target_tokens: usize,
) {
    for rank in snapshot.iter_mut() {
        for layer in rank.iter_mut() {
            if let LayerCacheSnapshot::FullAttn {
                k,
                v,
                current_tokens,
            } = layer
            {
                if *current_tokens <= n_target_tokens {
                    continue;
                }
                let bytes_per_token = k.len() / (*current_tokens).max(1);
                let new_bytes = bytes_per_token * n_target_tokens;
                k.truncate(new_bytes);
                v.truncate(new_bytes);
                k.shrink_to_fit();
                v.shrink_to_fit();
                *current_tokens = n_target_tokens;
            }
        }
    }
}

/// **#228 P2.10b** — restore a host-side KV snapshot into the active
/// inflight session.
/// `snapshot[r]` covers rank `r`'s full layer set (the same layout
/// `snapshot_layer_caches_to_host` produces). The caller is responsible
/// for ensuring the snapshot was captured under the same topology and
/// chunk size — the `PrefixCache::longest_match` lookup checks this
/// before this function is called.
/// Errors when the topology is `Hybrid` (V2 follow-up — the per-stage
/// per-rank shape doesn't match the flat `Vec<RankSnapshot>` layout).
/// Callers should skip prefix-cache restore for hybrid models in V1.
pub fn restore_kv_into_inflight(
    inflight: &mut Inflight,
    cluster: &HipCluster,
    snapshot: &[Vec<LayerCacheSnapshot>],
    model: &LoadedModel,
) -> Result<()> {
    use crate::model_extensions::KvSnapshot;
    if let (Inflight::Pp(s), Some(_)) = (&mut *inflight, model.as_pp()) {
        return s.restore(cluster, snapshot);
    }
    if let (Inflight::Tp(s), Some(_)) = (&mut *inflight, model.as_tp()) {
        return s.restore(cluster, snapshot);
    }
    if let (Inflight::Hybrid(s), Some(h)) = (&mut *inflight, model.as_hybrid()) {
        return s.restore_with_model(h, snapshot);
    }
    bail!("restore_kv_into_inflight: model/inflight variant mismatch")
}


