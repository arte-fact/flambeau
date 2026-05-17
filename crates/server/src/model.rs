//! Qwen3-moe-specific glue: per-request `Inflight` session, the
//! prefill / decode free functions, the shared `Qwen3MoeServerExtras`
//! workspaces, and the `qwen3moe_forward_decode_batched` dispatcher
//! called by `Model::forward_decode_batched`'s default impl.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{BarP2pAllReduce, HipCluster};
use flambeau_qwen3_moe::forward::{
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

/// Qwen3-moe-specific shared workspaces lazy-attached to
/// `ServerState`. `Some` on the qwen3-moe boot path; `None` on
/// non-qwen3-moe boots (gemma4) so they don't carry the
/// qwen3-moe-typed scratch fields.
pub struct Qwen3MoeServerExtras {
    /// Shared TP batched-decode workspace, lazy-init on first TP
    /// scheduler dispatch. Only the dispatcher leader touches it
    /// (gated by `ServerState::batched_dispatcher`); inner `Mutex`
    /// is just for safe lazy-init.
    pub tp_batched_scratch:
        std::sync::Mutex<Option<ShardedForwardPrefillScratchTp>>,
    /// Shared Hybrid (PP+TP) batched-decode workspace.
    pub hybrid_batched_scratch:
        std::sync::Mutex<Option<flambeau_qwen3_moe::ShardedForwardPrefillScratchHybrid>>,
    /// TP / Hybrid prefill serialiser. Caps peak per-call scratch
    /// alloc (~35 MB at chunk=512 for Qwen3.6-27B) at one instance
    /// regardless of N concurrent requests; the GPU stream is serial
    /// anyway, so this only serialises host-side launch + alloc.
    pub prefill_serialiser: std::sync::Mutex<()>,
    /// Shared TP prefill scratch, lazy-init on first TP prefill.
    /// Reused across every TP prefill call.
    pub tp_prefill_scratch:
        std::sync::Mutex<Option<ShardedForwardPrefillScratchTp>>,
}

impl Default for Qwen3MoeServerExtras {
    fn default() -> Self {
        Self {
            tp_batched_scratch: std::sync::Mutex::new(None),
            hybrid_batched_scratch: std::sync::Mutex::new(None),
            prefill_serialiser: std::sync::Mutex::new(()),
            tp_prefill_scratch: std::sync::Mutex::new(None),
        }
    }
}

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

/// Loaded weights + per-topology auxiliary state. `Arc<dyn Model>`
/// so per-request sessions can hold a cheap back-reference.
pub type LoadedModel = std::sync::Arc<dyn crate::model_handle::Model>;

/// Extension trait for qwen3-moe model topology accessors. Imported
/// by qwen3-moe-typed callers; downcasts through the trait's
/// `as_any()` hatch. Keeps the [`Model`](crate::model_handle::Model)
/// trait surface arch-clean (CLAUDE.md rule 12 + 13).
pub trait Qwen3MoeModelExt {
    fn as_pp(&self) -> Option<&PpHipModel>;
    fn as_tp(&self) -> Option<&TpHipModel>;
    fn as_hybrid(&self) -> Option<&HybridHipModel>;
}

impl<T: ?Sized + crate::model_handle::Model> Qwen3MoeModelExt for T {
    fn as_pp(&self) -> Option<&PpHipModel> {
        self.as_any().downcast_ref()
    }
    fn as_tp(&self) -> Option<&TpHipModel> {
        self.as_any().downcast_ref()
    }
    fn as_hybrid(&self) -> Option<&HybridHipModel> {
        self.as_any().downcast_ref()
    }
}

/// Extension trait for qwen3-moe session accessors. Imported by
/// qwen3-moe-typed callers; downcasts through `Session::as_any` /
/// `as_any_mut`.
pub trait Qwen3MoeSessionExt {
    fn as_pp(&self) -> Option<&PpHipSession>;
    fn as_pp_mut(&mut self) -> Option<&mut PpHipSession>;
    fn as_tp(&self) -> Option<&TpHipSession>;
    fn as_tp_mut(&mut self) -> Option<&mut TpHipSession>;
    fn as_hybrid(&self) -> Option<&HybridHipSession>;
    fn as_hybrid_mut(&mut self) -> Option<&mut HybridHipSession>;
}

impl<T: ?Sized + crate::model_handle::Session> Qwen3MoeSessionExt for T {
    fn as_pp(&self) -> Option<&PpHipSession> {
        self.as_any()
            .downcast_ref::<crate::model_handle::Qwen3MoeOwnedSession>()
            .and_then(|s| s.inflight.as_pp())
    }
    fn as_pp_mut(&mut self) -> Option<&mut PpHipSession> {
        self.as_any_mut()
            .downcast_mut::<crate::model_handle::Qwen3MoeOwnedSession>()
            .and_then(|s| s.inflight.as_pp_mut())
    }
    fn as_tp(&self) -> Option<&TpHipSession> {
        self.as_any()
            .downcast_ref::<crate::model_handle::Qwen3MoeOwnedSession>()
            .and_then(|s| s.inflight.as_tp())
    }
    fn as_tp_mut(&mut self) -> Option<&mut TpHipSession> {
        self.as_any_mut()
            .downcast_mut::<crate::model_handle::Qwen3MoeOwnedSession>()
            .and_then(|s| s.inflight.as_tp_mut())
    }
    fn as_hybrid(&self) -> Option<&HybridHipSession> {
        self.as_any()
            .downcast_ref::<crate::model_handle::Qwen3MoeOwnedSession>()
            .and_then(|s| s.inflight.as_hybrid())
    }
    fn as_hybrid_mut(&mut self) -> Option<&mut HybridHipSession> {
        self.as_any_mut()
            .downcast_mut::<crate::model_handle::Qwen3MoeOwnedSession>()
            .and_then(|s| s.inflight.as_hybrid_mut())
    }
}

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
            bail!("Inflight::new: model topology has no registered Model impl")
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
    inflight: &mut dyn crate::Session,
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
    // Phase 12.9 — gemma4 routes through ModelDriver, bypassing the
    // qwen3-moe sharded-session machinery. The driver owns its own
    // cluster + chunked-prefill story, so we ignore `tp_pool_prefill`
    // / `on_boundary` / `prefill_ubatch` here (per-arch follow-ups).
    // BOS prepend: gemma4 mandates BOS as token 0; the server-side
    // tokenizer doesn't add specials, so we consult the session's
    // recorded BOS id and prepend on a fresh prefill (start_position
    // == 0). Aligning the cache tail to N+1 makes the subsequent
    // decode positions from routes.rs (`prompt_ids.len() + step`)
    // land on the right slots.
    if inflight.as_model_driver_mut().is_some() {
        let _ = (cluster, tp_pool_prefill, prefill_ubatch);
        let _ = on_boundary;
        let bos_id = inflight.bos_id();
        let owned: Vec<u32>;
        let prompt_slice: &[u32] = if start_position == 0
            && bos_id.is_some()
            && prompt_ids.first() != bos_id.as_ref()
        {
            let bos = bos_id.expect("checked Some above");
            owned = std::iter::once(bos).chain(prompt_ids.iter().copied()).collect();
            owned.as_slice()
        } else {
            prompt_ids
        };
        tracing::debug!(
            target: "server.gemma4",
            in_len = prompt_ids.len(),
            out_len = prompt_slice.len(),
            start_position,
            bos = ?bos_id,
            "gemma4 prefill (BOS-prepended: {})",
            prompt_slice.len() > prompt_ids.len()
        );
        let driver = inflight
            .as_model_driver_mut()
            .expect("checked above");
        return driver
            .forward_prefill_logits(prompt_slice, start_position, logits_out)
            .context("gemma4 forward_prefill_logits");
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
        let tp = &t.tp;
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
                        model, decode, pool, tp, &mut session.caches,
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
                            model, decode, pool, tp, &mut session.caches,
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
                    model, decode, tp, &mut session.caches,
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
                        model, decode, tp, &mut session.caches,
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
    inflight: &dyn crate::Session,
    cluster: &HipCluster,
    model: &LoadedModel,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    use crate::model_extensions::KvSnapshot;
    if let (Some(s), Some(_)) = (inflight.as_pp(), model.as_pp()) {
        return s.capture(cluster);
    }
    if let (Some(s), Some(_)) = (inflight.as_tp(), model.as_tp()) {
        return s.capture(cluster);
    }
    if let (Some(s), Some(h)) = (inflight.as_hybrid(), model.as_hybrid()) {
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
    inflight: &mut dyn crate::Session,
    cluster: &HipCluster,
    snapshot: &[Vec<LayerCacheSnapshot>],
    model: &LoadedModel,
) -> Result<()> {
    use crate::model_extensions::KvSnapshot;
    if let (Some(s), Some(_)) = (inflight.as_pp_mut(), model.as_pp()) {
        return s.restore(cluster, snapshot);
    }
    if let (Some(s), Some(_)) = (inflight.as_tp_mut(), model.as_tp()) {
        return s.restore(cluster, snapshot);
    }
    if let (Some(s), Some(h)) = (inflight.as_hybrid_mut(), model.as_hybrid()) {
        return s.restore_with_model(h, snapshot);
    }
    bail!("restore_kv_into_inflight: model/inflight variant mismatch")
}

/// Qwen3-moe batched-decode dispatcher. Caller (per-arch
/// `Model::forward_decode_batched` impl) passes its `&self` as `model`,
/// plus the `SessionContext` for cluster + shared extras. Each inflight
/// downcasts to its concrete qwen3-moe session type; the shared scratch
/// comes from the context's `Qwen3MoeServerExtras` (lazy-allocated on
/// first call).
pub fn qwen3moe_forward_decode_batched(
    model: &dyn crate::Model,
    ctx: &dyn crate::model_handle::SessionContext,
    inflights: &mut [&mut dyn crate::Session],
    slots: &[crate::model_handle::BatchSlot],
    logits_refs: &mut [&mut Vec<f32>],
) -> Result<()> {
    use flambeau_qwen3_moe::forward::{
        forward_decode_batched_hybrid, forward_decode_batched_pp, forward_decode_batched_tp,
        BatchSlot as QwenBatchSlot,
    };
    let slots: Vec<QwenBatchSlot> = slots
        .iter()
        .map(|s| QwenBatchSlot {
            idx: s.idx,
            token_id: s.token_id,
            position: s.position,
        })
        .collect();
    let slots = slots.as_slice();
    let cluster: &HipCluster = ctx.cluster();
    let qwen3_moe = ctx
        .extras()
        .and_then(|a| a.downcast_ref::<Qwen3MoeServerExtras>());
    let max_inflight_slots = ctx.max_inflight_slots();
    let n = inflights.len();
    if n == 0 {
        bail!("qwen3moe_forward_decode_batched: empty inflight slice");
    }
    let inflights_ptr = inflights.as_mut_ptr();

    if let Some(pp_model) = model.as_pp() {
        let model = &pp_model.model;
        // N=1 fused fast-path. The batched `forward_decode_batched_pp`
        // at N=1 replays prefill-flavoured kernels (separate rmsnorm +
        // 2 quant variants per layer) that add ~7-8 ms/token of host
        // launch + HBM round-trip overhead vs the fused
        // `flambeau_rmsnorm_q8_1_fused` form used by the legacy
        // `forward_one_token_pp_logits` decode path. The fused path
        // owns its own `decode: ShardedForwardOneTokenScratch` scratch
        // on each inflight (separate from `prefill` which the batched
        // path uses), so N=1 routing through it is independent of any
        // batched bookkeeping.
        // See `certs/perf/tg_breakdown_2026_05_17/chat_path_breakdown.md`.
        if n == 1 {
            use flambeau_qwen3_moe::forward::forward_one_token_pp_logits;
            let slot = &slots[0];
            // SAFETY: n == 1; inflights[0] is the only borrow.
            let pp = unsafe {
                (&mut **inflights_ptr)
                    .as_pp_mut()
                    .context("N=1 fused PP: slot 0 is not PP")?
            };
            let logits_out: &mut Vec<f32> = logits_refs[0];
            logits_out.clear();
            return forward_one_token_pp_logits(
                model,
                &mut pp.session,
                cluster,
                &mut pp.decode,
                slot.token_id,
                slot.position,
                logits_out,
            )
            .context("forward_one_token_pp_logits (N=1 fused fast-path)");
        }
        let mut sessions: Vec<&mut Qwen3MoEShardedSession> = Vec::with_capacity(n);
        // SAFETY: n >= 1; disjoint reborrow of slot 0's `prefill` field
        // from the per-slot `session` borrows below.
        let prefill_scratch: &mut ShardedForwardPrefillScratch = unsafe {
            let g0: &mut dyn crate::Session = &mut **inflights_ptr;
            &mut g0
                .as_pp_mut()
                .context("batched decode: leader slot is not PP")?
                .prefill
        };
        for s in 0..n {
            // SAFETY: s in 0..n; inflights distinct by index.
            unsafe {
                let g: &mut dyn crate::Session = &mut **inflights_ptr.add(s);
                let pp = g
                    .as_pp_mut()
                    .with_context(|| format!("batched decode: slot {s} is not PP"))?;
                sessions.push(&mut pp.session);
            }
        }
        forward_decode_batched_pp(
            model,
            sessions.as_mut_slice(),
            cluster,
            prefill_scratch,
            slots,
            logits_refs,
        )
        .context("forward_decode_batched_pp")
    } else if let Some(tp_model) = model.as_tp() {
        let model = &tp_model.model;
        // N=1 fused fast-path — same rationale as PP.
        if n == 1 {
            use flambeau_qwen3_moe::forward::forward_one_token_tp_logits;
            let slot = &slots[0];
            let tp = unsafe {
                (&mut **inflights_ptr)
                    .as_tp_mut()
                    .context("N=1 fused TP: slot 0 is not TP")?
            };
            let logits_out: &mut Vec<f32> = logits_refs[0];
            logits_out.clear();
            return forward_one_token_tp_logits(
                model,
                &mut tp.decode,
                &tp_model.tp,
                &mut tp.session.caches,
                slot.token_id,
                slot.position,
                logits_out,
            )
            .context("forward_one_token_tp_logits (N=1 fused fast-path)");
        }
        let mut sessions: Vec<&mut flambeau_qwen3_moe::Qwen3MoETpSession> =
            Vec::with_capacity(n);
        for s in 0..n {
            // SAFETY: s in 0..n; inflights distinct.
            unsafe {
                let g: &mut dyn crate::Session = &mut **inflights_ptr.add(s);
                let tp = g
                    .as_tp_mut()
                    .with_context(|| format!("batched decode: slot {s} is not TP"))?;
                sessions.push(&mut tp.session);
            }
        }
        let qwen3_moe = qwen3_moe.context("TP batched dispatch requires qwen3-moe boot")?;
        let mut scratch_guard = qwen3_moe
            .tp_batched_scratch
            .lock()
            .expect("tp_batched_scratch poisoned");
        if scratch_guard.is_none() {
            let max_slots = max_inflight_slots.max(n);
            *scratch_guard = Some(
                ShardedForwardPrefillScratchTp::new(&model.config, cluster, max_slots)
                    .context("alloc tp_batched_scratch")?,
            );
        }
        let scratch = scratch_guard.as_mut().expect("just initialised");
        forward_decode_batched_tp(
            model,
            sessions.as_mut_slice(),
            &tp_model.tp,
            scratch,
            slots,
            logits_refs,
        )
        .context("forward_decode_batched_tp")
    } else if let Some(hybrid_model) = model.as_hybrid() {
        let model = &hybrid_model.model;
        let stage_ars = &hybrid_model.stage_ars();
        // N=1 fused fast-path — same rationale as PP.
        if n == 1 {
            use flambeau_qwen3_moe::forward::forward_one_token_hybrid_logits;
            let slot = &slots[0];
            let hyb = unsafe {
                (&mut **inflights_ptr)
                    .as_hybrid_mut()
                    .context("N=1 fused Hybrid: slot 0 is not Hybrid")?
            };
            let logits_out: &mut Vec<f32> = logits_refs[0];
            logits_out.clear();
            return forward_one_token_hybrid_logits(
                model,
                &mut hyb.decode,
                cluster,
                stage_ars,
                &mut hyb.session,
                slot.token_id,
                slot.position,
                logits_out,
            )
            .context("forward_one_token_hybrid_logits (N=1 fused fast-path)");
        }
        let mut sessions: Vec<&mut Qwen3MoEHybridSession> = Vec::with_capacity(n);
        for s in 0..n {
            // SAFETY: s in 0..n; inflights distinct.
            unsafe {
                let g: &mut dyn crate::Session = &mut **inflights_ptr.add(s);
                let hyb = g
                    .as_hybrid_mut()
                    .with_context(|| format!("batched decode: slot {s} is not Hybrid"))?;
                sessions.push(&mut hyb.session);
            }
        }
        let qwen3_moe = qwen3_moe.context("Hybrid batched dispatch requires qwen3-moe boot")?;
        let mut scratch_guard = qwen3_moe
            .hybrid_batched_scratch
            .lock()
            .expect("hybrid_batched_scratch poisoned");
        if scratch_guard.is_none() {
            let max_slots = max_inflight_slots.max(n);
            *scratch_guard = Some(
                flambeau_qwen3_moe::ShardedForwardPrefillScratchHybrid::new(model, max_slots)
                    .context("alloc hybrid_batched_scratch")?,
            );
        }
        let scratch = scratch_guard.as_mut().expect("just initialised");
        forward_decode_batched_hybrid(
            model,
            sessions.as_mut_slice(),
            cluster,
            stage_ars,
            scratch,
            slots,
            logits_refs,
        )
        .context("forward_decode_batched_hybrid")
    } else {
        bail!("qwen3moe_forward_decode_batched: model is not a qwen3-moe topology")
    }
}



