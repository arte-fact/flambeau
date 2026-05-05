//! **TP-5a-i2** — server-side LoadedModel + Inflight session abstractions.
//!
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
    forward_one_token_hybrid_logits, forward_one_token_pp_logits,
    forward_one_token_tp_keep_logits_on_device, forward_one_token_tp_logits,
    forward_prefill_hybrid_logits, forward_prefill_pp, forward_prefill_pp_logits,
    forward_prefill_tp_logits, forward_prefill_tp_logits_pooled,
    forward_speculative_pp_step, ShardedForwardOneTokenScratch,
    ShardedForwardOneTokenScratchHybrid, ShardedForwardOneTokenScratchTp,
    ShardedForwardPrefillScratch, ShardedForwardPrefillScratchTp, SpecStep,
};
use flambeau_qwen3_moe::session::{restore_layer_caches_from_host, LayerCacheSnapshot};
use flambeau_qwen3_moe::mtp::{MtpForwardScratch, MtpHeadWeights};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEHybridModel, Qwen3MoEHybridSession, Qwen3MoEShardedModel,
    Qwen3MoEShardedSession, Qwen3MoETpModel, Qwen3MoETpSession,
};
use flambeau_core::DevicePtr;

/// Loaded weights + per-topology auxiliary state.
///
/// Built once at startup. The PP variant just owns the sharded model;
/// the TP variant additionally owns a [`BarP2pAllReduce`] that holds an
/// `Arc<HipCluster>` against the same cluster the server uses.
pub enum LoadedModel {
    /// V1.8 pipeline-parallel sharded model. One whole layer per rank
    /// stage; cross-stage hand-off via host-bounce peer copy.
    /// MTP-5d: optional MTP attachment for spec-decode; loaded at
    /// startup when `FLAMBEAU_SPEC_MTP=path/to/mtp.gguf` is set, lives
    /// on the last rank.
    Pp {
        model: Qwen3MoEShardedModel,
        mtp: Option<MtpHeadWeights>,
    },
    /// V2 tensor-parallel sharded model. Every rank holds every layer
    /// (sliced); intra-layer Megatron splits + BAR1 P2P AllReduce.
    Tp {
        model: Qwen3MoETpModel,
        ar: BarP2pAllReduce,
    },
    /// **AUTO-4f** — hybrid PP-of-TP. `pp_size` contiguous layer
    /// stages, each owning a `tp_size`-rank TP subgroup. The
    /// per-stage `BarP2pAllReduce` instances live alongside the
    /// model; the inter-stage hand-off uses the server-owned global
    /// `HipCluster` passed through to [`prefill_logits`] /
    /// [`decode_logits`].
    Hybrid {
        model: Qwen3MoEHybridModel,
        stage_ars: Vec<BarP2pAllReduce>,
    },
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadedModel::Pp { .. } => f.debug_struct("LoadedModel::Pp").finish(),
            LoadedModel::Tp { .. } => f.debug_struct("LoadedModel::Tp").finish(),
            LoadedModel::Hybrid { model, .. } => f
                .debug_struct("LoadedModel::Hybrid")
                .field("spec", &model.spec)
                .finish(),
        }
    }
}

impl LoadedModel {
    /// Common config shared between variants.
    pub fn config(&self) -> &Qwen3MoEConfig {
        match self {
            LoadedModel::Pp { model: m, .. } => &m.config,
            LoadedModel::Tp { model, .. } => &model.config,
            LoadedModel::Hybrid { model, .. } => &model.config,
        }
    }

    /// Topology label for `tracing` / handler-side metrics.
    pub fn topology(&self) -> &'static str {
        match self {
            LoadedModel::Pp { .. } => "pp",
            LoadedModel::Tp { .. } => "tp",
            LoadedModel::Hybrid { .. } => "pp+tp",
        }
    }
}

/// Per-request session + scratch. Variant must match the
/// [`LoadedModel`] variant used to construct it; mismatches bail at
/// dispatch.
pub enum Inflight {
    Pp {
        session: Qwen3MoEShardedSession,
        prefill: ShardedForwardPrefillScratch,
        decode: ShardedForwardOneTokenScratch,
    },
    Tp {
        session: Qwen3MoETpSession,
        decode: ShardedForwardOneTokenScratchTp,
    },
    Hybrid {
        session: Qwen3MoEHybridSession,
        decode: ShardedForwardOneTokenScratchHybrid,
    },
}

impl Inflight {
    /// Allocate per-rank caches + scratches for a single request.
    /// `prompt_len` sizes the PP prefill scratch; ignored for TP (TP
    /// loops `forward_one_token_tp` over the prompt instead).
    pub fn new(model: &LoadedModel, cluster: &HipCluster, prompt_len: usize) -> Result<Self> {
        // **chunked prefill** (Phase B / task #237). The per-layer
        // attention scratch is O(heads × L²) and OOMs past ~5k tokens
        // on a 16 GB MI50. forward_prefill_pp recursively chunks when
        // L > scratch.max_tokens. Cap scratch ubatch to
        // FLAMBEAU_PREFILL_UBATCH (default 512).
        //
        // **Critical:** chunk size MUST stay in the same MMQ-dispatch
        // bucket as the model expects. On gfx906 / Q4_1 the boundary
        // is m=128: m<128 routes to MMVQ, m>=128 routes to MMQ-4warp;
        // mixing them across calls writes numerically-different K
        // bytes for the same input token (Phase A parity test
        // verified bit-equality only when chunks stay in one bucket).
        // 512 gives generous headroom past the 128 boundary;
        // override-floor is enforced at 128.
        let prefill_ubatch: usize = std::env::var("FLAMBEAU_PREFILL_UBATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|n: &usize| *n >= 128)
            .unwrap_or(512);
        let scratch_tokens = prompt_len.min(prefill_ubatch).max(1);
        let _ = model; // cfg lookups no longer needed; chunking handles size

        // **leak fix** — every alloc step here that ships a fresh GPU
        // resource must roll back the prior steps' GPU resources on
        // failure. Without this, an OOM in a downstream alloc (the
        // common case under long-context prompts) drops the
        // already-built earlier resource via its `Drop` impl, which
        // only WARNS and leaves the device buffers pinned until
        // process exit. One failed request used to lose ~3 GB of
        // VRAM on each rank.
        match model {
            LoadedModel::Pp { model: m, .. } => {
                let session =
                    Qwen3MoEShardedSession::new(m, cluster).context("create PP session")?;
                let prefill = match ShardedForwardPrefillScratch::new(
                    m,
                    cluster,
                    scratch_tokens,
                ) {
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
                Ok(Inflight::Pp {
                    session,
                    prefill,
                    decode,
                })
            }
            LoadedModel::Tp { model, .. } => {
                let session =
                    Qwen3MoETpSession::new(model, cluster).context("create TP session")?;
                let decode = match ShardedForwardOneTokenScratchTp::new(
                    &model.config,
                    cluster,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = session.dispose(cluster);
                        return Err(e).context("TP decode scratch");
                    }
                };
                Ok(Inflight::Tp { session, decode })
            }
            LoadedModel::Hybrid { model, .. } => {
                // `cluster` here is the server's global cluster; the
                // hybrid session/scratch are sized per-stage against
                // each stage's owning sub-cluster (no `cluster` arg
                // needed — sub-clusters live inside `model.stages`).
                let session = Qwen3MoEHybridSession::new(model)
                    .context("create hybrid session")?;
                let decode = match ShardedForwardOneTokenScratchHybrid::new(model) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = session.dispose(model);
                        return Err(e).context("hybrid decode scratch");
                    }
                };
                Ok(Inflight::Hybrid { session, decode })
            }
        }
    }

    /// Free per-rank caches + scratches. The hybrid variant also takes
    /// the parent [`LoadedModel`] (since each per-stage sub-cluster
    /// lives inside the model, not on the server-owned global
    /// cluster); pass `model` from the same registration.
    pub fn dispose(self, cluster: &HipCluster, model: &LoadedModel) -> Result<()> {
        match self {
            Inflight::Pp {
                session,
                prefill,
                decode,
            } => {
                decode
                    .dispose(cluster)
                    .context("dispose PP decode scratch")?;
                prefill
                    .dispose(cluster)
                    .context("dispose PP prefill scratch")?;
                session.dispose(cluster).context("dispose PP session")?;
                Ok(())
            }
            Inflight::Tp { session, decode } => {
                decode
                    .dispose(cluster)
                    .context("dispose TP decode scratch")?;
                session.dispose(cluster).context("dispose TP session")?;
                Ok(())
            }
            Inflight::Hybrid { session, decode } => {
                let LoadedModel::Hybrid { model, .. } = model else {
                    bail!(
                        "Inflight::Hybrid::dispose: paired LoadedModel variant is not Hybrid"
                    );
                };
                decode
                    .dispose(model)
                    .context("dispose hybrid decode scratch")?;
                session
                    .dispose(model)
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
        match (self, model) {
            (Inflight::Pp { session, .. }, LoadedModel::Pp { .. }) => session
                .reset_for_next_request(cluster)
                .context("reset PP session"),
            (Inflight::Tp { session, .. }, LoadedModel::Tp { .. }) => session
                .reset_for_next_request(cluster)
                .context("reset TP session"),
            (
                Inflight::Hybrid { session, .. },
                LoadedModel::Hybrid { model: hm, .. },
            ) => session
                .reset_for_next_request(hm)
                .context("reset Hybrid session"),
            _ => bail!("Inflight / LoadedModel variant mismatch in reset_for_next_request"),
        }
    }
}

/// Ingest the full prompt and write the logits row for the **last**
/// prompt position into `logits_out`. Each topology dispatches through
/// its own `forward_prefill_*_logits` entry point; the TP path is a
/// per-token loop today (AUTO-6a) and gets batched-across-L kernels
/// in AUTO-6b/c.
/// `tp_pool_prefill`: when `Some` and topology is TP, the pooled
/// `forward_prefill_tp_logits_pooled` is used, skipping per-call
/// alloc/dispose. Callers must already hold `prefill_serialiser`
/// (the TP/Hybrid chat handlers do for #321) — the scratch isn't
/// safe for parallel use. `None` falls back to alloc-per-call.
pub fn prefill_logits(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    prompt_ids: &[u32],
    logits_out: &mut Vec<f32>,
    tp_pool_prefill: Option<&mut ShardedForwardPrefillScratchTp>,
) -> Result<()> {
    if prompt_ids.is_empty() {
        bail!("prefill_logits: empty prompt");
    }
    match (model, inflight) {
        (
            LoadedModel::Pp { model: m, .. },
            Inflight::Pp {
                session, prefill, ..
            },
        ) => {
            // Chunked PP prefill. forward_prefill_pp recursively chunks
            // internally when L > scratch.max_tokens. Drive all-but-last
            // chunk through it (no logits needed), then a final
            // forward_prefill_pp_logits call on the last chunk to
            // harvest logits for sampling. KV / GDN state thread via
            // start_position. Verified bit-exact in Phase A.
            let chunk = prefill.per_rank[0].max_tokens;
            let l = prompt_ids.len();
            if l <= chunk {
                forward_prefill_pp_logits(
                    m, session, cluster, prefill, prompt_ids, 0, logits_out,
                )
                .context("PP prefill_logits")
            } else {
                let last = chunk.min(l);
                let split = l - last;
                tracing::debug!(
                    target: "server.prefill",
                    prompt_len = l,
                    chunk,
                    split,
                    "chunked PP prefill (prefix via forward_prefill_pp, final chunk via _logits)"
                );
                let _ = forward_prefill_pp(
                    m, session, cluster, prefill, &prompt_ids[..split], 0,
                )
                .context("PP prefill (prefix chunks)")?;
                forward_prefill_pp_logits(
                    m,
                    session,
                    cluster,
                    prefill,
                    &prompt_ids[split..],
                    split,
                    logits_out,
                )
                .context("PP prefill_logits (final chunk)")
            }
        }
        (LoadedModel::Tp { model, ar }, Inflight::Tp { session, decode }) => {
            // Chunked TP prefill (Phase B3a-TP). Phase A2-TP parity
            // test verified bit-exact KV at L=4096 chunk=512 (8 chunks),
            // so chunking is safe at chunk>=128. **#324** — when
            // `tp_pool_prefill` is `Some`, the caller (chat handler
            // holding `ServerState::prefill_serialiser`) hands us a
            // shared pre-allocated scratch and we skip per-call alloc.
            // When `None`, fall back to the legacy alloc-per-call
            // path inside `forward_prefill_tp_logits`.
            let chunk: usize = std::env::var("FLAMBEAU_PREFILL_UBATCH")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|n: &usize| *n >= 128)
                .unwrap_or(512);
            let l = prompt_ids.len();
            // We can't reuse `&mut tp_pool_prefill` across loop iterations
            // because the pooled call holds a reborrow; instead, take()
            // a local Option and re-store at end. But for the simple
            // single-chunk path we can just pass the &mut directly.
            if let Some(pool) = tp_pool_prefill {
                if l <= chunk {
                    forward_prefill_tp_logits_pooled(
                        model, decode, pool, cluster, ar, &mut session.caches,
                        prompt_ids, 0, logits_out,
                    )
                    .context("TP prefill_logits (pooled)")
                } else {
                    tracing::debug!(
                        target: "server.prefill",
                        prompt_len = l,
                        chunk,
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
                            &prompt_ids[start..end], start, dst,
                        )
                        .with_context(|| {
                            format!("TP prefill_logits chunk [{start}..{end}) (pooled)")
                        })?;
                        start = end;
                    }
                    Ok(())
                }
            } else if l <= chunk {
                forward_prefill_tp_logits(
                    model, decode, cluster, ar, &mut session.caches,
                    prompt_ids, 0, logits_out,
                )
                .context("TP prefill_logits")
            } else {
                tracing::debug!(
                    target: "server.prefill",
                    prompt_len = l,
                    chunk,
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
                        &prompt_ids[start..end], start, dst,
                    )
                    .with_context(|| format!("TP prefill_logits chunk [{start}..{end})"))?;
                    start = end;
                }
                Ok(())
            }
        }
        (
            LoadedModel::Hybrid {
                model: hmodel,
                stage_ars,
            },
            Inflight::Hybrid { session, decode },
        ) => {
            // Chunked Hybrid prefill (Phase B4a-Hybrid). Parity
            // verified bit-exact at L=4096 chunk=512 (8 chunks).
            let chunk: usize = std::env::var("FLAMBEAU_PREFILL_UBATCH")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|n: &usize| *n >= 128)
                .unwrap_or(512);
            let l = prompt_ids.len();
            if l <= chunk {
                forward_prefill_hybrid_logits(
                    hmodel, decode, cluster, stage_ars, session, prompt_ids, 0, logits_out,
                )
                .context("hybrid prefill_logits")
            } else {
                tracing::debug!(
                    target: "server.prefill",
                    prompt_len = l,
                    chunk,
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
                        &prompt_ids[start..end], start, dst,
                    )
                    .with_context(|| {
                        format!("hybrid prefill_logits chunk [{start}..{end})")
                    })?;
                    start = end;
                }
                Ok(())
            }
        }
        _ => bail!("LoadedModel/Inflight variant mismatch"),
    }
}

/// MTP-5d — per-request handle for spec-decode state. Owns the MTP
/// forward scratch (allocated lazily on the first spec call) and
/// tracks `h_for_mtp` between macro steps so the caller doesn't have
/// to thread it through. Dispose alongside `Inflight`.
pub struct SpecDecodePp {
    pub mtp_scratch: MtpForwardScratch,
    /// `h_for_mtp_dev` for the NEXT macro step's MTP draft. Lives
    /// inside `decode_scratch.per_rank[last_rank].hidden_a` after
    /// the most recent base step on the last rank.
    pub h_for_mtp: DevicePtr,
}

impl SpecDecodePp {
    pub fn new(model: &Qwen3MoEShardedModel, cluster: &HipCluster) -> Result<Self> {
        let last_rank = cluster.ranks() - 1;
        let last_device = cluster.device(last_rank);
        last_device.bind()?;
        let mtp_scratch = MtpForwardScratch::new(last_device, &model.config)
            .context("alloc MtpForwardScratch")?;
        // Placeholder; caller sets after the first base call (prefill_logits)
        // by reading `decode_scratch.per_rank[last].hidden_a` (or `hidden_b`).
        let h_for_mtp = mtp_scratch.h_t_post_norm; // arbitrary valid pointer; set by caller
        Ok(Self { mtp_scratch, h_for_mtp })
    }

    pub fn dispose(self, cluster: &HipCluster) -> Result<()> {
        let last_rank = cluster.ranks() - 1;
        let last_device = cluster.device(last_rank);
        last_device.bind()?;
        self.mtp_scratch.dispose(last_device)?;
        Ok(())
    }
}

/// MTP-5d — run one K=1 spec-decode macro step. Returns the
/// committed token(s) + telemetry. PP-only for now (matches
/// MTP-5c-shipped scope).
pub fn decode_spec_pp(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    spec: &mut SpecDecodePp,
    last_token: u32,
    position: usize,
) -> Result<SpecStep> {
    let (m, mtp) = match model {
        LoadedModel::Pp { model: m, mtp: Some(mtp) } => (m, mtp),
        LoadedModel::Pp { mtp: None, .. } => {
            bail!("decode_spec_pp called without MTP attachment (FLAMBEAU_SPEC_MTP not set)")
        }
        _ => bail!("decode_spec_pp requires LoadedModel::Pp"),
    };
    let (session, prefill, decode) = match inflight {
        Inflight::Pp { session, prefill, decode } => (session, prefill, decode),
        _ => bail!("decode_spec_pp requires Inflight::Pp"),
    };

    let last_rank = cluster.ranks() - 1;
    let last_shard = &m.shards[last_rank];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("PP last rank missing output_norm for spec-decode")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .context("PP last rank missing lm_head for spec-decode")?;

    let (step, h_next) = forward_speculative_pp_step(
        m, session, cluster, decode, prefill,
        mtp, &spec.mtp_scratch,
        output_norm, lm_head,
        last_token, spec.h_for_mtp, position,
    )?;
    spec.h_for_mtp = h_next;
    Ok(step)
}

/// MTP-5g — rejection-sampling variant. Same shape as
/// [`decode_spec_pp`] but threads a [`Sampling`] config and an `Rng`
/// through the spec macro so non-greedy sampling can be used with
/// vLLM-canonical rejection sampling.
pub fn decode_spec_pp_sampling(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    spec: &mut SpecDecodePp,
    last_token: u32,
    position: usize,
    sampling: &flambeau_runtime::Sampling,
    rng: &mut flambeau_runtime::Rng,
    // MTP-5g/h — per-turn generated-token slice for penalty application.
    history: &[u32],
) -> Result<SpecStep> {
    use flambeau_qwen3_moe::forward::forward_speculative_pp_step_sampling;

    let (m, mtp) = match model {
        LoadedModel::Pp { model: m, mtp: Some(mtp) } => (m, mtp),
        LoadedModel::Pp { mtp: None, .. } => {
            bail!("decode_spec_pp_sampling called without MTP attachment")
        }
        _ => bail!("decode_spec_pp_sampling requires LoadedModel::Pp"),
    };
    let (session, prefill, decode) = match inflight {
        Inflight::Pp { session, prefill, decode } => (session, prefill, decode),
        _ => bail!("decode_spec_pp_sampling requires Inflight::Pp"),
    };

    let last_rank = cluster.ranks() - 1;
    let last_shard = &m.shards[last_rank];
    let output_norm = last_shard
        .output_norm
        .as_ref()
        .context("PP last rank missing output_norm for spec-decode")?;
    let lm_head = last_shard
        .output
        .as_ref()
        .context("PP last rank missing lm_head for spec-decode")?;

    let (step, h_next) = forward_speculative_pp_step_sampling(
        m, session, cluster, decode, prefill,
        mtp, &spec.mtp_scratch,
        output_norm, lm_head,
        last_token, spec.h_for_mtp, position,
        sampling, rng, history,
    )?;
    spec.h_for_mtp = h_next;
    Ok(step)
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
    match (model, inflight) {
        (
            LoadedModel::Pp { model: m, .. },
            Inflight::Pp {
                session, decode, ..
            },
        ) => forward_one_token_pp_logits(m, session, cluster, decode, token, position, logits_out)
            .context("PP decode_logits"),
        (LoadedModel::Tp { model, ar }, Inflight::Tp { session, decode }) => {
            forward_one_token_tp_logits(
                model,
                decode,
                cluster,
                ar,
                &mut session.caches,
                token,
                position,
                logits_out,
            )
            .context("TP decode_logits")
        }
        (
            LoadedModel::Hybrid {
                model: hmodel,
                stage_ars,
            },
            Inflight::Hybrid { session, decode },
        ) => forward_one_token_hybrid_logits(
            hmodel,
            decode,
            cluster,
            stage_ars,
            session,
            token,
            position,
            logits_out,
        )
        .context("hybrid decode_logits"),
        _ => bail!("LoadedModel/Inflight variant mismatch"),
    }
}

/// **Sampler-D3 Phase B (#211)** — same as [`decode_logits`] but does
/// NOT DtoH the F32 logits row to host. Logits remain on the head
/// rank's `output_head.logits_f32` device pointer; the caller (the
/// GPU sampler hook in `gpu_sampler.rs`) consumes them in place via
/// `topk_softmax_f32` before the next forward call clobbers the
/// buffer. TP-only for now (matches Phase A coverage).
pub fn decode_keep_logits_on_device(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    token: u32,
    position: usize,
) -> Result<()> {
    match model {
        LoadedModel::Tp { model: m, ar } => match inflight {
            Inflight::Tp { session, decode } => forward_one_token_tp_keep_logits_on_device(
                m,
                decode,
                cluster,
                ar,
                &mut session.caches,
                token,
                position,
            )
            .context("TP decode_keep_logits_on_device"),
            _ => bail!("Inflight variant doesn't match LoadedModel::Tp"),
        },
        // **Hybrid GPU-sampler fallback (#258)** — there's no
        // forward_one_token_hybrid_keep_logits_on_device yet, so we
        // run the regular decode_logits (which does the host DtoH of
        // ~600 KB) and let the GPU sampler read logits_f32 from the
        // head stage's OutputHeadScratch on device anyway. The DtoH
        // is wasted bandwidth but the device buffer is still
        // correctly populated for the GPU topk kernel to consume.
        // A proper hybrid keep-on-device variant is a perf-only
        // follow-up.
        LoadedModel::Hybrid { .. } => {
            let mut sink: Vec<f32> = Vec::new();
            decode_logits(model, cluster, inflight, token, position, &mut sink)
                .context("Hybrid decode_keep_logits_on_device (decode_logits fallback)")
        }
        _ => bail!(
            "decode_keep_logits_on_device only wired for TP and Hybrid topologies"
        ),
    }
}

/// **#229 P2.10c** — capture the active inflight session's KV state
/// into a host-side snapshot, sized for the current `current_tokens`
/// of every layer.
///
/// Layout: `result[r]` covers rank `r`'s full layer set. PP and TP
/// supported; Hybrid bails (V2 follow-up — see `restore_kv_into_inflight`).
///
/// **Cost**: D→H copy of `total_bytes()` per rank. On 27B/TP2/ctx=4096
/// that's ~1 GB across 2 ranks, ~150 ms over PCIe 3.0 x16. On hit the
/// inverse H→D pays the same — still a net win vs the ~2-3 s prefill
/// it replaces.
pub fn capture_kv_from_inflight(
    inflight: &Inflight,
    cluster: &HipCluster,
    model: &LoadedModel,
) -> Result<Vec<Vec<LayerCacheSnapshot>>> {
    use flambeau_qwen3_moe::session::snapshot_layer_caches_to_host;
    match (inflight, model) {
        (Inflight::Pp { session, .. }, LoadedModel::Pp { .. }) => {
            let mut out = Vec::with_capacity(session.per_rank.len());
            for (rank_idx, rank_session) in session.per_rank.iter().enumerate() {
                let device = cluster.device(rank_idx);
                let snap = snapshot_layer_caches_to_host(&rank_session.caches, device)
                    .with_context(|| format!("PP capture rank {rank_idx}"))?;
                out.push(snap);
            }
            Ok(out)
        }
        (Inflight::Tp { session, .. }, LoadedModel::Tp { .. }) => {
            let mut out = Vec::with_capacity(session.caches.len());
            for (rank_idx, rank_caches) in session.caches.iter().enumerate() {
                let device = cluster.device(rank_idx);
                let snap = snapshot_layer_caches_to_host(rank_caches, device)
                    .with_context(|| format!("TP capture rank {rank_idx}"))?;
                out.push(snap);
            }
            Ok(out)
        }
        (Inflight::Hybrid { .. }, LoadedModel::Hybrid { .. }) => {
            bail!(
                "Hybrid prefix-cache capture not yet wired (see \
                 restore_kv_into_inflight — V2 follow-up)"
            );
        }
        _ => bail!("capture_kv_from_inflight: model/inflight variant mismatch"),
    }
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

/// **#228 P2.10b** — restore a host-side KV snapshot into the active
/// inflight session.
///
/// `snapshot[r]` covers rank `r`'s full layer set (the same layout
/// `snapshot_layer_caches_to_host` produces). The caller is responsible
/// for ensuring the snapshot was captured under the same topology and
/// chunk size — the `PrefixCache::longest_match` lookup checks this
/// before this function is called.
///
/// Errors when the topology is `Hybrid` (V2 follow-up — the per-stage
/// per-rank shape doesn't match the flat `Vec<RankSnapshot>` layout).
/// Callers should skip prefix-cache restore for hybrid models in V1.
pub fn restore_kv_into_inflight(
    inflight: &mut Inflight,
    cluster: &HipCluster,
    snapshot: &[Vec<LayerCacheSnapshot>],
    model: &LoadedModel,
) -> Result<()> {
    match (inflight, model) {
        (Inflight::Pp { session, .. }, LoadedModel::Pp { .. }) => {
            if snapshot.len() != session.per_rank.len() {
                bail!(
                    "PP restore: snapshot rank count {} != session ranks {}",
                    snapshot.len(),
                    session.per_rank.len()
                );
            }
            for (rank_idx, rank_session) in session.per_rank.iter_mut().enumerate() {
                let device = cluster.device(rank_idx);
                restore_layer_caches_from_host(
                    &snapshot[rank_idx],
                    rank_session.caches_mut(),
                    device,
                )
                .with_context(|| format!("PP restore rank {rank_idx}"))?;
            }
            Ok(())
        }
        (Inflight::Tp { session, .. }, LoadedModel::Tp { .. }) => {
            if snapshot.len() != session.caches.len() {
                bail!(
                    "TP restore: snapshot rank count {} != session ranks {}",
                    snapshot.len(),
                    session.caches.len()
                );
            }
            for (rank_idx, rank_caches) in session.caches.iter_mut().enumerate() {
                let device = cluster.device(rank_idx);
                restore_layer_caches_from_host(
                    &snapshot[rank_idx],
                    rank_caches,
                    device,
                )
                .with_context(|| format!("TP restore rank {rank_idx}"))?;
            }
            Ok(())
        }
        (Inflight::Hybrid { .. }, LoadedModel::Hybrid { .. }) => {
            bail!(
                "Hybrid prefix-cache restore not yet wired (per-stage per-rank \
                 layout doesn't match the flat snapshot type — V2 follow-up)"
            );
        }
        _ => bail!("restore_kv_into_inflight: model/inflight variant mismatch"),
    }
}


