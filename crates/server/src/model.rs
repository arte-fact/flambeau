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
    forward_prefill_hybrid_logits, forward_prefill_pp_logits, forward_prefill_tp_logits,
    forward_speculative_pp_step, ShardedForwardOneTokenScratch,
    ShardedForwardOneTokenScratchHybrid, ShardedForwardOneTokenScratchTp,
    ShardedForwardPrefillScratch, SpecStep,
};
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
        match model {
            LoadedModel::Pp { model: m, .. } => {
                let session =
                    Qwen3MoEShardedSession::new(m, cluster).context("create PP session")?;
                let prefill = ShardedForwardPrefillScratch::new(m, cluster, prompt_len.max(1))
                    .context("PP prefill scratch")?;
                let decode = ShardedForwardOneTokenScratch::new(m, cluster)
                    .context("PP decode scratch")?;
                Ok(Inflight::Pp {
                    session,
                    prefill,
                    decode,
                })
            }
            LoadedModel::Tp { model, .. } => {
                let session =
                    Qwen3MoETpSession::new(model, cluster).context("create TP session")?;
                let decode = ShardedForwardOneTokenScratchTp::new(&model.config, cluster)
                    .context("TP decode scratch")?;
                Ok(Inflight::Tp { session, decode })
            }
            LoadedModel::Hybrid { model, .. } => {
                // `cluster` here is the server's global cluster; the
                // hybrid session/scratch are sized per-stage against
                // each stage's owning sub-cluster (no `cluster` arg
                // needed — sub-clusters live inside `model.stages`).
                let session = Qwen3MoEHybridSession::new(model)
                    .context("create hybrid session")?;
                let decode = ShardedForwardOneTokenScratchHybrid::new(model)
                    .context("hybrid decode scratch")?;
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
}

/// Ingest the full prompt and write the logits row for the **last**
/// prompt position into `logits_out`. Each topology dispatches through
/// its own `forward_prefill_*_logits` entry point; the TP path is a
/// per-token loop today (AUTO-6a) and gets batched-across-L kernels
/// in AUTO-6b/c.
pub fn prefill_logits(
    model: &LoadedModel,
    cluster: &HipCluster,
    inflight: &mut Inflight,
    prompt_ids: &[u32],
    logits_out: &mut Vec<f32>,
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
        ) => forward_prefill_pp_logits(m, session, cluster, prefill, prompt_ids, 0, logits_out)
            .context("PP prefill_logits"),
        (LoadedModel::Tp { model, ar }, Inflight::Tp { session, decode }) => {
            forward_prefill_tp_logits(
                model,
                decode,
                cluster,
                ar,
                &mut session.caches,
                prompt_ids,
                0,
                logits_out,
            )
            .context("TP prefill_logits")
        }
        (
            LoadedModel::Hybrid {
                model: hmodel,
                stage_ars,
            },
            Inflight::Hybrid { session, decode },
        ) => forward_prefill_hybrid_logits(
            hmodel,
            decode,
            cluster,
            stage_ars,
            session,
            prompt_ids,
            0,
            logits_out,
        )
        .context("hybrid prefill_logits"),
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
    match (model, inflight) {
        (LoadedModel::Tp { model, ar }, Inflight::Tp { session, decode }) => {
            forward_one_token_tp_keep_logits_on_device(
                model,
                decode,
                cluster,
                ar,
                &mut session.caches,
                token,
                position,
            )
            .context("TP decode_keep_logits_on_device")
        }
        _ => bail!(
            "decode_keep_logits_on_device only wired for TP topology (Phase B)"
        ),
    }
}


