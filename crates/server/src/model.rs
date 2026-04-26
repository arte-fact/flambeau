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
    forward_one_token_pp_logits, forward_one_token_tp_logits, forward_prefill_pp_logits,
    ShardedForwardOneTokenScratch, ShardedForwardOneTokenScratchTp, ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession, Qwen3MoETpModel,
    Qwen3MoETpSession,
};

/// Loaded weights + per-topology auxiliary state.
///
/// Built once at startup. The PP variant just owns the sharded model;
/// the TP variant additionally owns a [`BarP2pAllReduce`] that holds an
/// `Arc<HipCluster>` against the same cluster the server uses.
pub enum LoadedModel {
    /// V1.8 pipeline-parallel sharded model. One whole layer per rank
    /// stage; cross-stage hand-off via host-bounce peer copy.
    Pp(Qwen3MoEShardedModel),
    /// V2 tensor-parallel sharded model. Every rank holds every layer
    /// (sliced); intra-layer Megatron splits + BAR1 P2P AllReduce.
    Tp {
        model: Qwen3MoETpModel,
        ar: BarP2pAllReduce,
    },
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadedModel::Pp(_) => f.debug_struct("LoadedModel::Pp").finish(),
            LoadedModel::Tp { .. } => f.debug_struct("LoadedModel::Tp").finish(),
        }
    }
}

impl LoadedModel {
    /// Common config shared between variants.
    pub fn config(&self) -> &Qwen3MoEConfig {
        match self {
            LoadedModel::Pp(m) => &m.config,
            LoadedModel::Tp { model, .. } => &model.config,
        }
    }

    /// Topology label for `tracing` / handler-side metrics.
    pub fn topology(&self) -> &'static str {
        match self {
            LoadedModel::Pp(_) => "pp",
            LoadedModel::Tp { .. } => "tp",
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
}

impl Inflight {
    /// Allocate per-rank caches + scratches for a single request.
    /// `prompt_len` sizes the PP prefill scratch; ignored for TP (TP
    /// loops `forward_one_token_tp` over the prompt instead).
    pub fn new(model: &LoadedModel, cluster: &HipCluster, prompt_len: usize) -> Result<Self> {
        match model {
            LoadedModel::Pp(m) => {
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
        }
    }

    /// Free per-rank caches + scratches.
    pub fn dispose(self, cluster: &HipCluster) -> Result<()> {
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
        }
    }
}

/// Ingest the full prompt and write the logits row for the **last**
/// prompt position into `logits_out`. PP runs through
/// [`forward_prefill_pp_logits`] in one batched pass; TP loops
/// [`forward_one_token_tp_logits`] per prompt token (no TP prefill
/// kernel exists yet — design accepted in TP-5b "prefill-PP +
/// decode-TP coexistence" where we treat per-token TP as the prefill
/// path until a real TP prefill lands).
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
            LoadedModel::Pp(m),
            Inflight::Pp {
                session, prefill, ..
            },
        ) => forward_prefill_pp_logits(m, session, cluster, prefill, prompt_ids, 0, logits_out)
            .context("PP prefill_logits"),
        (LoadedModel::Tp { model, ar }, Inflight::Tp { session, decode }) => {
            for (pos, &tok) in prompt_ids.iter().enumerate() {
                forward_one_token_tp_logits(
                    model,
                    decode,
                    cluster,
                    ar,
                    &mut session.caches,
                    tok,
                    pos,
                    logits_out,
                )
                .with_context(|| format!("TP prefill loop @ pos {pos}"))?;
            }
            Ok(())
        }
        _ => bail!("LoadedModel/Inflight variant mismatch"),
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
    match (model, inflight) {
        (
            LoadedModel::Pp(m),
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
        _ => bail!("LoadedModel/Inflight variant mismatch"),
    }
}

