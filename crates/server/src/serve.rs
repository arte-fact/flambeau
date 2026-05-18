//! `flambeau serve` entry point — loads the model + tokenizer, starts axum.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::GgufFile;
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoEHybridModel,
    Qwen3MoEShardedModel, Qwen3MoETpModel,
};
use flambeau_runtime::{LayerAssignment, Registry};
use tokio::sync::Mutex;
use tracing::info;

use crate::qwen3moe_handle::LoadedModel;

/// mesh topology selector. PP-V1 default; TP engages the
/// Qwen3MoETpModel loader + the BarP2pAllReduce-based forward path.
/// `Hybrid` adds a manual PP-of-TP composition where
/// `pp_size` contiguous layer stages each own a `tp_size`-rank TP
/// subgroup. Selection is operator-driven; flambeau does not autodetect
/// the right topology for a given rig (the bracket-bench harness in
/// produces the data, the operator picks the winner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshMode {
    /// Pipeline parallelism — V1 default. `LayerAssignment` distributes
    /// whole layers across ranks; one `peer_copy_via_host` per stage
    /// transition.
    Pp,
    /// Tensor parallelism — every rank holds every layer (sliced).
    /// `world` ranks; intra-layer Megatron splits + BAR1 P2P AllReduce.
    Tp { world: u32 },
    /// Hybrid PP-of-TP — `pp_size` contiguous layer stages, each owning
    /// a `tp_size`-rank TP subgroup. Total ranks = `pp_size * tp_size`.
    /// Devices are interpreted in stage-major order: `--devices d0,d1,...`
    /// with `tp_size = 2`, `pp_size = 2` means stage 0 = {d0, d1},
    /// stage 1 = {d2, d3}. Forward wiring lands in f; this
    /// variant currently boots up to the loader and bails.
    Hybrid { pp_size: u32, tp_size: u32 },
}

/// Runtime config for `flambeau serve`.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub gguf_path: PathBuf,
    pub device_ids: Vec<i32>,
    pub bind_addr: SocketAddr,
    pub model_id: String,
    /// mesh topology. Defaults to `Pp` for V1 callers that
    /// don't set the field explicitly (constructors use struct-update
    /// syntax with `..Default::default()`).
    pub mesh_mode: MeshMode,
    /// **#230 P2.11a** — optional path to a `qwen3` arch embedding
    /// GGUF loaded alongside the chat model. `None` disables the
    /// embedding subsystem; `/v1/embeddings` (#231) returns 503 when
    /// unset.
    pub embedding_gguf_path: Option<PathBuf>,
    /// **#230 P2.11a** — HIP device id for the embedding model. Must
    /// be one of `device_ids`; the embedding model reuses the
    /// chat-cluster's `HipDevice` handle for the matching rank.
    pub embedding_device_id: Option<i32>,
    /// Number of concurrent inflight decode slots (1-32). VRAM scales
    /// linearly with this value (each slot owns its own KV cache).
    pub inflight_slots: usize,
    /// Prefill chunk size in tokens. Default 512 is the production sweet
    /// spot across pp/tp/hybrid topologies; tune for short-prompt TTFT.
    pub prefill_ubatch: usize,
    /// #232 admission-control queue depth beyond the inflight pool. 0
    /// disables (legacy unbounded queue). Default 16.
    pub max_queue_depth: usize,
    /// Clamp the model's `context_length`. `None` keeps the GGUF's
    /// architectural max; many GGUFs ship 262 144 which OOMs the per-rank
    /// KV cache on 16 GB MI50. Only shrinks.
    pub ctx_cap: Option<usize>,
    /// On-device GPU sampler (top-k + softmax + penalties on the head
    /// rank, single DtoH per token). Drops sampler cost from ~12 ms to
    /// ~0 ms on chat workloads.
    pub gpu_sampler: bool,
    /// Batched-decode scheduler. Coalesces concurrent decode steps via
    /// the inflight-slot leader. Required for N>1 throughput.
    pub batched_decode: bool,
    /// #229 prompt prefix cache. Caches prompt-prefix KV across requests.
    pub prefix_cache: bool,
    /// Prefix-cache LRU size in GB. Tune to free VRAM minus model + KV.
    pub prefix_cache_max_gb: f64,
    /// KV cache layout: `"f16"` or `"q8"`.
    pub kv: String,
    /// Default system prompt prepended to chat-template requests when
    /// none is provided.
    pub default_system: Option<String>,
    /// /v1/embeddings per-prompt token cap.
    pub embedding_max_tokens: usize,
}

impl Default for MeshMode {
    fn default() -> Self {
        MeshMode::Pp
    }
}

/// Operator-visible forward-stack selector. `FLAMBEAU_V2=1` opts into
/// the v2 stack (flambeau-forward + per-arch v2 model crates); anything
/// else stays on the legacy qwen3-moe / gemma4 paths. Rule 1 of the
/// project CLAUDE.md bans env-flag-based variant selection — this one
/// is an explicit operator-facing migration switch, not a dispatch row.
fn v2_stack_requested() -> bool {
    matches!(
        std::env::var("FLAMBEAU_V2").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

/// Blocking serve loop — loads the model, starts the HTTP server, runs
/// until terminated. Caller owns the tokio runtime.
pub async fn serve(cfg: ServeConfig, registry: Registry) -> Result<()> {
    let v2 = v2_stack_requested();
    info!(
        forward_stack = if v2 { "v2" } else { "legacy" },
        "flambeau serve: loading model"
    );
    info!(?cfg, "flambeau serve config");

    let gguf = GgufFile::open(&cfg.gguf_path)
        .with_context(|| format!("open GGUF at {}", cfg.gguf_path.display()))?;

    // Reject unsupported GGUF arches before walking tokenizer / chat
    // template / model paths so the operator gets a clean diagnostic
    // instead of a downstream invariant error.
    let gguf_arch = gguf.metadata_str("general.architecture").unwrap_or("");
    let model_arch = registry
        .validate(gguf_arch)
        .with_context(|| format!("flambeau serve: unsupported GGUF arch `{gguf_arch}`"))?;
    info!(
        arch = gguf_arch,
        handler = model_arch.description(),
        "GGUF arch validated against registry"
    );

    let boot = crate::serve_common::BootMetadata::from_gguf(&gguf, &cfg)?;

    if v2 {
        return crate::serve::serve_inner_v2(cfg, gguf, boot).await;
    }

    // Sanity: device_ids must be valid.
    let n_available = device_count().unwrap_or(0);
    for d in &cfg.device_ids {
        if *d < 0 || *d >= n_available {
            bail!(
                "device {d} not available (have {n_available} HIP devices)"
            );
        }
    }

    // Phase 12.9 — gemma4 boot path. The qwen3-moe `Qwen3MoEConfig::from_gguf`
    // call below would fail with a config-mismatch error for gemma4 GGUFs;
    // detect the arch up-front and dispatch into the gemma4 inner before
    // touching qwen3-moe-specific code.
    if crate::gemma4_handle::arch_matches(gguf_arch) {
        return serve_inner_gemma4(cfg, gguf, boot).await;
    }

    let mut model_cfg = Qwen3MoEConfig::from_gguf(&gguf).context("model config from GGUF")?;
    // Allow operators to clamp the model's KV-cache provisioning ceiling
    // (mirrors the test-side FLAMBEAU_CTX_CAP). The on-disk
    // `context_length` is often the architectural max (262144 for
    // Qwen3.5/3.6) which would OOM the per-rank KV cache on consumer
    // VRAM. The clamp only shrinks; explicit increases are ignored.
    if let Some(cap) = cfg.ctx_cap {
        if cap > 0 && cap < model_cfg.context_length {
            info!(
                from = model_cfg.context_length,
                to = cap,
                "ctx-cap shrinking model.context_length"
            );
            model_cfg.context_length = cap;
        }
    }
    // construction order matters on this rig. For pure
    // PP/TP, the single global cluster is built first; for hybrid, the
    // per-stage sub-clusters are built first (inside
    // `Qwen3MoEHybridModel::load`) and the global cluster is built
    // *afterwards*. Reverse order leaves per-stage `peer_access_full`
    // reporting zero on the off-diagonal (`project_hybrid_cluster_order`).
    let (cluster, model): (Arc<HipCluster>, LoadedModel) = match cfg.mesh_mode {
        MeshMode::Pp => {
            let cluster: Arc<HipCluster> =
                Arc::new(HipCluster::new(&cfg.device_ids).context("HipCluster::new")?);
            let assignment =
                LayerAssignment::contiguous(model_cfg.num_layers, cluster.ranks() as u32);
            info!(
                num_layers = model_cfg.num_layers,
                ranks = cluster.ranks(),
                topology = "pp",
                "loading model weights"
            );
            let mut m = Qwen3MoEShardedModel::load(&gguf, &cluster, &assignment)
                .context("Qwen3MoEShardedModel::load")?;
            // Re-apply the FLAMBEAU_CTX_CAP clamp on the model-owned
            // cfg (mirrors the TP / Hybrid arms below). Without this,
            // the PP loader keeps the GGUF-embedded context_length
            // (often 128k+ on Qwen3.x), which OOMs SLOTS≥2 on
            // consumer-VRAM rigs at TP=1 (~512 MB/layer/slot KV at the
            // GGUF-default ctx).
            if m.config.context_length > model_cfg.context_length {
                m.config.context_length = model_cfg.context_length;
            }

            (
                cluster,
                std::sync::Arc::new(crate::qwen3moe_handle::PpHipModel { model: m }) as LoadedModel,
            )
        }
        MeshMode::Tp { world } => {
            let cluster: Arc<HipCluster> =
                Arc::new(HipCluster::new(&cfg.device_ids).context("HipCluster::new")?);
            if cluster.ranks() as u32 != world {
                bail!(
                    "--mesh-mode tp: --tp-size {world} but cluster has {} ranks",
                    cluster.ranks()
                );
            }
            let layout = Qwen35DenseTpLayout::new(&model_cfg, world)
                .context("--mesh-mode tp: layout validation")?;
            info!(
                num_layers = model_cfg.num_layers,
                ranks = cluster.ranks(),
                world,
                kv_replicated = layout.kv_replicated(),
                topology = "tp",
                "loading model weights"
            );
            let mut m = Qwen3MoETpModel::load(&gguf, &cluster, layout)
                .context("Qwen3MoETpModel::load")?;
            // Re-apply the FLAMBEAU_CTX_CAP clamp on the model-owned cfg.
            // `Qwen3MoETpModel::load` re-reads the GGUF for its embedded
            // config, so the clamp on `model_cfg` above doesn't propagate
            // here without an explicit second clamp.
            if m.config.context_length > model_cfg.context_length {
                m.config.context_length = model_cfg.context_length;
            }
            let tp = flambeau_blocks::TpCluster::from_arc(Arc::clone(&cluster))
                .context("TpCluster::from_arc (requires fully-connected peer-access matrix)")?;
            (
                cluster,
                std::sync::Arc::new(crate::qwen3moe_handle::TpHipModel { model: m, tp }) as LoadedModel,
            )
        }
        MeshMode::Hybrid { pp_size, tp_size } => {
            let spec = HybridMeshSpec { pp_size, tp_size };
            spec.validate(model_cfg.num_layers, cfg.device_ids.len())
                .context("--mesh-mode pp+tp: HybridMeshSpec::validate")?;
            info!(
                num_layers = model_cfg.num_layers,
                pp_size,
                tp_size,
                ranks = cfg.device_ids.len(),
                topology = "pp+tp",
                "loading model weights (hybrid)"
            );
            // 1. Per-stage sub-clusters (built inside `load`) + weights.
            let mut hybrid = Qwen3MoEHybridModel::load(&gguf, &cfg.device_ids, spec)
                .context("Qwen3MoEHybridModel::load")?;
            for stage in &hybrid.stages {
                info!(
                    stage = stage.stage_idx,
                    layer_range = ?stage.layer_range,
                    bytes = stage.tp_model.total_bytes(),
                    has_token_embd = stage.tp_model.has_token_embd,
                    has_output_head = stage.tp_model.has_output_head,
                    "hybrid stage loaded"
                );
            }
            // Re-apply the FLAMBEAU_CTX_CAP clamp on each stage's
            // model-owned cfg (same reason as the TP arm).
            for stage in hybrid.stages.iter_mut() {
                if stage.tp_model.config.context_length > model_cfg.context_length {
                    stage.tp_model.config.context_length = model_cfg.context_length;
                }
            }
            if hybrid.config.context_length > model_cfg.context_length {
                hybrid.config.context_length = model_cfg.context_length;
            }
            // 2. Build the global cluster, then construct `HybridCluster`
            // which enforces the sub-cluster-before-global ordering as
            // a type invariant. `HybridCluster::new` takes the
            // sub-cluster Arcs first, builds each stage's
            // `BarP2pAllReduce` on the still-fresh sub-cluster, then
            // accepts the global cluster — matching the legacy hand-
            // rolled order and the `project_hybrid_cluster_order`
            // memory note.
            let global_cluster: Arc<HipCluster> = Arc::new(
                HipCluster::new(&cfg.device_ids)
                    .context("HipCluster::new (global, for inter-stage hand-off)")?,
            );
            let sub_clusters: Vec<Arc<HipCluster>> = hybrid
                .stages
                .iter()
                .map(|s| Arc::clone(&s.sub_cluster))
                .collect();
            let hc = flambeau_blocks::HybridCluster::new(
                sub_clusters,
                Arc::clone(&global_cluster),
                tp_size as usize,
            )
            .context("HybridCluster::new (per-stage ARs + global cluster)")?;
            (
                global_cluster,
                std::sync::Arc::new(crate::qwen3moe_handle::HybridHipModel { model: hybrid, hc })
                    as LoadedModel,
            )
        }
    };

    let prefill_ubatch = cfg.prefill_ubatch.max(128);
    let inflight_slots = cfg.inflight_slots.clamp(1, 32);
    let max_queue_depth = cfg.max_queue_depth;
    info!(
        prefill_ubatch,
        inflight_slots,
        max_queue_depth,
        "pre-allocating inflight slot pool"
    );
    let mut inflight_pool: Vec<Mutex<Box<dyn crate::Session>>> =
        Vec::with_capacity(inflight_slots);
    for slot_idx in 0..inflight_slots {
        let slot = crate::create_qwen3moe_session(
            model.clone(),
            cluster.clone(),
            prefill_ubatch,
            flambeau_qwen3_moe::session::KvLayout::from_str(&cfg.kv),
        )
            .with_context(|| format!("pre-alloc inflight slot {slot_idx} at boot"))?;
        inflight_pool.push(Mutex::new(slot));
    }

    let topology_tag = crate::serve_common::topology_tag_from_mesh(
        cfg.mesh_mode,
        cfg.device_ids.len(),
    );
    let prefix_cache =
        crate::serve_common::build_prefix_cache(&cfg, topology_tag.mesh_kind);

    let embedding = load_embedding_qwen3(&cfg, &cluster)?;
    let embedding_rank = embedding.as_ref().map(|(_, _, r)| *r);
    let embedding = embedding.map(|(m, t, _)| (m, t));

    let state = crate::serve_common::build_server_state(
        crate::serve_common::ServerStateInputs {
            model_id: cfg.model_id.clone(),
            model_cfg: crate::model_cfg::ServerModelCfg::from(&model_cfg),
            model,
            cluster,
            inflight_pool,
            qwen3_moe: Some(crate::routes::Qwen3MoeServerExtras::default()),
            embedding,
            embedding_rank,
            gpu_sampler: cfg.gpu_sampler,
            batched_decode: cfg.batched_decode,
            max_queue_depth,
            prefill_ubatch,
            topology_tag,
            prefix_cache,
            boot,
        },
    );

    crate::serve_common::run_axum(state, cfg.bind_addr, "qwen3-moe").await
}

type EmbeddingTriple = (
    Arc<tokio::sync::Mutex<Box<dyn crate::embedding::EmbeddingHandle>>>,
    Arc<flambeau_quant::GgufTokenizer>,
    usize,
);

fn load_embedding_qwen3(
    cfg: &ServeConfig,
    cluster: &Arc<HipCluster>,
) -> Result<Option<EmbeddingTriple>> {
    let Some(path) = cfg.embedding_gguf_path.as_ref() else {
        info!("embedding model not configured (--embedding-model unset)");
        return Ok(None);
    };
    let device_id = cfg.embedding_device_id.unwrap_or(cfg.device_ids[0]);
    let rank = cfg
        .device_ids
        .iter()
        .position(|d| *d == device_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "embedding device {device_id} not in --devices {:?}",
                cfg.device_ids
            )
        })?;
    let device = cluster.device(rank);
    info!(path = %path.display(), device_id, rank, "loading embedding model");
    let efile = GgufFile::open(path)
        .with_context(|| format!("open embedding GGUF at {}", path.display()))?;
    let embedding_tokenizer =
        flambeau_quant::load_from_gguf(&efile).context("load embedding tokenizer from GGUF")?;
    let max_emb_tokens = cfg.embedding_max_tokens.clamp(16, 32768);
    let em = flambeau_qwen3_moe::EmbeddingModel::load(&efile, device, device_id, max_emb_tokens)
        .context("EmbeddingModel::load")?;
    info!(
        arch = em.arch(),
        hidden_size = em.hidden_size(),
        vocab_size = em.vocab_size(),
        pooling_type = em.pooling_type(),
        max_tokens = em.max_tokens,
        bytes = em.total_bytes(),
        device_id,
        "embedding model loaded"
    );
    let boxed: Box<dyn crate::embedding::EmbeddingHandle> = Box::new(em);
    Ok(Some((
        Arc::new(tokio::sync::Mutex::new(boxed)),
        Arc::new(embedding_tokenizer),
        rank,
    )))
}

/// Gemma4 boot path. Branched into from `serve_inner` when
/// `gguf.architecture()` matches a gemma4 family.
///
/// Remaining MVP constraints:
/// - Prefix cache silently misses for gemma4 sessions (gated upstream
///   in `prefix_cache_try_restore` by the qwen3-moe-typed early-out).
/// - GPU sampler disabled (`use_gpu_sampler=false`); host sampler runs.
/// - Embedding endpoint returns 503 (qwen3-only embedding model).
async fn serve_inner_gemma4(
    cfg: ServeConfig,
    gguf: GgufFile,
    boot: crate::serve_common::BootMetadata,
) -> Result<()> {
    use flambeau_gemma4::{
        partition_layers, Gemma4Config, Gemma4HybridDriver, Gemma4PpDriver, Gemma4TpDriver, ModelLayout,
    };
    use flambeau_runtime::ModelDriver;

    use crate::gemma4_handle::{build_gemma4_loaded_model, wrap_gemma4_driver};

    let topology_label: &'static str = match cfg.mesh_mode {
        MeshMode::Pp => "pp",
        MeshMode::Tp { .. } => "tp",
        MeshMode::Hybrid { .. } => "pp+tp",
    };

    let mut cfg_g4 = Gemma4Config::from_gguf(&gguf).context("Gemma4Config::from_gguf")?;
    if let Some(cap) = cfg.ctx_cap {
        if cap > 0 && cap < cfg_g4.context_length {
            info!(
                from = cfg_g4.context_length,
                to = cap,
                "ctx-cap shrinking gemma4 context_length"
            );
            cfg_g4.context_length = cap;
        }
    }

    // Sanity: device_ids valid.
    let n_available = device_count().unwrap_or(0);
    for d in &cfg.device_ids {
        if *d < 0 || *d >= n_available {
            bail!("device {d} not available (have {n_available} HIP devices)");
        }
    }

    let inflight_slots = cfg.inflight_slots.clamp(1, 32);
    let prefill_ubatch = cfg.prefill_ubatch.max(128);
    let max_queue_depth = cfg.max_queue_depth;

    // Build N slot-drivers per topology, each sharing one
    // `Arc<Gemma4*Model>` (weights uploaded once) plus its own
    // `Gemma4*Session` (KV + scratch).
    //   - PP: `Gemma4PpDriver::upload` consumes its cluster by value;
    //     we keep a SECOND state-side `Arc<HipCluster>` whose only
    //     purpose is satisfying `ServerState.cluster`'s type (handler
    //     paths that read it are qwen3-moe-gated).
    //   - TP / Hybrid: cluster goes inside the model, the state-side
    //     handle is the same Arc / global cluster.
    let (state_cluster, drivers): (Arc<HipCluster>, Vec<Box<dyn ModelDriver>>) = match cfg
        .mesh_mode
    {
        MeshMode::Pp => {
            let state_cluster: Arc<HipCluster> = Arc::new(
                HipCluster::new(&cfg.device_ids).context("HipCluster::new (state side)")?,
            );
            let driver_cluster =
                HipCluster::new(&cfg.device_ids).context("HipCluster::new (driver)")?;
            let mut layout = ModelLayout::from_config(&cfg_g4);
            let _shared_kv = layout.resolve_kv_sharing();
            let layer_to_rank = partition_layers(cfg.device_ids.len(), &layout)
                .context("partition_layers for gemma4 PP")?;
            info!(
                num_layers = cfg_g4.num_layers,
                ranks = cfg.device_ids.len(),
                topology = "pp",
                arch = cfg_g4.arch.as_str(),
                inflight_slots,
                "loading gemma4 weights"
            );
            // Slot 0 uploads weights; slots 1..N borrow them via Arc.
            let driver0 = Gemma4PpDriver::upload(
                &gguf,
                cfg_g4.clone(),
                layout,
                layer_to_rank,
                driver_cluster,
                cfg_g4.context_length,
            )
            .context("Gemma4PpDriver::upload")?;
            let model = Arc::clone(&driver0.model);
            let mut drivers: Vec<Box<dyn ModelDriver>> = Vec::with_capacity(inflight_slots);
            drivers.push(Box::new(driver0));
            for slot in 1..inflight_slots {
                let d = Gemma4PpDriver::new_session(Arc::clone(&model), cfg_g4.context_length)
                    .with_context(|| format!("Gemma4PpDriver::new_session slot {slot}"))?;
                drivers.push(Box::new(d));
            }
            (state_cluster, drivers)
        }
        MeshMode::Tp { world } => {
            let shared_cluster: Arc<HipCluster> =
                Arc::new(HipCluster::new(&cfg.device_ids).context("HipCluster::new")?);
            if shared_cluster.ranks() as u32 != world {
                bail!(
                    "--mesh-mode tp: --tp-size {world} but cluster has {} ranks",
                    shared_cluster.ranks()
                );
            }
            let mut layout = ModelLayout::from_config(&cfg_g4);
            let _shared_kv = layout.resolve_kv_sharing();
            info!(
                num_layers = cfg_g4.num_layers,
                ranks = shared_cluster.ranks(),
                topology = "tp",
                arch = cfg_g4.arch.as_str(),
                inflight_slots,
                "loading gemma4 weights"
            );
            let driver0 = Gemma4TpDriver::upload(
                &gguf,
                cfg_g4.clone(),
                layout,
                shared_cluster.clone(),
                cfg_g4.context_length,
            )
            .context("Gemma4TpDriver::upload")?;
            let model = Arc::clone(&driver0.model);
            let mut drivers: Vec<Box<dyn ModelDriver>> = Vec::with_capacity(inflight_slots);
            drivers.push(Box::new(driver0));
            for slot in 1..inflight_slots {
                let d = Gemma4TpDriver::new_session(Arc::clone(&model), cfg_g4.context_length)
                    .with_context(|| format!("Gemma4TpDriver::new_session slot {slot}"))?;
                drivers.push(Box::new(d));
            }
            (shared_cluster, drivers)
        }
        MeshMode::Hybrid { pp_size, tp_size } => {
            let pp = pp_size as usize;
            let tp = tp_size as usize;
            if pp * tp != cfg.device_ids.len() {
                bail!(
                    "--mesh-mode pp+tp: pp_size*tp_size ({pp}*{tp}) != device count ({})",
                    cfg.device_ids.len()
                );
            }
            // Per-stage sub-clusters (device-major: stage s owns
            // device_ids[s*tp .. (s+1)*tp]). Built BEFORE the global
            // cluster so HybridCluster::new can record the construction
            // order invariant (MEMORY.md `hybrid_cluster_order`).
            let mut sub_clusters: Vec<Arc<HipCluster>> = Vec::with_capacity(pp);
            for s in 0..pp {
                let stage_ids = &cfg.device_ids[s * tp..(s + 1) * tp];
                sub_clusters.push(Arc::new(
                    HipCluster::new(stage_ids)
                        .with_context(|| format!("HipCluster::new (sub_cluster stage {s})"))?,
                ));
            }
            let global_cluster: Arc<HipCluster> = Arc::new(
                HipCluster::new(&cfg.device_ids)
                    .context("HipCluster::new (global, for inter-stage hand-off)")?,
            );
            let hc = flambeau_blocks::HybridCluster::new(
                sub_clusters,
                Arc::clone(&global_cluster),
                tp,
            )
            .context("HybridCluster::new")?;
            let mut layout = ModelLayout::from_config(&cfg_g4);
            let _shared_kv = layout.resolve_kv_sharing();
            info!(
                num_layers = cfg_g4.num_layers,
                pp_size = pp,
                tp_size = tp,
                topology = "pp+tp",
                arch = cfg_g4.arch.as_str(),
                inflight_slots,
                "loading gemma4 weights"
            );
            let driver0 =
                Gemma4HybridDriver::upload(&gguf, cfg_g4.clone(), layout, hc, cfg_g4.context_length)
                    .context("Gemma4HybridDriver::upload")?;
            let model = Arc::clone(&driver0.model);
            let mut drivers: Vec<Box<dyn ModelDriver>> = Vec::with_capacity(inflight_slots);
            drivers.push(Box::new(driver0));
            for slot in 1..inflight_slots {
                let d = Gemma4HybridDriver::new_session(Arc::clone(&model), cfg_g4.context_length)
                    .with_context(|| format!("Gemma4HybridDriver::new_session slot {slot}"))?;
                drivers.push(Box::new(d));
            }
            (global_cluster, drivers)
        }
    };

    let model = build_gemma4_loaded_model(cfg_g4.clone(), topology_label);
    let bos_id = boot.tokenizer.bos_id;
    let inflight_pool: Vec<Mutex<Box<dyn crate::Session>>> = drivers
        .into_iter()
        .map(|d| Mutex::new(wrap_gemma4_driver(d, bos_id)))
        .collect();

    let topology_tag =
        crate::serve_common::topology_tag_from_mesh(cfg.mesh_mode, cfg.device_ids.len());
    let prefix_cache =
        crate::serve_common::build_prefix_cache(&cfg, topology_tag.mesh_kind);

    let state = crate::serve_common::build_server_state(
        crate::serve_common::ServerStateInputs {
            model_id: cfg.model_id.clone(),
            model_cfg: crate::model_cfg::ServerModelCfg::from(&cfg_g4),
            model,
            cluster: state_cluster,
            inflight_pool,
            qwen3_moe: None,
            embedding: None,
            embedding_rank: None,
            gpu_sampler: false,
            batched_decode: cfg.batched_decode,
            max_queue_depth,
            prefill_ubatch,
            topology_tag,
            prefix_cache,
            boot,
        },
    );

    crate::serve_common::run_axum(state, cfg.bind_addr, "gemma4").await
}

/// v2 forward-stack boot path. Selects `A: Arch` via the GGUF arch
/// string, builds one `Session<A>` per inflight slot, wraps each as a
/// `V2Session`, then assembles `ServerState` + `run_axum` exactly like
/// the legacy paths.
///
/// Arch dispatch is intentionally a `match` on the arch string in
/// `create_v2_driver`: one branch per supported arch crate. Adding a
/// new v2 arch = one new branch + a `flambeau-<arch>-v2` workspace
/// dep, with no churn at the routes/sampler/parser layer.
pub(crate) async fn serve_inner_v2(
    cfg: ServeConfig,
    gguf: GgufFile,
    boot: crate::serve_common::BootMetadata,
) -> Result<()> {
    use flambeau_runtime::ModelDriver;

    let gguf_arch_owned = gguf
        .metadata_str("general.architecture")
        .unwrap_or("")
        .to_string();
    let gguf_arch: &str = &gguf_arch_owned;

    let n_available = device_count().unwrap_or(0);
    for d in &cfg.device_ids {
        if *d < 0 || *d >= n_available {
            bail!("device {d} not available (have {n_available} HIP devices)");
        }
    }

    let model_cfg = parse_v2_server_model_cfg(gguf_arch, &cfg.gguf_path)?;
    let topology = topology_from_mesh(cfg.mesh_mode, &cfg.device_ids, model_cfg.num_layers)?;
    let topology_label: &'static str = match cfg.mesh_mode {
        MeshMode::Pp => "pp",
        MeshMode::Tp { .. } => "tp",
        MeshMode::Hybrid { .. } => "pp+tp",
    };

    let inflight_slots = cfg.inflight_slots.clamp(1, 32);
    let prefill_ubatch = cfg.prefill_ubatch.max(128);
    let max_queue_depth = cfg.max_queue_depth;
    info!(
        arch = gguf_arch,
        topology = topology_label,
        inflight_slots,
        prefill_ubatch,
        max_queue_depth,
        "v2: pre-allocating inflight slot pool"
    );

    // `GgufFile` is single-owner (mmap handle), but `Session<A>::new`
    // wraps it in `Arc` internally; re-opening per slot is the
    // simplest way to feed N owned `GgufFile`s without making the
    // type Cloneable. Open cost is two syscalls + an mmap walk —
    // dominated by the per-slot weight-upload that follows.
    drop(gguf);
    let gguf_path = cfg.gguf_path.clone();
    let chat_stops = crate::v2_handle::chat_stops_for(gguf_arch);
    let bos_id = if crate::v2_handle::wants_bos_prepend(gguf_arch) {
        boot.tokenizer.bos_id
    } else {
        None
    };
    let mut inflight_pool: Vec<Mutex<Box<dyn crate::Session>>> =
        Vec::with_capacity(inflight_slots);
    for slot_idx in 0..inflight_slots {
        let slot_gguf = GgufFile::open(&gguf_path)
            .with_context(|| format!("re-open GGUF for v2 slot {slot_idx}"))?;
        let driver: Box<dyn ModelDriver> = create_v2_driver(
            gguf_arch,
            slot_gguf,
            topology.clone(),
            cfg.ctx_cap,
            prefill_ubatch,
        )
        .with_context(|| format!("v2 driver slot {slot_idx} ({gguf_arch})"))?;
        let session: Box<dyn crate::Session> = Box::new(crate::v2_handle::V2Session {
            driver,
            bos_id,
            chat_stops,
        });
        inflight_pool.push(Mutex::new(session));
    }

    let cluster: Arc<HipCluster> =
        Arc::new(HipCluster::new(&cfg.device_ids).context("v2: HipCluster::new (state side)")?);
    let topology_tag =
        crate::serve_common::topology_tag_from_mesh(cfg.mesh_mode, cfg.device_ids.len());
    let prefix_cache =
        crate::serve_common::build_prefix_cache(&cfg, topology_tag.mesh_kind);

    // `--embedding-model` plumbing is topology-independent — the
    // embedding GGUF loads onto a single device and the handle runs
    // its own pooled-forward. Wire it in here so v2 chat servers
    // can serve `/v1/embeddings` alongside `/v1/chat/completions`.
    let embedding = load_embedding_qwen3(&cfg, &cluster)?;
    let embedding_rank = embedding.as_ref().map(|(_, _, r)| *r);
    let embedding = embedding.map(|(m, t, _)| (m, t));

    let model = std::sync::Arc::new(crate::v2_handle::V2Model {
        gguf_arch: gguf_arch_to_static(gguf_arch),
        topology: topology_label,
        chat_stops,
    }) as crate::qwen3moe_handle::LoadedModel;

    let state = crate::serve_common::build_server_state(
        crate::serve_common::ServerStateInputs {
            model_id: cfg.model_id.clone(),
            model_cfg,
            model,
            cluster,
            inflight_pool,
            qwen3_moe: None,
            embedding,
            embedding_rank,
            gpu_sampler: false,
            batched_decode: false,
            max_queue_depth,
            prefill_ubatch,
            topology_tag,
            prefix_cache,
            boot,
        },
    );

    crate::serve_common::run_axum(state, cfg.bind_addr, "v2").await
}

fn topology_from_mesh(
    mesh: MeshMode,
    device_ids: &[i32],
    num_layers: usize,
) -> Result<flambeau_forward::Topology> {
    use flambeau_forward::Topology;
    // `flambeau_forward`'s orchestrator falls back to (start=0, end=0)
    // empty per-rank layer ranges when `layer_split: None`; an
    // explicit even split is required for PP / Hybrid to actually
    // execute layers. SingleDevice avoids the issue entirely at N=1.
    let even_split = |n_groups: usize| -> Vec<usize> {
        let base = num_layers / n_groups;
        let rem = num_layers % n_groups;
        (0..n_groups)
            .map(|i| base + usize::from(i < rem))
            .collect()
    };
    match mesh {
        MeshMode::Pp if device_ids.len() == 1 => Ok(Topology::SingleDevice {
            device: device_ids[0],
        }),
        MeshMode::Pp => Ok(Topology::Pp {
            devices: device_ids.to_vec(),
            layer_split: Some(even_split(device_ids.len())),
        }),
        MeshMode::Tp { world } => {
            if device_ids.len() as u32 != world {
                bail!(
                    "--mesh-mode tp: --tp-size {world} but {} devices supplied",
                    device_ids.len()
                );
            }
            Ok(Topology::Tp {
                devices: device_ids.to_vec(),
            })
        }
        MeshMode::Hybrid { pp_size, tp_size } => {
            let pp = pp_size as usize;
            let tp = tp_size as usize;
            if pp * tp != device_ids.len() {
                bail!(
                    "--mesh-mode pp+tp: {pp}*{tp} != {} devices",
                    device_ids.len()
                );
            }
            let stages: Vec<Vec<i32>> = (0..pp)
                .map(|s| device_ids[s * tp..(s + 1) * tp].to_vec())
                .collect();
            Ok(Topology::Hybrid {
                stages,
                layer_split: Some(even_split(pp)),
            })
        }
    }
}

fn create_v2_driver(
    gguf_arch: &str,
    file: GgufFile,
    topology: flambeau_forward::Topology,
    ctx_cap: Option<usize>,
    prefill_ubatch: usize,
) -> Result<Box<dyn flambeau_runtime::ModelDriver>> {
    use flambeau_forward::Session;
    match gguf_arch {
        "qwen35" => {
            let s = Session::<flambeau_qwen35_v2::Qwen35V2>::new(
                file,
                topology,
                ctx_cap,
                prefill_ubatch,
            )?;
            Ok(Box::new(s))
        }
        "qwen35moe" => {
            let s = Session::<flambeau_qwen35moe_v2::Qwen35MoeV2>::new(
                file,
                topology,
                ctx_cap,
                prefill_ubatch,
            )?;
            Ok(Box::new(s))
        }
        "gemma3" | "gemma4" | "gemma4-26b-a4b" | "gemma4-31b" | "gemma4-9b" | "gemma4-2b" => {
            let s = Session::<flambeau_gemma4_v2::Gemma4V2>::new(
                file,
                topology,
                ctx_cap,
                prefill_ubatch,
            )?;
            Ok(Box::new(s))
        }
        other => bail!("v2 serve: unsupported GGUF arch `{other}`"),
    }
}

fn gguf_arch_to_static(arch: &str) -> &'static str {
    match arch {
        "qwen35" => "qwen35",
        "qwen35moe" => "qwen35moe",
        "gemma3" => "gemma3",
        "gemma4" => "gemma4",
        "gemma4-26b-a4b" => "gemma4-26b-a4b",
        "gemma4-31b" => "gemma4-31b",
        "gemma4-9b" => "gemma4-9b",
        "gemma4-2b" => "gemma4-2b",
        _ => "v2",
    }
}

fn parse_v2_server_model_cfg(
    gguf_arch: &str,
    gguf_path: &std::path::Path,
) -> Result<crate::model_cfg::ServerModelCfg> {
    let f = GgufFile::open(gguf_path).context("v2 cfg: re-open GGUF for metadata")?;
    // GGUF metadata keys are namespaced by `general.architecture`.
    let prefix = gguf_arch;
    let key = |suffix: &str| format!("{prefix}.{suffix}");
    let num_layers = f
        .metadata_u32(&key("block_count"))
        .ok_or_else(|| anyhow::anyhow!("v2 cfg: missing `{}.block_count`", prefix))?
        as usize;
    let context_length = f
        .metadata_u32(&key("context_length"))
        .ok_or_else(|| anyhow::anyhow!("v2 cfg: missing `{}.context_length`", prefix))?
        as usize;
    let vocab_size = f
        .info("token_embd.weight")
        .ok()
        .and_then(|ti| ti.dims.first().copied())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow::anyhow!("v2 cfg: token_embd.weight missing"))?;
    Ok(crate::model_cfg::ServerModelCfg {
        arch: gguf_arch.to_string(),
        vocab_size,
        context_length,
        num_layers,
    })
}
