//! `flambeau serve` entry point — loads the model + tokenizer, starts axum.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::routing::{get, post};
use axum::Router;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::{ChatTemplate, GgufFile};
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoEHybridModel,
    Qwen3MoEShardedModel, Qwen3MoETpModel,
};
use flambeau_runtime::{LayerAssignment, Registry};
use tokio::sync::Mutex;
use tracing::info;

use crate::model::LoadedModel;
use crate::routes::{
    agent_stats, chat_completions, completions, detokenize, embeddings, health, infill,
    messages_anthropic, models, tokenize, ServerState, SharedState,
};

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

/// Blocking serve loop — loads the model, starts the HTTP server, runs
/// until terminated. Caller owns the tokio runtime.
pub async fn serve(cfg: ServeConfig, registry: Registry) -> Result<()> {
    info!(?cfg, "flambeau serve: loading model");

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

    // Load tokenizer + chat template first (cheap, catch config errors early).
    let tokenizer = flambeau_quant::load_from_gguf(&gguf).context("load tokenizer")?;
    let chat_template =
        ChatTemplate::load_from_gguf(&gguf).context("load chat template")?;

    // L3 — detect tool-call format from the chat-template source. The
    // Unsloth UD Qwen3.6 GGUFs ship a Coder-XML template under the
    // qwen35moe arch tag; we can't decide from arch alone.
    let tpl_src = gguf
        .metadata_str("tokenizer.chat_template")
        .unwrap_or("");
    let tool_call_format_default =
        crate::tool_call_parser::detect_format_from_template(tpl_src);
    info!(
        format = ?tool_call_format_default,
        "tool-call format detected from chat template"
    );

    // **#235 P3.15** — `enable_thinking` Jinja variable detection.
    // Qwen3.6 templates render `<think>` blocks when this is true; the
    // bool feeds the `/v1/models` `"thinking"` capability so clients
    // can choose whether to expose the request flag (#233).
    let supports_thinking = tpl_src.contains("enable_thinking");

    // **#235 P3.15** — quantization label from GGUF `general.file_type`.
    // The integer enum mirrors llama.cpp's LLAMA_FTYPE; we only label
    // the families we actually load. Unknown values surface as
    // `"type=N"` rather than `None` so a new quant doesn't go silent.
    let quantization: Option<String> = gguf
        .metadata_u32("general.file_type")
        .map(|ft| match ft {
            0 => "F32".to_string(),
            1 => "F16".to_string(),
            2 => "Q4_0".to_string(),
            3 => "Q4_1".to_string(),
            6 => "Q5_0".to_string(),
            7 => "Q5_1".to_string(),
            8 => "Q8_0".to_string(),
            9 => "Q8_1".to_string(),
            10 => "Q2_K".to_string(),
            11 => "Q3_K_S".to_string(),
            12 => "Q3_K_M".to_string(),
            13 => "Q3_K_L".to_string(),
            14 => "Q4_K_S".to_string(),
            15 => "Q4_K_M".to_string(),
            16 => "Q5_K_S".to_string(),
            17 => "Q5_K_M".to_string(),
            18 => "Q6_K".to_string(),
            32 => "BF16".to_string(),
            other => format!("type={other}"),
        });

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
        return serve_inner_gemma4(
            cfg,
            gguf,
            tokenizer,
            chat_template,
            tool_call_format_default,
            supports_thinking,
            quantization,
        )
        .await;
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
                std::sync::Arc::new(crate::model::PpHipModel { model: m }) as LoadedModel,
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
                std::sync::Arc::new(crate::model::TpHipModel { model: m, tp }) as LoadedModel,
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
                std::sync::Arc::new(crate::model::HybridHipModel { model: hybrid, hc })
                    as LoadedModel,
            )
        }
    };

    let model_defaults = crate::state::ModelDefaults::from_gguf(&gguf);
    info!(
        temperature = ?model_defaults.temperature,
        top_p = ?model_defaults.top_p,
        top_k = ?model_defaults.top_k,
        min_p = ?model_defaults.min_p,
        "model sampling defaults from GGUF"
    );

    // P0.5 — boot-time default system prompt. Empty string treated as
    // unset so an operator can clear a system-level config by passing
    // `--default-system ""`.
    let default_system = cfg
        .default_system
        .as_ref()
        .filter(|s| !s.is_empty())
        .cloned();
    if let Some(s) = default_system.as_deref() {
        info!(len = s.len(), "default system prompt loaded");
    }

    // **P2.9b-i1 (multi-slot pool)** — pre-allocate N inflight slots
    // sized to FLAMBEAU_PREFILL_UBATCH (default 512). Each slot owns
    // its own session (KV cache, GDN state) and scratch buffers; a
    // request acquires any free slot via try-lock round-robin and
    // returns it to the pool on response. N defaults to 1 (P2.9a
    // behaviour); N>1 enables request-level concurrency. Decode
    // kernels still serialise on the GPU stream — true batched
    // throughput lands in P2.9b-i2.
    let prefill_ubatch = cfg.prefill_ubatch.max(128);
    let inflight_slots = cfg.inflight_slots.clamp(1, 32);
    // **#232 P2.12** — admission control. Cap at `inflight_slots +
    // max_queue_depth`; new requests beyond that get 503 +
    // Retry-After: 2. `0` disables (legacy behaviour). Default 16.
    let max_queue_depth = cfg.max_queue_depth;
    info!(
        prefill_ubatch,
        inflight_slots,
        max_queue_depth,
        "pre-allocating inflight slot pool"
    );
    let mut inflight_pool: Vec<Mutex<Box<dyn crate::HipSession>>> =
        Vec::with_capacity(inflight_slots);
    for slot_idx in 0..inflight_slots {
        // Phase 12.8 — pool holds the model-agnostic `HipSession` trait.
        // `create_hip_session` builds the qwen3-moe-typed `OwnedHipSession`
        // and erases it behind the trait. Gemma4 will add a parallel
        // factory in serve.rs's arch-dispatch branch.
        let slot = crate::create_hip_session(
            model.clone(),
            &cluster,
            prefill_ubatch,
            flambeau_qwen3_moe::session::KvLayout::from_str(&cfg.kv),
        )
            .with_context(|| format!("pre-alloc inflight slot {slot_idx} at boot"))?;
        inflight_pool.push(Mutex::new(slot));
    }

    // **P2.9b-i2-B (scheduler)** — request-lifetime claim flags +
    // empty pending queue + leader gate. These are used by the
    // scheduler-aware decode path (`FLAMBEAU_BATCHED_DECODE=1`)
    // to aggregate concurrent decode requests into batched dispatches.
    let slot_in_use: Vec<std::sync::atomic::AtomicBool> = (0..inflight_slots)
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();

    // **#229 P2.10c** — process-local prefix cache. Always constructed;
    // `prefix_cache.enabled()` (set from `cfg.prefix_cache`) controls
    // whether request handlers actually consult it. Empty index +
    // zero-byte LRU at boot.
    let prefix_cache = Arc::new(crate::prefix_cache::PrefixCache::new(
        crate::prefix_cache::PrefixCache::gb_to_bytes(cfg.prefix_cache_max_gb),
        cfg.prefix_cache,
    ));
    let topology_tag = match cfg.mesh_mode {
        MeshMode::Pp => crate::prefix_cache::TopologyTag {
            mesh_kind: "pp",
            ranks: cfg.device_ids.len() as u32,
            pp_size: cfg.device_ids.len() as u32,
            tp_size: 1,
        },
        MeshMode::Tp { world } => crate::prefix_cache::TopologyTag {
            mesh_kind: "tp",
            ranks: world,
            pp_size: 1,
            tp_size: world,
        },
        MeshMode::Hybrid { pp_size, tp_size } => crate::prefix_cache::TopologyTag {
            mesh_kind: "pp+tp",
            ranks: pp_size * tp_size,
            pp_size,
            tp_size,
        },
    };
    if prefix_cache.enabled() {
        info!(
            chunk_tokens = prefill_ubatch,
            budget_bytes = prefix_cache.vram_budget_bytes,
            mesh = topology_tag.mesh_kind,
            "prefix cache ENABLED (FLAMBEAU_PREFIX_CACHE=1)"
        );
    } else {
        info!("prefix cache disabled (set FLAMBEAU_PREFIX_CACHE=1 to enable)");
    }

    // **#230 P2.11a** — optional embedding model. Loaded after the
    // chat model + inflight pool so any boot-time OOM lands here
    // (where it's clearly an embedding-specific failure) rather than
    // mid-request. Reuses the chat cluster's per-device handle: we
    // resolve the requested embedding device id back to its rank in
    // the cluster, then pass the matching `&HipDevice`. Errors abort
    // the server boot — operator can omit `--embedding-model` to
    // disable.
    let mut embedding_rank: Option<usize> = None;
    let embedding_model: Option<(
        Arc<tokio::sync::Mutex<flambeau_qwen3_moe::EmbeddingModel>>,
        Arc<flambeau_quant::GgufTokenizer>,
    )> =
        if let Some(path) = cfg.embedding_gguf_path.as_ref() {
            let device_id = cfg.embedding_device_id.unwrap_or(cfg.device_ids[0]);
            let rank = cfg
                .device_ids
                .iter()
                .position(|d| *d == device_id)
                .ok_or_else(|| anyhow::anyhow!(
                    "embedding device {device_id} not in --devices {:?}",
                    cfg.device_ids
                ))?;
            embedding_rank = Some(rank);
            let device = cluster.device(rank);
            info!(
                path = %path.display(),
                device_id,
                rank,
                "loading embedding model"
            );
            let efile = GgufFile::open(path)
                .with_context(|| format!("open embedding GGUF at {}", path.display()))?;
            // **#231 quality fix** — load the embedding model's own
            // tokenizer (vocab_size differs from chat tokenizer:
            // Qwen3-Embedding ships 151669, Qwen3.5-9B ships 151424).
            // Token ids from the chat tokenizer dereference into the
            // wrong rows of the embedding model's `token_embd`,
            // producing non-discriminating output vectors.
            let embedding_tokenizer =
                flambeau_quant::load_from_gguf(&efile)
                    .context("load embedding tokenizer from GGUF")?;
            // **#231** — `max_tokens` caps the longest input the
            // `/v1/embeddings` endpoint will accept. Default 8192;
            // override with --embedding-max-tokens. Realistic RAG /
            // memory chunking patterns sit at 512–2048 tokens.
            let max_emb_tokens = cfg.embedding_max_tokens.clamp(16, 32768);
            let em = flambeau_qwen3_moe::EmbeddingModel::load(
                &efile,
                device,
                device_id,
                max_emb_tokens,
            )
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
            Some((
                Arc::new(tokio::sync::Mutex::new(em)),
                Arc::new(embedding_tokenizer),
            ))
        } else {
            info!("embedding model not configured (--embedding-model unset)");
            None
        };
    let (embedding_model, embedding_tokenizer): (
        Option<Arc<tokio::sync::Mutex<flambeau_qwen3_moe::EmbeddingModel>>>,
        Option<Arc<flambeau_quant::GgufTokenizer>>,
    ) = match embedding_model {
        Some((m, t)) => (Some(m), Some(t)),
        None => (None, None),
    };

    let state: SharedState = Arc::new(ServerState {
        model_id: cfg.model_id.clone(),
        cfg: crate::model_cfg::ServerModelCfg::from(&model_cfg),
        model,
        cluster,
        tokenizer,
        chat_template,
        inflight_pool,
        slot_in_use,
        batched_pending: std::sync::Mutex::new(Vec::new()),
        batched_dispatcher: std::sync::Mutex::new(()),
        tp_batched_scratch: std::sync::Mutex::new(None),
        hybrid_batched_scratch: std::sync::Mutex::new(None),
        prefill_serialiser: std::sync::Mutex::new(()),
        tp_prefill_scratch: std::sync::Mutex::new(None),
        prefix_cache,
        prefix_cache_chunk_tokens: prefill_ubatch,
        topology_tag,
        embedding_model,
        embedding_tokenizer,
        embedding_rank,
        in_flight: std::sync::atomic::AtomicUsize::new(0),
        max_queue_depth,
        prefill_ubatch,
        gpu_sampler: cfg.gpu_sampler,
        batched_decode: cfg.batched_decode,
        agent_stats: crate::agent_stats::AgentStatsRing::default(),
        tool_call_format_default,
        supports_thinking,
        quantization,
        model_defaults,
        default_system,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        // **#231 P2.11b** — OpenAI-compat embeddings endpoint. 503
        // when the server was started without `--embedding-model`.
        .route("/v1/embeddings", post(embeddings))
        // P1.6b — llama.cpp-compatible Fill-in-the-Middle. Both
        // top-level (`/infill`, llama.cpp + Continue) and
        // namespaced (`/v1/infill`) for clients that expect API-
        // versioned routes.
        .route("/infill", post(infill))
        .route("/v1/infill", post(infill))
        // **#234 P3.14** — llama.cpp-compatible tokenize / detokenize.
        // Both top-level and `/v1/` namespaced; same handler. No GPU
        // work — pure tokenizer round-trips for clients that need to
        // count tokens or render token boundaries.
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
        // P1.8a — Anthropic Messages API (text-only, non-streaming for
        // now). Tools (P1.8c) and SSE (P1.8b) layer in afterwards.
        .route("/v1/messages", post(messages_anthropic))
        // Read-only agent-loop telemetry snapshot.
        .route("/v1/agent/stats", get(agent_stats))
        .with_state(state);

    info!(bind = %cfg.bind_addr, "serving");
    let listener = tokio::net::TcpListener::bind(cfg.bind_addr)
        .await
        .context("bind listener")?;
    axum::serve(listener, app)
        .await
        .context("axum::serve failed")?;
    Ok(())
}

/// Phase 12.9 — gemma4 boot path. Branched into from `serve_inner`
/// when `gguf.architecture()` matches a gemma4 family.
///
/// MVP constraints (each tracked as a follow-up):
/// - PP only. TP / Hybrid drivers exist in gemma4 crate but their
///   server arch dispatch isn't plumbed yet.
/// - `FLAMBEAU_INFLIGHT_SLOTS=1` enforced. Gemma4 drivers bundle
///   weights + KV cache state in one struct; multi-slot would need
///   splitting (separate session-state struct) or N copies of weights.
/// - Prefix cache silently misses (gated upstream via the
///   `as_pp/as_tp/as_hybrid` early-out in `prefix_cache_try_restore`).
/// - GPU sampler disabled (`use_gpu_sampler=false`); host sampler runs.
/// - Embedding endpoint returns 503 (qwen3-only embedding model).
/// - `reset_for_next_request` bails — the slot can't be reused after
///   the first request. Stops the server from looping in production;
///   restart for a fresh request. Trade-off for shipping the MVP.
#[allow(clippy::too_many_arguments)]
async fn serve_inner_gemma4(
    cfg: ServeConfig,
    gguf: GgufFile,
    tokenizer: flambeau_quant::GgufTokenizer,
    chat_template: ChatTemplate,
    tool_call_format_default: crate::tool_call_parser::ToolCallFormat,
    supports_thinking: bool,
    quantization: Option<String>,
) -> Result<()> {
    use flambeau_gemma4::{
        partition_layers, Gemma4Config, Gemma4PpDriver, Gemma4TpDriver, ModelLayout,
    };
    use flambeau_runtime::ModelDriver;

    use crate::gemma4_handle::{build_gemma4_loaded_model, wrap_gemma4_driver};

    // PP and TP supported at MVP. Hybrid follow-up.
    let topology_label: &'static str = match cfg.mesh_mode {
        MeshMode::Pp => "pp",
        MeshMode::Tp { .. } => "tp",
        MeshMode::Hybrid { .. } => bail!(
            "gemma4 serve: --mesh-mode pp+tp not yet supported (got {:?}). Follow-up.",
            cfg.mesh_mode
        ),
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
    if inflight_slots != 1 {
        bail!(
            "gemma4 serve: FLAMBEAU_INFLIGHT_SLOTS must be 1 (got {inflight_slots}). \
             Multi-slot needs gemma4 weights/session split — follow-up."
        );
    }
    let prefill_ubatch = cfg.prefill_ubatch.max(128);
    let max_queue_depth = cfg.max_queue_depth;

    // Build the driver per topology.
    // - PP: `Gemma4PpDriver::upload` consumes its cluster by value, so
    //   we build a SECOND, state-side `Arc<HipCluster>` whose only
    //   purpose is satisfying `ServerState.cluster`'s type (the
    //   handler paths that read it are all qwen3-moe-gated).
    // - TP: `Gemma4TpDriver::upload` takes `Arc<HipCluster>`, so the
    //   state-side cluster and the driver's cluster are the same Arc.
    let (state_cluster, driver): (Arc<HipCluster>, Box<dyn ModelDriver>) = match cfg.mesh_mode {
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
                "loading gemma4 weights"
            );
            let pp_driver = Gemma4PpDriver::upload(
                &gguf,
                cfg_g4.clone(),
                layout,
                layer_to_rank,
                driver_cluster,
                prefill_ubatch,
            )
            .context("Gemma4PpDriver::upload")?;
            (state_cluster, Box::new(pp_driver))
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
                "loading gemma4 weights"
            );
            let tp_driver = Gemma4TpDriver::upload(
                &gguf,
                cfg_g4.clone(),
                layout,
                shared_cluster.clone(),
                prefill_ubatch,
            )
            .context("Gemma4TpDriver::upload")?;
            (shared_cluster, Box::new(tp_driver))
        }
        MeshMode::Hybrid { .. } => unreachable!("guarded above"),
    };

    let model = build_gemma4_loaded_model(cfg_g4.clone(), topology_label);
    let session = wrap_gemma4_driver(driver, tokenizer.bos_id);
    let inflight_pool: Vec<Mutex<Box<dyn crate::HipSession>>> = vec![Mutex::new(session)];

    let slot_in_use: Vec<std::sync::atomic::AtomicBool> = (0..inflight_slots)
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();

    // Prefix cache — keep the field populated but disabled by
    // default; gemma4 sessions skip via the `as_pp/as_tp/as_hybrid`
    // early-out in `prefix_cache_try_restore`.
    let prefix_cache = Arc::new(crate::prefix_cache::PrefixCache::new(
        crate::prefix_cache::PrefixCache::gb_to_bytes(cfg.prefix_cache_max_gb),
        cfg.prefix_cache,
    ));
    let topology_tag = match cfg.mesh_mode {
        MeshMode::Pp => crate::prefix_cache::TopologyTag {
            mesh_kind: "pp",
            ranks: cfg.device_ids.len() as u32,
            pp_size: cfg.device_ids.len() as u32,
            tp_size: 1,
        },
        MeshMode::Tp { world } => crate::prefix_cache::TopologyTag {
            mesh_kind: "tp",
            ranks: world,
            pp_size: 1,
            tp_size: world,
        },
        MeshMode::Hybrid { .. } => unreachable!("guarded above"),
    };

    let model_defaults = crate::state::ModelDefaults::from_gguf(&gguf);
    let default_system = cfg
        .default_system
        .as_ref()
        .filter(|s| !s.is_empty())
        .cloned();

    let state: SharedState = Arc::new(ServerState {
        model_id: cfg.model_id.clone(),
        cfg: crate::model_cfg::ServerModelCfg::from(&cfg_g4),
        model,
        cluster: state_cluster,
        tokenizer,
        chat_template,
        inflight_pool,
        slot_in_use,
        batched_pending: std::sync::Mutex::new(Vec::new()),
        batched_dispatcher: std::sync::Mutex::new(()),
        tp_batched_scratch: std::sync::Mutex::new(None),
        hybrid_batched_scratch: std::sync::Mutex::new(None),
        prefill_serialiser: std::sync::Mutex::new(()),
        tp_prefill_scratch: std::sync::Mutex::new(None),
        prefix_cache,
        prefix_cache_chunk_tokens: prefill_ubatch,
        topology_tag,
        embedding_model: None,
        embedding_tokenizer: None,
        embedding_rank: None,
        in_flight: std::sync::atomic::AtomicUsize::new(0),
        max_queue_depth,
        prefill_ubatch,
        gpu_sampler: false,
        batched_decode: cfg.batched_decode,
        agent_stats: crate::agent_stats::AgentStatsRing::default(),
        tool_call_format_default,
        supports_thinking,
        quantization,
        model_defaults,
        default_system,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/infill", post(infill))
        .route("/v1/infill", post(infill))
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/messages", post(messages_anthropic))
        .route("/v1/agent/stats", get(agent_stats))
        .with_state(state);

    info!(bind = %cfg.bind_addr, "serving (gemma4)");
    let listener = tokio::net::TcpListener::bind(cfg.bind_addr)
        .await
        .context("bind listener")?;
    axum::serve(listener, app)
        .await
        .context("axum::serve failed")?;
    Ok(())
}
