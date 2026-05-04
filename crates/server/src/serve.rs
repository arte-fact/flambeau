//! `flambeau serve` entry point — loads the model + tokenizer, starts axum.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::routing::{get, post};
use axum::Router;
use flambeau_backend_hip::{device_count, BarP2pAllReduce, HipCluster};
use flambeau_quant::{ChatTemplate, GgufFile};
use flambeau_qwen3_moe::{
    HybridMeshSpec, Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoEHybridModel,
    Qwen3MoEShardedModel, Qwen3MoETpModel,
};
use flambeau_runtime::LayerAssignment;
use tokio::sync::Mutex;
use tracing::info;

use crate::model::LoadedModel;
use crate::routes::{
    agent_stats, chat_completions, completions, health, index, infill, messages_anthropic,
    models, tools_endpoint, ServerState, SharedState,
};

/// **TP-5a** — mesh topology selector. PP-V1 default; TP engages the
/// Qwen3MoETpModel loader + the BarP2pAllReduce-based forward path.
/// **AUTO-4a** — `Hybrid` adds a manual PP-of-TP composition where
/// `pp_size` contiguous layer stages each own a `tp_size`-rank TP
/// subgroup. Selection is operator-driven; flambeau does not autodetect
/// the right topology for a given rig (the bracket-bench harness in
/// AUTO-5 produces the data, the operator picks the winner).
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
    /// stage 1 = {d2, d3}. Forward wiring lands in AUTO-4b..f; this
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
    /// **TP-5a** — mesh topology. Defaults to `Pp` for V1 callers that
    /// don't set the field explicitly (constructors use struct-update
    /// syntax with `..Default::default()`).
    pub mesh_mode: MeshMode,
    /// Upstream MCP servers to register as a tool source (ROADMAP-V2
    /// §M2.1). Each URL is enumerated once at boot and its tools are
    /// exposed to the model alongside the client-supplied `tools[]`
    /// on each chat completion. The agent loop that actually invokes
    /// the remote tools is M2.2.
    pub mcp_urls: Vec<String>,
}

impl Default for MeshMode {
    fn default() -> Self {
        MeshMode::Pp
    }
}

/// Blocking serve loop — loads the model, starts the HTTP server, runs
/// until terminated. Caller owns the tokio runtime.
pub async fn serve(cfg: ServeConfig) -> Result<()> {
    info!(?cfg, "flambeau serve: loading model");

    let gguf = GgufFile::open(&cfg.gguf_path)
        .with_context(|| format!("open GGUF at {}", cfg.gguf_path.display()))?;

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

    // Sanity: device_ids must be valid.
    let n_available = device_count().unwrap_or(0);
    for d in &cfg.device_ids {
        if *d < 0 || *d >= n_available {
            bail!(
                "device {d} not available (have {n_available} HIP devices)"
            );
        }
    }

    let mut model_cfg = Qwen3MoEConfig::from_gguf(&gguf).context("model config from GGUF")?;
    // Allow operators to clamp the model's KV-cache provisioning ceiling
    // (mirrors the test-side FLAMBEAU_CTX_CAP). The on-disk
    // `context_length` is often the architectural max (262144 for
    // Qwen3.5/3.6) which would OOM the per-rank KV cache on consumer
    // VRAM. The clamp only shrinks; explicit increases are ignored.
    if let Some(cap) = std::env::var("FLAMBEAU_CTX_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        if cap > 0 && cap < model_cfg.context_length {
            info!(
                from = model_cfg.context_length,
                to = cap,
                "FLAMBEAU_CTX_CAP shrinking model.context_length"
            );
            model_cfg.context_length = cap;
        }
    }
    // **AUTO-4f** — construction order matters on this rig. For pure
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

            // MTP-5d: opt-in MTP attachment. Loaded once, lives on
            // the last rank; per-request scratch allocated on each
            // chat completion.
            let mtp = match std::env::var("FLAMBEAU_SPEC_MTP") {
                Ok(path) if !path.is_empty() && path != "0" && path != "off" => {
                    let last_rank = (cluster.ranks() - 1) as usize;
                    let last_device = cluster.device(last_rank);
                    info!(path = %path, rank = last_rank, "loading MTP head for spec-decode");
                    let mtp_file = flambeau_quant::GgufFile::open(std::path::Path::new(&path))
                        .with_context(|| format!("MTP gguf {path}"))?;
                    let head = flambeau_qwen3_moe::mtp::load_mtp_head(&mtp_file, last_device)
                        .context("load_mtp_head")?;
                    info!(
                        bytes = head.total_bytes(),
                        "MTP head loaded — spec-decode ENABLED"
                    );
                    Some(head)
                }
                _ => {
                    info!("FLAMBEAU_SPEC_MTP not set — spec-decode disabled");
                    None
                }
            };
            (cluster, LoadedModel::Pp { model: m, mtp })
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
            let ar = BarP2pAllReduce::new(Arc::clone(&cluster))
                .context("BarP2pAllReduce::new (requires fully-connected peer-access matrix)")?;
            (cluster, LoadedModel::Tp { model: m, ar })
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
            // 2. Per-stage AllReduce, each on its own sub-cluster.
            let mut stage_ars: Vec<BarP2pAllReduce> = Vec::with_capacity(hybrid.stages.len());
            for stage in &hybrid.stages {
                let ar = BarP2pAllReduce::new(Arc::clone(&stage.sub_cluster))
                    .with_context(|| {
                        format!(
                            "BarP2pAllReduce::new for stage {} (devices need fully-\
                             connected BAR1 peer access)",
                            stage.stage_idx
                        )
                    })?;
                stage_ars.push(ar);
            }
            // 3. Global cluster LAST — used only for inter-stage
            //    `peer_copy_via_host` hand-off. Constructing it before
            //    the per-stage sub-clusters/ARs disables BAR1 on the
            //    sub-cluster off-diagonal (project_hybrid_cluster_order).
            let cluster: Arc<HipCluster> = Arc::new(
                HipCluster::new(&cfg.device_ids)
                    .context("HipCluster::new (global, for inter-stage hand-off)")?,
            );
            (
                cluster,
                LoadedModel::Hybrid {
                    model: hybrid,
                    stage_ars,
                },
            )
        }
    };

    // M2.1: enumerate tools on each `--mcp` URL in parallel. A failed
    // URL logs a warning and contributes zero tools; the server still
    // boots. Empty input → empty output, no network at all.
    let remote_tools = crate::mcp_client::enumerate(&cfg.mcp_urls)
        .await
        .context("enumerate mcp tools")?;
    if !cfg.mcp_urls.is_empty() {
        info!(
            mcp_urls = cfg.mcp_urls.len(),
            remote_tools = remote_tools.len(),
            "mcp upstream(s) registered"
        );
        // M2.3: context-budget warning. If the tools definitions alone
        // eat > ~10% of the context window, the operator is probably
        // configuring too many upstream MCP servers (r/LocalLLaMA
        // norm is 3–5 max). Don't error — the user knows their
        // environment.
        if let Some(tools_tokens) = crate::mcp_client::estimate_tools_token_cost(
            &chat_template,
            &tokenizer,
            &remote_tools,
            &[],
        ) {
            let ctx = model_cfg.context_length;
            let budget = ctx / 10;
            if tools_tokens > budget {
                // Surface the biggest offenders — most common cause
                // of bloat is one remote tool with a giant nested
                // parameters schema.
                let mut by_size: Vec<(&str, usize)> = remote_tools
                    .iter()
                    .map(|t| {
                        let size = serde_json::to_string(&crate::mcp_client::to_tool_json(t))
                            .map(|s| s.len())
                            .unwrap_or(0);
                        (t.name.as_str(), size)
                    })
                    .collect();
                by_size.sort_by(|a, b| b.1.cmp(&a.1));
                let largest: Vec<String> = by_size
                    .iter()
                    .take(3)
                    .map(|(n, sz)| format!("{n} (~{sz} bytes)"))
                    .collect();
                tracing::warn!(
                    target: "flambeau.server",
                    tools_tokens,
                    context_tokens = ctx,
                    budget_tokens = budget,
                    largest = %largest.join(", "),
                    "tools[] definitions consume > 10% of context — consider fewer --mcp upstreams"
                );
            } else {
                info!(
                    tools_tokens,
                    context_tokens = ctx,
                    "tools[] render cost within context budget"
                );
            }
        }
    }

    let model_defaults = crate::state::ModelDefaults::from_gguf(&gguf);
    info!(
        temperature = ?model_defaults.temperature,
        top_p = ?model_defaults.top_p,
        top_k = ?model_defaults.top_k,
        min_p = ?model_defaults.min_p,
        "model sampling defaults from GGUF"
    );

    // P0.5 — boot-time default system prompt. Empty string treated as
    // unset so an operator can clear a system-level config by exporting
    // `FLAMBEAU_DEFAULT_SYSTEM=`.
    let default_system = std::env::var("FLAMBEAU_DEFAULT_SYSTEM")
        .ok()
        .filter(|s| !s.is_empty());
    if let Some(s) = default_system.as_deref() {
        info!(
            len = s.len(),
            "default system prompt loaded from FLAMBEAU_DEFAULT_SYSTEM"
        );
    }

    // **P2.9b-i1 (multi-slot pool)** — pre-allocate N inflight slots
    // sized to FLAMBEAU_PREFILL_UBATCH (default 512). Each slot owns
    // its own session (KV cache, GDN state) and scratch buffers; a
    // request acquires any free slot via try-lock round-robin and
    // returns it to the pool on response. N defaults to 1 (P2.9a
    // behaviour); N>1 enables request-level concurrency. Decode
    // kernels still serialise on the GPU stream — true batched
    // throughput lands in P2.9b-i2.
    let prefill_ubatch: usize = std::env::var("FLAMBEAU_PREFILL_UBATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n: &usize| *n >= 128)
        .unwrap_or(512);
    let inflight_slots: usize = std::env::var("FLAMBEAU_INFLIGHT_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n: &usize| *n >= 1 && *n <= 32)
        .unwrap_or(1);
    info!(
        prefill_ubatch,
        inflight_slots, "pre-allocating inflight slot pool"
    );
    let mut inflight_pool: Vec<Mutex<crate::model::Inflight>> =
        Vec::with_capacity(inflight_slots);
    for slot_idx in 0..inflight_slots {
        let slot = crate::model::Inflight::new(&model, &cluster, prefill_ubatch)
            .with_context(|| format!("pre-alloc Inflight slot {slot_idx} at boot"))?;
        inflight_pool.push(Mutex::new(slot));
    }

    // **P2.9b-i2-B (scheduler)** — request-lifetime claim flags +
    // empty pending queue + leader gate. These are used by the
    // scheduler-aware decode path (`FLAMBEAU_BATCHED_DECODE=1`)
    // to aggregate concurrent decode requests into batched dispatches.
    let slot_in_use: Vec<std::sync::atomic::AtomicBool> = (0..inflight_slots)
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();

    let state: SharedState = Arc::new(ServerState {
        model_id: cfg.model_id.clone(),
        cfg: model_cfg,
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
        remote_tools,
        agent_stats: crate::agent_stats::AgentStatsRing::default(),
        tool_call_format_default,
        model_defaults,
        default_system,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        // P1.6b — llama.cpp-compatible Fill-in-the-Middle. Both
        // top-level (`/infill`, llama.cpp + Continue) and
        // namespaced (`/v1/infill`) for clients that expect API-
        // versioned routes.
        .route("/infill", post(infill))
        .route("/v1/infill", post(infill))
        // P1.8a — Anthropic Messages API (text-only, non-streaming for
        // now). Tools (P1.8c) and SSE (P1.8b) layer in afterwards.
        .route("/v1/messages", post(messages_anthropic))
        // M2.3 — read-only agent-loop telemetry snapshot.
        .route("/v1/agent/stats", get(agent_stats))
        // C4.1 — read-only introspection of `--mcp`-registered remote
        // tools. Used by the chat UI's tools panel (C4.2).
        .route("/v1/tools", get(tools_endpoint))
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
