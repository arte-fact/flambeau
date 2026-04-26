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
    Qwen35DenseTpLayout, Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoETpModel,
};
use flambeau_runtime::LayerAssignment;
use tokio::sync::Mutex;
use tracing::info;

use crate::model::LoadedModel;
use crate::routes::{
    agent_stats, chat_completions, completions, health, index, models, tools_endpoint,
    ServerState, SharedState,
};

/// **TP-5a** — mesh topology selector. PP-V1 default; TP engages the
/// Qwen3MoETpModel loader + the BarP2pAllReduce-based forward path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshMode {
    /// Pipeline parallelism — V1 default. `LayerAssignment` distributes
    /// whole layers across ranks; one `peer_copy_via_host` per stage
    /// transition.
    Pp,
    /// Tensor parallelism — every rank holds every layer (sliced).
    /// `world` ranks; intra-layer Megatron splits + BAR1 P2P AllReduce.
    Tp { world: u32 },
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
    let cluster: Arc<HipCluster> =
        Arc::new(HipCluster::new(&cfg.device_ids).context("HipCluster::new")?);

    // **TP-5a-i2** — branch on mesh topology. PP path uses the V1.8
    // sharded loader; TP path validates the layout, slices the GGUF,
    // and constructs a BarP2pAllReduce against the same cluster.
    let model: LoadedModel = match cfg.mesh_mode {
        MeshMode::Pp => {
            let assignment =
                LayerAssignment::contiguous(model_cfg.num_layers, cluster.ranks() as u32);
            info!(
                num_layers = model_cfg.num_layers,
                ranks = cluster.ranks(),
                topology = "pp",
                "loading model weights"
            );
            let m = Qwen3MoEShardedModel::load(&gguf, &cluster, &assignment)
                .context("Qwen3MoEShardedModel::load")?;
            LoadedModel::Pp(m)
        }
        MeshMode::Tp { world } => {
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
            LoadedModel::Tp { model: m, ar }
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

    let state: SharedState = Arc::new(ServerState {
        model_id: cfg.model_id.clone(),
        cfg: model_cfg,
        model,
        cluster,
        tokenizer,
        chat_template,
        inflight: Mutex::new(()),
        remote_tools,
        agent_stats: crate::agent_stats::AgentStatsRing::default(),
        tool_call_format_default,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
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
