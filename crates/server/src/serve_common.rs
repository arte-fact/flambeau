//! Shared boot helpers used by every serve_inner_* path. Two pieces:
//!
//! - `BootMetadata`: arch-neutral metadata derived from the GGUF + cfg
//!   once at startup (tokenizer, chat template, tool-call format,
//!   thinking flag, quantization label, sampling defaults, default
//!   system prompt). Computed before the arch dispatch.
//! - `build_server_state(...)` and `run_axum(state, bind_addr)`: the
//!   final two boot steps every path shares verbatim. Per-arch boot
//!   fills the arch-specific parts (model handle, cluster, inflight
//!   pool, qwen3_moe extras, embedding) and hands them in.

#![cfg(feature = "hip")]

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use flambeau_backend_hip::HipCluster;
use flambeau_quant::{ChatTemplate, GgufFile};
use tokio::sync::Mutex;
use tracing::info;

use crate::prefix_cache::{PrefixCache, TopologyTag};
use crate::routes::{
    agent_stats, chat_completions, completions, detokenize, embeddings, health, infill,
    messages_anthropic, models, tokenize, ServerState, SharedState,
};
use crate::serve::{MeshMode, ServeConfig};

/// Metadata pulled out of the GGUF once at boot — shared by every
/// arch-specific boot path. Construct via [`BootMetadata::from_gguf`].
pub struct BootMetadata {
    pub tokenizer: flambeau_quant::GgufTokenizer,
    pub chat_template: ChatTemplate,
    pub tool_call_format_default: crate::tool_call_parser::ToolCallFormat,
    pub supports_thinking: bool,
    pub quantization: Option<String>,
    pub model_defaults: crate::state::ModelDefaults,
    pub default_system: Option<String>,
}

impl BootMetadata {
    pub fn from_gguf(gguf: &GgufFile, cfg: &ServeConfig) -> Result<Self> {
        let tokenizer = flambeau_quant::load_from_gguf(gguf).context("load tokenizer")?;
        let chat_template = ChatTemplate::load_from_gguf(gguf).context("load chat template")?;

        let tpl_src = gguf.metadata_str("tokenizer.chat_template").unwrap_or("");
        let tool_call_format_default =
            crate::tool_call_parser::detect_format_from_template(tpl_src);
        info!(
            format = ?tool_call_format_default,
            "tool-call format detected from chat template"
        );
        let supports_thinking = tpl_src.contains("enable_thinking");

        let quantization: Option<String> = gguf
            .metadata_u32("general.file_type")
            .map(quantization_label);

        let model_defaults = crate::state::ModelDefaults::from_gguf(gguf);
        info!(
            temperature = ?model_defaults.temperature,
            top_p = ?model_defaults.top_p,
            top_k = ?model_defaults.top_k,
            min_p = ?model_defaults.min_p,
            "model sampling defaults from GGUF"
        );

        let default_system = cfg
            .default_system
            .as_ref()
            .filter(|s| !s.is_empty())
            .cloned();
        if let Some(s) = default_system.as_deref() {
            info!(len = s.len(), "default system prompt loaded");
        }

        Ok(Self {
            tokenizer,
            chat_template,
            tool_call_format_default,
            supports_thinking,
            quantization,
            model_defaults,
            default_system,
        })
    }
}

fn quantization_label(ft: u32) -> String {
    match ft {
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
    }
}

pub fn topology_tag_from_mesh(mesh: MeshMode, device_count: usize) -> TopologyTag {
    match mesh {
        MeshMode::Pp => TopologyTag {
            mesh_kind: "pp",
            ranks: device_count as u32,
            pp_size: device_count as u32,
            tp_size: 1,
        },
        MeshMode::Tp { world } => TopologyTag {
            mesh_kind: "tp",
            ranks: world,
            pp_size: 1,
            tp_size: world,
        },
        MeshMode::Hybrid { pp_size, tp_size } => TopologyTag {
            mesh_kind: "pp+tp",
            ranks: pp_size * tp_size,
            pp_size,
            tp_size,
        },
    }
}

/// Build the prefix cache. Always constructed; methods short-circuit
/// when `cfg.prefix_cache == false`. Boot-time logging here so every
/// path emits one consistent line.
pub fn build_prefix_cache(cfg: &ServeConfig, mesh_kind: &'static str) -> Arc<PrefixCache> {
    let prefix_cache = Arc::new(PrefixCache::new(
        PrefixCache::gb_to_bytes(cfg.prefix_cache_max_gb),
        cfg.prefix_cache,
    ));
    if prefix_cache.enabled() {
        info!(
            chunk_tokens = cfg.prefill_ubatch.max(128),
            budget_bytes = prefix_cache.vram_budget_bytes,
            mesh = mesh_kind,
            "prefix cache ENABLED (FLAMBEAU_PREFIX_CACHE=1)"
        );
    } else {
        info!("prefix cache disabled (set FLAMBEAU_PREFIX_CACHE=1 to enable)");
    }
    prefix_cache
}

pub type EmbeddingHandle = (
    Arc<tokio::sync::Mutex<Box<dyn crate::embedding::EmbeddingHandle>>>,
    Arc<flambeau_quant::GgufTokenizer>,
);

/// Per-path inputs assembled by the arch boot before state-building.
/// Layout mirrors `ServerState`'s arch-varying fields so `build_server_state`
/// is a flat assembly with no per-field branching.
pub struct ServerStateInputs {
    pub model_id: String,
    pub model_cfg: crate::model_cfg::ServerModelCfg,
    pub model: crate::model_handle::LoadedModel,
    pub cluster: Arc<HipCluster>,
    pub inflight_pool: Vec<Mutex<Box<dyn crate::Session>>>,
    pub embedding: Option<EmbeddingHandle>,
    pub embedding_rank: Option<usize>,
    pub gpu_sampler: bool,
    pub batched_decode: bool,
    pub max_queue_depth: usize,
    pub prefill_ubatch: usize,
    pub prefill_chunk_tokens: usize,
    pub topology_tag: TopologyTag,
    pub prefix_cache: Arc<PrefixCache>,
    pub boot: BootMetadata,
    pub decode_batch_window_us: u64,
}

pub fn build_server_state(inputs: ServerStateInputs) -> SharedState {
    let ServerStateInputs {
        model_id,
        model_cfg,
        model,
        cluster,
        inflight_pool,
        embedding,
        embedding_rank,
        gpu_sampler,
        batched_decode,
        max_queue_depth,
        prefill_ubatch,
        prefill_chunk_tokens,
        topology_tag,
        prefix_cache,
        boot,
        decode_batch_window_us,
    } = inputs;
    let slot_in_use: Vec<std::sync::atomic::AtomicBool> = (0..inflight_pool.len())
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();
    let (embedding_model, embedding_tokenizer) = match embedding {
        Some((m, t)) => (Some(m), Some(t)),
        None => (None, None),
    };
    Arc::new(ServerState {
        model_id,
        cfg: model_cfg,
        model,
        cluster,
        tokenizer: boot.tokenizer,
        chat_template: boot.chat_template,
        inflight_pool,
        slot_in_use,
        batched_pending: std::sync::Mutex::new(Vec::new()),
        batched_dispatcher: std::sync::Mutex::new(()),
        prefix_cache,
        prefix_cache_chunk_tokens: prefill_ubatch,
        topology_tag,
        embedding_model,
        embedding_tokenizer,
        embedding_rank,
        in_flight: std::sync::atomic::AtomicUsize::new(0),
        max_queue_depth,
        prefill_ubatch,
        prefill_chunk_tokens,
        gpu_sampler,
        batched_decode,
        decode_batch_window_us,
        agent_stats: crate::agent_stats::AgentStatsRing::default(),
        tool_call_format_default: boot.tool_call_format_default,
        supports_thinking: boot.supports_thinking,
        quantization: boot.quantization,
        model_defaults: boot.model_defaults,
        default_system: boot.default_system,
    })
}

pub async fn run_axum(
    state: SharedState,
    bind_addr: SocketAddr,
    boot_label: &'static str,
) -> Result<()> {
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

    info!(bind = %bind_addr, boot = boot_label, "serving");
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .context("bind listener")?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum::serve failed")?;
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = sigint.recv() => tracing::info!("SIGINT received — shutting down"),
        _ = sigterm.recv() => tracing::info!("SIGTERM received — shutting down"),
    }
}
