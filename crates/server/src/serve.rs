//! `flambeau serve` entry point — loads the model + tokenizer, starts axum.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::routing::{get, post};
use axum::Router;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_quant::{ChatTemplate, GgufFile};
use flambeau_qwen3_moe::{Qwen3MoEConfig, Qwen3MoEShardedModel};
use flambeau_runtime::LayerAssignment;
use tokio::sync::Mutex;
use tracing::info;

use crate::routes::{chat_completions, completions, health, models, ServerState, SharedState};

/// Runtime config for `flambeau serve`.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub gguf_path: PathBuf,
    pub device_ids: Vec<i32>,
    pub bind_addr: SocketAddr,
    pub model_id: String,
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

    // Sanity: device_ids must be valid.
    let n_available = device_count().unwrap_or(0);
    for d in &cfg.device_ids {
        if *d < 0 || *d >= n_available {
            bail!(
                "device {d} not available (have {n_available} HIP devices)"
            );
        }
    }

    let model_cfg = Qwen3MoEConfig::from_gguf(&gguf).context("model config from GGUF")?;
    let cluster =
        HipCluster::new(&cfg.device_ids).context("HipCluster::new")?;
    let assignment =
        LayerAssignment::contiguous(model_cfg.num_layers, cluster.ranks() as u32);

    info!(
        num_layers = model_cfg.num_layers,
        ranks = cluster.ranks(),
        "loading model weights"
    );
    let model = Qwen3MoEShardedModel::load(&gguf, &cluster, &assignment)
        .context("Qwen3MoEShardedModel::load")?;

    let state: SharedState = Arc::new(ServerState {
        model_id: cfg.model_id.clone(),
        cfg: model_cfg,
        model,
        cluster,
        tokenizer,
        chat_template,
        inflight: Mutex::new(()),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
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
