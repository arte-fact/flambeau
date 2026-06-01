//! Read-only endpoints: /health, /v1/agent/stats, /v1/models.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

use crate::api::{ModelObject, ModelsListResponse};
use crate::routes::SharedState;

/// GET /health — constant, no locks.
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// GET /v1/agent/stats — read-only agent-loop telemetry snapshot.
/// Returns the last N per-iteration stats (N = ring capacity).
pub async fn agent_stats(State(state): State<SharedState>) -> impl IntoResponse {
    let snap = state.agent_stats.snapshot();
    Json(json!({
        "iterations": snap,
        "count": snap.len(),
    }))
}

/// GET /v1/models — chat model + capabilities + tool-call format.
pub async fn models(State(state): State<SharedState>) -> impl IntoResponse {
    let mut capabilities: Vec<&'static str> = vec!["chat", "completion"];
    if state.tokenizer.fim.is_some() {
        capabilities.push("infill");
    }
    if state.embedding_model.is_some() {
        capabilities.push("embeddings");
    }
    capabilities.push("tools");
    if state.supports_thinking {
        capabilities.push("thinking");
    }

    let tool_call_format = match state.tool_call_format_default {
        crate::tool_call_parser::ToolCallFormat::Hermes => "hermes",
        crate::tool_call_parser::ToolCallFormat::QwenCoder => "qwen_coder",
        crate::tool_call_parser::ToolCallFormat::Gemma4 => "gemma4",
    };

    Json(ModelsListResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model",
            created: 0,
            owned_by: "flambeau",
            context_length: state.cfg.context_length as u32,
            max_output_tokens: 8192,
            architecture: state.cfg.arch.clone(),
            quantization: state.quantization.clone(),
            capabilities,
            tool_call_format,
        }],
    })
}
