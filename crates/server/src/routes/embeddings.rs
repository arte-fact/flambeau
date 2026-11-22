//! POST /v1/embeddings handler.
//!
//! Tokenises each input with the embedding model's OWN tokenizer (Qwen3-
//! Embedding has a different vocab size than Qwen3.5-9B's chat tokenizer),
//! then runs the pooled forward and returns L2-normalised F32 vectors.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::api::EmbeddingsRequest;
use crate::routes::{queue_full_response, SharedState};

pub async fn embeddings(
    State(state): State<SharedState>,
    Json(_req): Json<EmbeddingsRequest>,
) -> Response {
    let Some(_admission) = state.try_admit() else {
        return queue_full_response();
    };
    // The embedding subsystem is disabled (legacy `EmbeddingModel` was removed;
    // `embedding_model` is always `None`). Until a backend-generic embedding
    // forward lands, the endpoint reports unavailable on every backend.
    let _ = &state.embedding_model;
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": "embedding model not configured (server started without --embedding-model)",
                "type": "invalid_request_error",
                "code": "embedding_model_not_loaded"
            }
        })),
    )
        .into_response()
}
