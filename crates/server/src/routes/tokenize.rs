//! /tokenize + /detokenize handlers (llama.cpp-compat).
//!
//! No GPU work — both run on the tokio worker.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::api::{DetokenizeRequest, TokenizeRequest};
use crate::routes::SharedState;

/// Returns `{"tokens": [int, ...]}` for `content`. When
/// `add_special=true`, BOS/EOS are inserted by the underlying tokenizer
/// (the default `false` matches the chat path, which delegates specials
/// to the chat template). When `with_pieces=true`, each token is emitted
/// as `{"id", "piece"}` so a UI can render the surface form.
pub async fn tokenize(
    State(state): State<SharedState>,
    Json(req): Json<TokenizeRequest>,
) -> Response {
    let tokens = match state
        .tokenizer
        .inner
        .encode(req.content.as_str(), req.add_special)
    {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("tokenize failed: {e}") })),
            )
                .into_response();
        }
    };

    if req.with_pieces {
        let pieces: Vec<serde_json::Value> = tokens
            .iter()
            .map(|&id| {
                let piece = state.tokenizer.decode(&[id]).unwrap_or_default();
                json!({ "id": id, "piece": piece })
            })
            .collect();
        Json(json!({ "tokens": pieces })).into_response()
    } else {
        Json(json!({ "tokens": tokens })).into_response()
    }
}

/// Returns `{"content": "..."}` for `tokens`. Special tokens are NOT
/// skipped — clients sending a stop-id back through detokenize see its
/// surface form.
pub async fn detokenize(
    State(state): State<SharedState>,
    Json(req): Json<DetokenizeRequest>,
) -> Response {
    match state.tokenizer.decode(&req.tokens) {
        Ok(content) => Json(json!({ "content": content })).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("detokenize failed: {e}") })),
        )
            .into_response(),
    }
}
