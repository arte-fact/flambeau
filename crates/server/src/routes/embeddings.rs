//! POST /v1/embeddings handler.
//!
//! Tokenises each input with the embedding model's OWN tokenizer (Qwen3-
//! Embedding has a different vocab size than Qwen3.5-9B's chat tokenizer),
//! then runs the pooled forward and returns L2-normalised F32 vectors.

use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use flambeau_core::Device;
use serde_json::json;

use crate::api::{
    EmbeddingData, EmbeddingsInput, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage,
};
use crate::routes::{queue_full_response, SharedState};

pub async fn embeddings(
    State(state): State<SharedState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Response {
    let Some(_admission) = state.try_admit() else {
        return queue_full_response();
    };
    let Some(em_arc) = state.embedding_model.as_ref().cloned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "message": "embedding model not configured (server started without --embedding-model)",
                    "type": "invalid_request_error",
                    "code": "embedding_model_not_loaded"
                }
            })),
        )
            .into_response();
    };
    let Some(rank) = state.embedding_rank else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "embedding model present but rank unset",
                    "type": "internal_error"
                }
            })),
        )
            .into_response();
    };

    let inputs: Vec<String> = match req.input {
        EmbeddingsInput::Single(s) => vec![s],
        EmbeddingsInput::Batch(xs) => xs,
    };
    if inputs.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "input must be a non-empty string or array of strings",
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    let req_start = Instant::now();
    let model_id = req.model.clone().unwrap_or_else(|| state.model_id.clone());

    let tokenizer = state
        .embedding_tokenizer
        .as_ref()
        .cloned()
        .expect("embedding_model present without embedding_tokenizer");
    let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens: u32 = 0;
    for (i, s) in inputs.iter().enumerate() {
        let toks = match tokenizer.encode(s) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": format!("tokenize input[{i}]: {e}"),
                            "type": "invalid_request_error"
                        }
                    })),
                )
                    .into_response();
            }
        };
        if toks.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("input[{i}] tokenised to 0 tokens"),
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
        total_prompt_tokens = total_prompt_tokens.saturating_add(toks.len() as u32);
        all_tokens.push(toks);
    }

    let cluster = state.cluster.clone();
    let result: Result<Vec<Vec<f32>>> = tokio::task::spawn_blocking(move || {
        let mut em = em_arc.blocking_lock();
        let device = cluster.device(rank);
        let stream = device.default_stream();
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(all_tokens.len());
        for toks in &all_tokens {
            let max = em.max_tokens();
            let slice: &[u32] = if toks.len() > max {
                tracing::warn!(
                    target: "server.embeddings",
                    L = toks.len(),
                    max,
                    "input truncated to embedding max_tokens"
                );
                &toks[..max]
            } else {
                &toks[..]
            };
            let v = em
                .compute_pooled_embedding(device, stream, slice)
                .context("compute_pooled_embedding")?;
            out.push(v);
        }
        Ok(out)
    })
    .await
    .unwrap_or_else(|join_err| Err(anyhow!("embeddings join error: {join_err}")));

    let vectors = match result {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                target: "server.embeddings",
                error = %e,
                "embeddings forward failed"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": format!("embedding forward failed: {e}"),
                        "type": "internal_error"
                    }
                })),
            )
                .into_response();
        }
    };

    let data: Vec<EmbeddingData> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| EmbeddingData {
            object: "embedding",
            embedding: v,
            index: i as u32,
        })
        .collect();
    let resp = EmbeddingsResponse {
        object: "list",
        data,
        model: model_id,
        usage: EmbeddingsUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    };
    tracing::info!(
        target: "server.embeddings",
        n_inputs = resp.data.len(),
        prompt_tokens = total_prompt_tokens,
        elapsed_ms = req_start.elapsed().as_secs_f64() * 1000.0,
        "embeddings response"
    );
    Json(resp).into_response()
}
