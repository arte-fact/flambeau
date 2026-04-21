//! HTTP route handlers. Requires `hip` feature (loads real model).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp, forward_prefill_pp, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};
use flambeau_quant::{ChatMessage as TmplMessage, ChatTemplate, GgufTokenizer};
use flambeau_runtime::{sample as sample_token, Rng};
use serde_json::json;
use tokio::sync::Mutex;

use crate::api::*;
use crate::state::SamplingParams;

/// Server-wide shared state — built once at startup.
pub struct ServerState {
    pub model_id: String,
    pub cfg: Qwen3MoEConfig,
    pub model: Qwen3MoEShardedModel,
    pub cluster: HipCluster,
    pub tokenizer: GgufTokenizer,
    pub chat_template: ChatTemplate,
    /// Serialises all forward traffic through the model. Continuous batching
    /// is V2 — for now one request at a time.
    pub inflight: Mutex<()>,
}

pub type SharedState = Arc<ServerState>;

/// GET /health — constant, no locks.
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// GET /v1/models — lists just the loaded model.
pub async fn models(State(state): State<SharedState>) -> impl IntoResponse {
    Json(ModelsListResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model",
            created: 0,
            owned_by: "flambeau",
        }],
    })
}

/// POST /v1/chat/completions — non-streaming only in V1.8.B.
pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    if req.stream {
        return Err(ApiError::bad_request(
            "SSE streaming is not yet implemented (V1.8.C). Retry with stream=false.",
        ));
    }
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages[] is empty"));
    }

    // Render chat template → text prompt.
    let tmpl_msgs: Vec<TmplMessage> = req
        .messages
        .iter()
        .map(|m| TmplMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let prompt = state
        .chat_template
        .render(&tmpl_msgs, /*add_generation_prompt=*/ true)
        .map_err(ApiError::internal)?;

    let params = SamplingParams::from_parts(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.seed,
    );

    let (text, prompt_tokens, completion_tokens, finish) =
        run_completion(state.clone(), &prompt, params)
            .await
            .map_err(ApiError::internal)?;

    Ok(Json(ChatCompletionResponse {
        id: request_id("chatcmpl"),
        object: "chat.completion",
        created: now_unix(),
        model: state.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: finish,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// POST /v1/completions — legacy text-completion endpoint.
pub async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    if req.stream {
        return Err(ApiError::bad_request(
            "SSE streaming is not yet implemented (V1.8.C). Retry with stream=false.",
        ));
    }
    let params = SamplingParams::from_parts(
        req.temperature,
        req.top_p,
        req.max_tokens,
        req.seed,
    );
    let (text, prompt_tokens, completion_tokens, finish) =
        run_completion(state.clone(), &req.prompt, params)
            .await
            .map_err(ApiError::internal)?;

    Ok(Json(CompletionResponse {
        id: request_id("cmpl"),
        object: "text_completion",
        created: now_unix(),
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: finish,
        },],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Shared engine: text prompt → generated text + token counts + finish reason.
///
/// Runs inside `spawn_blocking` because HIP kernels + mutex hold are sync.
async fn run_completion(
    state: SharedState,
    prompt: &str,
    params: SamplingParams,
) -> Result<(String, u32, u32, String)> {
    let prompt = prompt.to_owned();
    tokio::task::spawn_blocking(move || run_completion_blocking(state, prompt, params))
        .await
        .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

fn run_completion_blocking(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
) -> Result<(String, u32, u32, String)> {
    // Serialise: one forward at a time through this server instance.
    let _guard = state
        .inflight
        .blocking_lock();

    // Tokenize prompt.
    let prompt_ids = state.tokenizer.encode(&prompt).context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }
    let prompt_tokens = prompt_ids.len() as u32;

    let cluster = &state.cluster;
    let model = &state.model;

    // Fresh session per request (no conversation state reuse in V1).
    let mut session =
        Qwen3MoEShardedSession::new(model, cluster).context("create session")?;
    let mut prefill_scratch =
        ShardedForwardPrefillScratch::new(model, cluster, prompt_ids.len())
            .context("prefill scratch")?;
    let mut decode_scratch =
        ShardedForwardOneTokenScratch::new(model, cluster).context("decode scratch")?;

    // Prefill — consumes all prompt_ids, returns argmax of the last position.
    let first_next = forward_prefill_pp(
        model,
        &mut session,
        cluster,
        &mut prefill_scratch,
        &prompt_ids,
        /*start_position=*/ 0,
    )
    .context("prefill")?;

    // Decode loop.
    let _rng = Rng::from_seed(params.seed);
    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let stop_ids = &state.tokenizer.stop_ids;
    let is_stop = |t: u32| stop_ids.contains(&t);

    // The prefill's argmax is our sampler's LOGITS-style hint, but we don't have
    // the logit row here — forward_prefill_pp only returns argmax. So for non-
    // greedy sampling the first generated token ALWAYS uses the argmax path.
    // To sample properly we'd need a variant that returns logits; punt to V1.9.
    generated.push(first_next);
    if is_stop(first_next) {
        return finalise(&state, prompt_tokens, &generated, "stop");
    }

    let mut finish_reason = "length";
    let mut last_token = first_next;
    for step in 1..params.max_tokens as usize {
        let next = forward_one_token_pp(
            model,
            &mut session,
            cluster,
            &mut decode_scratch,
            last_token,
            /*position=*/ prompt_ids.len() + step,
        )
        .context("decode step")?;
        generated.push(next);
        last_token = next;
        if is_stop(next) {
            finish_reason = "stop";
            break;
        }
        // V1.9: apply SamplingParams to full logits. Greedy already matches
        // the forward's argmax; temperature / top-p is implemented in
        // flambeau_runtime::sample but needs the logit row accessible from
        // forward_one_token_pp, which doesn't currently return it.
        let _ = sample_token; // reference to keep import used
        let _ = params.sampling;
    }

    // Dispose per-request scratch/session; keep model + cluster alive.
    decode_scratch
        .dispose(cluster)
        .context("dispose decode scratch")?;
    prefill_scratch
        .dispose(cluster)
        .context("dispose prefill scratch")?;
    session.dispose(cluster).context("dispose session")?;

    finalise(&state, prompt_tokens, &generated, finish_reason)
}

fn finalise(
    state: &ServerState,
    prompt_tokens: u32,
    generated: &[u32],
    reason: &str,
) -> Result<(String, u32, u32, String)> {
    // Strip ALL stop tokens (eos, <|im_end|>, etc.) from decoded text so the
    // client sees clean content. Raw count preserved for `usage` honesty.
    let stop_ids = &state.tokenizer.stop_ids;
    let completion_tokens = generated.len() as u32;
    let visible: Vec<u32> = generated
        .iter()
        .copied()
        .filter(|t| !stop_ids.contains(t))
        .collect();
    let text = state.tokenizer.decode(&visible).context("decode")?;
    Ok((text, prompt_tokens, completion_tokens, reason.to_owned()))
}

fn request_id(prefix: &str) -> String {
    format!("{prefix}-{}", now_unix())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---- error plumbing --------------------------------------------------------

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    pub fn internal(err: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": "invalid_request_error",
                }
            })),
        )
            .into_response()
    }
}
