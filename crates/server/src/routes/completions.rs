//! POST /v1/completions — legacy OpenAI text-completion endpoint.
//!
//! When `suffix` is set and the model carries FIM tokens, routes through
//! the shared FIM prompt assembly. Otherwise the prompt feeds the engine
//! verbatim with the chat-flavoured stop-mask policy enabled.

use axum::extract::State;
use axum::Json;

use crate::api::{CompletionChoice, CompletionRequest, CompletionResponse, Usage};
use crate::routes::{
    now_unix, request_id, run_completion, run_completion_ids, ApiError, SharedState,
};
use crate::state::{parse_stop, SamplingParams};

#[tracing::instrument(
    name = "server.completions",
    skip_all,
    fields(stream = req.stream)
)]
pub async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let _admission = state.try_admit().ok_or_else(ApiError::queue_full)?;
    if req.stream {
        return Err(ApiError::bad_request(
            "SSE streaming is not yet implemented. Retry with stream=false.",
        ));
    }
    let stop_strings = parse_stop(req.stop.as_ref());
    let params = SamplingParams::from_parts(
        crate::state::SamplingKnobs {
            temperature: req.temperature,
            top_p: req.top_p,
            ..Default::default()
        },
        crate::state::GenerationLimits {
            max_tokens: req.max_tokens,
            seed: req.seed,
            stop_strings,
        },
        crate::state::ResponseMode::default(),
        &state.model_defaults,
    );

    let suffix_fim = req.suffix.as_deref().filter(|s| !s.is_empty());
    let (text, prompt_tokens, completion_tokens, finish) = if let (Some(suffix), Some(fim)) =
        (suffix_fim, state.tokenizer.fim)
    {
        let prompt_ids = super::infill::build_fim_prompt_ids(
            &state.tokenizer,
            fim,
            &req.prompt,
            suffix,
            None,
            &[],
        )
        .map_err(ApiError::internal)?;
        let prompt_tokens = prompt_ids.len() as u32;
        let (text, _, completion_tokens, finish, _, _, _) =
            run_completion_ids(state.clone(), prompt_ids, params, true)
                .await
                .map_err(ApiError::internal)?;
        (text, prompt_tokens, completion_tokens, finish)
    } else {
        if suffix_fim.is_some() {
            tracing::warn!(
                target: "server.completions",
                "`suffix` provided but model carries no FIM tokens — falling back to non-FIM completion"
            );
        }
        let (t, p, c, f, _, _, _) = run_completion(state.clone(), &req.prompt, params, false)
            .await
            .map_err(ApiError::internal)?;
        (t, p, c, f)
    };

    Ok(Json(CompletionResponse {
        id: request_id("cmpl"),
        object: "text_completion",
        created: now_unix(),
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: finish,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        },
    }))
}
