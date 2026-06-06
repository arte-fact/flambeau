//! POST /infill (FIM code-completion) handler.
//!
//! Builds the FIM prompt from prefix/suffix (+ optional middle prime,
//! + optional repo-context blocks) and runs the standard sampling
//!   engine with `relax_stop_mask=true` (FIM completions sit outside the
//!   chat template, so the engine's chat-flavoured stop biases must be
//!   disabled).

use anyhow::Result;
use axum::extract::State;
use axum::Json;

use crate::api::{CompletionChoice, CompletionResponse, InfillExtra, InfillRequest, Usage};
use crate::routes::{now_unix, request_id, run_completion_ids, ApiError, SharedState};
use crate::state::{parse_stop, SamplingParams};

pub async fn infill(
    State(state): State<SharedState>,
    Json(req): Json<InfillRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    let _admission = state.try_admit().ok_or_else(ApiError::queue_full)?;
    if req.stream {
        return Err(ApiError::bad_request(
            "SSE streaming for /infill is not yet implemented. Retry with stream=false.",
        ));
    }
    let Some(fim) = state.tokenizer.fim else {
        return Err(ApiError::bad_request(
            "this model does not carry FIM tokens — /infill is only supported for code-completion models",
        ));
    };

    let prompt_ids = build_fim_prompt_ids(
        &state.tokenizer,
        fim,
        &req.input_prefix,
        &req.input_suffix,
        req.prompt.as_deref(),
        &req.input_extra,
    )
    .map_err(ApiError::internal)?;

    let stop_strings = parse_stop(req.stop.as_ref());
    let params = SamplingParams::from_parts(
        crate::state::SamplingKnobs {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            ..Default::default()
        },
        crate::state::GenerationLimits {
            max_tokens: req.n_predict,
            seed: req.seed,
            stop_strings,
        },
        crate::state::ResponseMode::default(),
        &state.model_defaults,
    );

    let prompt_tokens = prompt_ids.len() as u32;
    let (text, _, completion_tokens, finish, _, _) =
        run_completion_ids(state.clone(), prompt_ids, params, true)
            .await
            .map_err(ApiError::internal)?;

    Ok(Json(CompletionResponse {
        id: request_id("infill"),
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

pub(super) fn build_fim_prompt_ids(
    tok: &flambeau_quant::GgufTokenizer,
    fim: flambeau_quant::FimTokens,
    prefix: &str,
    suffix: &str,
    middle: Option<&str>,
    extra: &[InfillExtra],
) -> Result<Vec<u32>> {
    let mut prompt_ids: Vec<u32> = Vec::new();
    if !extra.is_empty() {
        match (fim.repo_name, fim.file_sep) {
            (Some(repo_tok), Some(sep_tok)) => {
                for f in extra {
                    prompt_ids.push(repo_tok);
                    prompt_ids.extend(tok.encode(&f.filename)?);
                    prompt_ids.push(sep_tok);
                    prompt_ids.extend(tok.encode(&f.text)?);
                }
            }
            _ => {
                tracing::warn!(
                    target: "server.fim",
                    "input_extra ignored: model lacks <|repo_name|> / <|file_sep|>"
                );
            }
        }
    }
    prompt_ids.push(fim.prefix);
    prompt_ids.extend(tok.encode(prefix)?);
    prompt_ids.push(fim.suffix);
    prompt_ids.extend(tok.encode(suffix)?);
    prompt_ids.push(fim.middle);
    if let Some(mid) = middle.filter(|s| !s.is_empty()) {
        prompt_ids.extend(tok.encode(mid)?);
    }
    Ok(prompt_ids)
}
