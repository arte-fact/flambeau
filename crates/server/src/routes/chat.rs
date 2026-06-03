//! POST /v1/chat/completions handler.
//!
//! Renders the request's chat-template prompt (with optional system
//! prefill + tool definitions + assistant-prefill continuation), runs
//! the engine via `run_completion` (or `stream_completion_sse` for SSE),
//! parses the response through the tool-call dispatcher, and emits the
//! OpenAI-shaped `ChatCompletionResponse` with `tool_calls` populated
//! when the model produced any.

use std::time::Instant;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatLogProbs, ChatMessage,
    ResponseFormat, Usage,
};
use crate::routes::{
    dev_flag, now_unix, queue_full_response, request_id, run_completion, stream_completion_sse,
    strip_trailing_assistant_terminator, ApiError, SharedState,
};
use crate::state::{parse_stop, SamplingParams};

#[tracing::instrument(name = "server.chat_completions", skip_all)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(raw): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let Some(_admission) = state.try_admit() else {
        return Ok(queue_full_response());
    };
    if dev_flag("FLAMBEAU_DUMP_RAW_REQ") {
        let s = serde_json::to_string(&raw).unwrap_or_else(|_| "<serialise fail>".into());
        tracing::info!(
            target: "server.req.raw",
            body_bytes = s.len(),
            body = %s,
            "raw chat completions body"
        );
    }
    if let Some(obj) = raw.as_object() {
        let mut keys: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        keys.sort();
        let has_tools = obj.get("tools").is_some();
        let n_tools = obj
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        tracing::info!(
            target: "server.req.shape",
            n_tools_field = n_tools,
            has_tools_field = has_tools,
            top_level_keys = ?keys,
            "raw chat completions body shape"
        );
    }
    let req: ChatCompletionRequest = serde_json::from_value(raw)
        .map_err(|e| ApiError::bad_request(format!("invalid request body: {e}")))?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages[] is empty"));
    }

    let last_user_preview: String = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.clone())
        .map(|s| {
            let mut t = s.replace('\n', "\\n");
            if t.chars().count() > 200 {
                t = t.chars().take(200).collect::<String>() + "…";
            }
            t
        })
        .unwrap_or_else(|| "<no user msg>".to_string());
    tracing::info!(
        target: "server.req",
        n_messages = req.messages.len(),
        stream = req.stream,
        temperature = ?req.temperature,
        top_p = ?req.top_p,
        top_k = ?req.top_k,
        repetition_penalty = ?req.repetition_penalty,
        max_tokens = ?req.max_tokens,
        n_tools = req.tools.as_deref().map(|t| t.len()).unwrap_or(0),
        last_user = %last_user_preview,
        "chat_completions request"
    );

    let has_real_system = req
        .messages
        .iter()
        .any(|m| m.role == "system" && !m.content_str().trim().is_empty());
    let injected_system: Option<ChatMessage> = if !has_real_system {
        state.default_system.as_deref().map(|sys| ChatMessage {
            role: "system".into(),
            content: Some(sys.to_owned()),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        })
    } else {
        None
    };

    let messages: Vec<ChatMessage> = injected_system
        .into_iter()
        .chain(req.messages.iter().map(normalise_message))
        .collect();

    let merged_tools = req.tools.as_deref().map(|ts| {
        ts.iter()
            .map(serde_json::to_value)
            .filter_map(Result::ok)
            .collect::<Vec<_>>()
    });

    let json_mode = matches!(
        req.response_format.as_ref(),
        Some(ResponseFormat::JsonObject) | Some(ResponseFormat::JsonSchema { .. }),
    );
    let stop_strings = parse_stop(req.stop.as_ref());
    let collect_logprobs: Option<u32> =
        if req.logprobs.unwrap_or(false) || req.top_logprobs.is_some() {
            Some(req.top_logprobs.unwrap_or(0))
        } else {
            None
        };
    let assistant_prefill: Option<String> = if json_mode {
        req.messages
            .last()
            .filter(|m| m.role == "assistant")
            .and_then(|m| {
                let s = m.content_str();
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            })
    } else {
        None
    };
    let json_prime_bytes: Vec<u8> = assistant_prefill
        .as_deref()
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_default();
    let params = SamplingParams::from_parts(
        crate::state::SamplingKnobs {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repetition_penalty: req.repetition_penalty,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
        },
        crate::state::GenerationLimits {
            max_tokens: req.max_tokens,
            seed: req.seed,
            stop_strings,
        },
        crate::state::ResponseMode {
            json_mode,
            collect_logprobs,
            enable_thinking: req.enable_thinking.unwrap_or(false),
            json_prime_bytes,
        },
        &state.model_defaults,
    );
    let assistant_prefill_active = assistant_prefill.is_some();

    let parallel_tool_calls = req.parallel_tool_calls.unwrap_or(true);
    let has_tools = merged_tools.is_some();
    let relax_stop_mask = has_tools;

    if req.stream {
        let add_generation_prompt = !assistant_prefill_active;
        let mut prompt = state
            .chat_template
            .render_with_tools(
                &messages,
                merged_tools.as_deref(),
                add_generation_prompt,
                Some(params.enable_thinking),
            )
            .map_err(ApiError::internal)?;
        if assistant_prefill_active {
            prompt = strip_trailing_assistant_terminator(&prompt);
        }
        if dev_flag("FLAMBEAU_DUMP_PROMPT") {
            eprintln!(
                "--- rendered prompt ({} bytes) ---\n{}\n--- end prompt ---",
                prompt.len(),
                prompt
            );
        }
        let include_usage = req
            .stream_options
            .as_ref()
            .and_then(|s| s.include_usage)
            .unwrap_or(false);
        return Ok(stream_completion_sse(
            state,
            prompt,
            params,
            req.tool_call_format.clone(),
            parallel_tool_calls,
            relax_stop_mask,
            include_usage,
        )
        .into_response());
    }

    let session_id = request_id("chatcmpl");
    let iter_start = Instant::now();
    let mut prompt = state
        .chat_template
        .render_with_tools(
            &messages,
            merged_tools.as_deref(),
            !assistant_prefill_active,
            Some(params.enable_thinking),
        )
        .map_err(ApiError::internal)?;
    if assistant_prefill_active {
        prompt = strip_trailing_assistant_terminator(&prompt);
    }
    if dev_flag("FLAMBEAU_DUMP_PROMPT") {
        eprintln!(
            "--- rendered prompt ({} bytes) ---\n{}\n--- end prompt ---",
            prompt.len(),
            prompt
        );
    }

    let (
        text,
        sum_prompt_tokens,
        sum_completion_tokens,
        iter_finish,
        final_logprobs,
        final_reasoning_content,
    ) = run_completion(state.clone(), &prompt, params.clone(), relax_stop_mask)
        .await
        .map_err(ApiError::internal)?;

    let (final_content, mut final_tool_calls) = {
        use crate::tool_call_parser::{dispatcher, split_events, ParserEvent};
        if dev_flag("FLAMBEAU_DEBUG_TOOL_RAW") {
            tracing::info!(
                target: "server.tool_raw",
                bytes = text.len(),
                text_dbg = ?text,
                "raw model text before tool-call parser"
            );
        }
        let mut parser = dispatcher(
            req.tool_call_format.as_deref(),
            state.tool_call_format_default,
        )
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let mut events = parser.push(&text);
        events.extend(parser.finish());
        split_events(ParserEvent::coalesce(events))
    };
    if !parallel_tool_calls && final_tool_calls.len() > 1 {
        final_tool_calls.truncate(1);
    }

    let iter_tool_names: Vec<String> = final_tool_calls
        .iter()
        .map(|tc| tc.function.name.clone())
        .collect();
    let final_finish = if final_tool_calls.is_empty() {
        iter_finish
    } else {
        "tool_calls".into()
    };
    state.agent_stats.push(crate::agent_stats::IterStat {
        session_id: session_id.clone(),
        iteration: 0,
        tool_names: iter_tool_names,
        remote_count: 0,
        latency_ms: iter_start.elapsed().as_millis() as u64,
        prompt_tokens: sum_prompt_tokens,
        completion_tokens: sum_completion_tokens,
        finish_reason: final_finish.clone(),
    });

    Ok(Json(ChatCompletionResponse {
        id: session_id,
        object: "chat.completion",
        created: now_unix(),
        model: state.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: if final_content.is_empty() && !final_tool_calls.is_empty() {
                    None
                } else {
                    Some(final_content)
                },
                tool_call_id: None,
                tool_calls: if final_tool_calls.is_empty() {
                    None
                } else {
                    Some(final_tool_calls)
                },
                reasoning_content: final_reasoning_content,
            },
            finish_reason: final_finish,
            logprobs: final_logprobs.map(|content| ChatLogProbs { content }),
        }],
        usage: Usage {
            prompt_tokens: sum_prompt_tokens,
            completion_tokens: sum_completion_tokens,
            total_tokens: sum_prompt_tokens.saturating_add(sum_completion_tokens),
        },
    })
    .into_response())
}

/// Wraps non-thinking assistant content in `<think>\n\n</think>\n\n…` so
/// Qwen3.6's chat template doesn't push the prior turn out-of-distribution
/// when re-rendering history. Skipped for messages that already carry a
/// `</think>` marker.
pub(super) fn normalise_message(m: &ChatMessage) -> ChatMessage {
    let body = m.content_str();
    if m.role == "assistant" && !body.contains("</think>") {
        ChatMessage {
            role: m.role.clone(),
            content: Some(format!("<think>\n\n</think>\n\n{}", body)),
            tool_call_id: m.tool_call_id.clone(),
            tool_calls: m.tool_calls.clone(),
            reasoning_content: m.reasoning_content.clone(),
        }
    } else {
        m.clone()
    }
}
