//! POST /v1/messages — Anthropic Messages API handler.
//!
//! Translates the Anthropic envelope (system + messages[] with content
//! blocks) into the OpenAI ChatMessage shape the engine already speaks,
//! runs the model, parses tool calls out of the response, and emits the
//! canonical Anthropic response (or SSE stream).

use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
    AnthropicMessagesResponse, AnthropicResponseBlock, AnthropicTool, AnthropicUsage, ChatMessage,
    FunctionCall, FunctionDef, ToolCall, ToolDef,
};
use crate::routes::{
    request_id, run_completion, run_completion_blocking_streaming, ApiError, SharedState,
};
use crate::state::SamplingParams;

/// Public `/v1/messages` entry point. Delegates to the inner handler and
/// re-renders any error in the Anthropic envelope (`{type:"error",…}`).
pub async fn messages_anthropic(
    state: State<SharedState>,
    req: Json<AnthropicMessagesRequest>,
) -> Response {
    match messages_anthropic_inner(state, req).await {
        Ok(resp) => resp,
        Err(e) => e.anthropic().into_response(),
    }
}

#[tracing::instrument(name = "server.messages", skip_all, fields(stream = req.stream))]
async fn messages_anthropic_inner(
    State(state): State<SharedState>,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages[] is empty"));
    }
    let _admission = state.try_admit().ok_or_else(ApiError::queue_full)?;
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(sys) = req.system.as_ref() {
        let body = sys.to_plain();
        if !body.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: Some(body),
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            });
        }
    }
    for m in &req.messages {
        translate_anthropic_message(m, &mut messages);
    }
    let messages: Vec<ChatMessage> = messages
        .iter()
        .map(|m| {
            if m.tool_calls.is_some() {
                m.clone()
            } else {
                super::chat::normalise_message(m)
            }
        })
        .collect();

    let messages = if !messages
        .iter()
        .any(|m| m.role == "system" && !m.content_str().trim().is_empty())
    {
        if let Some(sys) = state.default_system.as_deref() {
            let mut prepended = Vec::with_capacity(messages.len() + 1);
            prepended.push(ChatMessage {
                role: "system".into(),
                content: Some(sys.to_owned()),
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            });
            prepended.extend(messages);
            prepended
        } else {
            messages
        }
    } else {
        messages
    };

    let stop_strings = req.stop_sequences.clone().unwrap_or_default();
    let enable_thinking = req.thinking.as_ref().is_some_and(|t| t.is_enabled());
    let params = SamplingParams::from_parts(
        crate::state::SamplingKnobs {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            ..Default::default()
        },
        crate::state::GenerationLimits {
            max_tokens: Some(req.max_tokens),
            seed: None,
            stop_strings: stop_strings.clone(),
        },
        crate::state::ResponseMode {
            enable_thinking,
            reasoning_budget: req
                .thinking
                .as_ref()
                .filter(|t| t.is_enabled())
                .and_then(|t| t.budget_tokens),
            ..Default::default()
        },
        &state.model_defaults,
    );

    let tools_forbidden = req
        .tool_choice
        .as_ref()
        .is_some_and(|tc| tc.forbids_tools());
    let openai_tools: Option<Vec<ToolDef>> = if tools_forbidden {
        None
    } else {
        req.tools
            .as_deref()
            .map(anthropic_tools_to_openai)
            .filter(|v| !v.is_empty())
    };
    let merged_tools = openai_tools.as_ref().map(|ts| {
        ts.iter()
            .filter_map(|t| serde_json::to_value(t).ok())
            .collect::<Vec<_>>()
    });

    // B1 tool-choice forcing: when `tool_choice` requires a call, append the
    // arch's tool-call opening to the prompt so the model continues from a
    // guaranteed-valid call, and feed the same fragment to the parser ahead
    // of the model output so the reconstructed call parses.
    let tool_force_prefix: Option<String> = req
        .tool_choice
        .as_ref()
        .filter(|_| merged_tools.is_some())
        .and_then(|tc| tc.force_target())
        .map(|name| {
            crate::tool_call_parser::choose_format(None, state.tool_call_format_default)
                .map(|fmt| fmt.force_prefix(name))
                .unwrap_or_default()
        })
        .filter(|s| !s.is_empty());

    // A forced call goes straight to the tool, so suppress the thinking
    // prime — otherwise the prompt opens a `<think>` block immediately
    // followed by the forced tool-call fragment.
    let render_thinking = enable_thinking && tool_force_prefix.is_none();
    let mut prompt = state
        .chat_template
        .render_with_tools(&messages, merged_tools.as_deref(), true, Some(render_thinking))
        .map_err(ApiError::internal)?;
    if let Some(pfx) = &tool_force_prefix {
        prompt.push_str(pfx);
    }
    let has_tools = merged_tools.is_some();
    let relax_stop_mask = has_tools;

    if req.stream {
        let tcf = state.tool_call_format_default;
        return Ok(stream_messages_anthropic_sse(
            state,
            prompt,
            params,
            relax_stop_mask,
            tcf,
            tool_force_prefix,
        )
        .into_response());
    }

    let (text, prompt_tokens, completion_tokens, finish, _, reasoning, matched_stop) =
        run_completion(state.clone(), &prompt, params, relax_stop_mask)
            .await
            .map_err(ApiError::internal)?;

    let (text_out, tool_calls) = if has_tools {
        use crate::tool_call_parser::{dispatcher_with_prompt, split_events, ParserEvent};
        let mut parser =
            dispatcher_with_prompt(None, state.tool_call_format_default, &prompt, false)
                .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let mut events = Vec::new();
        if let Some(pfx) = &tool_force_prefix {
            events.extend(parser.push(pfx));
        }
        events.extend(parser.push(&text));
        events.extend(parser.finish());
        split_events(ParserEvent::coalesce(events))
    } else {
        (text, Vec::<ToolCall>::new())
    };

    let stop_reason: &str = if !tool_calls.is_empty() {
        "tool_use"
    } else if matched_stop.is_some() {
        "stop_sequence"
    } else {
        match finish.as_str() {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            other => other,
        }
    };
    // Anthropic reports the matched custom stop sequence alongside the
    // `stop_sequence` reason; null for a model-driven end-of-turn.
    let stop_sequence: Option<String> = if tool_calls.is_empty() {
        matched_stop
    } else {
        None
    };

    let mut content: Vec<AnthropicResponseBlock> = Vec::new();
    if let Some(reasoning) = reasoning.filter(|r| !r.is_empty()) {
        content.push(AnthropicResponseBlock::Thinking {
            thinking: reasoning,
            signature: String::new(),
        });
    }
    if !text_out.is_empty() {
        content.push(AnthropicResponseBlock::Text { text: text_out });
    }
    for tc in &tool_calls {
        let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
            .unwrap_or_else(|_| serde_json::json!({ "_raw": tc.function.arguments }));
        content.push(AnthropicResponseBlock::ToolUse {
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            input,
        });
    }
    if content.is_empty() {
        content.push(AnthropicResponseBlock::Text {
            text: String::new(),
        });
    }

    Ok(Json(AnthropicMessagesResponse {
        id: request_id("msg"),
        kind: "message",
        role: "assistant",
        content,
        model: state.model_id.clone(),
        stop_reason: stop_reason.to_owned(),
        stop_sequence,
        usage: AnthropicUsage {
            input_tokens: prompt_tokens,
            output_tokens: completion_tokens,
        },
    })
    .into_response())
}

/// Translate one Anthropic message into 0+ OpenAI `ChatMessage`s,
/// appending to `out`. Tool-result blocks emit independently as
/// `role="tool"` messages; image blocks are dropped.
fn translate_anthropic_message(m: &AnthropicMessage, out: &mut Vec<ChatMessage>) {
    let blocks: Vec<&AnthropicContentBlock> = match &m.content {
        AnthropicContent::Plain(s) => {
            out.push(ChatMessage {
                role: m.role.clone(),
                content: Some(s.clone()),
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            });
            return;
        }
        AnthropicContent::Blocks(bs) => bs.iter().collect(),
    };

    let mut text_buf = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => text_buf.push_str(text),
            AnthropicContentBlock::Image { .. } => {}
            // Replayed assistant reasoning: dropped from the prompt context.
            AnthropicContentBlock::Thinking { .. } => {}
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: input.to_string(),
                    },
                });
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: _,
            } => {
                if !text_buf.is_empty() || !tool_calls.is_empty() {
                    out.push(ChatMessage {
                        role: m.role.clone(),
                        content: if text_buf.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut text_buf))
                        },
                        tool_call_id: None,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut tool_calls))
                        },
                        reasoning_content: None,
                    });
                }
                let body = match content {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(arr) => arr
                        .iter()
                        .filter_map(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_owned))
                        .collect::<Vec<_>>()
                        .join(""),
                    other => other.to_string(),
                };
                out.push(ChatMessage {
                    role: "tool".into(),
                    content: Some(body),
                    tool_call_id: Some(tool_use_id.clone()),
                    tool_calls: None,
                    reasoning_content: None,
                });
            }
        }
    }
    if !text_buf.is_empty() || !tool_calls.is_empty() {
        out.push(ChatMessage {
            role: m.role.clone(),
            content: if text_buf.is_empty() {
                None
            } else {
                Some(text_buf)
            },
            tool_call_id: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            reasoning_content: None,
        });
    }
}

fn anthropic_tools_to_openai(tools: &[AnthropicTool]) -> Vec<ToolDef> {
    tools
        .iter()
        .map(|t| ToolDef {
            kind: "function".into(),
            function: FunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

/// Anthropic SSE streaming engine.
///
/// Emits the canonical Anthropic event sequence: `message_start` →
/// `content_block_start` → `content_block_delta`* → `content_block_stop`
/// → `message_delta` → `message_stop`. The stream closes without a
/// `[DONE]` sentinel (OpenAI-only). Each frame carries both `event:` and
/// `data:` SSE fields per the Anthropic spec.
fn stream_messages_anthropic_sse(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    relax_stop_mask: bool,
    tool_call_format_default: crate::tool_call_parser::ToolCallFormat,
    tool_force_prefix: Option<String>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let id = request_id("msg");
    let model = state.model_id.clone();

    let prompt_ids = match state.tokenizer.encode_for_inference(&prompt) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) | Err(_) => {
            let err = json!({"type":"error","error":{"type":"invalid_request_error","message":"prompt tokenized to 0 tokens"}});
            let _ = tx.try_send(Ok(Event::default().event("error").data(err.to_string())));
            return Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default());
        }
    };
    let input_tokens = prompt_ids.len() as u32;

    let message_start = json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": serde_json::Value::Null,
            "stop_sequence": serde_json::Value::Null,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
            },
        },
    });
    let _ = tx.try_send(Ok(Event::default()
        .event("message_start")
        .data(message_start.to_string())));

    let state_clone = state.clone();
    let id_clone = id.clone();
    let tx_clone = tx.clone();
    tokio::task::spawn_blocking(move || {
        use crate::tool_call_parser::{dispatcher_with_prompt, ParserEvent};
        // `<think>`-style templates prime the reasoning marker in the prompt
        // prefix when thinking is enabled, so start the parser in-think.
        let start_in_reasoning = params.enable_thinking
            && state_clone.model.reasoning_markers().style
                == crate::model_handle::ReasoningStyle::ThinkTag;
        let mut parser = match dispatcher_with_prompt(
            None,
            tool_call_format_default,
            &prompt,
            start_in_reasoning,
        ) {
            Ok(p) => p,
            Err(e) => {
                let err = json!({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": e.to_string()},
                });
                let _ = tx_clone
                    .blocking_send(Ok(Event::default().event("error").data(err.to_string())));
                return;
            }
        };

        // Block layout: when present, reasoning is block 0, the answer text is
        // the next block, tool calls follow. Indices are allocated in emission
        // order. Reasoning always precedes text/tools, so a thinking block is
        // closed (with a stub signature) on the first non-thinking event.
        let mut next_block_index: u32 = 0;
        let mut thinking_idx: Option<u32> = None;
        let mut thinking_closed = false;
        let mut text_idx: Option<u32> = None;
        let mut tool_open: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut emitted_any = false;
        let mut has_tool_calls = false;

        let send = |event: &str, value: serde_json::Value| -> bool {
            tx_clone
                .blocking_send(Ok(Event::default().event(event).data(value.to_string())))
                .is_ok()
        };

        let close_thinking = |thinking_idx: &Option<u32>, thinking_closed: &mut bool| -> bool {
            if let Some(ti) = *thinking_idx {
                if !*thinking_closed {
                    *thinking_closed = true;
                    return send(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": ti,
                            "delta": {"type": "signature_delta", "signature": ""},
                        }),
                    ) && send(
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": ti}),
                    );
                }
            }
            true
        };

        let drain_events = |events: Vec<ParserEvent>,
                            next_block_index: &mut u32,
                            thinking_idx: &mut Option<u32>,
                            thinking_closed: &mut bool,
                            text_idx: &mut Option<u32>,
                            tool_open: &mut std::collections::HashMap<u32, u32>,
                            emitted_any: &mut bool,
                            has_tool_calls: &mut bool|
         -> bool {
            for ev in events {
                match ev {
                    ParserEvent::ThinkDelta(s) => {
                        if s.is_empty() {
                            continue;
                        }
                        let idx = match *thinking_idx {
                            Some(i) => i,
                            None => {
                                let i = *next_block_index;
                                *next_block_index += 1;
                                *thinking_idx = Some(i);
                                if !send(
                                    "content_block_start",
                                    json!({
                                        "type": "content_block_start",
                                        "index": i,
                                        "content_block": {"type": "thinking", "thinking": ""},
                                    }),
                                ) {
                                    return false;
                                }
                                i
                            }
                        };
                        *emitted_any = true;
                        if !send(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "thinking_delta", "thinking": s},
                            }),
                        ) {
                            return false;
                        }
                    }
                    ParserEvent::TextDelta(s) => {
                        if s.is_empty() {
                            continue;
                        }
                        if !close_thinking(thinking_idx, thinking_closed) {
                            return false;
                        }
                        let idx = match *text_idx {
                            Some(i) => i,
                            None => {
                                let i = *next_block_index;
                                *next_block_index += 1;
                                *text_idx = Some(i);
                                if !send(
                                    "content_block_start",
                                    json!({
                                        "type": "content_block_start",
                                        "index": i,
                                        "content_block": {"type": "text", "text": ""},
                                    }),
                                ) {
                                    return false;
                                }
                                i
                            }
                        };
                        *emitted_any = true;
                        if !send(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": s},
                            }),
                        ) {
                            return false;
                        }
                    }
                    ParserEvent::ToolCallOpen { index, name } => {
                        *has_tool_calls = true;
                        if !close_thinking(thinking_idx, thinking_closed) {
                            return false;
                        }
                        let block_idx = *next_block_index;
                        *next_block_index += 1;
                        tool_open.insert(index, block_idx);
                        if !send(
                            "content_block_start",
                            json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": format!("toolu_{block_idx:08x}"),
                                    "name": name,
                                    "input": {},
                                },
                            }),
                        ) {
                            return false;
                        }
                    }
                    ParserEvent::ToolCallArgumentsDelta { index, arguments } => {
                        let Some(&block_idx) = tool_open.get(&index) else {
                            continue;
                        };
                        if !send(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {"type": "input_json_delta", "partial_json": arguments},
                            }),
                        ) {
                            return false;
                        }
                    }
                    ParserEvent::ToolCallClose { index } => {
                        let Some(block_idx) = tool_open.remove(&index) else {
                            continue;
                        };
                        if !send(
                            "content_block_stop",
                            json!({"type": "content_block_stop", "index": block_idx}),
                        ) {
                            return false;
                        }
                    }
                }
            }
            true
        };

        let mut emit_delta = |text: &str| -> bool {
            let events = parser.push(text);
            drain_events(
                events,
                &mut next_block_index,
                &mut thinking_idx,
                &mut thinking_closed,
                &mut text_idx,
                &mut tool_open,
                &mut emitted_any,
                &mut has_tool_calls,
            )
        };

        // B1: feed the forced tool-call opening to the parser first so the
        // model's continuation reconstructs a complete call (and any
        // already-complete events, e.g. a named ToolCallOpen, stream out).
        if let Some(pfx) = &tool_force_prefix {
            emit_delta(pfx);
        }

        let res = run_completion_blocking_streaming(
            state_clone,
            prompt,
            params,
            relax_stop_mask,
            &mut emit_delta,
        );

        let tail = parser.finish();
        let _ = drain_events(
            tail,
            &mut next_block_index,
            &mut thinking_idx,
            &mut thinking_closed,
            &mut text_idx,
            &mut tool_open,
            &mut emitted_any,
            &mut has_tool_calls,
        );

        for (_idx, block_idx) in tool_open.drain() {
            let _ = send(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": block_idx}),
            );
        }
        // Close a thinking block that never transitioned to an answer.
        let _ = close_thinking(&thinking_idx, &mut thinking_closed);
        if text_idx.is_none() && !emitted_any {
            let i = next_block_index;
            let _ = send(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                }),
            );
            text_idx = Some(i);
        }
        if let Some(ti) = text_idx {
            let _ = send(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": ti}),
            );
        }

        let (finish, _, output_tokens) = match &res {
            Ok(t) => (t.0.clone(), t.1, t.2),
            Err(e) => {
                tracing::warn!(
                    target: "server.messages",
                    "stream decode failed: {e:#}"
                );
                ("error".to_string(), 0u32, 0u32)
            }
        };
        let stop_reason: &str = if has_tool_calls {
            "tool_use"
        } else {
            match finish.as_str() {
                "stop" => "end_turn",
                "length" => "max_tokens",
                "tool_calls" => "tool_use",
                other => other,
            }
        };
        let mdelta = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": serde_json::Value::Null},
            "usage": {"output_tokens": output_tokens},
        });
        let _ = tx_clone.blocking_send(Ok(Event::default()
            .event("message_delta")
            .data(mdelta.to_string())));

        let mstop = json!({"type": "message_stop"});
        let _ = tx_clone.blocking_send(Ok(Event::default()
            .event("message_stop")
            .data(mstop.to_string())));

        if let Err(e) = res {
            let err = json!({
                "type": "error",
                "error": {"type": "internal_error", "message": e.to_string()},
            });
            let _ =
                tx_clone.blocking_send(Ok(Event::default().event("error").data(err.to_string())));
        }
        let _ = id_clone;
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}
