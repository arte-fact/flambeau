//! HTTP route handlers. Requires `hip` feature (loads real model).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use flambeau_backend_hip::HipCluster;
use flambeau_qwen3_moe::forward::{
    forward_one_token_pp_logits, forward_prefill_pp_logits, ShardedForwardOneTokenScratch,
    ShardedForwardPrefillScratch,
};
use flambeau_qwen3_moe::{
    Qwen3MoEConfig, Qwen3MoEShardedModel, Qwen3MoEShardedSession,
};
use flambeau_quant::{ChatTemplate, GgufTokenizer};
use flambeau_runtime::Sampler;
use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

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
    /// Tools discovered on the `--mcp <url>` upstreams at startup
    /// (ROADMAP-V2 §M2.1). Merged into each request's `tools[]` before
    /// rendering the Jinja template, so the model sees them alongside
    /// any client-supplied tools. Agent-loop bridging (actually calling
    /// them on a tool_call emission) is M2.2.
    pub remote_tools: Vec<crate::mcp_client::RemoteTool>,
    /// Per-iteration agent-loop telemetry (M2.3). Ring buffer; surfaced
    /// read-only at `GET /v1/agent/stats`.
    pub agent_stats: crate::agent_stats::AgentStatsRing,
}

pub type SharedState = Arc<ServerState>;

/// GET /health — constant, no locks.
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// GET / — embedded single-file chat UI (Nord palette, Claude aesthetic).
pub async fn index() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        INDEX_HTML,
    )
}

const INDEX_HTML: &str = include_str!("../assets/index.html");

/// GET /v1/agent/stats — M2.3 read-only agent-loop telemetry snapshot.
/// Returns the last N per-iteration stats (N = ring capacity).
pub async fn agent_stats(State(state): State<SharedState>) -> impl IntoResponse {
    let snap = state.agent_stats.snapshot();
    Json(json!({
        "iterations": snap,
        "count": snap.len(),
    }))
}

/// GET /v1/tools — C4.1 introspection for the chat UI's "what tools
/// does the model see?" panel. Surfaces only the `--mcp <url>`-
/// discovered remote tools (request-supplied `tools[]` are per-request
/// and not server state). Tools are reported with their alias-prefixed
/// `name` (what the model sees), the unprefixed `remote_name` (what
/// the upstream server knows), the source URL, and the parameters
/// schema.
pub async fn tools_endpoint(State(state): State<SharedState>) -> impl IntoResponse {
    Json(json!({
        "remote_tools": &state.remote_tools,
        "count": state.remote_tools.len(),
    }))
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

/// POST /v1/chat/completions — JSON when `stream=false`, SSE when `stream=true`.
#[tracing::instrument(
    name = "server.chat_completions",
    skip_all,
    fields(messages = req.messages.len(), stream = req.stream)
)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages[] is empty"));
    }

    // Qwen3.6's chat template prepends `<think>\n\n</think>\n\n` to the
    // current-turn assistant prefix (the "no-thinking" delimiter). Prior
    // assistant turns sent by the client are stored as plain content and,
    // when re-rendered, produce a prompt that's out-of-distribution — the
    // model greedy-stops on the first token. Normalise every assistant
    // message that lacks `</think>` by wrapping its content the same way
    // the template would for the current turn.
    let mut messages: Vec<ChatMessage> =
        req.messages.iter().map(normalise_message).collect();

    // T1.2 + M2.1: thread client tools + `--mcp`-discovered remote
    // tools into the Jinja render. Both visible to the model under one
    // namespace; remote tools are `alias.name`-prefixed.
    let merged_tools = merge_request_and_remote_tools(
        req.tools.as_deref(),
        &state.remote_tools,
    );

    // T4.b.2 / T4.b.1 sampler config.
    let params = SamplingParams::from_parts(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.repetition_penalty,
        req.presence_penalty,
        req.frequency_penalty,
        req.max_tokens,
        req.seed,
    );

    // T3.2: OpenAI default is `parallel_tool_calls=true`. `false` hard-
    // terminates decode at the first tool-call close (streaming) or
    // truncates the returned tool_calls to the first one (non-stream).
    let parallel_tool_calls = req.parallel_tool_calls.unwrap_or(true);
    // T4.1: tool-call turns bypass the first-24-token stop-mask — the
    // model should be free to emit `<|im_end|>` right after a
    // `</tool_call>` without being forced to pad.
    let has_tools = merged_tools.is_some();
    let relax_stop_mask = has_tools;

    // Streaming path delegates to stream_completion_sse which holds its
    // own parser. Agent-loop support inside streaming is M2.2.b; today
    // the streaming path returns whatever the model emits as-is,
    // without executing remote tool_calls server-side.
    if req.stream {
        let prompt = state
            .chat_template
            .render_with_tools(
                &messages,
                merged_tools.as_deref(),
                /*add_generation_prompt=*/ true,
                /*enable_thinking=*/ Some(false),
            )
            .map_err(ApiError::internal)?;
        if std::env::var("FLAMBEAU_DUMP_PROMPT").is_ok() {
            eprintln!(
                "--- rendered prompt ({} bytes) ---\n{}\n--- end prompt ---",
                prompt.len(),
                prompt
            );
        }
        return Ok(stream_completion_sse(
            state,
            prompt,
            params,
            req.tool_call_format.clone(),
            parallel_tool_calls,
            relax_stop_mask,
        )
        .into_response());
    }

    // Non-streaming agent loop (M2.2). On each iteration:
    //   1. Render the current `messages` into a prompt (with tools).
    //   2. Run decode to text completion.
    //   3. Parse text → (content, tool_calls).
    //   4. Partition tool_calls into client-visible vs self-executable
    //      (remote). If self-executable ones exist AND no client ones
    //      block us, execute them, append `assistant` + `role=tool`
    //      turns to `messages`, loop. Otherwise return.
    //   5. Cap at `MAX_TOOL_ITERATIONS`; hitting the cap returns
    //      whatever was produced on the final iteration as-is (no
    //      further execution).
    const MAX_TOOL_ITERATIONS: usize = 10;

    let mut sum_prompt_tokens: u32 = 0;
    let mut sum_completion_tokens: u32 = 0;
    let mut final_content = String::new();
    let mut final_tool_calls: Vec<crate::api::ToolCall> = Vec::new();
    let mut final_finish: String = "stop".into();
    let session_id = request_id("chatcmpl");

    for iter in 0..MAX_TOOL_ITERATIONS {
        let iter_start = Instant::now();
        let prompt = state
            .chat_template
            .render_with_tools(
                &messages,
                merged_tools.as_deref(),
                /*add_generation_prompt=*/ true,
                /*enable_thinking=*/ Some(false),
            )
            .map_err(ApiError::internal)?;
        if iter == 0 && std::env::var("FLAMBEAU_DUMP_PROMPT").is_ok() {
            eprintln!(
                "--- rendered prompt ({} bytes) ---\n{}\n--- end prompt ---",
                prompt.len(),
                prompt
            );
        }

        let (text, iter_prompt_tokens, iter_completion_tokens, iter_finish) =
            run_completion(state.clone(), &prompt, params.clone(), relax_stop_mask)
                .await
                .map_err(ApiError::internal)?;
        sum_prompt_tokens = sum_prompt_tokens.saturating_add(iter_prompt_tokens);
        sum_completion_tokens = sum_completion_tokens.saturating_add(iter_completion_tokens);

        let (content, mut tool_calls) = {
            use crate::tool_call_parser::{dispatcher, split_events, ParserEvent};
            let mut parser = dispatcher(
                req.tool_call_format.as_deref(),
                state.cfg.arch.as_str(),
            )
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
            let mut events = parser.push(&text);
            events.extend(parser.finish());
            split_events(ParserEvent::coalesce(events))
        };
        if !parallel_tool_calls && tool_calls.len() > 1 {
            tool_calls.truncate(1);
        }

        // Partition: remote tools we own vs client tools we hand back.
        let (remote_calls, client_calls): (Vec<_>, Vec<_>) = tool_calls
            .into_iter()
            .partition(|tc| {
                crate::mcp_client::find_by_prefixed_name(
                    &state.remote_tools,
                    &tc.function.name,
                )
                .is_some()
            });

        // Capture per-iteration telemetry (M2.3). We include both the
        // remote- and client-side tool names so the stats reflect what
        // the model actually emitted, not just what we executed.
        let iter_tool_names: Vec<String> = remote_calls
            .iter()
            .chain(client_calls.iter())
            .map(|tc| tc.function.name.clone())
            .collect();
        let iter_remote_count = remote_calls.len() as u32;
        let record_stat = |finish: &str| {
            state.agent_stats.push(crate::agent_stats::IterStat {
                session_id: session_id.clone(),
                iteration: iter as u32,
                tool_names: iter_tool_names.clone(),
                remote_count: iter_remote_count,
                latency_ms: iter_start.elapsed().as_millis() as u64,
                prompt_tokens: iter_prompt_tokens,
                completion_tokens: iter_completion_tokens,
                finish_reason: finish.to_owned(),
            });
        };

        // Stop conditions. The loop breaks in three cases:
        //   (a) no tool calls at all → plain "stop" response;
        //   (b) any client-visible tool calls → hand them back with
        //       finish_reason="tool_calls" (we don't execute them);
        //   (c) we've hit the iteration budget → return whatever
        //       remote_calls we have without running them, flagged so
        //       the client can follow up manually.
        if remote_calls.is_empty() && client_calls.is_empty() {
            record_stat(&iter_finish);
            final_content = content;
            final_finish = iter_finish;
            break;
        }
        if !client_calls.is_empty() {
            record_stat("tool_calls");
            final_content = content;
            final_tool_calls = [client_calls, remote_calls].concat();
            final_finish = "tool_calls".into();
            break;
        }
        if iter + 1 >= MAX_TOOL_ITERATIONS {
            tracing::warn!(
                target: "flambeau.server",
                iter = iter + 1,
                max = MAX_TOOL_ITERATIONS,
                "agent loop hit max_tool_iterations — returning remote tool_calls to the caller"
            );
            record_stat("tool_calls");
            final_content = content;
            final_tool_calls = remote_calls;
            final_finish = "tool_calls".into();
            break;
        }
        // Loop continuing: record the iteration before we re-enter.
        record_stat("tool_calls");

        // Execute the remote tool calls server-side, splice results
        // into `messages`, loop.
        tracing::info!(
            target: "flambeau.server",
            iter = iter + 1,
            remote_calls = remote_calls.len(),
            "agent loop: executing remote tool calls"
        );
        // Persist the assistant turn with the tool_calls into history —
        // Qwen3's template needs that slot for the role=tool follow-up
        // to land in the right conversational place.
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: if content.is_empty() {
                None
            } else {
                Some(format!("<think>\n\n</think>\n\n{}", content))
            },
            tool_call_id: None,
            tool_calls: Some(remote_calls.clone()),
        });
        for tc in &remote_calls {
            let Some(rt) = crate::mcp_client::find_by_prefixed_name(
                &state.remote_tools,
                &tc.function.name,
            ) else {
                // Shouldn't fire — partition check already matched.
                continue;
            };
            let tool_output = crate::mcp_client::call_remote(rt, &tc.function.arguments)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        target: "flambeau.server",
                        tool = %tc.function.name,
                        error = %e,
                        "remote tool call failed — surfacing error text to model"
                    );
                    format!(
                        "{{\"error\":\"remote tool {name} failed: {err}\"}}",
                        name = tc.function.name,
                        err = e.to_string().replace('"', "\\\""),
                    )
                });
            messages.push(ChatMessage {
                role: "tool".into(),
                content: Some(tool_output),
                tool_call_id: Some(tc.id.clone()),
                tool_calls: None,
            });
        }
        // continue the loop
    }

    Ok(Json(ChatCompletionResponse {
        // One id per HTTP request; the agent-loop stats ring (M2.3)
        // uses the same id so `/v1/agent/stats` can group iterations.
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
            },
            finish_reason: final_finish,
        }],
        usage: Usage {
            prompt_tokens: sum_prompt_tokens,
            completion_tokens: sum_completion_tokens,
            total_tokens: sum_prompt_tokens.saturating_add(sum_completion_tokens),
        },
    })
    .into_response())
}

/// Normalise ONE chat message's content for the Qwen3.6 template (see
/// the `<think>\n\n</think>\n\n` comment above for rationale).
fn normalise_message(m: &ChatMessage) -> ChatMessage {
    let body = m.content_str();
    if m.role == "assistant" && !body.contains("</think>") {
        ChatMessage {
            role: m.role.clone(),
            content: Some(format!("<think>\n\n</think>\n\n{}", body)),
            tool_call_id: m.tool_call_id.clone(),
            tool_calls: m.tool_calls.clone(),
        }
    } else {
        m.clone()
    }
}

/// POST /v1/completions — legacy text-completion endpoint.
#[tracing::instrument(
    name = "server.completions",
    skip_all,
    fields(stream = req.stream)
)]
pub async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    if req.stream {
        return Err(ApiError::bad_request(
            "SSE streaming is not yet implemented (V1.8.C). Retry with stream=false.",
        ));
    }
    // Legacy /v1/completions doesn't surface the full OpenAI sampler
    // knobs — only temperature/top_p/max_tokens/seed/stop. Pass None for
    // the penalty + top_k/min_p fields; `from_parts` applies Qwen3.5
    // published defaults.
    let params = SamplingParams::from_parts(
        req.temperature,
        req.top_p,
        /*top_k=*/ None,
        /*min_p=*/ None,
        /*repetition_penalty=*/ None,
        /*presence_penalty=*/ None,
        /*frequency_penalty=*/ None,
        req.max_tokens,
        req.seed,
    );
    // Legacy /v1/completions has no `tools` field — keep the default
    // stop-mask policy unchanged from V1.8.
    let (text, prompt_tokens, completion_tokens, finish) =
        run_completion(state.clone(), &req.prompt, params, /*relax_stop_mask=*/ false)
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

/// Merge client-supplied tools with any registered via `--mcp <url>`.
/// Returns `None` when both are empty so the Jinja template takes the
/// no-tools branch (same byte-for-byte output as V1.8). Returns a
/// `Vec<serde_json::Value>` with two kinds of entries:
/// - client tools — serialised from `api::ToolDef` directly;
/// - remote tools — rendered through `mcp_client::to_tool_json`, which
///   uses the `alias.name`-prefixed name so the two sources can share
///   a tool namespace without collisions.
fn merge_request_and_remote_tools(
    client_tools: Option<&[ToolDef]>,
    remote_tools: &[crate::mcp_client::RemoteTool],
) -> Option<Vec<serde_json::Value>> {
    let client_len = client_tools.map_or(0, <[ToolDef]>::len);
    let total = client_len + remote_tools.len();
    if total == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(total);
    if let Some(ts) = client_tools {
        for t in ts {
            match serde_json::to_value(t) {
                Ok(v) => out.push(v),
                Err(e) => {
                    // Shouldn't happen — ToolDef is a plain serde struct.
                    // If it does, skip the tool rather than fail the
                    // whole turn.
                    tracing::warn!(
                        target: "flambeau.server",
                        error = %e,
                        "failed to serialise client-supplied tool; skipping"
                    );
                }
            }
        }
    }
    for rt in remote_tools {
        out.push(crate::mcp_client::to_tool_json(rt));
    }
    Some(out)
}

/// Build an SSE stream from the blocking completion pipeline.
///
/// Emits OpenAI-compatible `chat.completion.chunk` frames: role
/// announcement, zero-or-more content / tool-call deltas, and a final
/// frame with `finish_reason` + `[DONE]` sentinel.
///
/// T3.1: tool-call deltas are emitted as the parser transitions — we
/// never buffer a tool call body to end-of-stream and then dump it as
/// raw text (the specific anti-pattern that produces the "raw XML at
/// end of stream" complaints against llama.cpp).
fn stream_completion_sse(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    tool_call_format: Option<String>,
    parallel_tool_calls: bool,
    relax_stop_mask: bool,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    use crate::tool_call_parser::{dispatcher, ParserEvent};

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let id = request_id("chatcmpl");
    let created = now_unix();
    let model = state.model_id.clone();

    // First chunk: role announcement, matches OpenAI's streaming shape.
    let role_frame = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant" },
            "finish_reason": null,
        }],
    });
    let _ = tx.try_send(Ok(Event::default().data(role_frame.to_string())));

    let state_clone = state.clone();
    let id_clone = id.clone();
    let model_clone = model.clone();
    let tx_clone = tx.clone();
    tokio::task::spawn_blocking(move || {
        // Build the parser once per request. Arch comes from the loaded
        // GGUF; the request may override the format explicitly.
        let parser_result = dispatcher(
            tool_call_format.as_deref(),
            state_clone.cfg.arch.as_str(),
        );
        let mut parser = match parser_result {
            Ok(p) => p,
            Err(e) => {
                // Bad tool_call_format — surface as a streamed error.
                let err = json!({ "error": { "message": e.to_string() } });
                let _ = tx_clone
                    .blocking_send(Ok(Event::default().data(err.to_string())));
                let _ = tx_clone
                    .blocking_send(Ok(Event::default().data("[DONE]")));
                return;
            }
        };

        // Shared helpers for emitting the different chunk shapes. Each
        // returns `true` on successful send, `false` if the receiver
        // dropped — the outer pipeline aborts early in that case.
        let emit_chunk = |delta_value: serde_json::Value| -> bool {
            let frame = json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [{
                    "index": 0,
                    "delta": delta_value,
                    "finish_reason": null,
                }],
            });
            tx_clone
                .blocking_send(Ok(Event::default().data(frame.to_string())))
                .is_ok()
        };

        // T3.1 / T3.2: state threaded through the per-token text feed.
        let mut has_tool_calls = false;
        // T3.2: when parallel_tool_calls=false, set on the first
        // ToolCallClose so the next emit_delta call returns false and
        // the decode loop hard-terminates.
        let mut abort_after_close = false;

        // Drain a batch of parser events into SSE chunks. Returns
        // false when the receiver dropped OR a !parallel close fired.
        let emit_events = |events: Vec<ParserEvent>,
                           has_tool_calls: &mut bool,
                           abort_after_close: &mut bool|
         -> bool {
            for e in events {
                let sent = match e {
                    ParserEvent::TextDelta(s) => emit_chunk(json!({ "content": s })),
                    ParserEvent::ThinkDelta(_) => true, // discard; V3 will surface
                    ParserEvent::ToolCallOpen { index, name } => {
                        *has_tool_calls = true;
                        emit_chunk(json!({
                            "tool_calls": [{
                                "index": index,
                                "id": format!("call_{index}"),
                                "type": "function",
                                "function": { "name": name },
                            }]
                        }))
                    }
                    ParserEvent::ToolCallArgumentsDelta { index, arguments } => {
                        *has_tool_calls = true;
                        emit_chunk(json!({
                            "tool_calls": [{
                                "index": index,
                                "function": { "arguments": arguments },
                            }]
                        }))
                    }
                    ParserEvent::ToolCallClose { .. } => {
                        if !parallel_tool_calls {
                            *abort_after_close = true;
                        }
                        true
                    }
                };
                if !sent {
                    return false;
                }
            }
            true
        };

        // Token-level text feed: the decoder emits one text delta per
        // sampled token (or coalesced multi-byte run). We push into the
        // parser and stream out events as they come. Never buffer to
        // end-of-stream.
        let mut emit_delta = |text: &str| -> bool {
            let events = parser.push(text);
            let sent = emit_events(events, &mut has_tool_calls, &mut abort_after_close);
            sent && !abort_after_close
        };

        let res = run_completion_blocking_streaming(
            state_clone.clone(),
            prompt,
            params,
            relax_stop_mask,
            &mut emit_delta,
        );

        // Flush the parser — dumps any buffered mid-tag content and
        // closes out pending think text.
        let tail = parser.finish();
        let _ = emit_events(tail, &mut has_tool_calls, &mut abort_after_close);

        // finish_reason flips to "tool_calls" when the turn produced
        // any, per OpenAI contract.
        let finish_reason = if has_tool_calls {
            "tool_calls"
        } else {
            match &res {
                Ok(r) => r.as_str(),
                Err(_) => "error",
            }
        };
        let done_frame = json!({
            "id": id_clone,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_clone,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        });
        let _ = tx_clone.blocking_send(Ok(Event::default().data(done_frame.to_string())));
        if let Err(e) = res {
            let err = json!({ "error": { "message": e.to_string() } });
            let _ = tx_clone.blocking_send(Ok(Event::default().data(err.to_string())));
        }
        let _ = tx_clone.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

/// Shared engine: text prompt → generated text + token counts + finish reason.
///
/// Runs inside `spawn_blocking` because HIP kernels + mutex hold are sync.
///
/// `relax_stop_mask`: when `true`, disables the first-N-token stop-token
/// suppression and the post-N stop-bias — used on turns where the client
/// supplied `tools[]` (T4.1). Tool-call responses are legitimately short
/// (a JSON blob fits in ~20 tokens); injecting the stop mask forces the
/// model to pad before emitting `</tool_call>`.
async fn run_completion(
    state: SharedState,
    prompt: &str,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<(String, u32, u32, String)> {
    let prompt = prompt.to_owned();
    tokio::task::spawn_blocking(move || {
        run_completion_blocking(state, prompt, params, relax_stop_mask)
    })
    .await
    .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

fn run_completion_blocking(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<(String, u32, u32, String)> {
    let request_start = Instant::now();

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

    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        "completion request accepted after queue wait"
    );

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

    // Sampler holds vocab-sized scratch reused across all decode steps
    // (C2 in RUST-PERF-CORRECTIONS.md). Reserve upfront to avoid the
    // first-token grow.
    let mut sampler = Sampler::from_seed(params.seed);
    sampler.reserve(state.cfg.vocab_size);
    let sampling = &params.sampling;
    let stop_ids = &state.tokenizer.stop_ids;
    let vocab = state.cfg.vocab_size;
    // Always allocate the logit buffer — we need it for the first-token
    // EOS mask regardless of sampling mode.
    let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);

    // Prefill. Always download logits so we can mask stop tokens on the
    // first generated token — Qwen3.6 sometimes argmaxes `<|im_end|>` as
    // the first response token on multi-turn prompts, producing an empty
    // reply. Suppress it until at least one content token is emitted.
    let prefill_start = Instant::now();
    forward_prefill_pp_logits(
        model,
        &mut session,
        cluster,
        &mut prefill_scratch,
        &prompt_ids,
        0,
        &mut logits_buf,
    )
    .context("prefill logits")?;
    for &sid in stop_ids {
        if (sid as usize) < logits_buf.len() {
            logits_buf[sid as usize] = f32::NEG_INFINITY;
        }
    }
    // First token: empty history, penalties are no-ops.
    let first_next = sampler.sample(&logits_buf, sampling, &[]);
    let is_greedy = sampling.is_greedy();
    tracing::info!(
        target: "server.completion.first_token",
        prompt_tokens,
        ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
        greedy = is_greedy,
        "first token produced (time-to-first-token)"
    );

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let is_stop = |t: u32| stop_ids.contains(&t);

    generated.push(first_next);
    // First-token stop mask means `is_stop(first_next)` cannot fire here,
    // but we keep the check as a defensive guard for future logit-mask
    // changes.
    if is_stop(first_next) {
        return finalise(&state, prompt_tokens, generated, "stop");
    }

    let mut finish_reason = "length";
    let mut last_token = first_next;
    // Mask stop tokens for the first MIN_RESPONSE_TOKENS steps. Qwen3.6
    // on multi-turn prompts otherwise emits `<|im_end|>` after 0-1 content
    // tokens, producing unusable one-word replies. MIN is small enough
    // that short on-topic answers ("Yes.", "42.") are still possible.
    //
    // T4.1: tool-call turns disable this entirely. When the model is
    // asked to emit a `<tool_call>…</tool_call>` body it may legitimately
    // take only ~10 tokens; forcing 24 content tokens before allowing
    // stop injects noise between the body and the `<|im_end|>` and
    // breaks downstream parsing. `relax_stop_mask` flips both knobs to
    // no-ops — trust the model on turns where `tools[]` is present.
    const MIN_RESPONSE_TOKENS: usize = 24;
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS,
    // so the model is free to stop but strongly disincentivised. The model's
    // EOS logit on multi-turn prompts sits ~1.3 nats above the next candidate;
    // -3.0 flips that to a solid discount without banning stop.
    const STOP_BIAS: f32 = 3.0;
    for step in 1..params.max_tokens as usize {
        let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
        forward_one_token_pp_logits(
            model,
            &mut session,
            cluster,
            &mut decode_scratch,
            last_token,
            prompt_ids.len() + step,
            &mut logits_buf,
        )
        .context("decode step logits")?;
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    if force_mask {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else {
                        logits_buf[sid as usize] -= STOP_BIAS;
                    }
                }
            }
        }
        // Pass `generated` as history so penalties can fire on repeats
        // / frequent tokens. T4.b.2 — without this, Qwen3.5/3.6 agent
        // loops degrade to long-CoT drift per the Ollama post-mortem.
        let next = sampler.sample(&logits_buf, sampling, &generated);
        generated.push(next);
        last_token = next;
        if is_stop(next) {
            finish_reason = "stop";
            break;
        }
    }

    // Dispose per-request scratch/session; keep model + cluster alive.
    decode_scratch
        .dispose(cluster)
        .context("dispose decode scratch")?;
    prefill_scratch
        .dispose(cluster)
        .context("dispose prefill scratch")?;
    session.dispose(cluster).context("dispose session")?;

    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens,
        completion_tokens = generated.len() as u32,
        finish_reason,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "completion request finished"
    );

    finalise(&state, prompt_tokens, generated, finish_reason)
}

/// Streaming variant: pushes text deltas through `emit` as each token is
/// produced. Returns the finish reason on success.
fn run_completion_blocking_streaming(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    relax_stop_mask: bool,
    emit: &mut dyn FnMut(&str) -> bool,
) -> Result<String> {
    let request_start = Instant::now();

    let _guard = state.inflight.blocking_lock();

    let prompt_ids = state.tokenizer.encode(&prompt).context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }
    let prompt_tokens = prompt_ids.len() as u32;

    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        stream = true,
        "streaming completion accepted"
    );

    let cluster = &state.cluster;
    let model = &state.model;
    let stop_ids = &state.tokenizer.stop_ids;
    let is_stop = |t: u32| stop_ids.contains(&t);

    let mut session =
        Qwen3MoEShardedSession::new(model, cluster).context("create session")?;
    let mut prefill_scratch =
        ShardedForwardPrefillScratch::new(model, cluster, prompt_ids.len())
            .context("prefill scratch")?;
    let mut decode_scratch =
        ShardedForwardOneTokenScratch::new(model, cluster).context("decode scratch")?;

    let mut sampler = Sampler::from_seed(params.seed);
    sampler.reserve(state.cfg.vocab_size);
    let sampling = &params.sampling;
    let is_greedy = sampling.is_greedy();
    let vocab = state.cfg.vocab_size;
    let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);

    // First token: always download logits and mask stop ids. See the
    // non-streaming path for the rationale (multi-turn Qwen3.6 argmaxes
    // `<|im_end|>` immediately otherwise).
    let prefill_start = Instant::now();
    forward_prefill_pp_logits(
        model,
        &mut session,
        cluster,
        &mut prefill_scratch,
        &prompt_ids,
        0,
        &mut logits_buf,
    )
    .context("prefill logits")?;
    for &sid in stop_ids {
        if (sid as usize) < logits_buf.len() {
            logits_buf[sid as usize] = f32::NEG_INFINITY;
        }
    }
    let first_next = sampler.sample(&logits_buf, sampling, &[]);
    tracing::info!(
        target: "server.completion.first_token",
        prompt_tokens,
        ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
        greedy = is_greedy,
        "first token produced (time-to-first-token)"
    );

    // Rolling decode: re-decode the full generated list each step and emit
    // the UTF-8 suffix beyond what was already emitted. Handles multi-byte
    // tokens without surfacing partial codepoints to the client.
    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let mut emitted_text = String::new();

    let mut push_and_emit = |tok: u32,
                             generated: &mut Vec<u32>,
                             emitted_text: &mut String|
     -> Result<bool> {
        generated.push(tok);
        let stop_hit = is_stop(tok);
        // Decode the full sequence *without* the trailing stop token so we
        // never emit the raw stop marker text.
        let slice: &[u32] = if stop_hit {
            &generated[..generated.len() - 1]
        } else {
            &generated[..]
        };
        let text = state.tokenizer.decode(slice).context("decode")?;
        if text.len() > emitted_text.len() && text.starts_with(emitted_text.as_str()) {
            let delta = &text[emitted_text.len()..];
            if !delta.is_empty() && !emit(delta) {
                // Receiver dropped — abort early.
                return Ok(false);
            }
            emitted_text.clear();
            emitted_text.push_str(&text);
        } else if text != *emitted_text {
            // Non-prefix change (rare — sentencepiece re-normalisation).
            // Emit full replacement as a single delta.
            if !emit(&text) {
                return Ok(false);
            }
            emitted_text.clear();
            emitted_text.push_str(&text);
        }
        Ok(!stop_hit)
    };

    let alive = push_and_emit(first_next, &mut generated, &mut emitted_text)?;
    if !alive {
        cleanup(cluster, decode_scratch, prefill_scratch, session)?;
        return Ok("stop".into());
    }

    let mut finish_reason: &str = "length";
    let mut last_token = first_next;
    // See non-streaming path for the MIN_RESPONSE_TOKENS rationale.
    const MIN_RESPONSE_TOKENS: usize = 24;
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS,
    // so the model is free to stop but strongly disincentivised. The model's
    // EOS logit on multi-turn prompts sits ~1.3 nats above the next candidate;
    // -3.0 flips that to a solid discount without banning stop.
    const STOP_BIAS: f32 = 3.0;
    for step in 1..params.max_tokens as usize {
        // T4.1: same relax-stop-mask behaviour as the non-streaming path.
        let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
        forward_one_token_pp_logits(
            model,
            &mut session,
            cluster,
            &mut decode_scratch,
            last_token,
            prompt_ids.len() + step,
            &mut logits_buf,
        )
        .context("decode step logits")?;
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    if force_mask {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else {
                        logits_buf[sid as usize] -= STOP_BIAS;
                    }
                }
            }
        }
        let next = sampler.sample(&logits_buf, sampling, &generated);
        let alive = push_and_emit(next, &mut generated, &mut emitted_text)?;
        last_token = next;
        if !alive {
            finish_reason = "stop";
            break;
        }
    }

    cleanup(cluster, decode_scratch, prefill_scratch, session)?;

    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens,
        completion_tokens = generated.len() as u32,
        finish_reason,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "streaming completion finished"
    );

    Ok(finish_reason.to_owned())
}

fn cleanup(
    cluster: &HipCluster,
    decode_scratch: ShardedForwardOneTokenScratch,
    prefill_scratch: ShardedForwardPrefillScratch,
    session: Qwen3MoEShardedSession,
) -> Result<()> {
    decode_scratch
        .dispose(cluster)
        .context("dispose decode scratch")?;
    prefill_scratch
        .dispose(cluster)
        .context("dispose prefill scratch")?;
    session.dispose(cluster).context("dispose session")?;
    Ok(())
}

fn finalise(
    state: &ServerState,
    prompt_tokens: u32,
    mut generated: Vec<u32>,
    reason: &str,
) -> Result<(String, u32, u32, String)> {
    // Strip ALL stop tokens (eos, <|im_end|>, etc.) from decoded text so the
    // client sees clean content. Raw count preserved for `usage` honesty.
    // C6: `retain` mutates in place instead of allocating a second Vec.
    let stop_ids = &state.tokenizer.stop_ids;
    let completion_tokens = generated.len() as u32;
    generated.retain(|t| !stop_ids.contains(t));
    let text = state.tokenizer.decode(&generated).context("decode")?;
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
