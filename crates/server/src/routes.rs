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
use flambeau_qwen3_moe::Qwen3MoEConfig;
use flambeau_quant::{ChatTemplate, GgufTokenizer};
use flambeau_runtime::json_grammar::JsonState;
use flambeau_runtime::Sampler;
use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

use crate::api::*;
use crate::gpu_sampler::{self, GpuSamplerScratch};
use crate::model::{
    decode_keep_logits_on_device, decode_logits, decode_spec_pp, decode_spec_pp_sampling,
    prefill_logits, Inflight, LoadedModel, SpecDecodePp,
};
use crate::state::{parse_stop, SamplingParams};

/// Server-wide shared state — built once at startup.
pub struct ServerState {
    pub model_id: String,
    pub cfg: Qwen3MoEConfig,
    /// **TP-5a-i2** — PP or TP loaded model. Handlers dispatch via the
    /// `crate::model::{prefill_logits, decode_logits}` helpers; they
    /// don't need to inspect this variant directly.
    pub model: LoadedModel,
    /// **TP-5a-i2** — `Arc` so the TP variant's `BarP2pAllReduce` can
    /// hold a peer reference to the same cluster the handlers borrow
    /// from.
    pub cluster: Arc<HipCluster>,
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
    /// L3 — tool-call format detected at boot from the GGUF chat
    /// template. `general.architecture=qwen35moe` alone is not enough
    /// to decide: the Unsloth UD Qwen3.6 GGUFs ship a Coder-XML
    /// template even though the arch tag says `qwen35moe`. Honoured
    /// when a request omits `tool_call_format` or sets it to "auto".
    pub tool_call_format_default: crate::tool_call_parser::ToolCallFormat,
    /// Sampling defaults read from the GGUF (`general.sampling.*`).
    /// Filled into omitted request fields by
    /// [`crate::state::SamplingParams::from_parts`]. Values that the
    /// GGUF doesn't carry stay `None`; the OpenAI-shaped fallback in
    /// `from_parts` then takes effect.
    pub model_defaults: crate::state::ModelDefaults,
    /// **P0.5** — fallback system prompt injected when the request
    /// carries no `role: "system"` message. Sourced at boot from
    /// `FLAMBEAU_DEFAULT_SYSTEM` (env), or `None` if unset. Some clients
    /// (Aider, plain `curl`, the embedded UI) routinely send only a
    /// `user` message, and Qwen3.6 then degrades into terse one-line
    /// replies because the chat template's neutral default doesn't
    /// frame the assistant role. Operator-controlled — never derived
    /// from request fields.
    pub default_system: Option<String>,
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
)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(raw): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    // Sampler-G debug — when FLAMBEAU_DUMP_RAW_REQ=1 is set, log the
    // complete JSON body the client sent (incl. fields flambeau
    // doesn't parse like `tools[]` if the client sent them under a
    // different shape). One log line per request, full body, no
    // truncation. Off by default — bodies can be 10s of KB.
    if std::env::var("FLAMBEAU_DUMP_RAW_REQ").is_ok() {
        let s = serde_json::to_string(&raw).unwrap_or_else(|_| "<serialise fail>".into());
        tracing::info!(
            target: "server.req.raw",
            body_bytes = s.len(),
            body = %s,
            "raw chat completions body"
        );
    }
    // Always log the top-level keys the client sent, even when not
    // dumping the full body — gives us a one-line answer to "did
    // OpenWebUI send tools/tool_choice/etc.?".
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

    // Sampler-G debug — emit the request envelope as a single
    // structured INFO line so the server log shows what the client
    // sent without needing FLAMBEAU_DUMP_PROMPT for the full prompt
    // (which is verbose). Last user message is truncated to 200 chars.
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

    // P0.5 — inject the operator-configured default system prompt when
    // the request carries no system turn. Single source of truth lives
    // on `ServerState`; per-request override is a system message in
    // the request, which silently wins (we never overwrite). Empty
    // body or whitespace-only content is treated as "no system turn"
    // so a client passing `[{"role":"system","content":""}, …]` still
    // gets the default.
    let has_real_system = req.messages.iter().any(|m| {
        m.role == "system" && !m.content_str().trim().is_empty()
    });
    let injected_system: Option<ChatMessage> =
        if !has_real_system {
            state
                .default_system
                .as_deref()
                .map(|sys| ChatMessage {
                    role: "system".into(),
                    content: Some(sys.to_owned()),
                    tool_call_id: None,
                    tool_calls: None,
                })
        } else {
            None
        };

    // Qwen3.6's chat template prepends `<think>\n\n</think>\n\n` to the
    // current-turn assistant prefix (the "no-thinking" delimiter). Prior
    // assistant turns sent by the client are stored as plain content and,
    // when re-rendered, produce a prompt that's out-of-distribution — the
    // model greedy-stops on the first token. Normalise every assistant
    // message that lacks `</think>` by wrapping its content the same way
    // the template would for the current turn.
    let mut messages: Vec<ChatMessage> = injected_system
        .into_iter()
        .chain(req.messages.iter().map(normalise_message))
        .collect();

    // T1.2 + M2.1: thread client tools + `--mcp`-discovered remote
    // tools into the Jinja render. Both visible to the model under one
    // namespace; remote tools are `alias.name`-prefixed.
    let merged_tools = merge_request_and_remote_tools(
        req.tools.as_deref(),
        &state.remote_tools,
    );

    // T4.b.2 / T4.b.1 sampler config.
    // P0.1: detect json_object response format and propagate.
    let json_mode = matches!(
        req.response_format.as_ref(),
        Some(crate::api::ResponseFormat::JsonObject)
            | Some(crate::api::ResponseFormat::JsonSchema { .. }),
    );
    // P0.2: parse OpenAI `stop` (string-or-array-of-strings, max 4).
    let stop_strings = parse_stop(req.stop.as_ref());
    // P1.7 — collect_logprobs is `Some(n)` when the request opts in.
    // `top_logprobs` implies `logprobs=true`. `logprobs=true` without
    // an explicit `top_logprobs` defaults to 0 (chosen-token only).
    let collect_logprobs: Option<u32> = if req.logprobs.unwrap_or(false)
        || req.top_logprobs.is_some()
    {
        Some(req.top_logprobs.unwrap_or(0))
    } else {
        None
    };
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
        json_mode,
        stop_strings,
        collect_logprobs,
        &state.model_defaults,
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
        // P0.3 — `stream_options.include_usage=true` opts the client in
        // to the canonical OpenAI separate `usage` final chunk.
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
    // P1.7 — logprobs of the FINAL iteration (the one whose text is
    // surfaced as `choices[0].message.content`). Only populated when
    // the request opted in.
    let mut final_logprobs: Option<Vec<ChatLogProbContent>> = None;
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

        let (text, iter_prompt_tokens, iter_completion_tokens, iter_finish, iter_logprobs) =
            run_completion(state.clone(), &prompt, params.clone(), relax_stop_mask)
                .await
                .map_err(ApiError::internal)?;
        sum_prompt_tokens = sum_prompt_tokens.saturating_add(iter_prompt_tokens);
        sum_completion_tokens = sum_completion_tokens.saturating_add(iter_completion_tokens);

        let (content, mut tool_calls) = {
            use crate::tool_call_parser::{dispatcher, split_events, ParserEvent};
            let mut parser = dispatcher(
                req.tool_call_format.as_deref(),
                state.tool_call_format_default,
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
            // P1.7 — only this break path surfaces user-visible text;
            // the tool-call paths emit JSON tool args without
            // user-visible content. Logprobs are most useful for the
            // text path so attach here.
            final_logprobs = iter_logprobs;
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
    let stop_strings = parse_stop(req.stop.as_ref());
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
        /*json_mode=*/ false,
        stop_strings,
        /*collect_logprobs=*/ None,
        &state.model_defaults,
    );

    // P1.6c — OpenAI suffix-style FIM. When the request carries a
    // `suffix` AND the model has FIM specials, route through the
    // shared FIM-id assembly used by /infill. Without `suffix` the
    // path is unchanged.
    let suffix_fim = req.suffix.as_deref().filter(|s| !s.is_empty());
    let (text, prompt_tokens, completion_tokens, finish) = if let (Some(suffix), Some(fim)) =
        (suffix_fim, state.tokenizer.fim)
    {
        let prompt_ids = build_fim_prompt_ids(
            &state.tokenizer,
            fim,
            &req.prompt,
            suffix,
            None,
            &[],
        )
        .map_err(ApiError::internal)?;
        let prompt_tokens = prompt_ids.len() as u32;
        // FIM has no chat template — bypass chat-flavoured stop-mask.
        let (text, _, completion_tokens, finish, _) = run_completion_ids(
            state.clone(),
            prompt_ids,
            params,
            /*relax_stop_mask=*/ true,
        )
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
        // Legacy path: text-only prompt, chat stop-mask policy.
        let (t, p, c, f, _) =
            run_completion(state.clone(), &req.prompt, params, /*relax_stop_mask=*/ false)
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
        },],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// **P1.7** — build one `ChatLogProbContent` entry for a single decoded
/// step. Calls [`flambeau_runtime::sampling::build_distribution`] to
/// reproduce the same penalty + temperature + top-k/top-p/min-p
/// transforms the sampler applied, then extracts the chosen token's
/// log-probability and the top-`top_n` alternatives.
///
/// Returns `None` if the chosen token is outside the post-filter
/// distribution (defensive — shouldn't happen because the sampler
/// drew from the same distribution). Logprobs below -100 are clamped
/// to -100, matching OpenAI's surface.
fn build_logprob_entry(
    tokenizer: &flambeau_quant::GgufTokenizer,
    logits: &[f32],
    sampling: &flambeau_runtime::Sampling,
    history: &[u32],
    chosen: u32,
    top_n: usize,
) -> Option<ChatLogProbContent> {
    use flambeau_runtime::sampling::build_distribution;
    let dist = build_distribution(logits, sampling, history);
    if dist.is_empty() {
        return None;
    }
    // Locate the chosen token's prob; if missing (filtered out), the
    // sampler couldn't have picked it, so skip the entry.
    let chosen_prob = dist
        .iter()
        .find(|(id, _)| *id == chosen)
        .map(|(_, p)| *p)?;
    let chosen_logprob = log_clamped(chosen_prob);

    // Top alternatives are already in the front of `dist` if it was
    // sorted (when filters were active). When filters are off,
    // `build_distribution` returns an unsorted full-vocab list — sort
    // a partial copy. Skip the chosen token from the alternatives;
    // OpenAI-spec lists only OTHER candidates here.
    let mut alts: Vec<(u32, f32)> = dist
        .iter()
        .filter(|(id, _)| *id != chosen)
        .copied()
        .collect();
    if alts.len() > top_n {
        alts.select_nth_unstable_by(top_n, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        alts.truncate(top_n);
    }
    alts.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let chosen_text = tokenizer.decode(&[chosen]).unwrap_or_default();
    let chosen_bytes = chosen_text.as_bytes().to_vec();

    let top_logprobs: Vec<TopLogProb> = alts
        .into_iter()
        .map(|(id, p)| {
            let t = tokenizer.decode(&[id]).unwrap_or_default();
            let bytes = t.as_bytes().to_vec();
            TopLogProb {
                token: t,
                logprob: log_clamped(p),
                bytes,
            }
        })
        .collect();

    Some(ChatLogProbContent {
        token: chosen_text,
        logprob: chosen_logprob,
        bytes: chosen_bytes,
        top_logprobs,
    })
}

/// log(p) clamped to OpenAI's [-100, 0] surface. Zero-prob → -100.
fn log_clamped(p: f32) -> f32 {
    if p <= 0.0 {
        -100.0
    } else {
        p.ln().max(-100.0)
    }
}

/// Assemble a PSM-shaped FIM token stream from prefix / suffix / middle
/// fragments. Returns the full prompt-id vector ready for the engine.
///
/// Shared by `/infill` (P1.6b) and `/v1/completions?suffix=…` (P1.6c).
/// `extra` carries optional repo-context files (Qwen-Coder PSM); silently
/// skipped when the vocab lacks `<|repo_name|>` / `<|file_sep|>`.
fn build_fim_prompt_ids(
    tok: &flambeau_quant::GgufTokenizer,
    fim: flambeau_quant::FimTokens,
    prefix: &str,
    suffix: &str,
    middle: Option<&str>,
    extra: &[crate::api::InfillExtra],
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

/// POST /infill — llama.cpp-compatible Fill-in-the-Middle endpoint.
///
/// Composes a PSM-shaped FIM prompt of the form
/// `[<|repo_name|>name<|file_sep|>body…]<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{prompt}`
/// using the FIM specials detected at boot (P1.6a). The leading repo
/// block is omitted when `input_extra` is empty or the vocab lacks
/// `<|repo_name|>` / `<|file_sep|>`.
///
/// The `text` returned in the OpenAI-shaped response carries ONLY the
/// generated middle — caller is expected to splice it back at the
/// cursor between `input_prefix` and `input_suffix`.
#[tracing::instrument(name = "server.infill", skip_all, fields(stream = req.stream))]
pub async fn infill(
    State(state): State<SharedState>,
    Json(req): Json<InfillRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
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
        req.temperature,
        req.top_p,
        req.top_k,
        /*min_p=*/ None,
        /*repetition_penalty=*/ None,
        /*presence_penalty=*/ None,
        /*frequency_penalty=*/ None,
        req.n_predict,
        req.seed,
        /*json_mode=*/ false,
        stop_strings,
        /*collect_logprobs=*/ None,
        &state.model_defaults,
    );

    // FIM completions live entirely outside the chat template — there
    // are no `<|im_end|>` markers to mask early. Pass relax_stop_mask
    // so the engine doesn't apply chat-flavoured biases.
    let prompt_tokens = prompt_ids.len() as u32;
    let (text, _, completion_tokens, finish, _) =
        run_completion_ids(state.clone(), prompt_ids, params, /*relax_stop_mask=*/ true)
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
    include_usage: bool,
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
        // Build the parser once per request. Default format is
        // boot-detected from the chat template (L3); the request may
        // override explicitly.
        let parser_result = dispatcher(
            tool_call_format.as_deref(),
            state_clone.tool_call_format_default,
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
                Ok((r, _, _)) => r.as_str(),
                Err(_) => "error",
            }
        };
        // L3 — emit `usage` on the final chunk so the chat UI can
        // compute prefill / token-generation rates. Token counts come
        // from run_completion_blocking_streaming's tuple return.
        let (prompt_tokens, completion_tokens) = match &res {
            Ok((_, p, c)) => (*p, *c),
            Err(_) => (0u32, 0u32),
        };
        // Finish chunk. When `include_usage=true`, the canonical OpenAI
        // shape carries `usage: null` here and ships the real usage on
        // the next (choices=[]) chunk; that's the form expected by the
        // openai-python SDK and LangChain. When false, embed `usage`
        // inline so existing clients that only watch the finish chunk
        // still get token counts.
        let done_frame = if include_usage {
            json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": serde_json::Value::Null,
            })
        } else {
            json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens.saturating_add(completion_tokens),
                },
            })
        };
        let _ = tx_clone.blocking_send(Ok(Event::default().data(done_frame.to_string())));

        // P0.3 — canonical separate usage chunk. Emitted strictly after
        // the finish chunk and strictly before [DONE], with choices=[]
        // so clients that match on `choices[0].finish_reason` don't
        // double-fire.
        if include_usage {
            let usage_frame = json!({
                "id": id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_clone,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens.saturating_add(completion_tokens),
                },
            });
            let _ = tx_clone.blocking_send(Ok(Event::default().data(usage_frame.to_string())));
        }

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
/// Engine return shape: text, prompt_tokens, completion_tokens,
/// finish_reason, optional per-token logprobs (P1.7).
type CompletionOutput = (String, u32, u32, String, Option<Vec<ChatLogProbContent>>);

async fn run_completion(
    state: SharedState,
    prompt: &str,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    let prompt = prompt.to_owned();
    tokio::task::spawn_blocking(move || {
        let ids = state
            .tokenizer
            .encode(&prompt)
            .context("tokenize prompt")?;
        run_completion_blocking_ids(state, ids, params, relax_stop_mask)
    })
    .await
    .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

/// **P1.6b** — token-id entry point used by the FIM `/infill` route.
/// Skips the tokenizer string round-trip so callers that build prompts
/// directly out of pre-tokenized fragments (e.g., FIM specials wrapping
/// user prefix/suffix) don't depend on the tokenizer's special-token
/// auto-registration heuristic.
async fn run_completion_ids(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    tokio::task::spawn_blocking(move || {
        run_completion_blocking_ids(state, prompt_ids, params, relax_stop_mask)
    })
    .await
    .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

fn run_completion_blocking_ids(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    let request_start = Instant::now();

    // Serialise: one forward at a time through this server instance.
    let _guard = state
        .inflight
        .blocking_lock();

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

    let cluster: &HipCluster = &state.cluster;
    let model = &state.model;

    // Fresh session per request (no conversation state reuse in V1).
    let mut inflight =
        Inflight::new(model, cluster, prompt_ids.len()).context("create inflight session")?;

    // Sampler holds vocab-sized scratch reused across all decode steps
    // (C2 in RUST-PERF-CORRECTIONS.md). Reserve upfront to avoid the
    // first-token grow.
    let mut sampler = Sampler::from_seed(params.seed);
    sampler.reserve(state.cfg.vocab_size);
    let sampling = &params.sampling;
    let stop_ids = &state.tokenizer.stop_ids;
    let always_stop_ids = &state.tokenizer.always_stop_ids;
    let vocab = state.cfg.vocab_size;
    // Always allocate the logit buffer — we need it for the first-token
    // EOS mask regardless of sampling mode.
    let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);

    // **Sampler-D3 / D4** — GPU-side top-K sampler. Opts in via
    // `FLAMBEAU_GPU_SAMPLER=1` for any non-greedy TP request; the
    // penalty-active path goes through `run_gpu_topk_with_penalties`
    // (D4 — applies repetition / presence / frequency on device
    // before topk) instead of `run_gpu_topk` (D3 — bare topk).
    let use_gpu_sampler = std::env::var("FLAMBEAU_GPU_SAMPLER").is_ok()
        && matches!(model, LoadedModel::Tp { .. })
        && !sampling.is_greedy();
    let mut gpu_scratch: Option<GpuSamplerScratch> = if use_gpu_sampler {
        let head_rank = match &inflight {
            Inflight::Tp { decode, .. } => decode.head_rank.0 as usize,
            _ => 0,
        };
        Some(
            // K=2048 matches Sampler-A's `effective_top_k` default
            // for `top_p`/`min_p` callers without explicit `top_k`.
            // Smaller K caused the GPU sampler to bias multinomial
            // toward EOS at natural-endpoint positions.
            GpuSamplerScratch::new(cluster, head_rank, 2048)
                .context("alloc GpuSamplerScratch")?,
        )
    } else {
        None
    };

    // Prefill. Always download logits so we can mask stop tokens on the
    // first generated token — Qwen3.6 sometimes argmaxes `<|im_end|>` as
    // the first response token on multi-turn prompts, producing an empty
    // reply. Suppress it until at least one content token is emitted.
    let prefill_start = Instant::now();
    prefill_logits(model, cluster, &mut inflight, &prompt_ids, &mut logits_buf)
        .context("prefill logits")?;
    for &sid in stop_ids {
        if (sid as usize) < logits_buf.len() {
            logits_buf[sid as usize] = f32::NEG_INFINITY;
        }
    }
    // First token: ALWAYS goes through the host path. Prefill writes
    // logits to its own local scratch (disposed at end of prefill); the
    // decode scratch's `output_head.logits_f32` is uninitialised at
    // this point. The GPU sampler path activates from the SECOND token
    // onwards once the decode forward populates the right buffer.
    // (One-shot first token doesn't matter perf-wise; D3 Phase B's
    // bigger win is the per-step decode DtoH-skip.)
    let inv_temp = if sampling.temperature > 0.0 {
        1.0 / sampling.temperature
    } else {
        1.0
    };
    let first_next = sampler.sample(&logits_buf, sampling, &[]);
    let is_greedy = sampling.is_greedy();
    tracing::info!(
        target: "server.completion.first_token",
        prompt_tokens,
        ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
        greedy = is_greedy,
        topology = model.topology(),
        "first token produced (time-to-first-token)"
    );

    // **P1.7** — accumulate logprobs when the request opted in AND
    // the path supports it (host sampler only; GPU sampler / spec-
    // decode silently disable). `top_logprobs` may be 0 → just the
    // chosen token's logprob.
    let mut logprobs_acc: Option<Vec<ChatLogProbContent>> =
        if params.collect_logprobs.is_some() && !use_gpu_sampler {
            Some(Vec::with_capacity(params.max_tokens as usize))
        } else {
            if params.collect_logprobs.is_some() {
                tracing::warn!(
                    target: "server.logprobs",
                    "logprobs requested but path is GPU sampler — returning null logprobs"
                );
            }
            None
        };
    if let Some(lp) = logprobs_acc.as_mut() {
        if let Some(entry) = build_logprob_entry(
            &state.tokenizer,
            &logits_buf,
            sampling,
            &[],
            first_next,
            params.collect_logprobs.unwrap_or(0) as usize,
        ) {
            lp.push(entry);
        }
    }

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let is_stop = |t: u32| stop_ids.contains(&t);

    // **P0.1** — JSON-grammar state. Active only when the request set
    // `response_format: {"type": "json_object"}`. Each chosen token's
    // bytes advance the state; the GPU-sampler path also masks
    // candidates against this state before the multinomial draw.
    let mut json_state: Option<JsonState> =
        if params.json_mode { Some(JsonState::new()) } else { None };
    if let Some(js) = json_state.as_mut() {
        if let Ok(text) = state.tokenizer.decode(&[first_next]) {
            let _ = js.feed_slice(text.as_bytes());
        }
    }

    generated.push(first_next);
    // First-token stop mask means `is_stop(first_next)` cannot fire here,
    // but we keep the check as a defensive guard for future logit-mask
    // changes.
    if is_stop(first_next) {
        inflight
            .dispose(cluster, model)
            .context("dispose inflight (early-stop)")?;
        return finalise(&state, prompt_tokens, generated, "stop", &params.stop_strings, logprobs_acc);
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
    //
    // **Sampler-G (2026-04-30)** — was 24, lowered to 8. With Qwen3.6-27B
    // at temp=0.7+top_p=0.8 the 24-token floor forced the model to keep
    // generating 13+ tokens past natural endpoints like
    // "Hello! How can I help you today?" (~10 tokens), at which point it
    // wandered into reasoning-marker leaks (`</think>`, `<end_thought>`),
    // hallucinated chat formats (`<|user|>\n<|assistant|>`), or duplicate-
    // the-response loops. 8 lets short greetings stop naturally; the
    // first-token NEG_INFINITY mask still prevents immediate-EOS on
    // multi-turn prompts.
    const MIN_RESPONSE_TOKENS: usize = 24;
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS.
    // CN-80B-18 — was 0.5 (was 3.0 before that). Even 0.5 is enough to push
    // EOS below the next-best continuation when the model wants to stop at
    // the end of a paragraph; under top_k=20 sampling the next-best is
    // typically "regenerate the paragraph" → whole-block repetition loops
    // (observed live on Coder-Next acknowledgement responses). The
    // first-token NEG_INFINITY mask + MIN_RESPONSE_TOKENS=24 hard mask are
    // sufficient on their own to prevent immediate-EOS failure modes;
    // beyond that, do not bias the model's natural stopping decision.
    // **Sampler-G follow-up (2026-04-30)** — non-zero STOP_BIAS subtracts
    // from every stop-token logit beyond MIN_RESPONSE_TOKENS. Was 0.0;
    // raised to 3.0 because Qwen3.6-27B at default sampling
    // (temp=1.0 / top_p=0.95 / top_k=20 — what Open-WebUI sends when
    // the user hasn't set an override) picks `<|im_end|>` at the
    // first natural sentence break, e.g. "I will include the full
    // implementation below." → STOP, instead of actually delivering
    // the implementation. 1.5 nats was insufficient at temp=1.0;
    // 3.0 is enough to keep EOS below the next-best continuation at
    // most natural endpoints. The prior CN-80B-18 concern about `0.5
    // → repeat loops` was driven by Coder-Next-80B specifically;
    // Qwen3.6 doesn't show that failure at this bias on chat tests.
    const STOP_BIAS: f32 = 3.0;
    // MTP-5d/5g/h: spec-decode fast path. Active when MTP head is loaded
    // (FLAMBEAU_SPEC_MTP=path at startup). Greedy uses strict-match verify;
    // non-greedy uses vLLM-canonical rejection sampling. Penalties
    // (repetition / presence / frequency) are now applied to base AND
    // MTP distributions inside `build_distribution` via the threaded
    // `history` slice (MTP-5g/h #194), so penalty-active requests no
    // longer have to fall through.
    let spec_available = matches!(model, LoadedModel::Pp { mtp: Some(_), .. });
    let use_spec = spec_available;
    if use_spec {
        // h_for_mtp at first macro step = h@(prompt_len-1), which lives in
        // the prefill scratch's hidden_a buffer at offset (prompt_len-1)*row_bytes.
        let last_rank = cluster.ranks() - 1;
        let row_bytes = state.cfg.hidden_size * 2;
        let h_initial = match &inflight {
            Inflight::Pp { prefill, .. } => prefill.per_rank[last_rank]
                .hidden_a
                .offset_bytes((prompt_ids.len() - 1) * row_bytes),
            _ => bail!("spec-decode requires Inflight::Pp"),
        };
        let m = match model {
            LoadedModel::Pp { model, .. } => model,
            _ => bail!("spec-decode requires LoadedModel::Pp"),
        };
        let mut spec_state = SpecDecodePp::new(m, cluster).context("alloc SpecDecodePp")?;
        spec_state.h_for_mtp = h_initial;

        let mut accept_count = 0usize;
        let mut macro_count = 0usize;
        let mut position = prompt_ids.len();
        'spec_loop: while generated.len() < params.max_tokens as usize {
            let step = if is_greedy {
                decode_spec_pp(
                    model, cluster, &mut inflight, &mut spec_state, last_token, position,
                )
                .context("spec macro step (greedy)")?
            } else {
                decode_spec_pp_sampling(
                    model, cluster, &mut inflight, &mut spec_state, last_token, position,
                    sampling, sampler.rng_mut(), &generated,
                )
                .context("spec macro step (sampling)")?
            };
            macro_count += 1;
            if step.accepted { accept_count += 1; }
            for tok in step.committed.iter().copied() {
                generated.push(tok);
                last_token = tok;
                if is_stop(tok) {
                    finish_reason = "stop";
                    break 'spec_loop;
                }
                if generated.len() >= params.max_tokens as usize {
                    break 'spec_loop;
                }
            }
            position = step.new_position;
        }
        let accept_pct = if macro_count == 0 {
            0.0
        } else {
            100.0 * accept_count as f64 / macro_count as f64
        };
        tracing::info!(
            target: "server.spec_decode",
            macro_steps = macro_count,
            tokens = generated.len() as u32,
            accept_pct,
            "spec-decode loop complete"
        );
        spec_state.dispose(cluster).ok();
    } else {
        for step in 1..params.max_tokens as usize {
            let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
            // **Sampler-D3 Phase B** — GPU sampler path skips the
            // 600 KB host-logits DtoH entirely; logits stay on device
            // and `run_gpu_topk` consumes them via topk_softmax_f32.
            // Host path keeps the existing `decode_logits` DtoH so
            // penalty / non-TP / fallback callers still get host
            // logits.
            let next = if let Some(scratch) = gpu_scratch.as_mut() {
                decode_keep_logits_on_device(
                    model,
                    cluster,
                    &mut inflight,
                    last_token,
                    prompt_ids.len() + step,
                )
                .context("decode step keep-on-device")?;
                if sampling.has_penalties() {
                    // D4 — apply penalties on GPU before topk.
                    gpu_sampler::run_gpu_topk_with_penalties(
                        model,
                        cluster,
                        &inflight,
                        scratch,
                        &generated,
                        sampling,
                        inv_temp,
                    )
                    .context("decode-step GPU topk (with penalties)")?;
                } else {
                    gpu_sampler::run_gpu_topk(
                        model, cluster, &inflight, scratch, inv_temp,
                    )
                    .context("decode-step GPU topk")?;
                }
                if force_mask && !relax_stop_mask {
                    gpu_sampler::apply_stop_mask(
                        &scratch.host_ids,
                        &mut scratch.host_probs,
                        stop_ids,
                    );
                }
                // P0.1 — JSON-grammar mask before multinomial.
                if let Some(js) = json_state.as_ref() {
                    gpu_sampler::apply_json_mask(
                        js,
                        &state.tokenizer,
                        &scratch.host_ids,
                        &mut scratch.host_probs,
                    );
                }
                sampler.sample_from_topk(
                    &scratch.host_ids,
                    &scratch.host_probs,
                    sampling,
                )
            } else {
                decode_logits(
                    model,
                    cluster,
                    &mut inflight,
                    last_token,
                    prompt_ids.len() + step,
                    &mut logits_buf,
                )
                .context("decode step logits")?;
                if !relax_stop_mask {
                    for &sid in stop_ids {
                        if (sid as usize) < logits_buf.len() {
                            // Sampler-G — `<think>` / `</think>` always
                            // get NEG_INFINITY, even outside the early
                            // window: the chat template ran with
                            // `enable_thinking=false` so the model
                            // should never emit them.
                            if always_stop_ids.contains(&sid) {
                                logits_buf[sid as usize] = f32::NEG_INFINITY;
                            } else if force_mask {
                                logits_buf[sid as usize] = f32::NEG_INFINITY;
                            } else {
                                logits_buf[sid as usize] -= STOP_BIAS;
                            }
                        }
                    }
                }
                // Pass `generated` as history so penalties can fire on
                // repeats / frequent tokens. T4.b.2 — without this,
                // Qwen3.5/3.6 agent loops degrade to long-CoT drift.
                let next = sampler.sample(&logits_buf, sampling, &generated);
                // P1.7 — collect per-token logprobs (host path only).
                if let Some(lp) = logprobs_acc.as_mut() {
                    if let Some(entry) = build_logprob_entry(
                        &state.tokenizer,
                        &logits_buf,
                        sampling,
                        &generated,
                        next,
                        params.collect_logprobs.unwrap_or(0) as usize,
                    ) {
                        lp.push(entry);
                    }
                }
                next
            };
            // P0.1 — advance JSON state with the chosen token's bytes.
            if let Some(js) = json_state.as_mut() {
                if let Ok(text) = state.tokenizer.decode(&[next]) {
                    let _ = js.feed_slice(text.as_bytes());
                }
            }
            generated.push(next);
            last_token = next;
            if is_stop(next) {
                finish_reason = "stop";
                break;
            }
            // **Sampler-G** — string-level stop on reasoning markers.
            // The model can route around the single-token `</think>`
            // mask by emitting the multi-token text form. Detokenize
            // the recent tail and stop if a leak is present. Final
            // response cleanup happens in `finalise`.
            //
            // **P0.2** — same mechanism extended to the per-request
            // `stop` strings. Tail window grows with the longest
            // user stop so multi-token caller stops are catchable.
            let user_stop_max = params
                .stop_strings
                .iter()
                .map(|s| s.len())
                .max()
                .unwrap_or(0);
            let need_user_check = !params.stop_strings.is_empty();
            if (need_user_check || !relax_stop_mask)
                && (step % 4 == 0 || step >= MIN_RESPONSE_TOKENS)
            {
                let n = generated.len();
                // 16 tokens covers ≥48 chars typical; widen if a user
                // stop string is longer than ~32 chars.
                let token_window = 16.max((user_stop_max / 2).min(64));
                let from = n.saturating_sub(token_window);
                if let Ok(tail) = state.tokenizer.decode(&generated[from..]) {
                    let marker_hit = !relax_stop_mask
                        && (tail.contains("</think>")
                            || tail.contains("<end_thought>")
                            || tail.contains("<end_think>")
                            || tail.contains("</thought>"));
                    let user_hit = params
                        .stop_strings
                        .iter()
                        .any(|s| tail.contains(s.as_str()));
                    if marker_hit || user_hit {
                        finish_reason = "stop";
                        break;
                    }
                }
            }
        }
    }

    // Dispose GPU sampler scratch (if allocated) before tearing down
    // the inflight session.
    if let Some(scratch) = gpu_scratch.take() {
        scratch.dispose(cluster).context("dispose GpuSamplerScratch")?;
    }

    // Dispose per-request scratch/session; keep model + cluster alive.
    inflight.dispose(cluster, model).context("dispose inflight")?;

    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens,
        completion_tokens = generated.len() as u32,
        finish_reason,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "completion request finished"
    );

    let result = finalise(
        &state,
        prompt_tokens,
        generated,
        finish_reason,
        &params.stop_strings,
        logprobs_acc,
    )?;
    // Sampler-G debug — emit the completed response text (head + tail
    // preview) so we can correlate request shape with what the model
    // actually produced. Truncated to 240 chars on each end so the
    // log line stays readable.
    let resp_preview = preview_text(&result.0, 240);
    tracing::info!(
        target: "server.resp",
        completion_tokens = result.2,
        finish_reason = %result.3,
        resp = %resp_preview,
        "chat_completions response"
    );
    Ok(result)
}

/// Truncate `text` to a head/tail preview suitable for log lines.
/// Replaces newlines with `\n` for single-line readability.
fn preview_text(text: &str, head: usize) -> String {
    let n_chars = text.chars().count();
    let escape = |s: &str| s.replace('\n', "\\n");
    if n_chars <= head * 2 + 20 {
        return escape(text);
    }
    let head_str: String = text.chars().take(head).collect();
    let tail_str: String = text.chars().skip(n_chars - head).collect();
    format!("{} … <{}c omitted> … {}", escape(&head_str), n_chars - head * 2, escape(&tail_str))
}

/// Streaming variant: pushes text deltas through `emit` as each token is
/// produced. Returns the finish reason on success.
/// Streaming variant. Returns `(finish_reason, prompt_tokens,
/// completion_tokens)` so the SSE producer can emit a final usage
/// chunk (L3 — pp/tg indicators in the chat UI).
fn run_completion_blocking_streaming(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    relax_stop_mask: bool,
    emit: &mut dyn FnMut(&str) -> bool,
) -> Result<(String, u32, u32)> {
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

    let cluster: &HipCluster = &state.cluster;
    let model = &state.model;
    let stop_ids = &state.tokenizer.stop_ids;
    let always_stop_ids = &state.tokenizer.always_stop_ids;
    let is_stop = |t: u32| stop_ids.contains(&t);

    let mut inflight =
        Inflight::new(model, cluster, prompt_ids.len()).context("create inflight session")?;

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
    prefill_logits(model, cluster, &mut inflight, &prompt_ids, &mut logits_buf)
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
        topology = model.topology(),
        "first token produced (time-to-first-token)"
    );

    // Rolling decode: re-decode the full generated list each step and emit
    // the UTF-8 suffix beyond what was already emitted. Handles multi-byte
    // tokens without surfacing partial codepoints to the client.
    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
    let mut emitted_text = String::new();
    // CN-80B-19 — incremental detokenize state. `decode_cursor` is the
    // index of the first token NOT yet decoded into clean emitted bytes.
    // `pending_emitted_in_segment` tracks how many bytes of the current
    // open segment (`generated[decode_cursor..]`) we've already streamed
    // to the client, so a re-decode after a multi-byte boundary doesn't
    // re-emit safe bytes. When the segment finishes cleanly (no
    // trailing U+FFFD), `decode_cursor` advances and the segment resets.
    //
    // Reduces per-step decode cost from O(generated.len()) to ~O(1) —
    // typical "open segment" is 1-3 tokens, only growing when a multi-
    // byte glyph straddles a BPE-token boundary.
    let mut decode_cursor: usize = 0;
    let mut pending_emitted_in_segment: usize = 0;

    let mut push_and_emit = |tok: u32,
                             generated: &mut Vec<u32>,
                             emitted_text: &mut String|
     -> Result<bool> {
        generated.push(tok);
        let stop_hit = is_stop(tok);
        // Decode only the still-open segment, not the full sequence.
        // `end` excludes the trailing stop token so its raw text never leaks.
        let end = if stop_hit { generated.len() - 1 } else { generated.len() };
        if end <= decode_cursor {
            return Ok(!stop_hit);
        }
        let raw = state
            .tokenizer
            .decode(&generated[decode_cursor..end])
            .context("decode")?;
        // Trim trailing U+FFFD: the HF BPE+ByteLevel decoder substitutes
        // it when the byte tail is an incomplete UTF-8 codepoint
        // (multi-byte glyph straddling two BPE tokens). The trimmed
        // bytes will resolve once the next token's bytes arrive.
        let safe = raw.trim_end_matches('\u{FFFD}');

        // Emit anything new within the current open segment.
        if safe.len() > pending_emitted_in_segment {
            let delta = &safe[pending_emitted_in_segment..];
            if !emit(delta) {
                return Ok(false);
            }
            emitted_text.push_str(delta);
            pending_emitted_in_segment = safe.len();
        }

        // If nothing was trimmed, the segment closed cleanly: advance
        // the cursor and reset segment-local emit accounting.
        if safe.len() == raw.len() {
            decode_cursor = end;
            pending_emitted_in_segment = 0;
        }
        Ok(!stop_hit)
    };

    let alive = push_and_emit(first_next, &mut generated, &mut emitted_text)?;
    if !alive {
        inflight
            .dispose(cluster, model)
            .context("dispose inflight (stream early-stop)")?;
        return Ok(("stop".into(), prompt_tokens, generated.len() as u32));
    }

    let mut finish_reason: &str = "length";
    let mut last_token = first_next;
    // See non-streaming path for the MIN_RESPONSE_TOKENS rationale.
    // Sampler-G — lowered from 24 to 8 to let short greetings stop
    // at natural endpoints instead of wandering into leak territory.
    const MIN_RESPONSE_TOKENS: usize = 24;
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS.
    // CN-80B-18 — was 0.5 (was 3.0 before that). Even 0.5 is enough to push
    // EOS below the next-best continuation when the model wants to stop at
    // the end of a paragraph; under top_k=20 sampling the next-best is
    // typically "regenerate the paragraph" → whole-block repetition loops
    // (observed live on Coder-Next acknowledgement responses). The
    // first-token NEG_INFINITY mask + MIN_RESPONSE_TOKENS=24 hard mask are
    // sufficient on their own to prevent immediate-EOS failure modes;
    // beyond that, do not bias the model's natural stopping decision.
    // **Sampler-G follow-up (2026-04-30)** — non-zero STOP_BIAS subtracts
    // from every stop-token logit beyond MIN_RESPONSE_TOKENS. Was 0.0;
    // raised to 3.0 because Qwen3.6-27B at default sampling
    // (temp=1.0 / top_p=0.95 / top_k=20 — what Open-WebUI sends when
    // the user hasn't set an override) picks `<|im_end|>` at the
    // first natural sentence break, e.g. "I will include the full
    // implementation below." → STOP, instead of actually delivering
    // the implementation. 1.5 nats was insufficient at temp=1.0;
    // 3.0 is enough to keep EOS below the next-best continuation at
    // most natural endpoints. The prior CN-80B-18 concern about `0.5
    // → repeat loops` was driven by Coder-Next-80B specifically;
    // Qwen3.6 doesn't show that failure at this bias on chat tests.
    const STOP_BIAS: f32 = 3.0;
    // CN-80B-22 — env-gated TP-decode profiling. When FLAMBEAU_PROFILE_DECODE
    // is set, enable HipEvent section recording for `n` warm-up-skipped decode
    // steps, then flush + dump aggregate per-section ms to stderr. Skips the
    // first 8 steps (cold-cache effects, allocator warmup).
    let profile_decode_n: usize = std::env::var("FLAMBEAU_PROFILE_DECODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let profile_skip: usize = 8;

    // MTP-5d streaming + 5g/h penalty-aware: spec-decode SSE path. Active
    // when MTP head is loaded. Penalties applied in build_distribution via
    // the threaded `&generated` history. Mirrors the non-streaming branch.
    let spec_available = matches!(model, LoadedModel::Pp { mtp: Some(_), .. });
    let use_spec = spec_available;
    if use_spec {
        let last_rank = cluster.ranks() - 1;
        let row_bytes = state.cfg.hidden_size * 2;
        let h_initial = match &inflight {
            Inflight::Pp { prefill, .. } => prefill.per_rank[last_rank]
                .hidden_a
                .offset_bytes((prompt_ids.len() - 1) * row_bytes),
            _ => bail!("spec-decode requires Inflight::Pp"),
        };
        let m = match model {
            LoadedModel::Pp { model, .. } => model,
            _ => bail!("spec-decode requires LoadedModel::Pp"),
        };
        let mut spec_state = SpecDecodePp::new(m, cluster).context("alloc SpecDecodePp")?;
        spec_state.h_for_mtp = h_initial;

        let mut accept_count = 0usize;
        let mut macro_count = 0usize;
        let mut position = prompt_ids.len();
        'spec_stream_loop: while generated.len() < params.max_tokens as usize {
            let step_result = if is_greedy {
                decode_spec_pp(
                    model, cluster, &mut inflight, &mut spec_state, last_token, position,
                )
                .context("spec macro step (greedy, streaming)")
            } else {
                decode_spec_pp_sampling(
                    model, cluster, &mut inflight, &mut spec_state, last_token, position,
                    sampling, sampler.rng_mut(), &generated,
                )
                .context("spec macro step (sampling, streaming)")
            };
            let step = match step_result {
                Ok(s) => s,
                Err(e) => {
                    spec_state.dispose(cluster).ok();
                    return Err(e);
                }
            };
            macro_count += 1;
            if step.accepted { accept_count += 1; }
            for tok in step.committed.iter().copied() {
                let alive = push_and_emit(tok, &mut generated, &mut emitted_text)?;
                last_token = tok;
                if !alive {
                    finish_reason = "stop";
                    break 'spec_stream_loop;
                }
                if generated.len() >= params.max_tokens as usize {
                    break 'spec_stream_loop;
                }
            }
            position = step.new_position;
        }
        let accept_pct = if macro_count == 0 {
            0.0
        } else {
            100.0 * accept_count as f64 / macro_count as f64
        };
        tracing::info!(
            target: "server.spec_decode",
            macro_steps = macro_count,
            tokens = generated.len() as u32,
            accept_pct,
            stream = true,
            "spec-decode SSE loop complete"
        );
        spec_state.dispose(cluster).ok();

        inflight
            .dispose(cluster, model)
            .context("dispose inflight (stream end)")?;
        tracing::info!(
            target: "server.completion.finish",
            prompt_tokens,
            completion_tokens = generated.len() as u32,
            finish_reason,
            total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            stream = true,
            "streaming completion finished (spec)"
        );
        return Ok((finish_reason.into(), prompt_tokens, generated.len() as u32));
    }

    for step in 1..params.max_tokens as usize {
        if profile_decode_n > 0 {
            if step == profile_skip + 1 {
                flambeau_backend_hip::profile::enable();
                tracing::info!(target: "server.profile", "enabled decode profiling");
            } else if step == profile_skip + profile_decode_n + 1 {
                match flambeau_backend_hip::profile::flush() {
                    Ok(stats) => {
                        let total: f32 = stats.iter().map(|s| s.total_ms).sum();
                        let mut msg = String::from(
                            "\n=== TP decode profile ===\nsection                    total_ms     count    mean_ms  ms/token\n",
                        );
                        for s in &stats {
                            msg.push_str(&format!(
                                "{:<24}  {:>10.2}  {:>8}  {:>10.4}  {:>9.3}\n",
                                s.name,
                                s.total_ms,
                                s.count,
                                s.mean_ms,
                                s.total_ms / profile_decode_n as f32
                            ));
                        }
                        msg.push_str(&format!(
                            "{:<24}  {:>10.2}  {:>8}  {:>10}  {:>9.3}\n",
                            "TOTAL_RECORDED",
                            total,
                            "-",
                            "-",
                            total / profile_decode_n as f32
                        ));
                        eprintln!("{}", msg);
                    }
                    Err(e) => tracing::error!(target: "server.profile", "flush: {e}"),
                }
            }
        }
        // T4.1: same relax-stop-mask behaviour as the non-streaming path.
        let force_mask = step < MIN_RESPONSE_TOKENS && !relax_stop_mask;
        decode_logits(
            model,
            cluster,
            &mut inflight,
            last_token,
            prompt_ids.len() + step,
            &mut logits_buf,
        )
        .context("decode step logits")?;
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    // Sampler-G — always_stop_ids (`<think>`/`</think>`)
                    // get NEG_INFINITY in every step, even outside the
                    // early-window mask. The model should never emit
                    // them when `enable_thinking=false`.
                    if always_stop_ids.contains(&sid) {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else if force_mask {
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
        // **Sampler-G** string-level stop. The model can route around
        // single-token `</think>` masks by emitting the multi-token
        // text form (`</`, `think`, `>` or `<end`, `_thought`, `>`).
        // Detect on the running emitted text and cut cleanly. Cheap:
        // string contains check on the trailing window only. Use a
        // char-safe slice — `emitted_text.len() - 64` can land inside
        // a multi-byte UTF-8 codepoint (e.g. `’` at byte 805..808),
        // which would panic. Walk back to the nearest char boundary.
        // P0.2 — user-supplied stop sequences also use this tail-window
        // detector. Window size grows with the longest user stop so
        // multi-codepoint caller stops are catchable.
        let user_stop_max = params
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        if !relax_stop_mask || !params.stop_strings.is_empty() {
            let window_bytes = 64.max(user_stop_max + 16);
            let tail_window = if emitted_text.len() > window_bytes {
                let mut start = emitted_text.len() - window_bytes;
                while start < emitted_text.len() && !emitted_text.is_char_boundary(start) {
                    start += 1;
                }
                &emitted_text[start..]
            } else {
                emitted_text.as_str()
            };
            let marker_hit = !relax_stop_mask
                && (tail_window.contains("</think>")
                    || tail_window.contains("<end_thought>")
                    || tail_window.contains("<end_think>")
                    || tail_window.contains("</thought>"));
            let user_hit = params
                .stop_strings
                .iter()
                .any(|s| tail_window.contains(s.as_str()));
            if marker_hit || user_hit {
                finish_reason = "stop";
                tracing::info!(
                    target: "server.completion.string_stop",
                    user_stop = user_hit,
                    "string-level stop on reasoning-marker or user-stop leak"
                );
                break;
            }
        }
    }

    inflight
        .dispose(cluster, model)
        .context("dispose inflight (stream end)")?;

    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens,
        completion_tokens = generated.len() as u32,
        finish_reason,
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        "streaming completion finished"
    );

    // Sampler-G debug — preview of what we actually streamed.
    let resp_preview = preview_text(&emitted_text, 240);
    tracing::info!(
        target: "server.resp",
        completion_tokens = generated.len() as u32,
        finish_reason = %finish_reason,
        resp = %resp_preview,
        "streaming chat_completions response"
    );

    Ok((finish_reason.to_owned(), prompt_tokens, generated.len() as u32))
}

fn finalise(
    state: &ServerState,
    prompt_tokens: u32,
    mut generated: Vec<u32>,
    reason: &str,
    stop_strings: &[String],
    mut logprobs: Option<Vec<ChatLogProbContent>>,
) -> Result<CompletionOutput> {
    // Strip ALL stop tokens (eos, <|im_end|>, etc.) from decoded text so the
    // client sees clean content. Raw count preserved for `usage` honesty.
    // C6: `retain` mutates in place instead of allocating a second Vec.
    let stop_ids = &state.tokenizer.stop_ids;
    let completion_tokens = generated.len() as u32;
    // P1.7 — when logprobs were collected, drop entries that line up
    // with stop-token positions so the per-token list mirrors the
    // text-content `retain` filter.
    if let Some(lp) = logprobs.as_mut() {
        if lp.len() == generated.len() {
            let mut idx = 0;
            generated.retain(|t| {
                let keep = !stop_ids.contains(t);
                if !keep {
                    if idx < lp.len() {
                        lp.remove(idx);
                    }
                } else {
                    idx += 1;
                }
                keep
            });
        } else {
            // Length mismatch — keep the original retain semantics for
            // text and clear logprobs to avoid a misleading partial
            // mapping (should never happen in practice).
            generated.retain(|t| !stop_ids.contains(t));
            lp.clear();
        }
    } else {
        generated.retain(|t| !stop_ids.contains(t));
    }
    let mut text = state.tokenizer.decode(&generated).context("decode")?;
    // **Sampler-G** — truncate at any leaked reasoning marker. The
    // string-level stop in the decode loop catches these mid-flight,
    // but the marker itself is already in `text`; cut before its
    // first occurrence so the client sees a clean response.
    for marker in ["</think>", "<end_thought>", "<end_think>", "</thought>"] {
        if let Some(idx) = text.find(marker) {
            text.truncate(idx);
        }
    }
    // **P0.2** — same treatment for caller-supplied stop sequences.
    // OpenAI semantics: the stop sequence itself is not part of the
    // returned content. Truncate at the earliest match across all
    // user stops.
    let mut earliest = text.len();
    for s in stop_strings {
        if let Some(idx) = text.find(s.as_str()) {
            if idx < earliest {
                earliest = idx;
            }
        }
    }
    text.truncate(earliest);
    // Trim trailing whitespace introduced by the now-removed marker.
    let trimmed_len = text.trim_end().len();
    text.truncate(trimmed_len);
    Ok((text, prompt_tokens, completion_tokens, reason.to_owned(), logprobs))
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
        let chain = format!("{err:#}");
        let top = err.to_string();
        // Log the full chain so the operator can see what bailed; only the
        // topmost context reaches the HTTP client (matches OpenAI surface).
        // anyhow's `{:#}` walks `.source()`; for non-anyhow Display impls
        // it's identical to `to_string()`, so this is safe in both cases.
        tracing::error!(
            target: "server.api.error",
            top = %top,
            chain = %chain,
            "ApiError::internal",
        );
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: top,
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
