//! Decode-loop engine: text-prompt → generated text + token counts +
//! finish reason. All variants (single-call, scheduler-aggregated,
//! ID-only, streaming) share the same prefill / sampler / stop-policy
//! pipeline; they differ only in how they admit requests, batch with
//! peers, and emit output (sync collect vs SSE emit callback).

use std::convert::Infallible;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use axum::response::sse::{Event, KeepAlive, Sse};
use flambeau_runtime::json_schema::JsonConstraint;
use flambeau_runtime::Sampler;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::*;
use crate::gpu_sampler;
use crate::routes::{now_unix, request_id, ServerState, SharedState};
use crate::state::SamplingParams;

use super::finalise::{build_logprob_entry, finalise, preview_text, CompletionOutput};
use super::token_emit::{Discard, Emitter, Finish, Forward, TokenSink};

#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_completion_sse(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    tool_call_format: Option<String>,
    parallel_tool_calls: bool,
    relax_stop_mask: bool,
    include_usage: bool,
    tool_force_prefix: Option<String>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    use crate::model_handle::ReasoningStyle;
    use crate::tool_call_parser::{dispatcher_with_prompt, ParserEvent};

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
        // `<think>`-style templates prime the reasoning open marker in the
        // prompt prefix when thinking is enabled, so the model streams
        // reasoning with no literal `<think>` to detect — start the parser
        // in its in-think state so those tokens route to `reasoning_content`.
        let start_in_reasoning = params.enable_thinking
            && state_clone.model.reasoning_markers().style == ReasoningStyle::ThinkTag;
        let parser_result = dispatcher_with_prompt(
            tool_call_format.as_deref(),
            state_clone.tool_call_format_default,
            &prompt,
            start_in_reasoning,
        );
        let mut parser = match parser_result {
            Ok(p) => p,
            Err(e) => {
                // Bad tool_call_format — surface as a streamed error.
                let err = json!({ "error": { "message": e.to_string() } });
                let _ = tx_clone.blocking_send(Ok(Event::default().data(err.to_string())));
                let _ = tx_clone.blocking_send(Ok(Event::default().data("[DONE]")));
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

        // State threaded through the per-token text feed.
        let mut has_tool_calls = false;
        // When parallel_tool_calls=false, set on the first
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
                    ParserEvent::ThinkDelta(s) => emit_chunk(json!({ "reasoning_content": s })),
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

        // B1: feed the forced tool-call opening to the parser first so the
        // model's continuation reconstructs a complete call.
        if let Some(pfx) = &tool_force_prefix {
            emit_delta(pfx);
        }

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

        // Canonical separate usage chunk. Emitted strictly after
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
            tracing::error!(
                target: "server.completion.error",
                err = format!("{e:#}"),
                "streaming chat error (full chain)"
            );
            let err = json!({ "error": { "message": format!("{e:#}") } });
            let _ = tx_clone.blocking_send(Ok(Event::default().data(err.to_string())));
        }
        let _ = tx_clone.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

/// Shared engine: text prompt → generated text + token counts + finish
/// reason. Runs inside `spawn_blocking` because HIP kernels + mutex hold
/// are sync. `relax_stop_mask=true` disables the first-N-token stop-id
/// suppression and post-N stop-bias — used when the client supplied
/// `tools[]`, since tool-call responses can be legitimately short.
pub(crate) async fn run_completion(
    state: SharedState,
    prompt: &str,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    let prompt = prompt.to_owned();
    tokio::task::spawn_blocking(move || {
        let ids = state
            .tokenizer
            .encode_for_inference(&prompt)
            .context("tokenize prompt")?;
        run_completion_blocking_ids(state, ids, params, relax_stop_mask)
    })
    .await
    .map_err(|e| anyhow!("spawn_blocking join failed: {e}"))?
}

/// Token-id entry point used by the FIM `/infill` route.
/// Skips the tokenizer string round-trip so callers that build prompts
/// directly out of pre-tokenized fragments (e.g., FIM specials wrapping
/// user prefix/suffix) don't depend on the tokenizer's special-token
/// auto-registration heuristic.
pub(crate) async fn run_completion_ids(
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

/// Gate predicate for the scheduler-aware
/// chat handler. The scheduler path is a focused subset of the full
/// `run_completion_blocking_ids` flow: greedy, PP topology, no
/// spec-decode, no GPU sampler, no JSON mode, no logprobs, no tools.
/// All other configurations fall through to the legacy handler.
fn scheduler_can_engage(state: &ServerState, params: &SamplingParams) -> bool {
    if !state.batched_decode {
        return false;
    }
    // MTP draft/verify cycles run on the legacy handler.
    if state.mtp_max_draft > 0 {
        return false;
    }
    // JSON / logprobs paths still go through the legacy handler —
    // those features carry extra device-side state (JSON DFA, top-K
    // logprobs grab) that isn't yet plumbed through the scheduler.
    if !state.model.supports_scheduler_batching() {
        return false;
    }
    !params.json_mode && params.collect_logprobs.is_none()
}

/// Chat handler that uses
/// the scheduler. Engaged when [`scheduler_can_engage`] returns true.
/// Releases the slot's mutex during the decode loop so the scheduler-
/// leader can `blocking_lock` other slots' mutexes for batched dispatch.
/// Supports the full host-sampler feature set: greedy or temp/top_k/
/// top_p/min_p sampling, repetition / presence / frequency penalties.
/// Does NOT (yet) support: spec-decode (MTP), JSON-grammar masking,
/// logprobs, GPU sampler. Those still go through the legacy
/// `run_completion_blocking_ids`.
/// Sarathi-Serve style chunked prefill. Splits the prompt into
/// fixed-size chunks and releases `inflight_pool[slot_idx]`'s mutex
/// between chunks so concurrent slots' decode steps can interleave,
/// bounding the per-step stall a long prompt inflicts on peers. Chunk
/// size from `--prefill-chunk-tokens` (default 512 — Sarathi paper's
/// recommendation).
///
/// On entry the slot is claimed but unlocked. The first chunk's guard
/// scope runs `reset_for_next_request`. After return the slot remains
/// claimed; caller must explicitly relock for decode if it needs to.
///
/// `logits_out` is populated with the LAST chunk's final-row logits
/// — the prefill-side input for first-token sampling.
/// Returns `true` when a full-prompt prefix-cache hit served the logits
/// (the caller must skip the post-prefill capture — the entry it would
/// write is the one that was just hit).
fn chunked_prefill_pp(
    state: &ServerState,
    slot_idx: usize,
    prompt_ids: &[u32],
    logits_out: &mut Vec<f32>,
) -> Result<bool> {
    let prefill_chunk = state.prefill_chunk_tokens.max(1);
    // Phase K4c — mixed-batch engagement on every arch that
    // implements `Model::supports_mixed_batch` (qwen35 / qwen35moe /
    // gemma3 / gemma4 today). Each per-chunk lock acquisition tries
    // to also become the batched-decode leader. If we get the lock
    // AND `batched_pending` is non-empty, we drain it, build a mixed
    // forward (K prefill rows + N decode rows), demux logits, and
    // send decode logits back to the pending response senders. The
    // prefill side's logits land in `logits_out` exactly like the
    // pure path. Single-request workloads see ~0 % overhead (the
    // empty drain costs one batch-window sleep per chunk, ~1.5 ms).
    let mixed_on = state.model.supports_mixed_batch();
    let mut prefill_start = 0usize;
    {
        let mut guard = state.inflight_pool[slot_idx].blocking_lock();
        guard
            .reset_for_next_request()
            .context("reset inflight for new request")?;
        // Prefix-cache restore replaces the prefix's prefill: a FULL hit
        // returns the cached first-token logits with no forward at all; a
        // PREFIX hit advances the loop to the matched boundary (the
        // restored GDN state is exactly at that position, so the tail
        // MUST start there).
        match state.prefix_cache_try_restore(&mut **guard, prompt_ids)? {
            crate::routes::PrefixCacheRestore::FullHit { logits } => {
                logits_out.clear();
                logits_out.extend_from_slice(&logits);
                return Ok(true);
            }
            crate::routes::PrefixCacheRestore::PrefixHit { n_matched } => {
                prefill_start = n_matched;
            }
            crate::routes::PrefixCacheRestore::Miss => {}
        }
    }
    // Intermediate capture targets: the last two full-chunk boundaries
    // strictly inside the prompt. A grown conversation diverges from this
    // prompt only near its end, so these are the chains its next turn can
    // hit; earlier boundaries are shadowed by them, and capturing every
    // boundary would multiply host-RAM cost for no extra hit coverage.
    let capture_boundaries: Vec<usize> = (1..=prompt_ids.len() / prefill_chunk)
        .map(|i| i * prefill_chunk)
        .filter(|&b| b < prompt_ids.len())
        .rev()
        .take(2)
        .collect();
    let restored_at = prefill_start;
    while prefill_start < prompt_ids.len() {
        let end = (prefill_start + prefill_chunk).min(prompt_ids.len());
        let chunk = &prompt_ids[prefill_start..end];
        let mut guard = state.inflight_pool[slot_idx].blocking_lock();
        // The previous chunk ran the full layer stack, so the slot state at
        // `prefill_start` is committed — snapshot it here, under the same
        // freshly-acquired guard the chunk forward uses. Skip the restored
        // boundary itself (its entry is the one we just hit).
        if prefill_start > restored_at && capture_boundaries.contains(&prefill_start) {
            state.prefix_cache_insert_intermediate(&mut **guard, prompt_ids, prefill_start);
        }
        if mixed_on {
            // Become the dispatch leader for this chunk. `lock()`
            // (blocking) — we wait for the current decode tick to
            // finish its dispatch. The decode loops PUSH then RELEASE
            // their slot mutex before rx.recv(), so there is no
            // dependency cycle. Once we have the lock we sleep one
            // batch window for late arrivals, drain, and fire mixed
            // if pendings exist; otherwise fall through to pure
            // prefill while still holding the lock.
            let _dispatch_lock = state.batched_dispatcher.lock().expect("dispatcher poisoned");
            {
                // Brief window for active decode loops to push their next
                // token into `batched_pending` before we drain. Mirrors
                // the decode-batch coalescence sleep in
                // `decode_via_scheduler_into`. Without this the leader/
                // follower race usually drains an empty queue.
                std::thread::sleep(std::time::Duration::from_micros(
                    state.decode_batch_window_us,
                ));
                let drained: Vec<crate::routes::PendingDecode> = {
                    let mut q = state
                        .batched_pending
                        .lock()
                        .expect("batched_pending mutex poisoned");
                    std::mem::take(&mut *q)
                };
                if !drained.is_empty() {
                    // Capacity guard. The ScratchPool was sized for
                    // `max_prefill_tokens = --prefill-ubatch` at boot; a
                    // K + N row mixed forward overruns that buffer when
                    // K + N > prefill_ubatch. Until the boot-time scratch
                    // sizing reserves `chunk_budget + max_slots`, fall
                    // through to pure prefill instead of crashing. The
                    // pendings get re-pushed via their already-held
                    // response channels — but they're already drained
                    // here, so re-push.
                    let total = chunk.len() + drained.len();
                    if total > prefill_chunk {
                        let mut q = state
                            .batched_pending
                            .lock()
                            .expect("batched_pending mutex poisoned");
                        for p in drained {
                            q.push(p);
                        }
                        drop(q);
                    } else {
                    tracing::info!(
                        target: "server.scheduler.mixed",
                        slot_p = slot_idx,
                        k = chunk.len(),
                        n_dec = drained.len(),
                        prefill_start,
                        "mixed-batch engaged"
                    );
                    let prefill_logits = state
                        .dispatch_mixed_with_pending(&mut guard, chunk, prefill_start, &drained)
                        .with_context(|| {
                            format!(
                                "chunked_prefill mixed chunk [{}..{}] of {} (N_decode={})",
                                prefill_start,
                                end,
                                prompt_ids.len(),
                                drained.len(),
                            )
                        })?;
                    logits_out.clear();
                    logits_out.extend_from_slice(&prefill_logits);
                    prefill_start = end;
                    continue;
                    }
                }
            }
            drop(_dispatch_lock);
        }
        let driver = guard
            .as_model_driver_mut()
            .context("chunked_prefill: session is not a v2 ModelDriver")?;
        driver
            .forward_prefill_logits(chunk, prefill_start, logits_out)
            .with_context(|| {
                format!(
                    "chunked_prefill chunk [{}..{}] of {}",
                    prefill_start,
                    end,
                    prompt_ids.len()
                )
            })?;
        prefill_start = end;
    }
    Ok(false)
}

fn run_completion_scheduler_pp_blocking(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    let request_start = Instant::now();
    let slot_idx = state.claim_slot_blocking();
    // Wrap the body in a closure so any error path runs `release_slot`.
    let result = (|| -> Result<CompletionOutput> {
        if prompt_ids.is_empty() {
            bail!("prompt tokenized to 0 tokens");
        }
        let prompt_tokens = prompt_ids.len() as u32;

        tracing::info!(
            target: "server.completion.start",
            prompt_tokens,
            max_tokens = params.max_tokens,
            slot_idx,
            scheduler = true,
            "completion request accepted (scheduler path)"
        );

        let model = &state.model;
        let stop_ids = &state.tokenizer.stop_ids;
        let always_stop_ids = &state.tokenizer.always_stop_ids;
        let is_stop = |t: u32| stop_ids.contains(&t);
        let vocab = state.cfg.vocab_size;
        let sampling = &params.sampling;

        // Sampler shared across first-token + decode loop. Mirrors the
        // legacy handler's setup.
        let mut sampler = Sampler::from_seed(params.seed);
        sampler.reserve(vocab);

        let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);
        let _ = model;
        let full_hit = chunked_prefill_pp(&state, slot_idx, &prompt_ids, &mut logits_buf)?;
        if !full_hit {
            let mut guard = state.inflight_pool[slot_idx].blocking_lock();
            state.prefix_cache_try_capture_full(&mut **guard, &prompt_ids, &logits_buf);
        }
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    logits_buf[sid as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let first_next = sampler.sample(&logits_buf, sampling, &[]);
        tracing::info!(
            target: "server.completion.first_token",
            prompt_tokens,
            ttft_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            greedy = sampling.is_greedy(),
            scheduler = true,
            "first token produced (time-to-first-token)"
        );

        let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
        generated.push(first_next);
        if is_stop(first_next) {
            return finalise(
                &state,
                prompt_tokens,
                generated,
                "stop",
                &params.stop_strings,
                None,
                params.enable_thinking,
            );
        }

        let mut last_token = first_next;
        let mut finish_reason = "length";
        let min_response_tokens = crate::v2_handle::min_response_tokens_for(&state.cfg.arch);
        let stop_bias = crate::v2_handle::stop_bias_for(&state.cfg.arch);
        let mut reasoning_closer = ReasoningCloser::new(reasoning_close_ids(&state, &params));
        let user_stop_max = params
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        // ---------- Stage 2: decode loop via scheduler. NO mutex held.
        // **Cycle 3** — reuse one logits buffer across the whole decode
        // loop instead of allocating a fresh ~600 KB Vec per step.
        let mut logits: Vec<f32> = Vec::with_capacity(vocab);
        for step in 1..params.max_tokens as usize {
            let force_mask = step < min_response_tokens && !relax_stop_mask;
            state
                .decode_via_scheduler_into(
                    slot_idx,
                    last_token,
                    prompt_ids.len() + step - 1,
                    &mut logits,
                )
                .context("scheduler-path decode step")?;
            if !relax_stop_mask {
                for &sid in stop_ids {
                    if (sid as usize) < logits.len() {
                        if force_mask || always_stop_ids.contains(&sid) {
                            logits[sid as usize] = f32::NEG_INFINITY;
                        } else {
                            logits[sid as usize] -= stop_bias;
                        }
                    }
                }
            }
            // Force the reasoning-close marker once the thinking budget is
            // spent so the model stops thinking and answers.
            let next = match reasoning_closer.next_forced(&generated, params.reasoning_budget) {
                Some(forced) => forced,
                None => sampler.sample(&logits, sampling, &generated),
            };
            if is_stop(next) {
                finish_reason = "stop";
                break;
            }
            generated.push(next);
            last_token = next;
            if generated.len() >= params.max_tokens as usize {
                break;
            }
            // Stop-string detection on the recent decoded window
            // (mirrors the legacy handler).
            let need_user_check = !params.stop_strings.is_empty();
            if (need_user_check || !relax_stop_mask)
                && (step % 4 == 0 || step >= min_response_tokens)
            {
                let n = generated.len();
                let token_window = 16.max((user_stop_max / 2).min(64));
                let from = n.saturating_sub(token_window);
                if let Ok(tail) = state.tokenizer.decode(&generated[from..]) {
                    let marker_hit = !relax_stop_mask
                        && !params.enable_thinking
                        && (tail.contains("</think>")
                            || tail.contains("<end_thought>")
                            || tail.contains("<end_think>")
                            || tail.contains("</thought>"));
                    let arch_hit = state
                        .model
                        .chat_stop_markers()
                        .iter()
                        .any(|m| tail.contains(*m));
                    let user_hit = params
                        .stop_strings
                        .iter()
                        .any(|s| tail.contains(s.as_str()));
                    if marker_hit || arch_hit || user_hit {
                        finish_reason = "stop";
                        break;
                    }
                }
            }
        }

        tracing::info!(
            target: "server.completion.finish",
            prompt_tokens,
            completion_tokens = generated.len() as u32,
            finish_reason,
            total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            scheduler = true,
            "completion request finished (scheduler path)"
        );

        finalise(
            &state,
            prompt_tokens,
            generated,
            finish_reason,
            &params.stop_strings,
            None,
            params.enable_thinking,
        )
    })();
    state.release_slot(slot_idx);
    result
}

/// Streaming counterpart of [`run_completion_scheduler_pp_blocking`]: the
/// decode loop coalesces via `decode_via_scheduler_into` (no slot mutex held
/// across the loop), and each sampled token is streamed through the same
/// incremental-detokenise `push_and_emit` the legacy streaming path uses.
/// Engaged from `run_completion_blocking_streaming` when
/// [`scheduler_can_engage`] holds. Output behaviour matches the legacy
/// streaming path; only the decode source differs (coalescing vs N=1).
#[allow(clippy::too_many_lines)]
fn run_completion_streaming_scheduler_pp_blocking(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
    emit: &mut dyn FnMut(&str) -> bool,
) -> Result<(String, u32, u32)> {
    let request_start = Instant::now();
    let slot_idx = state.claim_slot_blocking();
    let result = (|| -> Result<(String, u32, u32)> {
        if prompt_ids.is_empty() {
            bail!("prompt tokenized to 0 tokens");
        }
        let prompt_tokens = prompt_ids.len() as u32;

        tracing::info!(
            target: "server.completion.start",
            prompt_tokens,
            max_tokens = params.max_tokens,
            stream = true,
            slot_idx,
            scheduler = true,
            "streaming completion accepted (scheduler path)"
        );

        let stop_ids = &state.tokenizer.stop_ids;
        let always_stop_ids = &state.tokenizer.always_stop_ids;
        let is_stop = |t: u32| stop_ids.contains(&t);
        let vocab = state.cfg.vocab_size;
        let sampling = &params.sampling;

        let mut sampler = Sampler::from_seed(params.seed);
        sampler.reserve(vocab);

        let mut logits_buf: Vec<f32> = Vec::with_capacity(vocab);
        let prefill_start = Instant::now();
        let full_hit = chunked_prefill_pp(&state, slot_idx, &prompt_ids, &mut logits_buf)?;
        // Capture the unmasked prefill logits into the prefix cache under a
        // brief lock (the decode loop below holds no mutex).
        if !full_hit {
            let mut guard = state.inflight_pool[slot_idx].blocking_lock();
            state.prefix_cache_try_capture_full(&mut **guard, &prompt_ids, &logits_buf);
        }
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    logits_buf[sid as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let first_next = sampler.sample(&logits_buf, sampling, &[]);
        tracing::info!(
            target: "server.completion.first_token",
            prompt_tokens,
            ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
            scheduler = true,
            "first token produced (time-to-first-token)"
        );

        let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens as usize);
        let mut emitted_text = String::new();
        let mut decode_cursor: usize = 0;
        let mut pending_emitted_in_segment: usize = 0;
        let mut push_and_emit =
            |tok: u32, generated: &mut Vec<u32>, emitted_text: &mut String| -> Result<bool> {
                generated.push(tok);
                let stop_hit = is_stop(tok);
                let end = if stop_hit { generated.len() - 1 } else { generated.len() };
                if end <= decode_cursor {
                    return Ok(!stop_hit);
                }
                let raw = state
                    .tokenizer
                    .decode(&generated[decode_cursor..end])
                    .context("decode")?;
                let safe = raw.trim_end_matches('\u{FFFD}');
                if safe.len() > pending_emitted_in_segment {
                    let delta = &safe[pending_emitted_in_segment..];
                    if !emit(delta) {
                        return Ok(false);
                    }
                    emitted_text.push_str(delta);
                    pending_emitted_in_segment = safe.len();
                }
                if safe.len() == raw.len() {
                    decode_cursor = end;
                    pending_emitted_in_segment = 0;
                }
                Ok(!stop_hit)
            };

        let alive = push_and_emit(first_next, &mut generated, &mut emitted_text)?;
        if !alive {
            return Ok(("stop".into(), prompt_tokens, generated.len() as u32));
        }

        let mut finish_reason: &str = "length";
        let mut last_token = first_next;
        let min_response_tokens = crate::v2_handle::min_response_tokens_for(&state.cfg.arch);
        let stop_bias = crate::v2_handle::stop_bias_for(&state.cfg.arch);
        let mut reasoning_closer = ReasoningCloser::new(reasoning_close_ids(&state, &params));
        let user_stop_max = params.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut logits: Vec<f32> = Vec::with_capacity(vocab);
        for step in 1..params.max_tokens as usize {
            let force_mask = step < min_response_tokens && !relax_stop_mask;
            state
                .decode_via_scheduler_into(
                    slot_idx,
                    last_token,
                    prompt_ids.len() + step - 1,
                    &mut logits,
                )
                .context("scheduler-path decode step")?;
            if !relax_stop_mask {
                for &sid in stop_ids {
                    if (sid as usize) < logits.len() {
                        if force_mask || always_stop_ids.contains(&sid) {
                            logits[sid as usize] = f32::NEG_INFINITY;
                        } else {
                            logits[sid as usize] -= stop_bias;
                        }
                    }
                }
            }
            let next = match reasoning_closer.next_forced(&generated, params.reasoning_budget) {
                Some(forced) => forced,
                None => sampler.sample(&logits, sampling, &generated),
            };
            let alive = push_and_emit(next, &mut generated, &mut emitted_text)?;
            last_token = next;
            if !alive {
                finish_reason = "stop";
                break;
            }
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
                    && !params.enable_thinking
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
                    break;
                }
            }
        }

        tracing::info!(
            target: "server.completion.finish",
            prompt_tokens,
            completion_tokens = generated.len() as u32,
            finish_reason,
            total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            stream = true,
            scheduler = true,
            "streaming completion finished (scheduler path)"
        );

        Ok((finish_reason.to_owned(), prompt_tokens, generated.len() as u32))
    })();
    state.release_slot(slot_idx);
    result
}

/// The token sequence for the arch's reasoning-close marker, resolved once
/// per request — only when a thinking budget is in play. Empty otherwise.
/// The marker is often multi-token (qwen `</think>` = `[510, 26003, 29]`),
/// so the whole sequence must be forced to actually close the block.
fn reasoning_close_ids(state: &ServerState, params: &SamplingParams) -> Vec<u32> {
    if !(params.enable_thinking && params.reasoning_budget.is_some()) {
        return Vec::new();
    }
    let close = state.model.reasoning_markers().close;
    state.tokenizer.encode(close).unwrap_or_default()
}

/// True if `needle` appears as a contiguous run anywhere in `haystack`.
fn tokens_contain(haystack: &[u32], needle: &[u32]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// State for forcing the reasoning-close marker once the thinking budget is
/// spent. While the model is still inside its `<think>` block past `budget`
/// generated tokens, the close marker's token sequence is emitted verbatim
/// (one token per step) so the model moves on to the answer.
#[derive(Default)]
struct ReasoningCloser {
    close_ids: Vec<u32>,
    queue: std::collections::VecDeque<u32>,
    fired: bool,
}

impl ReasoningCloser {
    fn new(close_ids: Vec<u32>) -> Self {
        Self {
            close_ids,
            ..Default::default()
        }
    }

    /// Next forced token (the marker sequence), or `None` to sample normally.
    /// `generated` is the per-turn token history; `budget` the cap.
    fn next_forced(&mut self, generated: &[u32], budget: Option<u32>) -> Option<u32> {
        if self.queue.is_empty()
            && !self.fired
            && !self.close_ids.is_empty()
            && budget.is_some_and(|b| generated.len() as u32 >= b)
            && !tokens_contain(generated, &self.close_ids)
        {
            self.fired = true;
            self.queue.extend(self.close_ids.iter().copied());
        }
        self.queue.pop_front()
    }
}

fn run_completion_blocking_ids(
    state: SharedState,
    prompt_ids: Vec<u32>,
    params: SamplingParams,
    relax_stop_mask: bool,
) -> Result<CompletionOutput> {
    if scheduler_can_engage(&state, &params) {
        return run_completion_scheduler_pp_blocking(state, prompt_ids, params, relax_stop_mask);
    }
    let out = decode_tokens(
        &state,
        &prompt_ids,
        &params,
        relax_stop_mask,
        DecodeFeatures { structured: true, stream: false },
        &mut Discard,
    )?;
    let result = finalise(
        &state,
        out.prompt_tokens,
        out.generated,
        out.finish.reason(),
        &params.stop_strings,
        out.logprobs,
        params.enable_thinking,
    )?;
    tracing::info!(
        target: "server.resp",
        completion_tokens = result.2,
        finish_reason = %result.3,
        resp = %preview_text(&result.0, 240),
        "chat_completions response"
    );
    Ok(result)
}

/// Streaming variant of the decode loop. Pushes text deltas through
/// `emit` as each token is produced; returns `(finish_reason,
/// prompt_tokens, completion_tokens)` so the SSE producer can emit a
/// final usage chunk.
pub(crate) fn run_completion_blocking_streaming(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    relax_stop_mask: bool,
    emit: &mut dyn FnMut(&str) -> bool,
) -> Result<(String, u32, u32)> {
    let prompt_ids = state
        .tokenizer
        .encode_for_inference(&prompt)
        .context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }
    if scheduler_can_engage(&state, &params) {
        return run_completion_streaming_scheduler_pp_blocking(
            state,
            prompt_ids,
            params,
            relax_stop_mask,
            emit,
        );
    }
    let out = decode_tokens(
        &state,
        &prompt_ids,
        &params,
        relax_stop_mask,
        DecodeFeatures { structured: false, stream: true },
        &mut Forward { emit },
    )?;
    tracing::info!(
        target: "server.resp",
        completion_tokens = out.generated.len() as u32,
        finish_reason = %out.finish.reason(),
        resp = %preview_text(&out.text, 240),
        "streaming chat_completions response"
    );
    Ok((out.finish.reason().to_owned(), out.prompt_tokens, out.generated.len() as u32))
}

/// Which request surfaces a decode-loop caller serves. The SSE path has no
/// JSON-constraint or logprobs output yet, so it decodes without them.
#[derive(Clone, Copy)]
struct DecodeFeatures {
    structured: bool,
    stream: bool,
}

struct DecodeOutcome {
    prompt_tokens: u32,
    generated: Vec<u32>,
    finish: Finish,
    logprobs: Option<Vec<ChatLogProbContent>>,
    mtp_cycles: u32,
    text: String,
}

/// The mutex-held single-request decode loop behind both legacy paths:
/// prefill, first token, then plain steps or MTP cycles until a stop rule
/// fires. Every token, including each token of an MTP cycle, goes through
/// `Emitter::accept`, so stop tokens, stop sequences, `max_tokens` and a
/// gone client end generation at the exact token on both paths.
fn decode_tokens(
    state: &SharedState,
    prompt_ids: &[u32],
    params: &SamplingParams,
    relax_stop_mask: bool,
    features: DecodeFeatures,
    sink: &mut dyn TokenSink,
) -> Result<DecodeOutcome> {
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }
    let request_start = Instant::now();
    let slot_idx = state.claim_slot_blocking();
    let out = decode_in_slot(state, slot_idx, prompt_ids, params, relax_stop_mask, features, sink);
    state.release_slot(slot_idx);
    let out = out?;
    tracing::info!(
        target: "server.completion.finish",
        prompt_tokens = out.prompt_tokens,
        completion_tokens = out.generated.len() as u32,
        finish_reason = out.finish.reason(),
        total_ms = request_start.elapsed().as_secs_f64() * 1000.0,
        mtp_cycles = out.mtp_cycles,
        stream = features.stream,
        "completion request finished"
    );
    Ok(out)
}

/// Stop sequences checked on the decoded text: leaked reasoning markers
/// (unless thinking is enabled or the stop mask is relaxed), the arch's
/// chat-turn markers, and the request's stop strings.
fn stop_patterns(state: &ServerState, params: &SamplingParams, relax_stop_mask: bool) -> Vec<String> {
    let mut patterns: Vec<String> = Vec::new();
    if !relax_stop_mask && !params.enable_thinking {
        patterns.extend(
            ["</think>", "<end_thought>", "<end_think>", "</thought>"]
                .iter()
                .map(|s| (*s).to_owned()),
        );
    }
    patterns.extend(state.model.chat_stop_markers().iter().map(|s| (*s).to_owned()));
    patterns.extend(params.stop_strings.iter().filter(|s| !s.is_empty()).cloned());
    patterns
}

#[allow(clippy::too_many_arguments)]
fn decode_in_slot(
    state: &SharedState,
    slot_idx: usize,
    prompt_ids: &[u32],
    params: &SamplingParams,
    relax_stop_mask: bool,
    features: DecodeFeatures,
    sink: &mut dyn TokenSink,
) -> Result<DecodeOutcome> {
    let prompt_tokens = prompt_ids.len() as u32;
    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        stream = features.stream,
        slot_idx,
        "completion request accepted after queue wait"
    );

    let mut sampler = Sampler::from_seed(params.seed);
    sampler.reserve(state.cfg.vocab_size);
    let sampling = &params.sampling;
    let stop_ids = &state.tokenizer.stop_ids;
    let always_stop_ids = &state.tokenizer.always_stop_ids;
    let decode = |ids: &[u32]| -> Result<String> { state.tokenizer.decode(ids).context("decode") };
    let mut logits_buf: Vec<f32> = Vec::with_capacity(state.cfg.vocab_size);

    // Prefill. The first token's stop ids are always masked: Qwen3.6 can
    // argmax `<|im_end|>` first on multi-turn prompts, yielding an empty
    // reply.
    let prefill_start = Instant::now();
    let full_hit = chunked_prefill_pp(state, slot_idx, prompt_ids, &mut logits_buf)?;
    // The prefix-cache snapshot (0.1-2.2 s of DtoH) is taken after the first
    // token is out, from the unmasked prefill logits.
    let prefill_logits_for_cache = if full_hit { Vec::new() } else { logits_buf.clone() };
    let mut inflight_guard = state.inflight_pool[slot_idx].blocking_lock();
    let inflight: &mut dyn crate::Session = &mut **inflight_guard;
    for &sid in stop_ids {
        if (sid as usize) < logits_buf.len() {
            logits_buf[sid as usize] = f32::NEG_INFINITY;
        }
    }
    // JSON state is primed with the assistant prefill before the first
    // token so the mask holds from the model's actual position.
    let json_state: Option<JsonConstraint> = if features.structured && params.json_mode {
        let mut js = match params.json_schema.as_ref() {
            Some(schema) => JsonConstraint::for_schema(schema),
            None => JsonConstraint::object(),
        };
        if !params.json_prime_bytes.is_empty() {
            let _ = js.feed_slice(&params.json_prime_bytes);
        }
        Some(js)
    } else {
        None
    };
    if let Some(js) = json_state.as_ref() {
        gpu_sampler::apply_json_mask_to_logits(js, &state.tokenizer, &mut logits_buf, 2048);
    }
    let first_next = sampler.sample(&logits_buf, sampling, &[]);
    tracing::info!(
        target: "server.completion.first_token",
        prompt_tokens,
        ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0,
        greedy = sampling.is_greedy(),
        topology = state.model.topology(),
        "first token produced (time-to-first-token)"
    );
    let mut logprobs_acc: Option<Vec<ChatLogProbContent>> =
        if features.structured && params.collect_logprobs.is_some() {
            Some(Vec::with_capacity(params.max_tokens as usize))
        } else {
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

    let mtp_on = state.mtp_max_draft > 0
        && json_state.is_none()
        && logprobs_acc.is_none()
        && params.reasoning_budget.is_none();
    let mut emitter = Emitter::new(
        stop_patterns(state, params, relax_stop_mask),
        json_state,
        stop_ids,
        params.max_tokens as usize,
        sink,
    );
    let outcome = |emitter: Emitter<'_>, finish, logprobs, mtp_cycles| DecodeOutcome {
        prompt_tokens,
        text: emitter.text.text().to_owned(),
        generated: emitter.generated,
        finish,
        logprobs,
        mtp_cycles,
    };
    if let Some(finish) = emitter.accept(first_next, &decode)? {
        return Ok(outcome(emitter, finish, logprobs_acc, 0));
    }
    if !full_hit {
        state.prefix_cache_try_capture_full(inflight, prompt_ids, &prefill_logits_for_cache);
    }

    let min_response_tokens = crate::v2_handle::min_response_tokens_for(&state.cfg.arch);
    let stop_bias = crate::v2_handle::stop_bias_for(&state.cfg.arch);
    let mut reasoning_closer = ReasoningCloser::new(reasoning_close_ids(state, params));
    let mut last_token = first_next;
    let mut mtp_cycles = 0u32;
    let finish = 'decode: loop {
        let generated_len = emitter.generated.len();
        let pos = prompt_ids.len() + generated_len - 1;
        // Stop ids are masked while fewer than `min_response_tokens` tokens
        // exist and biased by `stop_bias` after. A cycle emits at most
        // `n_draft + 1` tokens from index `generated_len`, so the draft
        // count is capped to keep a cycle on one side of that boundary.
        let in_stop_window = !relax_stop_mask && generated_len < min_response_tokens;
        let n_draft = if in_stop_window {
            state
                .mtp_max_draft
                .min((min_response_tokens - generated_len).saturating_sub(1))
        } else {
            state.mtp_max_draft
        };
        if mtp_on && n_draft > 0 {
            let factory = mtp_sample_factory(
                sampling,
                &emitter.generated,
                stop_ids,
                always_stop_ids,
                relax_stop_mask,
                in_stop_window,
                stop_bias,
            );
            let seed = params.seed ^ (pos as u64).wrapping_mul(0x9E3779B9);
            if let Some(emitted) = inflight
                .mtp_step(last_token, pos, n_draft, factory, seed, &mut logits_buf)
                .context("mtp step")?
            {
                mtp_cycles += 1;
                for t in emitted {
                    last_token = t;
                    if let Some(finish) = emitter.accept(t, &decode)? {
                        break 'decode finish;
                    }
                }
                continue;
            }
        }
        state
            .dispatch_decode_one(&mut *inflight, last_token, pos, &mut logits_buf)
            .context("decode step logits")?;
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    if always_stop_ids.contains(&sid) || in_stop_window {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else {
                        logits_buf[sid as usize] -= stop_bias;
                    }
                }
            }
        }
        if let Some(js) = emitter.json.as_ref() {
            gpu_sampler::apply_json_mask_to_logits(js, &state.tokenizer, &mut logits_buf, 2048);
        }
        // Once the thinking budget is spent the reasoning-close marker is
        // forced verbatim; otherwise `generated` is the penalty history.
        let next = match reasoning_closer.next_forced(&emitter.generated, params.reasoning_budget) {
            Some(forced) => forced,
            None => sampler.sample(&logits_buf, sampling, &emitter.generated),
        };
        if let Some(lp) = logprobs_acc.as_mut() {
            if let Some(entry) = build_logprob_entry(
                &state.tokenizer,
                &logits_buf,
                sampling,
                &emitter.generated,
                next,
                params.collect_logprobs.unwrap_or(0) as usize,
            ) {
                lp.push(entry);
            }
        }
        last_token = next;
        if let Some(finish) = emitter.accept(next, &decode)? {
            break 'decode finish;
        }
    };
    Ok(outcome(emitter, finish, logprobs_acc, mtp_cycles))
}

/// Sampler handed to an MTP cycle: the plain path's stop-id rule (mask
/// inside the window, bias after, `always_stop_ids` always masked) applied
/// to every draft and verify row, then the request's sampling transforms.
#[allow(clippy::too_many_arguments)]
fn mtp_sample_factory(
    sampling: &flambeau_runtime::Sampling,
    history: &[u32],
    stop_ids: &[u32],
    always_stop_ids: &[u32],
    relax_stop_mask: bool,
    in_stop_window: bool,
    stop_bias: f32,
) -> std::sync::Arc<dyn Fn(u64) -> flambeau_server_core::MtpSampleFn + Send + Sync> {
    let sampling = sampling.clone();
    let history = history.to_vec();
    let stop_adjust: std::sync::Arc<Vec<(u32, f32)>> = std::sync::Arc::new(if relax_stop_mask {
        Vec::new()
    } else {
        stop_ids
            .iter()
            .map(|&sid| {
                let delta = if in_stop_window || always_stop_ids.contains(&sid) {
                    f32::NEG_INFINITY
                } else {
                    -stop_bias
                };
                (sid, delta)
            })
            .collect()
    });
    std::sync::Arc::new(move |seed: u64| {
        let sampling = sampling.clone();
        let history = history.clone();
        let stop_adjust = std::sync::Arc::clone(&stop_adjust);
        let mut adjusted: Vec<f32> = Vec::new();
        let mut rng = flambeau_runtime::Rng::from_seed(seed);
        Box::new(move |logits: &[f32]| -> (u32, Vec<(u32, f32)>) {
            adjusted.clear();
            adjusted.extend_from_slice(logits);
            for &(sid, delta) in stop_adjust.iter() {
                if let Some(l) = adjusted.get_mut(sid as usize) {
                    *l += delta;
                }
            }
            let dist = flambeau_runtime::build_distribution(&adjusted, &sampling, &history);
            let r = rng.next_f32();
            let mut acc = 0.0f32;
            let mut tok = dist.last().map(|(t, _)| *t).unwrap_or(0);
            for &(t, p) in &dist {
                acc += p;
                if r <= acc {
                    tok = t;
                    break;
                }
            }
            (tok, dist)
        })
    })
}
