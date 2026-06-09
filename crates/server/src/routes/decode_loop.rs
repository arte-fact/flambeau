//! Decode-loop engine: text-prompt → generated text + token counts +
//! finish reason. All variants (single-call, scheduler-aggregated,
//! ID-only, streaming) share the same prefill / sampler / stop-policy
//! pipeline; they differ only in how they admit requests, batch with
//! peers, and emit output (sync collect vs SSE emit callback).

use std::convert::Infallible;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use axum::response::sse::{Event, KeepAlive, Sse};
use flambeau_backend_hip::HipCluster;
use flambeau_runtime::json_schema::JsonConstraint;
use flambeau_runtime::Sampler;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::*;
use crate::gpu_sampler::{self, GpuSamplerScratch};
use crate::routes::{dev_flag, dev_usize, now_unix, request_id, ServerState, SharedState};
use crate::state::SamplingParams;

use super::finalise::{build_logprob_entry, finalise, preview_text, CompletionOutput};

pub(crate) fn stream_completion_sse(
    state: SharedState,
    prompt: String,
    params: SamplingParams,
    tool_call_format: Option<String>,
    parallel_tool_calls: bool,
    relax_stop_mask: bool,
    include_usage: bool,
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

/// **P1.6b** — token-id entry point used by the FIM `/infill` route.
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

/// **P2.9b-i2-B-wire** — gate predicate for the scheduler-aware
/// chat handler. The scheduler path is a focused subset of the full
/// `run_completion_blocking_ids` flow: greedy, PP topology, no
/// spec-decode, no GPU sampler, no JSON mode, no logprobs, no tools.
/// All other configurations fall through to the legacy handler.
fn scheduler_can_engage(state: &ServerState, params: &SamplingParams) -> bool {
    if !state.batched_decode {
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

/// **P2.9b-i2-B-wire (cleanup 2026-05-03)** — chat handler that uses
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
fn chunked_prefill_pp(
    state: &ServerState,
    slot_idx: usize,
    prompt_ids: &[u32],
    logits_out: &mut Vec<f32>,
) -> Result<()> {
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
                return Ok(());
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
    {
        let mut guard = state.inflight_pool[slot_idx].blocking_lock();
        state.prefix_cache_try_capture_full(&mut **guard, prompt_ids, logits_out);
    }
    Ok(())
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

        let cluster: &flambeau_backend_hip::HipCluster = &state.cluster;
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
        let _ = cluster;
        let _ = model;
        chunked_prefill_pp(&state, slot_idx, &prompt_ids, &mut logits_buf)?;
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    logits_buf[sid as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let first_next = sampler.sample(&logits_buf, sampling, &[]);

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
                    prompt_ids.len() + step,
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
    // **P2.9b-i2-B-wire** — opt into the scheduler path when
    // conditions allow. Greedy + PP + no MTP + no JSON + no logprobs.
    if scheduler_can_engage(&state, &params) {
        return run_completion_scheduler_pp_blocking(state, prompt_ids, params, relax_stop_mask);
    }

    let request_start = Instant::now();

    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }

    let slot_idx = state.claim_slot_blocking();
    let result = (|| -> Result<CompletionOutput> {
    let prompt_tokens = prompt_ids.len() as u32;

    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        slot_idx,
        "completion request accepted after queue wait"
    );

    let cluster: &HipCluster = &state.cluster;
    let model = &state.model;

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

    // GPU sampler scratch alloc is gated on the keep-logits-on-device
    // optimisation, which was disabled when decode collapsed onto
    // `forward_decode_batched_*` (Phase 12.5) — the batched output head
    // writes to a different scratch buffer than the keep-on-device
    // kernels read from. Re-wiring `resolve_head_logits` to the batched
    // scratch is a follow-up; for now scratch stays `None`.
    let _ = state.gpu_sampler;
    let use_gpu_sampler = false;
    let mut gpu_scratch: Option<GpuSamplerScratch> = None;

    // Prefill. Always download logits so we can mask stop tokens on the
    // first generated token — Qwen3.6 sometimes argmaxes `<|im_end|>` as
    // the first response token on multi-turn prompts, producing an empty
    // reply. Suppress it until at least one content token is emitted.
    let prefill_start = Instant::now();
    let _ = cluster;
    let _ = model;
    // **Phase 5 S2** — chunked prefill releases `inflight_pool[slot_idx]`'s
    // mutex between chunks so concurrent slots' decode steps interleave
    // (bounds the per-step stall a long prompt inflicts on peers).
    // Reset for new request happens inside `chunked_prefill_pp` on the
    // first chunk's lock scope. Decode reacquires the guard below.
    chunked_prefill_pp(&state, slot_idx, &prompt_ids, &mut logits_buf)?;
    let mut inflight_guard = state.inflight_pool[slot_idx].blocking_lock();
    let inflight: &mut dyn crate::Session = &mut **inflight_guard;
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
    // Phase 12.5 — `inv_temp` was the GPU-sampler entry, retired with
    // the keep-on-device branch. Re-introduce when `run_gpu_topk` re-
    // wires onto the batched output buffer.
    let _inv_temp = if sampling.temperature > 0.0 {
        1.0 / sampling.temperature
    } else {
        1.0
    };
    // **#236 P0.1b** — JSON state primed BEFORE first_next. When the
    // request is in json_mode AND the chat handler detected an
    // assistant prefill (last message had `role: "assistant"`),
    // `params.json_prime_bytes` carries those bytes; the JSON state
    // machine advances through them before the first decoded token,
    // so the mask honours the model's actual continuation position.
    // For non-prefill json_mode requests the prime is empty and the
    // state starts at `JsonState::new()`. Created here (rather than
    // post-first_next as before) so the mask can apply to the very
    // first sampled token.
    let mut json_state: Option<JsonConstraint> = if params.json_mode {
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
    // **#236 P0.1b** — mask first-token logits when json_mode is
    // active. Without this the very first sampled token can violate
    // the grammar (e.g. emit `Hello` before `{`), and the existing
    // post-sample feed_slice would silently advance into a doomed
    // state. Top-K=2048 candidate cap keeps this cheap (~1–2 ms) on
    // a 151 k Qwen vocab.
    if let Some(js) = json_state.as_ref() {
        gpu_sampler::apply_json_mask_to_logits(
            js,
            &state.tokenizer,
            &mut logits_buf,
            /*max_candidates=*/ 2048,
        );
    }
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

    // **P0.1** — advance JSON state by first_next's bytes (the
    // pre-sample mask above already filtered candidates that would
    // invalidate the running JSON, so this should always succeed
    // when the mask was active).
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
        // Slot stays pooled; mutex releases on function return.
        return finalise(
            &state,
            prompt_tokens,
            generated,
            "stop",
            &params.stop_strings,
            logprobs_acc,
            params.enable_thinking,
        );
    }

    let mut finish_reason = "length";
    let mut last_token = first_next;
    // Mask stop tokens for the first MIN_RESPONSE_TOKENS steps. Qwen3.6
    // on multi-turn prompts otherwise emits `<|im_end|>` after 0-1 content
    // tokens, producing unusable one-word replies. MIN is small enough
    // that short on-topic answers ("Yes.", "42.") are still possible.
    // T4.1: tool-call turns disable this entirely. When the model is
    // asked to emit a `<tool_call>…</tool_call>` body it may legitimately
    // take only ~10 tokens; forcing 24 content tokens before allowing
    // stop injects noise between the body and the `<|im_end|>` and
    // breaks downstream parsing. `relax_stop_mask` flips both knobs to
    // no-ops — trust the model on turns where `tools[]` is present.
    // **Sampler-G (2026-04-30)** — was 24, lowered to 8. With Qwen3.6-27B
    // at temp=0.7+top_p=0.8 the 24-token floor forced the model to keep
    // generating 13+ tokens past natural endpoints like
    // "Hello! How can I help you today?" (~10 tokens), at which point it
    // wandered into reasoning-marker leaks (`</think>`, `<end_thought>`),
    // hallucinated chat formats (`<|user|>\n<|assistant|>`), or duplicate-
    // the-response loops. 8 lets short greetings stop naturally; the
    // first-token NEG_INFINITY mask still prevents immediate-EOS on
    // multi-turn prompts.
    let min_response_tokens = crate::v2_handle::min_response_tokens_for(&state.cfg.arch);
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS.
    // was 0.5 (was 3.0 before that). Even 0.5 is enough to push
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
    // most natural endpoints. The prior concern about `0.5
    // → repeat loops` was driven by Coder-Next-80B specifically;
    // Qwen3.6 doesn't show that failure at this bias on chat tests.
    let stop_bias = crate::v2_handle::stop_bias_for(&state.cfg.arch);
    let mut reasoning_closer = ReasoningCloser::new(reasoning_close_ids(&state, &params));
    for step in 1..params.max_tokens as usize {
        let force_mask = step < min_response_tokens && !relax_stop_mask;
        // Phase 12.5 — decode goes through the host-path DtoH always.
        // The GPU sampler keep-on-device branch was retired with the
        // decode/batched collapse (see `use_gpu_sampler` initialiser).
        let next = {
            state
                .dispatch_decode_one(
                    &mut *inflight,
                    last_token,
                    prompt_ids.len() + step,
                    &mut logits_buf,
                )
                .context("decode step logits")?;
            if !relax_stop_mask {
                for &sid in stop_ids {
                    if (sid as usize) < logits_buf.len() {
                        if always_stop_ids.contains(&sid) || force_mask {
                            logits_buf[sid as usize] = f32::NEG_INFINITY;
                        } else {
                            logits_buf[sid as usize] -= stop_bias;
                        }
                    }
                }
            }
            // **#236 P0.1b** — host-path JSON mask. Mirrors the
            // GPU-path mask above: set logit of any top-K candidate
            // that would invalidate the running JSON to NEG_INFINITY
            // before the sampler picks. Only active when the request
            // set `response_format=json_object`. Top-K=2048 cap keeps
            // the per-token cost in the low-ms range.
            if let Some(js) = json_state.as_ref() {
                gpu_sampler::apply_json_mask_to_logits(
                    js,
                    &state.tokenizer,
                    &mut logits_buf,
                    /*max_candidates=*/ 2048,
                );
            }
            // Once the thinking budget is spent, emit the reasoning-close
            // marker verbatim so the model stops thinking and answers.
            // Otherwise pass `generated` as history so penalties can fire on
            // repeats / frequent tokens (T4.b.2 — without this, Qwen3.5/3.6
            // agent loops degrade to long-CoT drift).
            let next = match reasoning_closer.next_forced(&generated, params.reasoning_budget) {
                Some(forced) => forced,
                None => sampler.sample(&logits_buf, sampling, &generated),
            };
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
        let mut json_complete = false;
        if let Some(js) = json_state.as_mut() {
            if let Ok(text) = state.tokenizer.decode(&[next]) {
                let _ = js.feed_slice(text.as_bytes());
            }
            json_complete = js.is_complete();
        }
        generated.push(next);
        last_token = next;
        if is_stop(next) {
            finish_reason = "stop";
            break;
        }
        // Structured-output: stop once the value is structurally complete so
        // the model can't drift into trailing prose past a valid JSON value.
        if json_complete {
            finish_reason = "stop";
            break;
        }
        // **Sampler-G** — string-level stop on reasoning markers.
        // The model can route around the single-token `</think>`
        // mask by emitting the multi-token text form. Detokenize
        // the recent tail and stop if a leak is present. Final
        // response cleanup happens in `finalise`.
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
        if (need_user_check || !relax_stop_mask) && (step % 4 == 0 || step >= min_response_tokens) {
            let n = generated.len();
            // 16 tokens covers ≥48 chars typical; widen if a user
            // stop string is longer than ~32 chars.
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

    // gpu_scratch is always None today (see initialiser above); dispose
    // wiring will land when the keep-logits-on-device path is re-wired
    // for the batched decode kernels.
    let _ = gpu_scratch.take();

    // **P2.9a (slot pool)** — no dispose. The pooled inflight stays
    // allocated; releasing the mutex returns the slot to the pool
    // for the next request, which will reset_for_next_request on
    // claim. Removing per-request dispose saves the ~ms cost of
    // session/scratch teardown + re-alloc.
    let _ = inflight; // silence unused-warn after removing dispose

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
        params.enable_thinking,
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
    })();
    state.release_slot(slot_idx);
    result
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
    let request_start = Instant::now();

    let prompt_ids = state
        .tokenizer
        .encode_for_inference(&prompt)
        .context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to 0 tokens");
    }

    let slot_idx = state.claim_slot_blocking();
    let result = (|| -> Result<(String, u32, u32)> {
    let prompt_tokens = prompt_ids.len() as u32;

    tracing::info!(
        target: "server.completion.start",
        prompt_tokens,
        max_tokens = params.max_tokens,
        stream = true,
        slot_idx,
        "streaming completion accepted"
    );

    let cluster: &HipCluster = &state.cluster;
    let model = &state.model;
    let stop_ids = &state.tokenizer.stop_ids;
    let always_stop_ids = &state.tokenizer.always_stop_ids;
    let is_stop = |t: u32| stop_ids.contains(&t);

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
    let _ = cluster;
    // **Phase 5 S2** — chunked prefill on the streaming path. Same
    // mutex-release-between-chunks shape as the legacy path; decode
    // reacquires once below for the SSE-emit loop.
    chunked_prefill_pp(&state, slot_idx, &prompt_ids, &mut logits_buf)?;
    let mut inflight_guard = state.inflight_pool[slot_idx].blocking_lock();
    let inflight: &mut dyn crate::Session = &mut **inflight_guard;
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
    // incremental detokenize state. `decode_cursor` is the
    // index of the first token NOT yet decoded into clean emitted bytes.
    // `pending_emitted_in_segment` tracks how many bytes of the current
    // open segment (`generated[decode_cursor..]`) we've already streamed
    // to the client, so a re-decode after a multi-byte boundary doesn't
    // re-emit safe bytes. When the segment finishes cleanly (no
    // trailing U+FFFD), `decode_cursor` advances and the segment resets.
    // Reduces per-step decode cost from O(generated.len()) to ~O(1) —
    // typical "open segment" is 1-3 tokens, only growing when a multi-
    // byte glyph straddles a BPE-token boundary.
    let mut decode_cursor: usize = 0;
    let mut pending_emitted_in_segment: usize = 0;

    let mut push_and_emit =
        |tok: u32, generated: &mut Vec<u32>, emitted_text: &mut String| -> Result<bool> {
            generated.push(tok);
            let stop_hit = is_stop(tok);
            // Decode only the still-open segment, not the full sequence.
            // `end` excludes the trailing stop token so its raw text never leaks.
            let end = if stop_hit {
                generated.len() - 1
            } else {
                generated.len()
            };
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
        // Slot stays pooled; mutex releases on function return.
        return Ok(("stop".into(), prompt_tokens, generated.len() as u32));
    }

    let mut finish_reason: &str = "length";
    let mut last_token = first_next;
    // See non-streaming path for the MIN_RESPONSE_TOKENS rationale.
    // Sampler-G — lowered from 24 to 8 to let short greetings stop
    // at natural endpoints instead of wandering into leak territory.
    let min_response_tokens = crate::v2_handle::min_response_tokens_for(&state.cfg.arch);
    // Nats subtracted from every stop-token logit beyond MIN_RESPONSE_TOKENS.
    // was 0.5 (was 3.0 before that). Even 0.5 is enough to push
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
    // most natural endpoints. The prior concern about `0.5
    // → repeat loops` was driven by Coder-Next-80B specifically;
    // Qwen3.6 doesn't show that failure at this bias on chat tests.
    let stop_bias = crate::v2_handle::stop_bias_for(&state.cfg.arch);
    let mut reasoning_closer = ReasoningCloser::new(reasoning_close_ids(&state, &params));
    // env-gated TP-decode profiling. When FLAMBEAU_PROFILE_DECODE
    // is set, enable HipEvent section recording for `n` warm-up-skipped decode
    // steps, then flush + dump aggregate per-section ms to stderr. Skips the
    // first 8 steps (cold-cache effects, allocator warmup).
    let profile_decode_n: usize = dev_usize("FLAMBEAU_PROFILE_DECODE", 0);
    let profile_skip: usize = 8;

    // **Lever B (host profile)** — per-section host wall accumulator,
    // gated by FLAMBEAU_HOST_PROFILE=1. Skips the first 8 decode
    // steps to dodge cold-cache effects, accumulates ms-per-section
    // over the rest, dumps a one-line summary at end-of-loop.
    let host_profile_on = dev_flag("FLAMBEAU_HOST_PROFILE");
    let mut hp_n: usize = 0;
    let mut hp_decode_us: u128 = 0;
    let mut hp_mask_us: u128 = 0;
    let mut hp_sample_us: u128 = 0;
    let mut hp_emit_us: u128 = 0;
    let mut hp_stopstr_us: u128 = 0;
    let mut hp_step_us: u128 = 0;

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
        let force_mask = step < min_response_tokens && !relax_stop_mask;
        let hp_step_t0 = if host_profile_on {
            Some(Instant::now())
        } else {
            None
        };
        state
            .dispatch_decode_one(
                &mut *inflight,
                last_token,
                prompt_ids.len() + step,
                &mut logits_buf,
            )
            .context("decode step logits")?;
        let hp_after_decode = if host_profile_on {
            Some(Instant::now())
        } else {
            None
        };
        if !relax_stop_mask {
            for &sid in stop_ids {
                if (sid as usize) < logits_buf.len() {
                    if always_stop_ids.contains(&sid) || force_mask {
                        logits_buf[sid as usize] = f32::NEG_INFINITY;
                    } else {
                        logits_buf[sid as usize] -= stop_bias;
                    }
                }
            }
        }
        let hp_after_mask = if host_profile_on {
            Some(Instant::now())
        } else {
            None
        };
        // Once the thinking budget is spent, emit the reasoning-close marker
        // verbatim so the model stops thinking and answers.
        let next = match reasoning_closer.next_forced(&generated, params.reasoning_budget) {
            Some(forced) => forced,
            None => sampler.sample(&logits_buf, sampling, &generated),
        };
        let hp_after_sample = if host_profile_on {
            Some(Instant::now())
        } else {
            None
        };
        let alive = push_and_emit(next, &mut generated, &mut emitted_text)?;
        let hp_after_emit = if host_profile_on {
            Some(Instant::now())
        } else {
            None
        };
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
                tracing::info!(
                    target: "server.completion.string_stop",
                    user_stop = user_hit,
                    "string-level stop on reasoning-marker or user-stop leak"
                );
                break;
            }
        }
        // **Lever B (host profile)** — accumulate per-section wall.
        // Skip first 8 steps (cold cache / allocator warmup).
        if host_profile_on && step > profile_skip {
            let t_step_end = Instant::now();
            if let (Some(t0), Some(td), Some(tm), Some(ts), Some(te)) = (
                hp_step_t0,
                hp_after_decode,
                hp_after_mask,
                hp_after_sample,
                hp_after_emit,
            ) {
                hp_n += 1;
                hp_decode_us += (td - t0).as_micros();
                hp_mask_us += (tm - td).as_micros();
                hp_sample_us += (ts - tm).as_micros();
                hp_emit_us += (te - ts).as_micros();
                hp_stopstr_us += (t_step_end - te).as_micros();
                hp_step_us += (t_step_end - t0).as_micros();
            }
        }
    }

    if host_profile_on && hp_n > 0 {
        let f = hp_n as f64;
        eprintln!(
            "\n=== HOST decode profile (n={hp_n}, post-warmup) ===\n  decode_logits  : {:>7.3} ms/tok\n  stop-mask      : {:>7.3} ms/tok\n  sampler.sample : {:>7.3} ms/tok\n  push_and_emit  : {:>7.3} ms/tok\n  stopstr_check  : {:>7.3} ms/tok\n  TOTAL_per_step : {:>7.3} ms/tok\n",
            hp_decode_us as f64 / f / 1000.0,
            hp_mask_us as f64 / f / 1000.0,
            hp_sample_us as f64 / f / 1000.0,
            hp_emit_us as f64 / f / 1000.0,
            hp_stopstr_us as f64 / f / 1000.0,
            hp_step_us as f64 / f / 1000.0,
        );
    }

    // Slot stays pooled; mutex releases on function return.

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

    Ok((
        finish_reason.to_owned(),
        prompt_tokens,
        generated.len() as u32,
    ))
    })();
    state.release_slot(slot_idx);
    result
}
