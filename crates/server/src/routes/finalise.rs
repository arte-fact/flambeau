//! Decode-output post-processing: stop-id stripping, `<think>` block
//! extraction, stop-string truncation, logprob entry construction.
//!
//! `finalise` is the seam between the engine's raw token stream and the
//! OpenAI-shaped `CompletionOutput` returned to the client.

use anyhow::{Context, Result};

use crate::api::{ChatLogProbContent, TopLogProb};
use crate::routes::ServerState;

/// Engine return shape: text, prompt_tokens, completion_tokens,
/// finish_reason, optional per-token logprobs, optional reasoning_content
/// (populated when `enable_thinking=true` and the model emitted a
/// `<think>...</think>` block; the leading reasoning is split off and
/// returned here while `text` keeps only the post-think answer).
pub(super) type CompletionOutput = (
    String,
    u32,
    u32,
    String,
    Option<Vec<ChatLogProbContent>>,
    Option<String>,
);

/// Build one `ChatLogProbContent` entry for a single decoded step. Calls
/// `flambeau_runtime::sampling::build_distribution` to reproduce the
/// same penalty + temperature + top-k/top-p/min-p transforms the sampler
/// applied, then extracts the chosen token's log-probability and the
/// top-`top_n` alternatives.
///
/// Returns `None` if the chosen token is outside the post-filter
/// distribution (defensive — shouldn't happen because the sampler drew
/// from the same distribution). Logprobs below -100 are clamped to -100.
pub(super) fn build_logprob_entry(
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
    let chosen_prob = dist.iter().find(|(id, _)| *id == chosen).map(|(_, p)| *p)?;
    let chosen_logprob = log_clamped(chosen_prob);

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
    alts.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
pub(super) fn log_clamped(p: f32) -> f32 {
    if p <= 0.0 {
        -100.0
    } else {
        p.ln().max(-100.0)
    }
}

/// Truncate `text` to a head/tail preview suitable for log lines.
/// Replaces newlines with `\n` for single-line readability.
pub(super) fn preview_text(text: &str, head: usize) -> String {
    let n_chars = text.chars().count();
    let escape = |s: &str| s.replace('\n', "\\n");
    if n_chars <= head * 2 + 20 {
        return escape(text);
    }
    let head_str: String = text.chars().take(head).collect();
    let tail_str: String = text.chars().skip(n_chars - head).collect();
    format!(
        "{} … <{}c omitted> … {}",
        escape(&head_str),
        n_chars - head * 2,
        escape(&tail_str)
    )
}

/// Strip stop tokens, split out the `<think>...</think>` block (when
/// thinking is enabled), truncate at any leaked reasoning marker or
/// arch-specific chat-template fragment, then apply caller-supplied
/// stop sequences. Returns the OpenAI-shaped `CompletionOutput` tuple.
pub(super) fn finalise(
    state: &ServerState,
    prompt_tokens: u32,
    mut generated: Vec<u32>,
    reason: &str,
    stop_strings: &[String],
    mut logprobs: Option<Vec<ChatLogProbContent>>,
    enable_thinking: bool,
) -> Result<CompletionOutput> {
    let stop_ids = &state.tokenizer.stop_ids;
    let completion_tokens = generated.len() as u32;
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
            generated.retain(|t| !stop_ids.contains(t));
            lp.clear();
        }
    } else {
        generated.retain(|t| !stop_ids.contains(t));
    }
    let mut text = state.tokenizer.decode(&generated).context("decode")?;

    // Reasoning-channel split is per output FORMAT, detected from the text
    // (not an arch string): gemma4 emits a harmony-style channel
    // `<|channel>thought {cot} <channel|> {answer}`; qwen/deepseek emit
    // `<think> {cot} </think> {answer}`. Either way the chain-of-thought is
    // lifted into `reasoning_content` and the user-facing answer is kept clean.
    let reasoning_content: Option<String> = if let Some(close) = text.find("<channel|>") {
        // `text[..close]` is `<|channel>{name}\n{reasoning}` (or just the
        // marker for an empty thought). Drop the open marker + channel-name
        // line; keep the reasoning body.
        let head = text[..close].trim();
        let head = head.strip_prefix("<|channel>").unwrap_or(head);
        let cot = head
            .split_once('\n')
            .map(|(_name, body)| body.trim())
            .unwrap_or("")
            .to_string();
        let mut answer = text[close + "<channel|>".len()..].to_string();
        // Drop a leading channel name on the answer span (e.g. "final\n…").
        if let Some(nl) = answer.find('\n') {
            let head = answer[..nl].trim();
            if head.is_empty() || head == "final" {
                answer = answer[nl + 1..].to_string();
            }
        }
        // Truncate at any further channel scaffolding the model leaks.
        for m in ["<|channel>", "<channel|>"] {
            if let Some(i) = answer.find(m) {
                answer.truncate(i);
            }
        }
        text = answer.trim().to_string();
        if cot.is_empty() {
            None
        } else {
            Some(cot)
        }
    } else if enable_thinking {
        if let Some(end_idx) = text.find("</think>") {
            let cot_raw = &text[..end_idx];
            let cot = cot_raw.trim_start_matches("<think>").trim().to_string();
            let answer_start = end_idx + "</think>".len();
            let answer = text[answer_start..].trim_start().to_string();
            text = answer;
            if cot.is_empty() {
                None
            } else {
                Some(cot)
            }
        } else {
            let cot = text.trim_start_matches("<think>").trim().to_string();
            text = String::new();
            if cot.is_empty() {
                None
            } else {
                Some(cot)
            }
        }
    } else {
        for marker in ["</think>", "<end_thought>", "<end_think>", "</thought>"] {
            if let Some(idx) = text.find(marker) {
                text.truncate(idx);
            }
        }
        for marker in state.model.chat_stop_markers() {
            if let Some(idx) = text.find(*marker) {
                text.truncate(idx);
            }
        }
        None
    };

    let mut earliest = text.len();
    for s in stop_strings {
        if let Some(idx) = text.find(s.as_str()) {
            if idx < earliest {
                earliest = idx;
            }
        }
    }
    text.truncate(earliest);
    let trimmed_len = text.trim_end().len();
    text.truncate(trimmed_len);
    Ok((
        text,
        prompt_tokens,
        completion_tokens,
        reason.to_owned(),
        logprobs,
        reasoning_content,
    ))
}
