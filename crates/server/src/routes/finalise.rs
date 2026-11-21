//! Decode-output post-processing: stop-id stripping, `<think>` block
//! extraction, stop-string truncation, logprob entry construction.
//!
//! `finalise` is the seam between the engine's raw token stream and the
//! OpenAI-shaped `CompletionOutput` returned to the client.

use anyhow::{Context, Result};

use crate::api::{ChatLogProbContent, TopLogProb};
use crate::model_handle::{ReasoningMarkers, ReasoningStyle};
use crate::routes::ServerState;

/// Engine return shape: text, prompt_tokens, completion_tokens,
/// finish_reason, optional per-token logprobs, optional reasoning_content
/// (populated when `enable_thinking=true` and the model emitted a
/// `<think>...</think>` block; the leading reasoning is split off and
/// returned here while `text` keeps only the post-think answer), and the
/// caller-supplied stop string that matched (so the Anthropic path can
/// report `stop_reason="stop_sequence"`).
pub(super) type CompletionOutput = (
    String,
    u32,
    u32,
    String,
    Option<Vec<ChatLogProbContent>>,
    Option<String>,
    Option<String>,
);

/// Build one `ChatLogProbContent` entry for a single decoded step. Calls
/// `flambeau_runtime::sampling::build_distribution` to reproduce the
/// same penalty + temperature + top-k/top-p/min-p transforms the sampler
/// applied, then extracts the chosen token's log-probability and the
/// `top_n` most likely tokens (the chosen one included, as OpenAI does).
/// Under greedy sampling the distribution is the unfiltered softmax.
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
    // Greedy sampling reduces the distribution to its argmax; logprobs must
    // still report the model's softmax (penalties and bias applied).
    let unfiltered;
    let sampling = if sampling.is_greedy() {
        unfiltered = flambeau_runtime::Sampling {
            temperature: 1.0,
            top_p: None,
            top_k: None,
            min_p: None,
            ..sampling.clone()
        };
        &unfiltered
    } else {
        sampling
    };
    let dist = build_distribution(logits, sampling, history);
    if dist.is_empty() {
        return None;
    }
    let chosen_prob = dist.iter().find(|(id, _)| *id == chosen).map(|(_, p)| *p)?;
    let chosen_logprob = log_clamped(chosen_prob);

    let mut alts: Vec<(u32, f32)> = dist.clone();
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

    // Reasoning split is per-arch: the model reports its delimiters via
    // `reasoning_markers()`. gemma4 uses a harmony-style channel, qwen/
    // deepseek use `<think>…</think>`. Either way the chain-of-thought is
    // lifted into `reasoning_content` and the answer is kept clean. The
    // channel style is prompt-triggered (not flag-gated); the think style
    // splits only when `enable_thinking`, else truncates a leaked marker.
    let markers = state.model.reasoning_markers();
    let reasoning_content: Option<String> = match markers.style {
        ReasoningStyle::Channel => split_channel_reasoning(&mut text, markers, state),
        ReasoningStyle::ThinkTag if enable_thinking => split_think_reasoning(&mut text, markers),
        ReasoningStyle::ThinkTag => {
            truncate_leaked_reasoning(&mut text, state);
            None
        }
    };

    let matched_stop = cut_at_stop_strings(&mut text, stop_strings);
    Ok((
        text,
        prompt_tokens,
        completion_tokens,
        reason.to_owned(),
        logprobs,
        reasoning_content,
        matched_stop,
    ))
}

/// Cut `text` before the earliest caller stop sequence and return the one
/// that matched. Everything before it is kept, whitespace included, as
/// OpenAI and llama-server return it, so the response is the concatenation
/// of the deltas the streamed path sends. A trailing U+FFFD (a glyph whose
/// bytes the cut split) is dropped, as the stream holds it back.
fn cut_at_stop_strings(text: &mut String, stop_strings: &[String]) -> Option<String> {
    let mut earliest = text.len();
    let mut matched_stop: Option<String> = None;
    for s in stop_strings {
        if let Some(idx) = text.find(s.as_str()) {
            if idx < earliest {
                earliest = idx;
                matched_stop = Some(s.clone());
            }
        }
    }
    text.truncate(earliest);
    let complete = text.trim_end_matches('\u{FFFD}').len();
    text.truncate(complete);
    matched_stop
}

/// Split a harmony-style channel (`<|channel>{name}\n{cot} <channel|>
/// {answer}`) into reasoning + clean answer. With no close marker present
/// (the model never opened a thought) this degrades to leaked-marker
/// truncation. `text` is rewritten to the answer; the reasoning is returned.
fn split_channel_reasoning(
    text: &mut String,
    markers: ReasoningMarkers,
    state: &ServerState,
) -> Option<String> {
    let Some(close) = text.find(markers.close) else {
        truncate_leaked_reasoning(text, state);
        return None;
    };
    // `text[..close]` is `{open}{name}\n{reasoning}` (or just the marker for
    // an empty thought). Drop the open marker + channel-name line.
    let head = text[..close].trim();
    let head = head.strip_prefix(markers.open).unwrap_or(head);
    let cot = head
        .split_once('\n')
        .map(|(_name, body)| body.trim())
        .unwrap_or("")
        .to_string();
    let mut answer = text[close + markers.close.len()..].to_string();
    // Drop a leading channel name on the answer span (e.g. "final\n…").
    if let Some(nl) = answer.find('\n') {
        let h = answer[..nl].trim();
        if h.is_empty() || h == "final" {
            answer = answer[nl + 1..].to_string();
        }
    }
    // Truncate at any further channel scaffolding the model leaks.
    for m in [markers.open, markers.close] {
        if let Some(i) = answer.find(m) {
            answer.truncate(i);
        }
    }
    *text = answer.trim().to_string();
    if cot.is_empty() {
        None
    } else {
        Some(cot)
    }
}

/// Split a `<think>{cot}</think>{answer}` span. With no close marker the
/// whole output is treated as an unterminated thought (answer cleared).
/// `text` is rewritten to the answer; the reasoning is returned.
fn split_think_reasoning(text: &mut String, markers: ReasoningMarkers) -> Option<String> {
    let cot = if let Some(end_idx) = text.find(markers.close) {
        let cot = text[..end_idx]
            .trim_start_matches(markers.open)
            .trim()
            .to_string();
        let answer = text[end_idx + markers.close.len()..].trim_start().to_string();
        *text = answer;
        cot
    } else {
        let cot = text.trim_start_matches(markers.open).trim().to_string();
        text.clear();
        cot
    };
    if cot.is_empty() {
        None
    } else {
        Some(cot)
    }
}

/// When thinking is off, cut the answer at any leaked reasoning marker or
/// arch chat-template fragment so confused-mode scaffolding never reaches
/// the client.
fn truncate_leaked_reasoning(text: &mut String, state: &ServerState) {
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
}

#[cfg(test)]
mod tests {
    use super::cut_at_stop_strings;
    use crate::routes::token_emit::{Emitter, TokenSink};

    /// Token i decodes to PIECES[i].
    const PIECES: &[&str] = &["class LRUCache:", "\n", "    ", "def", " get", "(self):", "\n\n", " ", "x"];
    const STOP_TOKEN: u32 = 99;

    fn decode(ids: &[u32]) -> anyhow::Result<String> {
        Ok(ids.iter().map(|&t| PIECES[t as usize]).collect())
    }

    struct Collect(String);
    impl TokenSink for Collect {
        fn text(&mut self, delta: &str) -> bool {
            self.0.push_str(delta);
            true
        }
    }

    /// The streamed deltas of `run` and the non-streamed response
    /// (`finalise`'s cut over the decoded ids the same run kept).
    fn both(run: &[u32], stops: &[&str], max_tokens: usize) -> (String, String) {
        let patterns: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let stop_ids = [STOP_TOKEN];
        let mut sink = Collect(String::new());
        let mut e = Emitter::new(patterns.clone(), None, &stop_ids, max_tokens, &mut sink);
        let mut finish = None;
        for &t in run {
            finish = e.accept(t, &decode).unwrap();
            if finish.is_some() {
                break;
            }
        }
        assert!(finish.is_some(), "run {run:?} must end");
        let kept: Vec<u32> = e.generated.iter().copied().filter(|&t| t != STOP_TOKEN).collect();
        drop(e);
        let mut text = decode(&kept).unwrap();
        cut_at_stop_strings(&mut text, &patterns);
        (sink.0, text)
    }

    #[test]
    fn response_equals_streamed_deltas_whitespace_included() {
        let cases: [(&[u32], &[&str], usize); 5] = [
            // Stop sequence right after indentation: the indentation stays.
            (&[0, 1, 2, 3, 4, 5], &["def "], 100),
            // Stop sequence that starts with whitespace: cut before it.
            (&[0, 1, 2, 3, 4, 5], &["\n    def"], 100),
            // Trailing blank lines before a stop token.
            (&[0, 5, 6, STOP_TOKEN], &[], 100),
            // Trailing space at max_tokens.
            (&[0, 8, 7, 8], &["zzz"], 3),
            // Earliest of several stop sequences.
            (&[0, 1, 2, 3, 4, 5], &["(self)", " get"], 100),
        ];
        for (run, stops, max_tokens) in cases {
            let (streamed, response) = both(run, stops, max_tokens);
            assert_eq!(response, streamed, "run {run:?} stops {stops:?}");
        }
    }

    #[test]
    fn stop_cut_keeps_trailing_whitespace() {
        let mut text = "class LRUCache:\n    def get(self):".to_owned();
        let matched = cut_at_stop_strings(&mut text, &["def ".to_owned()]);
        assert_eq!(text, "class LRUCache:\n    ");
        assert_eq!(matched.as_deref(), Some("def "));
    }
}
