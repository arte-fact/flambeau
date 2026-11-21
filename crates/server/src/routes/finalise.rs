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
    relax_stop_mask: bool,
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
        ReasoningStyle::Channel => split_channel_reasoning(&mut text, markers, state, relax_stop_mask),
        ReasoningStyle::ThinkTag if enable_thinking => split_think_reasoning(&mut text, markers),
        ReasoningStyle::ThinkTag => {
            truncate_leaked_reasoning(&mut text, state.model.chat_stop_markers(), !relax_stop_mask);
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
    relax_stop_mask: bool,
) -> Option<String> {
    let Some(close) = text.find(markers.close) else {
        truncate_leaked_reasoning(text, state.model.chat_stop_markers(), !relax_stop_mask);
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

/// Reasoning close markers that end a non-thinking reply unless the stop
/// mask is relaxed (requests with `tools`): stop sequences of the decode
/// loop then, cut points here.
pub(super) const LEAKED_REASONING_MARKERS: [&str; 4] = ["</think>", "<end_thought>", "<end_think>", "</thought>"];

/// When thinking is off, cut the answer where the decode loop stopped it:
/// at an arch chat-template fragment, and at a leaked reasoning marker when
/// those are stop sequences (`reasoning_markers_stop`). The text is then the
/// one the streamed path sends.
fn truncate_leaked_reasoning(text: &mut String, chat_stop_markers: &[&str], reasoning_markers_stop: bool) {
    if reasoning_markers_stop {
        for marker in LEAKED_REASONING_MARKERS {
            if let Some(idx) = text.find(marker) {
                text.truncate(idx);
            }
        }
    }
    for marker in chat_stop_markers {
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

    /// The streamed `(reasoning_content, content)` of `raw`, fed in chunks of
    /// `chunk` bytes to the template-primed (in-think) parser of the SSE path.
    fn streamed_thinking(raw: &str, chunk: usize) -> (String, String) {
        use crate::tool_call_parser::{hermes::HermesJsonParser, ParserEvent, ReasoningTrim, ToolCallParser};
        let mut parser = HermesJsonParser::new_in_think();
        let mut trim = ReasoningTrim::default();
        let mut events = Vec::new();
        let bytes: Vec<char> = raw.chars().collect();
        for piece in bytes.chunks(chunk) {
            let piece: String = piece.iter().collect();
            events.extend(trim.apply(parser.push(&piece)));
        }
        events.extend(trim.apply(parser.finish()));
        let (mut think, mut text) = (String::new(), String::new());
        for e in events {
            match e {
                ParserEvent::ThinkDelta(s) => think.push_str(&s),
                ParserEvent::TextDelta(s) => text.push_str(&s),
                other => panic!("unexpected event {other:?}"),
            }
        }
        (think, text)
    }

    #[test]
    fn thinking_response_equals_streamed_deltas() {
        let markers = crate::model_handle::ReasoningMarkers {
            style: crate::model_handle::ReasoningStyle::ThinkTag,
            open: "<think>",
            close: "</think>",
        };
        let raws = [
            "\nLet me think.\n\nTwo steps.\n</think>\n\nThe answer is 4.\n",
            "plain reasoning</think>answer",
            "  spaced  out  \n</think>  \n  indented answer  ",
            "\n\n</think>\n\nOnly an answer.",
            "unterminated reasoning \n with a tail   ",
            "a</think>\n\n",
        ];
        for raw in raws {
            let mut text = raw.to_owned();
            let cot = super::split_think_reasoning(&mut text, markers).unwrap_or_default();
            for chunk in [1, 2, 3, 7, 64] {
                let (think, content) = streamed_thinking(raw, chunk);
                assert_eq!(think, cot, "reasoning of {raw:?} in {chunk}-char chunks");
                assert_eq!(content, text, "content of {raw:?} in {chunk}-char chunks");
            }
        }
    }

    /// Token i decodes to TOOLS_PIECES[i]: a non-thinking reply that leaks
    /// a think block mid-answer.
    const TOOLS_PIECES: &[&str] = &["Sure.", "<think>", "\n  hmm, ", "check  ", "</think>", "\n\n", "Done.", "\n"];

    /// Streamed `(content, reasoning)` and non-streamed `(content, reasoning)`
    /// of a non-thinking request, `relax_stop_mask` as for a request with
    /// `tools`.
    fn both_non_thinking(run: &[u32], relax_stop_mask: bool) -> ((String, String), (String, String)) {
        use crate::tool_call_parser::{
            hermes::HermesJsonParser, split_events_with_reasoning, ParserEvent, ReasoningTrim, ToolCallParser,
        };
        let decode = |ids: &[u32]| -> anyhow::Result<String> {
            Ok(ids.iter().map(|&t| TOOLS_PIECES[t as usize]).collect())
        };
        // The decode loop's stop sequences (decode_loop::stop_patterns).
        let patterns: Vec<String> = if relax_stop_mask {
            Vec::new()
        } else {
            super::LEAKED_REASONING_MARKERS.iter().map(|m| (*m).to_owned()).collect()
        };
        let stop_ids = [STOP_TOKEN];
        let mut sink = Collect(String::new());
        let mut e = Emitter::new(patterns, None, &stop_ids, 100, &mut sink);
        for &t in run {
            if e.accept(t, &decode).unwrap().is_some() {
                break;
            }
        }
        let kept: Vec<u32> = e.generated.iter().copied().filter(|&t| t != STOP_TOKEN).collect();
        drop(e);

        // SSE path: the streamed text, one char at a time, through the parser.
        let mut parser = HermesJsonParser::new();
        let mut trim = ReasoningTrim::default();
        let (mut s_content, mut s_reasoning) = (String::new(), String::new());
        let mut events = Vec::new();
        for c in sink.0.chars() {
            events.extend(trim.apply(parser.push(&c.to_string())));
        }
        events.extend(trim.apply(parser.finish()));
        for ev in events {
            match ev {
                ParserEvent::TextDelta(s) => s_content.push_str(&s),
                ParserEvent::ThinkDelta(s) => s_reasoning.push_str(&s),
                other => panic!("unexpected event {other:?}"),
            }
        }

        // Non-streamed path: finalise's cuts, then chat's parse.
        let mut text = decode(&kept).unwrap();
        super::truncate_leaked_reasoning(&mut text, &[], !relax_stop_mask);
        cut_at_stop_strings(&mut text, &[]);
        let mut parser = HermesJsonParser::new();
        let mut events = parser.push(&text);
        events.extend(parser.finish());
        let (content, reasoning, tool_calls) = split_events_with_reasoning(events);
        assert!(tool_calls.is_empty());
        ((s_content, s_reasoning), (content, reasoning))
    }

    #[test]
    fn non_thinking_response_equals_streamed_deltas_with_and_without_tools() {
        let runs: [&[u32]; 3] = [&[0, 1, 2, 3, 4, 5, 6, 7, STOP_TOKEN], &[0, 1, 2, 3, STOP_TOKEN], &[0, 5, 6, 7, STOP_TOKEN]];
        for run in runs {
            for relax_stop_mask in [false, true] {
                let (streamed, response) = both_non_thinking(run, relax_stop_mask);
                assert_eq!(response, streamed, "run {run:?} relax_stop_mask {relax_stop_mask}");
            }
        }
        // With tools the reply goes on past the leaked block, which is
        // reported as reasoning on both paths.
        let (streamed, _) = both_non_thinking(runs[0], true);
        assert_eq!(streamed, ("Sure.Done.\n".to_owned(), "hmm, check".to_owned()));
    }

    /// Streamed and non-streamed `(text, reasoning)` of the /v1/messages
    /// route for `raw`: thinking on (template-primed parser, finalise's
    /// split) or off (finalise's cuts, `relax_stop_mask` for tools), the
    /// non-streamed reply always parsed like the stream.
    fn both_messages(raw: &str, enable_thinking: bool, relax_stop_mask: bool) -> ((String, String), (String, String)) {
        use crate::tool_call_parser::{
            hermes::HermesJsonParser, split_events_with_reasoning, ParserEvent, ReasoningTrim, ToolCallParser,
        };
        let markers = crate::model_handle::ReasoningMarkers {
            style: crate::model_handle::ReasoningStyle::ThinkTag,
            open: "<think>",
            close: "</think>",
        };
        // The decode loop stops the text at its stop sequences.
        let mut streamed_text = raw.to_owned();
        if !relax_stop_mask && !enable_thinking {
            cut_at_stop_strings(
                &mut streamed_text,
                &super::LEAKED_REASONING_MARKERS.map(str::to_owned),
            );
        }
        let mut parser =
            if enable_thinking { HermesJsonParser::new_in_think() } else { HermesJsonParser::new() };
        let mut trim = ReasoningTrim::default();
        let mut events = Vec::new();
        for c in streamed_text.chars() {
            events.extend(trim.apply(parser.push(&c.to_string())));
        }
        events.extend(trim.apply(parser.finish()));
        let (mut s_text, mut s_reasoning) = (String::new(), String::new());
        for ev in events {
            match ev {
                ParserEvent::TextDelta(s) => s_text.push_str(&s),
                ParserEvent::ThinkDelta(s) => s_reasoning.push_str(&s),
                other => panic!("unexpected event {other:?}"),
            }
        }

        let mut text = raw.to_owned();
        let split = if enable_thinking {
            super::split_think_reasoning(&mut text, markers)
        } else {
            super::truncate_leaked_reasoning(&mut text, &[], !relax_stop_mask);
            None
        };
        let mut parser = HermesJsonParser::new();
        let mut events = parser.push(&text);
        events.extend(parser.finish());
        let (content, parsed, _) = split_events_with_reasoning(events);
        let reasoning = split.unwrap_or(parsed);
        ((s_text, s_reasoning), (content, reasoning))
    }

    #[test]
    fn messages_response_equals_streamed_blocks() {
        let raws = [
            "\nweighing it\n</think>\n\nAnswer.\n",
            "Sure.<think>\n  hmm  </think>\n\nDone.",
            "no reasoning at all  ",
        ];
        for raw in raws {
            for enable_thinking in [false, true] {
                for relax_stop_mask in [false, true] {
                    let (streamed, response) = both_messages(raw, enable_thinking, relax_stop_mask);
                    assert_eq!(
                        response, streamed,
                        "{raw:?} thinking {enable_thinking} tools {relax_stop_mask}"
                    );
                }
            }
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
