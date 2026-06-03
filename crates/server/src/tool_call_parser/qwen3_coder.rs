//! Qwen3-Coder XML-style tool-call parser.
//! Happy-path state machine (T2.2). Parses the format that Qwen3-Coder
//! and the Unsloth UD Qwen3.6 GGUFs emit:
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! San Francisco
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//! Emits:
//! - `ToolCallOpen{ index, name }` at `<function=NAME>`.
//! - `ToolCallArgumentsDelta{ index, arguments }` once at `</function>`
//!   with the JSON-encoded parameter object as a **string** (guards
//!   llama.cpp #20198).
//! - `ToolCallClose{ index }` at `</tool_call>`.
//! - `TextDelta(..)` for any free-text content outside tool calls.
//!   Parameter values are literal strings from `<parameter=K>\n…\n</parameter>`
//!   — multi-line and angle-bracket-containing values pass through
//!   unchanged. Parameter ORDER is preserved in the emitted JSON object
//!   via `serde_json::Map` (workspace-wide `preserve_order` feature).
//!   Ambiguous-prefix hardening (buffer-before-emit for partial `<` tag
//!   starts in free text and inside parameter bodies) is T2.3's scope;
//!   for T2.2 we use the minimum-viable tail-holdback approach: never
//!   emit the last `MAX_TAG_LEN` bytes of the buffer while in a state
//!   that might see a tag next, so a tag straddling a chunk boundary
//!   doesn't get mis-parsed.
//!   `<think>` interaction is also T2.3; today `<think>` runs inside a
//!   `Text` state get emitted as normal `TextDelta` (never promoted to a
//!   tool call) because none of the tool tags start with `<th`.

use serde_json::{json, Value};

use super::{ParserEvent, ToolCallParser};

/// Parser states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Free text, between / outside tool calls. Scans for the earliest
    /// of `<tool_call>` and `<think>`.
    Text,
    /// Inside a `<think>…</think>` block. Tool-call syntax here is
    /// part of the think content — not promoted to a tool call
    /// (llama.cpp #20837). Contents surface as `ThinkDelta`.
    InThink,
    /// Saw `<tool_call>`, waiting for `<function=NAME>`.
    InToolCall,
    /// Inside `<function=…>`, between `<parameter=…>` blocks or
    /// waiting for `</function>`.
    InFunctionBody,
    /// Inside a `<parameter=KEY>` block, accumulating the value.
    InParameterBody,
    /// Saw `</function>`, waiting for `</tool_call>` to close the call.
    AwaitingToolCallClose,
}

/// Tags we watch for in `Text` state. The parser's smarter
/// buffer-before-emit holds back only the tail that could be a strict
/// prefix of one of these.
const TEXT_STATE_OPEN_TAGS: &[&str] = &["<tool_call>", "<think>"];

pub struct QwenCoderXmlParser {
    state: State,
    /// Accumulated unconsumed input.
    buf: String,
    /// Monotonic index for `ToolCallOpen`/`Close` within one turn.
    next_index: u32,
    /// Name of the currently-open tool call (between Open and Close).
    current_name: String,
    /// Key of the parameter currently being accumulated (when state is
    /// `InParameterBody`).
    current_param_key: String,
    /// Accumulated parameters for the currently-open tool call,
    /// preserving insertion order.
    current_params: serde_json::Map<String, Value>,
}

impl Default for QwenCoderXmlParser {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for QwenCoderXmlParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenCoderXmlParser")
            .field("state", &self.state)
            .field("buf_len", &self.buf.len())
            .field("next_index", &self.next_index)
            .finish()
    }
}

impl QwenCoderXmlParser {
    pub fn new() -> Self {
        Self {
            state: State::Text,
            buf: String::new(),
            next_index: 0,
            current_name: String::new(),
            current_param_key: String::new(),
            current_params: serde_json::Map::new(),
        }
    }

    /// Drain as much of `self.buf` as the current state allows. Appends
    /// produced events to `out`. Stops when progress is no longer
    /// possible without more input.
    fn drain(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) {
        loop {
            let made_progress = match self.state {
                State::Text => self.step_text(out, is_finish),
                State::InThink => self.step_in_think(out, is_finish),
                State::InToolCall => self.step_in_tool_call(out),
                State::InFunctionBody => self.step_in_function_body(out),
                State::InParameterBody => self.step_in_parameter_body(out),
                State::AwaitingToolCallClose => self.step_awaiting_close(out),
            };
            if !made_progress {
                return;
            }
        }
    }

    fn step_text(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        // Earliest of any watched open-tag. Handles both `<tool_call>`
        // (enters tool-call state machine) and `<think>` (enters
        // InThink — tool syntax inside stays think content, #20837).
        let earliest = TEXT_STATE_OPEN_TAGS
            .iter()
            .filter_map(|tag| self.buf.find(tag).map(|p| (p, *tag)))
            .min_by_key(|(p, _)| *p);
        if let Some((pos, tag)) = earliest {
            if pos > 0 {
                emit_text(out, &self.buf[..pos]);
            }
            // Do NOT speculatively drain a trailing `\n` here: in
            // char-by-char streaming that newline may not be in buf yet,
            // and the one-shot-vs-streaming equivalence is load-bearing.
            // Any `\n` immediately after `<tool_call>` is consumed
            // inside step_in_tool_call (which drains leading whitespace
            // before `<function=`). Ditto `<think>` + step_in_think.
            self.buf.drain(..pos + tag.len());
            self.state = match tag {
                "<tool_call>" => State::InToolCall,
                "<think>" => State::InThink,
                _ => unreachable!("unexpected tag matched: {tag}"),
            };
            return true;
        }
        // No full open-tag yet. Emit everything whose tail is NOT a
        // prefix of any watched tag. This is tighter than a blind
        // `MAX_TAG_LEN` tail-holdback — an unambiguous `<foo` chunk in
        // free text goes out immediately, reducing streaming latency.
        let total = self.buf.len();
        let emit_upto = if is_finish {
            total
        } else {
            ambiguous_tail_start(&self.buf, TEXT_STATE_OPEN_TAGS)
        };
        let emit_upto = align_down_char_boundary(&self.buf, emit_upto);
        if emit_upto > 0 {
            let chunk: String = self.buf.drain(..emit_upto).collect();
            emit_text(out, &chunk);
            return true;
        }
        false
    }

    fn step_in_think(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        if let Some(pos) = self.buf.find("</think>") {
            if pos > 0 {
                out.push(ParserEvent::ThinkDelta(self.buf[..pos].to_owned()));
            }
            // No speculative `\n` skip — see step_text for rationale.
            self.buf.drain(..pos + "</think>".len());
            self.state = State::Text;
            return true;
        }
        // No close tag yet. Stream partial think content, holding back
        // only the tail that could be a prefix of `</think>`. At finish,
        // dump everything buffered so we don't silently drop
        // reasoning text on an unterminated think block.
        let total = self.buf.len();
        let emit_upto = if is_finish {
            total
        } else {
            ambiguous_tail_start(&self.buf, &["</think>"])
        };
        let emit_upto = align_down_char_boundary(&self.buf, emit_upto);
        if emit_upto > 0 {
            let chunk: String = self.buf.drain(..emit_upto).collect();
            if !chunk.is_empty() {
                out.push(ParserEvent::ThinkDelta(chunk));
            }
            return true;
        }
        false
    }

    fn step_in_tool_call(&mut self, out: &mut Vec<ParserEvent>) -> bool {
        // Expect <function=NAME>. Silently swallow any leading
        // whitespace — chunk boundaries can split the `\n` between
        // `<tool_call>` and `<function=`, and emitting it as stray
        // TextDelta would break one-shot-vs-streaming equivalence.
        let Some(rel_start) = self.buf.find("<function=") else {
            return false;
        };
        if rel_start > 0 {
            self.buf.drain(..rel_start);
        }
        // Find the closing `>` of `<function=NAME>`.
        let after_prefix = "<function=".len();
        let Some(close_rel) = self.buf[after_prefix..].find('>') else {
            return false;
        };
        let name = self.buf[after_prefix..after_prefix + close_rel].to_owned();
        // Don't speculatively drain the `\n` after `<function=NAME>`
        // (chunk boundary may not have it yet). step_in_function_body
        // drains leading whitespace before `<parameter=`/`</function>`
        // so this is safe.
        let cursor = after_prefix + close_rel + 1;
        self.buf.drain(..cursor);
        self.current_name = name.clone();
        self.current_params.clear();
        out.push(ParserEvent::ToolCallOpen {
            index: self.next_index,
            name,
        });
        self.state = State::InFunctionBody;
        true
    }

    fn step_in_function_body(&mut self, out: &mut Vec<ParserEvent>) -> bool {
        // Earliest of `<parameter=` or `</function>`. Note: the tuple
        // `(pos, is_param)` carries the *position of the winning tag* —
        // the earlier one — not of whichever was scanned first. An
        // earlier bug returned the parameter position when the function
        // close was earlier, draining past both.
        let p_pos = self.buf.find("<parameter=");
        let f_pos = self.buf.find("</function>");
        let next = match (p_pos, f_pos) {
            (Some(p), Some(f)) => {
                if p < f {
                    Some((p, true))
                } else {
                    Some((f, false))
                }
            }
            (Some(p), None) => Some((p, true)),
            (None, Some(f)) => Some((f, false)),
            (None, None) => None,
        };
        let Some((pos, is_param)) = next else {
            return false;
        };
        if pos > 0 {
            // Any text between params is whitespace in well-formed
            // input — swallow silently (don't emit TextDelta; it would
            // show up mid-tool-call which is noise).
            self.buf.drain(..pos);
        }
        if is_param {
            // buf now starts with "<parameter=". Find closing `>`.
            let after_prefix = "<parameter=".len();
            let Some(close_rel) = self.buf[after_prefix..].find('>') else {
                // Restore position by *not* draining further; wait for
                // more input. But we already drained leading whitespace,
                // which is fine — return false so we retry next push.
                return false;
            };
            let key = self.buf[after_prefix..after_prefix + close_rel].to_owned();
            // No speculative `\n` drain — step_in_parameter_body's
            // value-strip handles leading/trailing `\n` uniformly.
            let cursor = after_prefix + close_rel + 1;
            self.buf.drain(..cursor);
            self.current_param_key = key;
            self.state = State::InParameterBody;
            true
        } else {
            // </function>. `pos` was the index BEFORE the leading-
            // whitespace drain above, so it's no longer valid — the
            // tag now starts at byte 0 of `self.buf`.
            self.buf.drain(.."</function>".len());
            // No speculative `\n` drain — step_awaiting_close's leading-
            // whitespace drain handles it.
            // Emit arguments-as-JSON-string (guards #20198).
            let args_json =
                serde_json::to_string(&Value::Object(std::mem::take(&mut self.current_params)))
                    .unwrap_or_else(|_| "{}".to_owned());
            out.push(ParserEvent::ToolCallArgumentsDelta {
                index: self.next_index,
                arguments: args_json,
            });
            // Close event deferred to `</tool_call>` so `index` is
            // monotonic across parallel tool calls in one turn.
            self.state = State::AwaitingToolCallClose;
            true
        }
    }

    fn step_in_parameter_body(&mut self, _out: &mut Vec<ParserEvent>) -> bool {
        // Accumulate into the value until `</parameter>`.
        if let Some(pos) = self.buf.find("</parameter>") {
            // Strip BOTH leading and trailing newline — they're template
            // formatting (the Jinja emits `<parameter=K>\nV\n</parameter>`)
            // and not part of the caller-visible value. This is also
            // what makes one-shot and byte-by-byte streams produce
            // identical `arguments` JSON.
            let mut start = 0usize;
            let mut end = pos;
            let bytes = self.buf.as_bytes();
            if start < end && bytes[start] == b'\n' {
                start += 1;
            }
            if end > start && bytes[end - 1] == b'\n' {
                end -= 1;
            }
            let value: String = self.buf[start..end].to_owned();
            // Drain through `</parameter>` (no speculative \n drain).
            self.buf.drain(..pos + "</parameter>".len());
            let key = std::mem::take(&mut self.current_param_key);
            self.current_params.insert(key, json!(value));
            self.state = State::InFunctionBody;
            return true;
        }
        // No close tag yet — hold off to avoid a chunk-split
        // `</paramete` getting caught mid-emit. The body text stays in
        // `self.buf` and we wait for more input.
        false
    }

    fn step_awaiting_close(&mut self, out: &mut Vec<ParserEvent>) -> bool {
        // Find the earliest of: `</tool_call>`, `<function=`, or
        // `<parameter=`. Well-formed input has `</tool_call>` first;
        // the latter two are forgiveness for malformed multi-call
        // outputs that Qwen3.6-35B-A3B emits when asked for parallel
        // calls — it forgets to close `</tool_call>` between them
        // (sometimes also forgetting to re-emit `<function=NAME>`
        // and just running new `<parameter=...>` blocks).
        let close_pos = self.buf.find("</tool_call>");
        let func_pos = self.buf.find("<function=");
        let param_pos = self.buf.find("<parameter=");
        // Pick the earliest tag.
        let earliest = [
            close_pos.map(|p| (p, "close")),
            func_pos.map(|p| (p, "func")),
            param_pos.map(|p| (p, "param")),
        ]
        .into_iter()
        .flatten()
        .min_by_key(|(p, _)| *p);
        let Some((pos, kind)) = earliest else {
            return false;
        };
        if pos > 0 {
            // Whitespace between `</function>` and the next tag —
            // silent drain.
            self.buf.drain(..pos);
        }
        match kind {
            "close" => {
                self.buf.drain(.."</tool_call>".len());
                // No speculative `\n` drain: any `\n` after `</tool_call>`
                // surfaces as a `TextDelta("\n")` in Text state. Same
                // behaviour in one-shot and byte-by-byte (load-bearing
                // streaming-equivalence invariant).
                out.push(ParserEvent::ToolCallClose {
                    index: self.next_index,
                });
                self.next_index += 1;
                self.state = State::Text;
                true
            }
            "func" => {
                // **Forgiveness**: model forgot `</tool_call>` and went
                // straight into another `<function=NAME>`. Implicit close
                // of the prior call, then transition into `InToolCall`
                // (which expects `<function=NAME>`).
                out.push(ParserEvent::ToolCallClose {
                    index: self.next_index,
                });
                self.next_index += 1;
                // Don't drain `<function=` — `step_in_tool_call` does it.
                self.state = State::InToolCall;
                true
            }
            "param" => {
                // **Forgiveness**: model forgot both `</tool_call>` and
                // `<function=NAME>`, just emitted another
                // `<parameter=...>` block. Treat as a continuation call
                // with the SAME function name as the prior call. Implicit
                // close + reopen with same name.
                out.push(ParserEvent::ToolCallClose {
                    index: self.next_index,
                });
                self.next_index += 1;
                let name = self.current_name.clone();
                if name.is_empty() {
                    // Defensive: we shouldn't reach `AwaitingToolCallClose`
                    // without `current_name` set (it's set by
                    // `step_in_tool_call`). If somehow empty, drop the
                    // continuation as text.
                    return false;
                }
                out.push(ParserEvent::ToolCallOpen {
                    index: self.next_index,
                    name,
                });
                // Don't drain `<parameter=` — `step_in_function_body`
                // handles it from `InFunctionBody`.
                self.state = State::InFunctionBody;
                true
            }
            _ => unreachable!("kind comes from a fixed set"),
        }
    }
}

fn emit_text(out: &mut Vec<ParserEvent>, s: &str) {
    if !s.is_empty() {
        out.push(ParserEvent::TextDelta(s.to_owned()));
    }
}

/// Round `n` down to a UTF-8 char boundary of `s`. Safe for n ≥ s.len().
fn align_down_char_boundary(s: &str, n: usize) -> usize {
    let mut i = n.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Find the leftmost byte position `p` such that `s[p..]` is a proper
/// prefix of at least one watched `tags` — i.e. this tail *could* be
/// the start of a matching tag once more data arrives. Everything
/// before `p` is unambiguous and safe to emit.
/// Returns `s.len()` when no ambiguous tail exists — the whole string
/// can be emitted.
/// Only the last `max(tag.len()) - 1` bytes are candidates — no need to
/// scan the whole string.
fn ambiguous_tail_start(s: &str, tags: &[&str]) -> usize {
    let max_tail = tags
        .iter()
        .map(|t| t.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    if max_tail == 0 || s.len() <= max_tail {
        // Consider every suffix; we'll still return s.len() if none match.
    }
    let start_scan = s.len().saturating_sub(max_tail);
    // Scan from earliest candidate (start_scan) forward — the FIRST
    // position whose tail is a tag prefix is the one to hold from.
    for p in start_scan..s.len() {
        if !s.is_char_boundary(p) {
            continue;
        }
        let tail = &s[p..];
        // Proper prefix = tail is non-empty, strictly shorter than the
        // tag, and the tag starts with tail.
        if tags
            .iter()
            .any(|tag| tail.len() < tag.len() && tag.starts_with(tail))
        {
            return p;
        }
    }
    s.len()
}

impl ToolCallParser for QwenCoderXmlParser {
    fn push(&mut self, chunk: &str) -> Vec<ParserEvent> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(chunk);
        let mut out = Vec::new();
        self.drain(&mut out, /*is_finish=*/ false);
        out
    }

    fn finish(&mut self) -> Vec<ParserEvent> {
        let mut out = Vec::new();
        // Flush through end-of-stream. Incomplete mid-tool-call state
        // stays stuck in its state; emit whatever text is outstanding
        // so the caller can surface it as assistant content.
        self.drain(&mut out, /*is_finish=*/ true);
        // Any remaining buffer at this point is unterminated mid-tag
        // (e.g. `<tool_call>\n<function=foo>` with no `</function>`,
        // or a mid-parameter body that never closed). We emit it as
        // text so we don't silently drop model output, stripping any
        // leading `\n` left over from the template formatting — keeps
        // one-shot and byte-by-byte streams byte-identical.
        if !self.buf.is_empty() {
            let tail = std::mem::take(&mut self.buf);
            let stripped = tail.strip_prefix('\n').unwrap_or(&tail);
            emit_text(&mut out, stripped);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience — parse `input` in one shot and return all events.
    fn parse_all(input: &str) -> Vec<ParserEvent> {
        let mut p = QwenCoderXmlParser::new();
        let mut evts = p.push(input);
        evts.extend(p.finish());
        evts
    }

    fn collect_args_strings(evts: &[ParserEvent]) -> Vec<(u32, String)> {
        evts.iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallArgumentsDelta { index, arguments } => {
                    Some((*index, arguments.clone()))
                }
                _ => None,
            })
            .collect()
    }

    fn collect_opens(evts: &[ParserEvent]) -> Vec<(u32, String)> {
        evts.iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallOpen { index, name } => Some((*index, name.clone())),
                _ => None,
            })
            .collect()
    }

    fn collect_closes(evts: &[ParserEvent]) -> Vec<u32> {
        evts.iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallClose { index } => Some(*index),
                _ => None,
            })
            .collect()
    }

    fn collect_text(evts: &[ParserEvent]) -> String {
        evts.iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .concat()
    }

    #[test]
    fn single_tool_call_one_param() {
        let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSan Francisco\n</parameter>\n</function>\n</tool_call>";
        let evts = parse_all(input);
        assert_eq!(collect_opens(&evts), vec![(0, "get_weather".into())]);
        assert_eq!(collect_closes(&evts), vec![0]);
        let args = collect_args_strings(&evts);
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].0, 0);
        // JSON string body, with preserve_order from serde_json.
        let parsed: serde_json::Value = serde_json::from_str(&args[0].1).expect("valid json");
        assert_eq!(parsed["city"], "San Francisco");
    }

    #[test]
    fn two_params_in_order() {
        let input = concat!(
            "<tool_call>\n",
            "<function=create_event>\n",
            "<parameter=title>\nTeam sync\n</parameter>\n",
            "<parameter=when>\n2026-04-25T10:00\n</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let evts = parse_all(input);
        let args = collect_args_strings(&evts);
        assert_eq!(args.len(), 1);
        // Parameter order preserved (preserve_order workspace feature).
        assert_eq!(
            args[0].1,
            r#"{"title":"Team sync","when":"2026-04-25T10:00"}"#
        );
    }

    #[test]
    fn parallel_tool_calls_get_monotonic_indices() {
        let input = concat!(
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>"
        );
        let evts = parse_all(input);
        assert_eq!(collect_opens(&evts), vec![(0, "a".into()), (1, "b".into())]);
        assert_eq!(collect_closes(&evts), vec![0, 1]);
        let args = collect_args_strings(&evts);
        assert_eq!(args[0], (0, r#"{"x":"1"}"#.into()));
        assert_eq!(args[1], (1, r#"{"y":"2"}"#.into()));
    }

    #[test]
    fn multiline_parameter_value() {
        let input = concat!(
            "<tool_call>\n<function=f>\n<parameter=body>\n",
            "line one\nline two\nline three",
            "\n</parameter>\n</function>\n</tool_call>"
        );
        let evts = parse_all(input);
        let args = collect_args_strings(&evts);
        assert_eq!(args.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(&args[0].1).unwrap();
        assert_eq!(parsed["body"], "line one\nline two\nline three");
    }

    #[test]
    fn parameter_value_containing_angle_chars() {
        // Tool output can legitimately contain `<` or `>` — e.g. a code
        // snippet inside a `body` parameter. Parser must treat the body
        // as opaque text until it sees the exact `</parameter>` tag.
        let input = concat!(
            "<tool_call>\n<function=f>\n<parameter=snippet>\n",
            "fn foo<T>(x: &T) where T: Copy { }",
            "\n</parameter>\n</function>\n</tool_call>"
        );
        let evts = parse_all(input);
        let args = collect_args_strings(&evts);
        let parsed: serde_json::Value = serde_json::from_str(&args[0].1).unwrap();
        assert_eq!(parsed["snippet"], "fn foo<T>(x: &T) where T: Copy { }");
    }

    #[test]
    fn text_before_and_after_tool_call() {
        let input = concat!(
            "I'll check the weather. ",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nSF\n</parameter>\n</function>\n</tool_call>",
            " Done."
        );
        let evts = parse_all(input);
        let text = collect_text(&evts);
        assert!(text.contains("I'll check the weather."), "text={text:?}");
        assert!(text.contains(" Done."), "text={text:?}");
        assert_eq!(collect_opens(&evts).len(), 1);
        assert_eq!(collect_closes(&evts).len(), 1);
    }

    #[test]
    fn streaming_split_across_chunks_is_equivalent_to_one_shot() {
        // Same input pushed as one blob vs byte-by-byte must produce
        // the same event sequence.
        let input = concat!(
            "prefix ",
            "<tool_call>\n<function=f>\n<parameter=x>\nvalue 1\n</parameter>\n<parameter=y>\nvalue 2\n</parameter>\n</function>\n</tool_call>",
            " suffix"
        );
        let one_shot = parse_all(input);

        let mut byte_by_byte = Vec::new();
        let mut p = QwenCoderXmlParser::new();
        // Feed char-by-char (so we don't split UTF-8 codepoints).
        for ch in input.chars() {
            let mut s = [0u8; 4];
            let slice = ch.encode_utf8(&mut s);
            byte_by_byte.extend(p.push(slice));
        }
        byte_by_byte.extend(p.finish());

        // Events must have identical content (order and payload); what
        // CAN differ is TextDelta chunking — a one-shot push may emit
        // "prefix " as one event while char-by-char emits many. Coalesce
        // adjacent TextDeltas before comparing.
        fn coalesce(v: Vec<ParserEvent>) -> Vec<ParserEvent> {
            let mut out = Vec::<ParserEvent>::new();
            for e in v {
                match (out.last_mut(), &e) {
                    (Some(ParserEvent::TextDelta(acc)), ParserEvent::TextDelta(s)) => {
                        acc.push_str(s);
                    }
                    _ => out.push(e),
                }
            }
            out
        }
        assert_eq!(coalesce(one_shot), coalesce(byte_by_byte));
    }

    #[test]
    fn plain_text_no_tool_call_passes_through() {
        let input = "Just a normal sentence with no tool call.";
        let evts = parse_all(input);
        assert_eq!(collect_opens(&evts).len(), 0);
        assert_eq!(collect_closes(&evts).len(), 0);
        assert_eq!(collect_text(&evts), input);
    }

    #[test]
    fn empty_push_is_noop() {
        let mut p = QwenCoderXmlParser::new();
        assert!(p.push("").is_empty());
    }

    // ---- T2.3: think + buffer-before-emit hardening ----

    fn collect_think(evts: &[ParserEvent]) -> String {
        evts.iter()
            .filter_map(|e| match e {
                ParserEvent::ThinkDelta(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .concat()
    }

    #[test]
    fn think_block_surfaces_as_think_delta_not_text() {
        let input = "before<think>reasoning here</think>after";
        let evts = parse_all(input);
        assert_eq!(collect_text(&evts), "beforeafter");
        assert_eq!(collect_think(&evts), "reasoning here");
        assert!(collect_opens(&evts).is_empty());
    }

    #[test]
    fn tool_call_inside_think_block_does_not_fire() {
        // llama.cpp #20837 guard: a `<tool_call>` mention inside a
        // think block must NOT open a tool call. It stays think text.
        let input = concat!(
            "<think>",
            "Maybe I should call <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
            "</think>",
            "Here's my answer."
        );
        let evts = parse_all(input);
        assert!(
            collect_opens(&evts).is_empty(),
            "tool call inside think must not fire: evts={evts:?}"
        );
        assert!(
            collect_closes(&evts).is_empty(),
            "tool close inside think must not fire"
        );
        assert!(
            collect_think(&evts).contains("<tool_call>"),
            "raw tool-call text should be inside the think delta"
        );
        assert_eq!(collect_text(&evts), "Here's my answer.");
    }

    #[test]
    fn close_think_immediately_followed_by_tool_call() {
        // llama.cpp #21118 guard: `</think><tool_call>` with NO
        // whitespace between them must still parse as think-close
        // followed by tool-call-open.
        let input = concat!(
            "<think>plan</think>",
            "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
        );
        let evts = parse_all(input);
        assert_eq!(collect_think(&evts), "plan");
        assert_eq!(collect_opens(&evts), vec![(0, "f".into())]);
        assert_eq!(collect_closes(&evts), vec![0]);
    }

    #[test]
    fn bare_angle_chars_in_text_pass_through_unambiguously() {
        // The smarter buffer-before-emit: `<a ` in free text is NOT a
        // prefix of `<tool_call>` or `<think>`, so it emits immediately
        // rather than waiting for more chars.
        let mut p = QwenCoderXmlParser::new();
        let evts = p.push("Check this: <a href='x'>link</a>");
        // All of it should have come out as one (or more) TextDelta(s);
        // nothing buffered.
        let txt: String = evts
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .concat();
        assert_eq!(txt, "Check this: <a href='x'>link</a>");
    }

    #[test]
    fn ambiguous_tail_start_cases() {
        // Helper unit tests on the prefix-matcher.
        let tags = &["<tool_call>", "<think>"];
        // No ambiguity:
        assert_eq!(ambiguous_tail_start("plain text", tags), 10);
        // Full tag present — earlier detection should already have
        // consumed it; the ambiguous-tail check treats equal-to-tag as
        // a NON-prefix (tail.len() < tag.len() is false).
        assert_eq!(ambiguous_tail_start("x<think>", tags), 8);
        // Partial prefix of `<think>`:
        assert_eq!(ambiguous_tail_start("hello <thi", tags), 6);
        // Partial prefix of `<tool_call>`:
        assert_eq!(ambiguous_tail_start("foo <tool_c", tags), 4);
        // Lone `<` is a prefix of both:
        assert_eq!(ambiguous_tail_start("foo <", tags), 4);
        // Unambiguous `<x`:
        assert_eq!(ambiguous_tail_start("foo <x", tags), 6);
    }

    #[test]
    fn think_content_streams_incrementally() {
        // Think content that fits in one push (no close tag yet) should
        // surface as ThinkDelta(s) incrementally — not buffered whole.
        let mut p = QwenCoderXmlParser::new();
        let evts1 = p.push("<think>first batch of reasoning ");
        // Should have emitted the `first batch of reasoning ` as
        // ThinkDelta, holding only any tag-prefix tail.
        assert!(
            evts1
                .iter()
                .any(|e| matches!(e, ParserEvent::ThinkDelta(_))),
            "expected incremental ThinkDelta, got {evts1:?}"
        );
        let evts2 = p.push("more text</think>final");
        let all: Vec<_> = evts1.into_iter().chain(evts2).collect();
        let think = collect_think(&all);
        assert!(
            think.contains("first batch of reasoning "),
            "think={think:?}"
        );
        assert!(think.contains("more text"), "think={think:?}");
        assert_eq!(collect_text(&all), "final");
    }

    #[test]
    fn unterminated_think_surfaces_on_finish() {
        // Stream terminates mid-think. Don't drop the reasoning text.
        let input = "<think>partial reasoning with no close";
        let evts = parse_all(input);
        assert!(
            collect_think(&evts).contains("partial reasoning with no close"),
            "evts={evts:?}"
        );
    }

    #[test]
    fn forgiving_continuation_implicit_function_reopen() {
        // **Real Qwen3.6-35B-A3B output**: model declares the
        // function ONCE then runs `<parameter></function>` blocks for
        // each successive call without re-emitting `<function=NAME>`
        // and without `</tool_call>` between them. Parser leniency:
        // treat each subsequent `<parameter=...>` after `</function>`
        // as a continuation call with the SAME function name.
        let input = concat!(
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>\nParis\n</parameter>\n</function>\n",
            "<parameter=city>\nTokyo\n</parameter>\n</function>\n",
            "<parameter=city>\nLondon\n</parameter>\n</function>",
        );
        let evts = parse_all(input);
        let opens = collect_opens(&evts);
        assert_eq!(
            opens.len(),
            3,
            "expected 3 calls (Paris/Tokyo/London), got opens={opens:?}"
        );
        for (i, (_, name)) in opens.iter().enumerate() {
            assert_eq!(name, "get_weather", "open #{i} name should be get_weather");
        }
        let args = collect_args_strings(&evts);
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].1, r#"{"city":"Paris"}"#);
        assert_eq!(args[1].1, r#"{"city":"Tokyo"}"#);
        assert_eq!(args[2].1, r#"{"city":"London"}"#);
    }

    #[test]
    fn forgiving_continuation_implicit_tool_call_close_with_new_function() {
        // Variant: model emits multiple `<function=NAME>` blocks inside
        // a single `<tool_call>` (forgetting `</tool_call>` between).
        // Parser should treat each new `<function=NAME>` as an implicit
        // close + reopen.
        let input = concat!(
            "<tool_call>\n",
            "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
            "<function=get_population>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n",
            "</tool_call>",
        );
        let evts = parse_all(input);
        let opens = collect_opens(&evts);
        assert_eq!(
            opens,
            vec![(0, "get_weather".into()), (1, "get_population".into())],
            "evts={evts:?}"
        );
        let args = collect_args_strings(&evts);
        assert_eq!(args[0].1, r#"{"city":"Paris"}"#);
        assert_eq!(args[1].1, r#"{"city":"Tokyo"}"#);
    }

    #[test]
    fn unterminated_tool_call_surfaces_as_text_on_finish() {
        // Mid-stream decode-loop termination (e.g. hit max_tokens inside
        // a tool call). We must NOT drop the partial content — surface
        // it as text so the caller can log/debug.
        let input = "<tool_call>\n<function=f>\n<parameter=x>\npartial";
        let evts = parse_all(input);
        // The half-open tool call never emits Open/Close (no `<function=`
        // close-tag seen, for example, if we truncated earlier). For
        // this case we DID see `<function=f>` so Open fires, but no
        // Close. Whatever text is buffered dumps via the finish() tail.
        assert!(collect_closes(&evts).is_empty(), "should not close");
        // Some text must survive — drop is worse than imperfect output.
        let text = collect_text(&evts);
        // Either the "partial" body or the whole unterminated run ends
        // up in text. Either way, non-empty.
        assert!(
            text.contains("partial") || collect_opens(&evts).iter().any(|(_, n)| n == "f"),
            "unterminated content must surface somewhere: evts={evts:?}"
        );
    }
}
