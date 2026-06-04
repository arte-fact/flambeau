//! Hermes-style `<tool_call>\n{JSON}\n</tool_call>` parser.
//!
//! Wire format emitted by the model:
//!
//! ```text
//! <tool_call>
//! {"name": "get_weather", "arguments": {"location": "Paris"}}
//! </tool_call>
//! ```
//!
//! - `<tool_call>` / `</tool_call>` delimit one call; multiple calls
//!   may appear back-to-back in a turn.
//! - The body is a JSON object with a `name` string and an `arguments`
//!   object. `arguments` is re-serialized to a JSON **string** for the
//!   OpenAI wire contract (never an object — llama.cpp #20198).
//! - `<think>...</think>` reasoning surfaces as
//!   [`ParserEvent::ThinkDelta`] and is dropped from `content`.
//!
//! Streaming + deterministic across chunk boundaries (buffer-before-emit
//! for partial open markers; the JSON-object scan is string/escape
//! aware so a `</tool_call>` inside an argument value doesn't close the
//! call early).

use serde_json::Value;

use super::{ParserEvent, ToolCallParser};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TEXT_OPEN_TAGS: &[&str] = &[TOOL_OPEN, THINK_OPEN];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum State {
    #[default]
    Text,
    InThink,
    InTool,
    AwaitingToolClose,
}

#[derive(Debug, Default)]
pub struct HermesJsonParser {
    buf: String,
    state: State,
    next_index: u32,
}

impl HermesJsonParser {
    pub fn new() -> Self {
        Self::default()
    }

    fn drain(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) {
        loop {
            let progress = match self.state {
                State::Text => self.step_text(out, is_finish),
                State::InThink => self.step_in_think(out, is_finish),
                State::InTool => self.step_in_tool(out, is_finish),
                State::AwaitingToolClose => self.step_awaiting_close(is_finish),
            };
            if !progress {
                return;
            }
        }
    }

    fn step_text(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        let earliest = TEXT_OPEN_TAGS
            .iter()
            .filter_map(|t| self.buf.find(t).map(|p| (p, *t)))
            .min_by_key(|(p, _)| *p);
        if let Some((pos, tag)) = earliest {
            if pos > 0 {
                push_text(out, &self.buf[..pos]);
            }
            self.buf.drain(..pos + tag.len());
            self.state = if tag == THINK_OPEN {
                State::InThink
            } else {
                State::InTool
            };
            return true;
        }
        let emit_upto = if is_finish {
            self.buf.len()
        } else {
            ambiguous_tail_start(&self.buf, TEXT_OPEN_TAGS)
        };
        let emit_upto = floor_char_boundary(&self.buf, emit_upto);
        if emit_upto > 0 {
            let chunk: String = self.buf.drain(..emit_upto).collect();
            push_text(out, &chunk);
            return true;
        }
        false
    }

    fn step_in_think(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        if let Some(pos) = self.buf.find(THINK_CLOSE) {
            if pos > 0 {
                out.push(ParserEvent::ThinkDelta(self.buf[..pos].to_owned()));
            }
            self.buf.drain(..pos + THINK_CLOSE.len());
            self.state = State::Text;
            return true;
        }
        let emit_upto = if is_finish {
            self.buf.len()
        } else {
            ambiguous_tail_start(&self.buf, &[THINK_CLOSE])
        };
        let emit_upto = floor_char_boundary(&self.buf, emit_upto);
        if emit_upto > 0 {
            let chunk: String = self.buf.drain(..emit_upto).collect();
            if !chunk.is_empty() {
                out.push(ParserEvent::ThinkDelta(chunk));
            }
            return true;
        }
        false
    }

    fn step_in_tool(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        match json_object_span(&self.buf) {
            Some((start, end)) => {
                let json = &self.buf[start..=end];
                let (name, args) = match serde_json::from_str::<Value>(json) {
                    Ok(v) => {
                        let name = v
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let args = match v.get("arguments") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => {
                                serde_json::to_string(other).unwrap_or_else(|_| "{}".to_owned())
                            }
                            None => "{}".to_owned(),
                        };
                        (name, args)
                    }
                    Err(_) => (String::new(), "{}".to_owned()),
                };
                out.push(ParserEvent::ToolCallOpen {
                    index: self.next_index,
                    name,
                });
                out.push(ParserEvent::ToolCallArgumentsDelta {
                    index: self.next_index,
                    arguments: args,
                });
                out.push(ParserEvent::ToolCallClose {
                    index: self.next_index,
                });
                self.next_index += 1;
                self.buf.drain(..=end);
                self.state = State::AwaitingToolClose;
                true
            }
            None => {
                if is_finish && !self.buf.trim().is_empty() {
                    // Malformed open with no JSON object — surface it so
                    // the turn isn't silently dropped.
                    let leak = format!("{TOOL_OPEN}{}", std::mem::take(&mut self.buf));
                    push_text(out, &leak);
                    self.state = State::Text;
                    return true;
                }
                false
            }
        }
    }

    fn step_awaiting_close(&mut self, is_finish: bool) -> bool {
        let trimmed = self.buf.trim_start();
        let lead_ws = self.buf.len() - trimmed.len();
        if let Some(rel) = trimmed.find(TOOL_CLOSE) {
            self.buf.drain(..lead_ws + rel + TOOL_CLOSE.len());
            self.state = State::Text;
            return true;
        }
        // The close hasn't fully landed. If what we have is still a
        // prefix of (whitespace* + `</tool_call>`), wait. Otherwise the
        // close is absent/malformed — resume as text from here.
        if !is_finish && (trimmed.is_empty() || TOOL_CLOSE.starts_with(trimmed)) {
            return false;
        }
        if trimmed.is_empty() {
            self.buf.clear();
            self.state = State::Text;
            return false;
        }
        self.state = State::Text;
        true
    }
}

impl ToolCallParser for HermesJsonParser {
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
        self.drain(&mut out, /*is_finish=*/ true);
        // Flush any residue (an unterminated think block, leftover text).
        if !self.buf.is_empty() {
            let leftover = std::mem::take(&mut self.buf);
            match self.state {
                State::InThink => out.push(ParserEvent::ThinkDelta(leftover)),
                _ => push_text(&mut out, &leftover),
            }
            self.state = State::Text;
        }
        out
    }
}

fn push_text(out: &mut Vec<ParserEvent>, s: &str) {
    if !s.is_empty() {
        out.push(ParserEvent::TextDelta(s.to_owned()));
    }
}

/// Byte span `(start, end_inclusive)` of the first balanced JSON object
/// in `s`, or `None` if no complete object is present yet. String/escape
/// aware: braces inside string literals don't count.
fn json_object_span(s: &str) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &c) in b.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((start, i));
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Largest prefix length safe to emit without risking that the tail is
/// the start of one of `tags`. Mirrors the gemma4 helper.
fn ambiguous_tail_start(buf: &str, tags: &[&str]) -> usize {
    let n = buf.len();
    let max_tag = tags.iter().map(|t| t.len()).max().unwrap_or(0);
    let lo = n.saturating_sub(max_tag.saturating_sub(1));
    for cut in lo..n {
        let tail = &buf[cut..];
        if tags.iter().any(|t| t.starts_with(tail)) {
            return cut;
        }
    }
    n
}

fn floor_char_boundary(s: &str, mut n: usize) -> usize {
    if n >= s.len() {
        return s.len();
    }
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(input: &str) -> Vec<ParserEvent> {
        let mut p = HermesJsonParser::new();
        let mut out = p.push(input);
        out.extend(p.finish());
        ParserEvent::coalesce(out)
    }

    fn names(events: &[ParserEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallOpen { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    fn args(events: &[ParserEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallArgumentsDelta { arguments, .. } => Some(arguments.clone()),
                _ => None,
            })
            .collect()
    }

    fn text(events: &[ParserEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn happy_path_single_call() {
        let e = collect(
            "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}\n</tool_call>",
        );
        assert_eq!(names(&e), ["get_weather"]);
        assert_eq!(args(&e), [r#"{"location":"Paris"}"#]);
        assert!(text(&e).is_empty());
    }

    #[test]
    fn text_then_call() {
        let e = collect(
            "Let me check. <tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
        );
        assert_eq!(text(&e), "Let me check. ");
        assert_eq!(names(&e), ["f"]);
    }

    #[test]
    fn two_calls() {
        let e = collect(
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>",
        );
        assert_eq!(names(&e), ["a", "b"]);
    }

    #[test]
    fn close_marker_inside_string_arg() {
        let e = collect(
            "<tool_call>{\"name\":\"echo\",\"arguments\":{\"t\":\"</tool_call> x\"}}</tool_call>",
        );
        assert_eq!(names(&e), ["echo"]);
        let v: Value = serde_json::from_str(&args(&e)[0]).unwrap();
        assert_eq!(v["t"], "</tool_call> x");
    }

    #[test]
    fn think_block_is_dropped() {
        let e = collect("<think>reasoning</think>Answer.");
        assert_eq!(text(&e), "Answer.");
        assert!(e
            .iter()
            .any(|x| matches!(x, ParserEvent::ThinkDelta(s) if s == "reasoning")));
    }

    #[test]
    fn streaming_char_by_char_matches_one_shot() {
        let input =
            "ok <think>r</think><tool_call>{\"name\":\"f\",\"arguments\":{\"x\":1}}</tool_call> done";
        let one = collect(input);
        let mut p = HermesJsonParser::new();
        let mut out: Vec<ParserEvent> = Vec::new();
        for ch in input.chars() {
            out.extend(p.push(&ch.to_string()));
        }
        out.extend(p.finish());
        assert_eq!(ParserEvent::coalesce(out), one);
    }

    #[test]
    fn pure_text_passthrough() {
        let e = collect("The capital of France is Paris.");
        assert_eq!(text(&e), "The capital of France is Paris.");
        assert!(names(&e).is_empty());
    }

    #[test]
    fn empty_push_is_noop() {
        let mut p = HermesJsonParser::new();
        assert!(p.push("").is_empty());
    }
}
