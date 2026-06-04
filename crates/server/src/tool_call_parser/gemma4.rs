//! Gemma 4 tool-call parser.
//!
//! Wire format emitted by the model (mirrors the GGUF-embedded chat
//! template):
//!
//! ```text
//! <|tool_call>call:NAME{key1:<|"|>val<|"|>,key2:42,key3:true,
//!                       key4:[<|"|>a<|"|>,<|"|>b<|"|>],
//!                       key5:{nested:<|"|>x<|"|>}}<tool_call|>
//! ```
//!
//! - `<|tool_call>` / `<tool_call|>` — open/close special tokens.
//! - `call:NAME` — literal prefix followed by the function name.
//! - `{ key:value, ... }` — comma-separated `key:value` pairs.
//! - String values are wrapped in `<|"|>...<|"|>` (special-token quotes,
//!   distinct from ASCII `"`). Content between quotes is the literal value.
//! - Booleans render as `true` / `false`; numbers render bare.
//! - Arrays render `[v1,v2,...]`; objects render `{k:v,...}` (nestable).
//! - Object keys are bare (no quotes around the key in the wire format).
//!
//! Reasoning blocks (mirroring qwen3_coder's `<think>` interaction):
//!
//! ```text
//! visible<|channel>hidden reasoning<channel|>more visible
//! ```
//!
//! `<|channel>` opens a reasoning block; `<channel|>` closes it. Content
//! between them surfaces as [`ParserEvent::ThinkDelta`] and is dropped
//! from the assistant `content` field.
//!
//! The parser is streaming + deterministic across chunk boundaries: the
//! same byte stream pushed in any chunk decomposition produces the same
//! event sequence after [`ParserEvent::coalesce`]. The buffer-before-emit
//! discipline applies (never ship a `TextDelta` whose tail could still
//! be the prefix of a tool-call / channel open marker).
//!
//! Bit-exact parity vs llama.cpp's gemma4 tool-call extraction is the
//! follow-up; this parser handles the happy path + the streaming
//! invariants the rest of the framework relies on.

use serde_json::{Map, Value};

use super::{ParserEvent, ToolCallParser};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Outside a tool call / channel; scanning for opens.
    Text,
    /// Inside a `<|channel>...<channel|>` reasoning block.
    InChannel,
    /// Saw `<|tool_call>`, awaiting the `call:NAME{` header.
    InCallHeader,
    /// Inside the `{...}` arguments body (depth-tracked).
    InArgs,
    /// Saw the matching `}`, awaiting `<tool_call|>` close.
    AwaitingClose,
}

/// Tags watched while in [`State::Text`]. Order doesn't matter; the
/// parser picks the earliest match.
const TEXT_OPEN_TAGS: &[&str] = &["<|tool_call>", "<|channel>", CHANNEL_CLOSE];

const CALL_PREFIX: &str = "call:";
const STRING_QUOTE: &str = "<|\"|>";
const TOOL_CALL_CLOSE: &str = "<tool_call|>";
const CHANNEL_CLOSE: &str = "<channel|>";

pub struct Gemma4ToolCallParser {
    state: State,
    /// Accumulated unconsumed input.
    buf: String,
    /// Monotonic index for `ToolCallOpen` / `Close` within one turn.
    next_index: u32,
    /// Function name of the currently-open call (carried from
    /// `InCallHeader` to `InArgs`).
    current_name: String,
    /// Args body collected verbatim between the opening `{` and the
    /// matching `}` (with depth tracking that ignores delimiters inside
    /// `<|"|>...<|"|>` strings). Parsed once the close is seen.
    args_body: String,
    /// `{` / `[` nesting depth inside [`State::InArgs`]. The opening
    /// `{` of the args body counts as depth 1; the matching close
    /// drops it back to 0 and transitions to [`State::AwaitingClose`].
    args_depth: u32,
    /// `true` while scanning past a `<|"|>` open marker, looking for
    /// the matching `<|"|>` close. Suppresses brace / bracket
    /// counting inside strings.
    args_in_string: bool,
    /// `true` when the chat-template prompt ended with
    /// `<|channel>thought\n<channel|>` and the model may echo
    /// `thought\n<channel|>` (or just `<channel|>`) as the first
    /// decoded bytes. Cleared after the first `push()` that sees
    /// non-prefix content. See [`Self::with_pending_channel_close`].
    pending_thought_echo: bool,
}

impl Default for Gemma4ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Gemma4ToolCallParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4ToolCallParser")
            .field("state", &self.state)
            .field("buf_len", &self.buf.len())
            .field("next_index", &self.next_index)
            .finish()
    }
}

impl Gemma4ToolCallParser {
    pub fn new() -> Self {
        Self::with_thought_echo_guard(false)
    }

    /// Construct a parser that knows the rendered prompt ended with
    /// `<|channel>thought\n<channel|>` (a closed empty thought block —
    /// the gemma4 default when `add_generation_prompt=true` and
    /// reasoning is disabled). The model often echoes the
    /// `thought\n<channel|>` tail before producing real content; with
    /// the echo-guard the parser drops that leading prefix from the
    /// first push() without consuming the rest of the reply, so a
    /// straight `<|tool_call>...` continuation still parses normally
    /// and a plain prose reply lands intact in `content`.
    pub fn with_pending_channel_close() -> Self {
        Self::with_thought_echo_guard(true)
    }

    fn with_thought_echo_guard(pending_thought_echo: bool) -> Self {
        Self {
            state: State::Text,
            buf: String::new(),
            next_index: 0,
            current_name: String::new(),
            args_body: String::new(),
            args_depth: 0,
            args_in_string: false,
            pending_thought_echo,
        }
    }

    /// True if `prompt` ends in the gemma4 `<channel|>` close marker
    /// (with optional trailing whitespace). Cheap byte-suffix match —
    /// the chat route calls this once per request after rendering the
    /// chat-template prompt.
    pub fn prompt_ends_with_channel_close(prompt: &str) -> bool {
        prompt.trim_end().ends_with(CHANNEL_CLOSE)
    }

    /// Strip the gemma4 `thought\n<channel|>` echo prefix once, if
    /// present, then clear the guard so it never fires again on the
    /// same parser. Tolerates streaming: if not enough bytes have
    /// landed to disambiguate, leaves the buffer untouched and returns
    /// false; the next `push()` retries. Once enough bytes are
    /// available, accepts any of:
    /// - `<channel|>...` — bare close, drop just the close.
    /// - `thought\n<channel|>...` — full echo, drop both.
    /// - anything else — clear the guard, the model is going straight
    ///   to a reply without echoing.
    fn try_consume_echo_prefix(&mut self, is_finish: bool) {
        if !self.pending_thought_echo {
            return;
        }
        const ECHO_FULL: &str = "thought\n<channel|>";
        if self.buf.starts_with(ECHO_FULL) {
            self.buf.drain(..ECHO_FULL.len());
            self.pending_thought_echo = false;
            return;
        }
        if self.buf.starts_with(CHANNEL_CLOSE) {
            self.buf.drain(..CHANNEL_CLOSE.len());
            self.pending_thought_echo = false;
            return;
        }
        // Streaming guard: while more bytes might still land that
        // complete the echo, wait. At EOF (`is_finish`) we have to
        // make a call.
        let is_strict_prefix = !self.buf.is_empty()
            && (ECHO_FULL.starts_with(self.buf.as_str())
                || CHANNEL_CLOSE.starts_with(self.buf.as_str()));
        if is_strict_prefix && !is_finish {
            return;
        }
        if is_strict_prefix && is_finish {
            // The model emitted a truncated echo prefix and then
            // stopped (e.g. just `thought`). Drop it — it was meant to
            // be reasoning, not content.
            self.buf.clear();
            self.pending_thought_echo = false;
            return;
        }
        // Any other content: the model isn't echoing. Disarm so the
        // first byte lands in `content` (or starts a tool call).
        self.pending_thought_echo = false;
    }

    fn drain(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) {
        loop {
            let progress = match self.state {
                State::Text => self.step_text(out, is_finish),
                State::InChannel => self.step_in_channel(out, is_finish),
                State::InCallHeader => self.step_in_call_header(out),
                State::InArgs => self.step_in_args(out),
                State::AwaitingClose => self.step_awaiting_close(out),
            };
            if !progress {
                return;
            }
        }
    }

    fn step_text(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        let earliest = TEXT_OPEN_TAGS
            .iter()
            .filter_map(|tag| self.buf.find(tag).map(|p| (p, *tag)))
            .min_by_key(|(p, _)| *p);
        if let Some((pos, tag)) = earliest {
            if pos > 0 {
                emit_text(out, &self.buf[..pos]);
            }
            self.buf.drain(..pos + tag.len());
            self.state = match tag {
                "<|tool_call>" => State::InCallHeader,
                "<|channel>" => State::InChannel,
                // The chat template ends with `<|channel>thought\n<channel|>`,
                // so the model's first emitted token is often a stuttered
                // `<channel|>`. With no preceding `<|channel>` open, we're
                // already in State::Text — drop the close marker and stay.
                t if t == CHANNEL_CLOSE => State::Text,
                _ => unreachable!("unexpected tag matched: {tag}"),
            };
            return true;
        }
        let total = self.buf.len();
        let emit_upto = if is_finish {
            total
        } else {
            ambiguous_tail_start(&self.buf, TEXT_OPEN_TAGS)
        };
        let emit_upto = align_down_char_boundary(&self.buf, emit_upto);
        if emit_upto > 0 {
            let chunk: String = self.buf.drain(..emit_upto).collect();
            emit_text(out, &chunk);
            return true;
        }
        false
    }

    fn step_in_channel(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        if let Some(pos) = self.buf.find(CHANNEL_CLOSE) {
            if pos > 0 {
                out.push(ParserEvent::ThinkDelta(self.buf[..pos].to_owned()));
            }
            self.buf.drain(..pos + CHANNEL_CLOSE.len());
            self.state = State::Text;
            return true;
        }
        let total = self.buf.len();
        let emit_upto = if is_finish {
            total
        } else {
            ambiguous_tail_start(&self.buf, &[CHANNEL_CLOSE])
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

    fn step_in_call_header(&mut self, out: &mut Vec<ParserEvent>) -> bool {
        // Expect `call:NAME{`. Hold until we've seen everything through
        // the opening `{` — the name terminates there.
        if !self.buf.starts_with(CALL_PREFIX) {
            // Streaming may have only landed the leading bytes (e.g.
            // `cal`); require the whole prefix to advance.
            if CALL_PREFIX.starts_with(self.buf.as_str()) {
                return false;
            }
            // Malformed: didn't see `call:` after `<|tool_call>`. Treat
            // the whole malformed block as text and bail back to Text.
            // We can't reliably recover the structure; surface the
            // sequence so the assistant turn isn't silently swallowed.
            let leak = format!("<|tool_call>{}", self.buf);
            emit_text(out, &leak);
            self.buf.clear();
            self.state = State::Text;
            return true;
        }
        let after_prefix = CALL_PREFIX.len();
        let Some(brace_rel) = self.buf[after_prefix..].find('{') else {
            return false;
        };
        let name = self.buf[after_prefix..after_prefix + brace_rel].to_owned();
        let cursor = after_prefix + brace_rel + 1;
        self.buf.drain(..cursor);
        self.current_name = name.clone();
        self.args_body.clear();
        self.args_depth = 1;
        self.args_in_string = false;
        out.push(ParserEvent::ToolCallOpen {
            index: self.next_index,
            name,
        });
        self.state = State::InArgs;
        true
    }

    fn step_in_args(&mut self, out: &mut Vec<ParserEvent>) -> bool {
        // Walk the buffer char-by-char, tracking depth + string state.
        // Bytes are appended to `args_body`; when depth returns to 0 at
        // a `}`, we've found the close.
        // We work on byte indices and explicitly check string-quote
        // boundaries via `STRING_QUOTE` literal.
        let bytes = self.buf.as_bytes();
        let q_len = STRING_QUOTE.len();
        let mut i = 0usize;
        let mut closed_at: Option<usize> = None;
        while i < bytes.len() {
            if self.args_in_string {
                // Look for closing `<|"|>` at i.
                if i + q_len <= bytes.len() && &self.buf[i..i + q_len] == STRING_QUOTE {
                    self.args_in_string = false;
                    i += q_len;
                    continue;
                }
                // Tail ambiguity: a partial `<|"` etc. straddling the
                // chunk boundary may complete to the close marker once
                // more input arrives. Hold rather than swallow the
                // bytes into the string body.
                if bytes[i] == b'<' && i + q_len > bytes.len() {
                    let tail = &self.buf[i..];
                    if STRING_QUOTE.starts_with(tail) {
                        break;
                    }
                }
                i += 1;
                continue;
            }
            // Outside strings: check for opening string quote.
            if i + q_len <= bytes.len() && &self.buf[i..i + q_len] == STRING_QUOTE {
                self.args_in_string = true;
                i += q_len;
                continue;
            }
            // Tail-of-buffer ambiguity: a partial `<|"` etc. that may
            // be the start of `<|"|>`. We can't safely advance past it
            // without confirming the full marker. Hold for more input.
            if !self.args_in_string && bytes[i] == b'<' {
                let tail = &self.buf[i..];
                if STRING_QUOTE.starts_with(tail) {
                    break;
                }
            }
            match bytes[i] {
                b'{' | b'[' => {
                    self.args_depth += 1;
                    i += 1;
                }
                b']' => {
                    if self.args_depth == 0 {
                        // Malformed: extra closing bracket. Treat as a
                        // literal byte so we don't lose data.
                        i += 1;
                    } else {
                        self.args_depth -= 1;
                        i += 1;
                    }
                }
                b'}' => {
                    if self.args_depth == 0 {
                        i += 1;
                        continue;
                    }
                    self.args_depth -= 1;
                    if self.args_depth == 0 {
                        // Found the matching close of the args body.
                        // Don't include this `}` in args_body.
                        closed_at = Some(i);
                        break;
                    }
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }
        // Append the consumed slice to args_body (excluding the close
        // `}` when closed_at fires).
        let consume_upto = closed_at.unwrap_or(i);
        if consume_upto > 0 {
            self.args_body.push_str(&self.buf[..consume_upto]);
        }
        if let Some(close_pos) = closed_at {
            // Drain through the close brace.
            self.buf.drain(..close_pos + 1);
            // Parse args_body → JSON object, emit deltas, advance.
            let parsed =
                parse_args_body(&self.args_body).unwrap_or_else(|_| Value::Object(Map::new()));
            let json = serde_json::to_string(&parsed).unwrap_or_else(|_| "{}".to_owned());
            out.push(ParserEvent::ToolCallArgumentsDelta {
                index: self.next_index,
                arguments: json,
            });
            self.args_body.clear();
            self.args_depth = 0;
            self.args_in_string = false;
            self.state = State::AwaitingClose;
            return true;
        }
        if consume_upto > 0 {
            self.buf.drain(..consume_upto);
            return true;
        }
        false
    }

    fn step_awaiting_close(&mut self, out: &mut Vec<ParserEvent>) -> bool {
        if let Some(pos) = self.buf.find(TOOL_CALL_CLOSE) {
            // Drain leading whitespace + the close.
            self.buf.drain(..pos + TOOL_CALL_CLOSE.len());
            out.push(ParserEvent::ToolCallClose {
                index: self.next_index,
            });
            self.next_index += 1;
            self.state = State::Text;
            return true;
        }
        // Could be a prefix; hold.
        false
    }
}

/// Parse a gemma4 args-body (everything between the outer `{` and
/// matching `}`) into a JSON Map. The grammar is a tiny custom format:
///
/// - `key:value` pairs separated by top-level `,`.
/// - Keys are bare (no quotes); read until `:`.
/// - Values:
///   - `<|"|>...<|"|>` → string (literal content)
///   - `{...}` → nested object
///   - `[...]` → array
///   - `true` / `false` → boolean
///   - bare number → JSON number
///   - bare token (anything else) → string fallback (matches llama.cpp
///     leniency for unrecognised literals)
///
/// Returns an empty object on parse failure rather than erroring — the
/// model produced *something* tool-call-shaped; surfacing the call with
/// empty args beats dropping it entirely.
fn parse_args_body(body: &str) -> Result<Value, String> {
    let mut p = Parser::new(body);
    p.skip_ws();
    if p.is_eof() {
        return Ok(Value::Object(Map::new()));
    }
    let v = p.parse_object_body()?;
    Ok(v)
}

struct Parser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.s.len()
    }

    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn starts_with(&self, s: &str) -> bool {
        self.s[self.pos..].starts_with(s)
    }

    /// Parse the body of a `{...}` block (without the outer braces). At
    /// entry `pos` points to the first non-whitespace byte after `{`.
    /// At exit `pos` points at either the trailing `}` of the parent
    /// scope or eof.
    fn parse_object_body(&mut self) -> Result<Value, String> {
        let mut obj = Map::new();
        loop {
            self.skip_ws();
            if self.is_eof() || self.peek() == Some(b'}') {
                break;
            }
            // Parse key (bare ident, up to ':').
            let key_start = self.pos;
            while let Some(b) = self.peek() {
                if b == b':' {
                    break;
                }
                self.pos += 1;
            }
            let key = self.s[key_start..self.pos].trim().to_owned();
            if self.peek() != Some(b':') {
                return Err(format!(
                    "expected `:` after key `{key}` at byte {}",
                    self.pos
                ));
            }
            self.pos += 1; // consume ':'
            self.skip_ws();
            let val = self.parse_value()?;
            obj.insert(key, val);
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.pos += 1;
                continue;
            } else {
                break;
            }
        }
        Ok(Value::Object(obj))
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        if self.is_eof() {
            return Ok(Value::Null);
        }
        // String quote.
        if self.starts_with(STRING_QUOTE) {
            self.pos += STRING_QUOTE.len();
            let start = self.pos;
            let close_rel = self.s[self.pos..]
                .find(STRING_QUOTE)
                .ok_or_else(|| format!("unterminated string at byte {}", start))?;
            let s = self.s[start..start + close_rel].to_owned();
            self.pos = start + close_rel + STRING_QUOTE.len();
            return Ok(Value::String(s));
        }
        match self.peek() {
            Some(b'{') => {
                self.pos += 1;
                let v = self.parse_object_body()?;
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                }
                Ok(v)
            }
            Some(b'[') => {
                self.pos += 1;
                let mut arr: Vec<Value> = Vec::new();
                loop {
                    self.skip_ws();
                    if self.is_eof() || self.peek() == Some(b']') {
                        break;
                    }
                    let v = self.parse_value()?;
                    arr.push(v);
                    self.skip_ws();
                    if self.peek() == Some(b',') {
                        self.pos += 1;
                        continue;
                    } else {
                        break;
                    }
                }
                if self.peek() == Some(b']') {
                    self.pos += 1;
                }
                Ok(Value::Array(arr))
            }
            Some(_) => {
                // Bare token: bool, number, or fallback string.
                let start = self.pos;
                while let Some(b) = self.peek() {
                    if b == b',' || b == b'}' || b == b']' {
                        break;
                    }
                    self.pos += 1;
                }
                let raw = self.s[start..self.pos].trim();
                if raw == "true" {
                    return Ok(Value::Bool(true));
                }
                if raw == "false" {
                    return Ok(Value::Bool(false));
                }
                if raw == "null" {
                    return Ok(Value::Null);
                }
                if let Ok(n) = raw.parse::<i64>() {
                    return Ok(Value::from(n));
                }
                if let Ok(n) = raw.parse::<f64>() {
                    if let Some(jn) = serde_json::Number::from_f64(n) {
                        return Ok(Value::Number(jn));
                    }
                }
                Ok(Value::String(raw.to_owned()))
            }
            None => Ok(Value::Null),
        }
    }
}

fn emit_text(out: &mut Vec<ParserEvent>, s: &str) {
    if !s.is_empty() {
        out.push(ParserEvent::TextDelta(s.to_owned()));
    }
}

fn align_down_char_boundary(s: &str, n: usize) -> usize {
    let mut i = n.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ambiguous_tail_start(s: &str, tags: &[&str]) -> usize {
    let max_tail = tags
        .iter()
        .map(|t| t.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    if max_tail == 0 {
        return s.len();
    }
    let start_scan = s.len().saturating_sub(max_tail);
    for p in start_scan..s.len() {
        if !s.is_char_boundary(p) {
            continue;
        }
        let tail = &s[p..];
        if tags
            .iter()
            .any(|tag| tail.len() < tag.len() && tag.starts_with(tail))
        {
            return p;
        }
    }
    s.len()
}

impl ToolCallParser for Gemma4ToolCallParser {
    fn push(&mut self, chunk: &str) -> Vec<ParserEvent> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(chunk);
        self.try_consume_echo_prefix(/*is_finish=*/ false);
        let mut out = Vec::new();
        self.drain(&mut out, /*is_finish=*/ false);
        out
    }

    fn finish(&mut self) -> Vec<ParserEvent> {
        let mut out = Vec::new();
        self.try_consume_echo_prefix(/*is_finish=*/ true);
        self.drain(&mut out, /*is_finish=*/ true);
        // If we're stuck mid-call at EOF, surface what we have so the
        // assistant turn isn't dropped.
        match self.state {
            State::InCallHeader => {
                let leak = format!("<|tool_call>{}", std::mem::take(&mut self.buf));
                emit_text(&mut out, &leak);
                self.state = State::Text;
            }
            State::InArgs => {
                // Emit what we collected with empty args.
                out.push(ParserEvent::ToolCallArgumentsDelta {
                    index: self.next_index,
                    arguments: "{}".to_owned(),
                });
                out.push(ParserEvent::ToolCallClose {
                    index: self.next_index,
                });
                self.next_index += 1;
                self.args_body.clear();
                self.state = State::Text;
            }
            State::AwaitingClose => {
                out.push(ParserEvent::ToolCallClose {
                    index: self.next_index,
                });
                self.next_index += 1;
                self.state = State::Text;
            }
            State::InChannel | State::Text => {
                // Already drained on is_finish=true above.
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(input: &str) -> Vec<ParserEvent> {
        let mut p = Gemma4ToolCallParser::new();
        let mut out = p.push(input);
        out.extend(p.finish());
        ParserEvent::coalesce(out)
    }

    #[test]
    fn happy_path_one_string_arg() {
        let events =
            collect("Hello! <|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>");
        assert_eq!(events.len(), 4, "events: {events:#?}");
        assert!(matches!(&events[0], ParserEvent::TextDelta(s) if s == "Hello! "));
        assert!(
            matches!(&events[1], ParserEvent::ToolCallOpen { index: 0, name } if name == "get_weather")
        );
        assert!(
            matches!(&events[2], ParserEvent::ToolCallArgumentsDelta { index: 0, arguments } if arguments == r#"{"location":"Paris"}"#)
        );
        assert!(matches!(events[3], ParserEvent::ToolCallClose { index: 0 }));
    }

    #[test]
    fn mixed_types_bool_int_string_array_object() {
        let events = collect(
            "<|tool_call>call:do{a:<|\"|>x<|\"|>,b:42,c:true,d:[<|\"|>p<|\"|>,<|\"|>q<|\"|>],e:{n:<|\"|>v<|\"|>}}<tool_call|>",
        );
        let args = events
            .iter()
            .find_map(|e| match e {
                ParserEvent::ToolCallArgumentsDelta { arguments, .. } => Some(arguments.clone()),
                _ => None,
            })
            .unwrap();
        let v: Value = serde_json::from_str(&args).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.get("a"), Some(&Value::String("x".into())));
        assert_eq!(obj.get("b"), Some(&Value::from(42i64)));
        assert_eq!(obj.get("c"), Some(&Value::Bool(true)));
        assert_eq!(
            obj.get("d"),
            Some(&Value::Array(vec![
                Value::String("p".into()),
                Value::String("q".into())
            ])),
        );
        let nested = obj.get("e").and_then(|v| v.as_object()).unwrap();
        assert_eq!(nested.get("n"), Some(&Value::String("v".into())));
    }

    #[test]
    fn two_calls_sequential() {
        let events = collect(
            "<|tool_call>call:a{x:1}<tool_call|>between<|tool_call>call:b{y:2}<tool_call|>",
        );
        let opens: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallOpen { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(opens, vec!["a", "b"]);
        let indices: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallClose { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, vec![0, 1]);
        assert!(events
            .iter()
            .any(|e| matches!(e, ParserEvent::TextDelta(s) if s == "between")));
    }

    #[test]
    fn channel_block_emits_think_delta() {
        let events = collect(
            "before<|channel>reasoning content<channel|>after<|tool_call>call:f{}<tool_call|>",
        );
        assert!(events
            .iter()
            .any(|e| matches!(e, ParserEvent::ThinkDelta(s) if s == "reasoning content")));
        assert!(events
            .iter()
            .any(|e| matches!(e, ParserEvent::TextDelta(s) if s == "before")));
        assert!(events
            .iter()
            .any(|e| matches!(e, ParserEvent::TextDelta(s) if s == "after")));
        assert!(events
            .iter()
            .any(|e| matches!(e, ParserEvent::ToolCallOpen { name, .. } if name == "f")));
    }

    #[test]
    fn leading_channel_close_is_dropped() {
        // Gemma4's chat template ends `...<|channel>thought\n<channel|>`,
        // so the model often emits a stuttered `<channel|>` as its first
        // token. With no preceding `<|channel>` in the model output, the
        // parser stays in State::Text and just drops the close marker.
        let events = collect("<channel|>Hello world");
        assert!(events
            .iter()
            .any(|e| matches!(e, ParserEvent::TextDelta(s) if s == "Hello world")));
        // No ThinkDelta should fire — there was no real reasoning content.
        assert!(events
            .iter()
            .all(|e| !matches!(e, ParserEvent::ThinkDelta(_))));
        // And the literal `<channel|>` should not appear in any TextDelta.
        assert!(events.iter().all(|e| !matches!(
            e,
            ParserEvent::TextDelta(s) if s.contains("<channel|>")
        )));
    }

    #[test]
    fn with_pending_channel_close_drops_full_echo_then_keeps_text() {
        // gemma4 prompt ends with `<|channel>thought\n<channel|>` so the
        // model often echoes `thought\n<channel|>` as its first decoded
        // bytes. The echo guard drops that prefix and keeps the real
        // reply intact in `content`.
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut events = p.push("thought\n<channel|>Paris is the capital.");
        events.extend(p.finish());
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Paris is the capital.");
    }

    #[test]
    fn with_pending_channel_close_then_tool_call() {
        // Echo followed by a clean tool call.
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut events = p.push(
            "thought\n<channel|><|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>",
        );
        events.extend(p.finish());
        let names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallOpen { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["get_weather"]);
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.is_empty(), "expected no leaked text, got {text:?}");
    }

    #[test]
    fn with_pending_channel_close_no_echo_passes_text_through() {
        // Model goes straight to text without echoing the prompt tail.
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut events = p.push("Paris is the capital.");
        events.extend(p.finish());
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Paris is the capital.");
    }

    #[test]
    fn with_pending_channel_close_no_echo_goes_straight_to_tool_call() {
        // No echo, model emits a tool call directly.
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut events = p.push(
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>",
        );
        events.extend(p.finish());
        let names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::ToolCallOpen { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["get_weather"]);
    }

    #[test]
    fn with_pending_channel_close_bare_close_only() {
        // Model echoes just the close, no `thought\n` prefix.
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut events = p.push("<channel|>The answer is 42.");
        events.extend(p.finish());
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn prompt_ends_with_channel_close_detects_marker() {
        assert!(Gemma4ToolCallParser::prompt_ends_with_channel_close(
            "<|turn>model\n<|channel>thought\n<channel|>"
        ));
        // Trailing whitespace tolerated.
        assert!(Gemma4ToolCallParser::prompt_ends_with_channel_close(
            "...<channel|>\n  "
        ));
        // qwen3.6 / non-gemma4 prompts don't match.
        assert!(!Gemma4ToolCallParser::prompt_ends_with_channel_close(
            "<|im_start|>assistant\n"
        ));
        assert!(!Gemma4ToolCallParser::prompt_ends_with_channel_close(""));
    }

    #[test]
    fn streaming_char_by_char_matches_one_shot() {
        let input =
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>,unit:<|\"|>c<|\"|>}<tool_call|>";
        let one_shot = collect(input);

        let mut p = Gemma4ToolCallParser::new();
        let mut out: Vec<ParserEvent> = Vec::new();
        // Push one char (one UTF-8 code point) at a time.
        for ch in input.chars() {
            let s = ch.to_string();
            out.extend(p.push(&s));
        }
        out.extend(p.finish());
        let streamed = ParserEvent::coalesce(out);
        assert_eq!(streamed, one_shot, "streamed != one_shot");
    }

    #[test]
    fn empty_args() {
        let events = collect("<|tool_call>call:noargs{}<tool_call|>");
        let args = events
            .iter()
            .find_map(|e| match e {
                ParserEvent::ToolCallArgumentsDelta { arguments, .. } => Some(arguments.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(args, "{}");
    }

    #[test]
    fn free_text_with_no_tool_call() {
        let events = collect("just some text, nothing special");
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ParserEvent::TextDelta(s) if s == "just some text, nothing special")
        );
    }

    #[test]
    fn buffer_before_emit_tail_holdback() {
        // Half a `<|tool_call>` prefix shouldn't be emitted on first push.
        let mut p = Gemma4ToolCallParser::new();
        let first = p.push("hello <|tool_ca");
        // Only "hello " should leak; the rest is ambiguous tail.
        let coalesced = ParserEvent::coalesce(first);
        assert!(
            coalesced
                .iter()
                .all(|e| matches!(e, ParserEvent::TextDelta(s) if !s.contains("<|tool_ca"))),
            "leaked ambiguous prefix: {coalesced:#?}"
        );
        let rest = p.push("ll>call:f{}<tool_call|>");
        let all = ParserEvent::coalesce(first_then(coalesced, rest, p.finish()));
        assert!(all
            .iter()
            .any(|e| matches!(e, ParserEvent::ToolCallOpen { name, .. } if name == "f")));
    }

    fn first_then(
        a: Vec<ParserEvent>,
        b: Vec<ParserEvent>,
        c: Vec<ParserEvent>,
    ) -> Vec<ParserEvent> {
        let mut v = a;
        v.extend(b);
        v.extend(c);
        v
    }
}
