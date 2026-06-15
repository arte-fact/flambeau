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
//! Alternate `tool_code` block (agent-prose convention):
//!
//! ```text
//! <|tool_code|>NAME
//! {"arg":"value"}
//! </tool_code>
//! ```
//!
//! `<|tool_code|>` is **not** a trained gemma special token — the model
//! emits it as plain text when a coding agent describes its tools in the
//! prompt using the Gemini-CLI "tool_code" convention instead of the
//! native `tools` array (which renders the `<|tool_call>` form above).
//! The first line after the open is the function name; the remainder is
//! a JSON arguments object. We recognize this shape so flambeau is a
//! drop-in for those agents — strictly additive: the native
//! `<|tool_call>` path is unchanged, and a block we can't structure
//! (no name tag, no JSON object) is surfaced verbatim rather than
//! silently dropped.
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
    /// Saw `<|tool_code|>`, buffering until `</tool_code>`.
    InToolCode,
    /// Saw `<|turn>`, dropping the turn-header line until `\n`.
    InTurnHeader,
    /// Saw `<|tool_response>`, dropping the echoed tool-result block
    /// until `<tool_response|>`. The model sometimes regurgitates the
    /// tool result that was fed back to it; it is never real content.
    InToolResponse,
}

/// Tags watched while in [`State::Text`]. Order doesn't matter; the
/// parser picks the earliest match.
///
/// gemma4 wobbles on the tool-call *open* marker — the canonical
/// `<|tool_call>` aside, the model emits `<|tool>` (the definition
/// marker), `<|call:` (the `call:` prefix run into the open), and the
/// Gemini-CLI `<|tool_code|>`, all closed by the reliable `<tool_call|>`.
/// All of them route into the (lenient) call-header path. `<think>` /
/// `</think>` are the qwen-style reasoning markers the model sometimes
/// substitutes for its native `<|channel>` channel; they route to the
/// reasoning stream rather than leaking into content. Matches a
/// permissive Gemma-4 tool-call grammar.
const TEXT_OPEN_TAGS: &[&str] = &[
    "<|tool_call>",
    TOOL_CALL_OPEN_SYM,
    TOOL_DEF_OPEN,
    CALL_OPEN_BARE,
    HERMES_CALL_OPEN,
    "<|channel>",
    THINK_OPEN,
    THINK_CLOSE,
    CHANNEL_CLOSE,
    TOOL_CODE_OPEN,
    TOOL_RESPONSE_OPEN,
    TURN_OPEN,
    "<turn|>",
    "<|end_of_turn>",
    "<end_of_turn>",
    "<|file_separator|>",
    "<|endoftext|>",
];

/// Structural gemma special tokens that are never legitimate content —
/// dropped wherever they appear (the model echoes / hallucinates them,
/// especially at higher temperature).
const DROP_TOKENS: &[&str] = &[
    "<turn|>",
    "<|end_of_turn>",
    "<end_of_turn>",
    "<|file_separator|>",
    "<|endoftext|>",
];

const CALL_PREFIX: &str = "call:";
const STRING_QUOTE: &str = "<|\"|>";
const TOOL_CALL_CLOSE: &str = "<tool_call|>";
const CHANNEL_CLOSE: &str = "<channel|>";
const TOOL_CODE_OPEN: &str = "<|tool_code|>";
const TOOL_CODE_CLOSE: &str = "</tool_code>";
const TURN_OPEN: &str = "<|turn>";
/// Tool-*definition* open marker. The model occasionally emits it as a
/// tool-*call* open (the two markers are adjacent in its vocabulary).
const TOOL_DEF_OPEN: &str = "<|tool>";
/// Mangled open where the `call:` prefix is run straight into the `<|`
/// marker start (`<|call:NAME{…}`).
const CALL_OPEN_BARE: &str = "<|call:";
/// Symmetric-pipe open variant (`<|tool_call|>…`) — neither the native
/// open `<|tool_call>` nor the close `<tool_call|>`. Distinct literal;
/// the close `<tool_call|>` never matches inside it.
const TOOL_CALL_OPEN_SYM: &str = "<|tool_call|>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
/// Hermes-style markers the model sometimes substitutes for its native
/// `<|tool_call>` / `<tool_call|>` pair. Distinct strings — they never
/// collide with the gemma markers (which carry the inner `|`).
const HERMES_CALL_OPEN: &str = "<tool_call>";
const HERMES_CALL_CLOSE: &str = "</tool_call>";
/// Tool-result block markers. These are *input* scaffolding (the chat
/// template feeds tool results back wrapped in them), but the model
/// sometimes echoes the block into its reply — dropped, never content.
const TOOL_RESPONSE_OPEN: &str = "<|tool_response>";
const TOOL_RESPONSE_CLOSE: &str = "<tool_response|>";

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

    /// Strip the gemma4 forced-scaffolding echo from the response start,
    /// then disarm.
    ///
    /// The chat template ends the generation prompt with
    /// `<|turn>model\n<|channel>thought\n<channel|>`. gemma4-26B
    /// frequently regurgitates a *corrupted* echo of that scaffolding
    /// before the real reply — one or more lines drawn from the
    /// scaffolding vocabulary, with the `|`/bracket characters dropped
    /// or mangled. Observed live: `thought\n…`, `thought**\n…`,
    /// `thought>\n…`, `<channelthought>\n…`, `thought\n<|turn>model\n…`,
    /// bare `<channel|>…`.
    ///
    /// Rule: while the leading whole line is a "scaffolding line" —
    /// short, contains a scaffolding word (`turn`/`channel`/`thought`/
    /// `model`/`system`/`user`), and consists only of ASCII-lowercase +
    /// the marker chars `<>|*/` + whitespace (so it can't be prose:
    /// real replies carry capitals, digits, or punctuation) — drop it
    /// and check the next line. Stop at the first real-content line.
    /// Streaming: wait for the newline that ends a candidate line.
    fn try_consume_echo_prefix(&mut self, is_finish: bool) {
        if !self.pending_thought_echo {
            return;
        }
        loop {
            match self.buf.find('\n') {
                Some(nl) => {
                    if is_scaffolding_line(&self.buf[..nl]) {
                        self.buf.drain(..nl + 1);
                        // Check the next line too (e.g. `thought\n<|turn>model\n`).
                        continue;
                    }
                    self.pending_thought_echo = false;
                    return;
                }
                None => {
                    // No newline yet. The current partial line is either
                    // a scaffolding line still streaming, or real content.
                    if !is_finish && could_extend_to_scaffolding_line(&self.buf) {
                        return; // wait for the newline
                    }
                    if is_finish && is_scaffolding_line(&self.buf) {
                        self.buf.clear();
                    }
                    self.pending_thought_echo = false;
                    return;
                }
            }
        }
    }

    fn drain(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) {
        loop {
            let progress = match self.state {
                State::Text => self.step_text(out, is_finish),
                State::InChannel => self.step_in_channel(out, is_finish),
                State::InCallHeader => self.step_in_call_header(out),
                State::InArgs => self.step_in_args(out),
                State::AwaitingClose => self.step_awaiting_close(out),
                State::InToolCode => self.step_in_tool_code(out, is_finish),
                State::InToolResponse => self.step_in_tool_response(is_finish),
                State::InTurnHeader => self.step_in_turn_header(is_finish),
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
                // Every tool-call open variant funnels into the lenient
                // call header. `<|call:` carries the `call:` prefix in the
                // marker itself; the header treats `call:` as optional, so
                // the residual `NAME{…}` parses uniformly.
                "<|tool_call>" | TOOL_CALL_OPEN_SYM | TOOL_DEF_OPEN | CALL_OPEN_BARE
                | HERMES_CALL_OPEN => State::InCallHeader,
                "<|channel>" | THINK_OPEN => State::InChannel,
                t if t == TOOL_CODE_OPEN => State::InToolCode,
                t if t == TOOL_RESPONSE_OPEN => State::InToolResponse,
                t if t == TURN_OPEN => State::InTurnHeader,
                // The chat template ends with `<|channel>thought\n<channel|>`,
                // so the model's first emitted token is often a stuttered
                // `<channel|>`. With no preceding `<|channel>` open, we're
                // already in State::Text — drop the close marker and stay.
                // A stray `</think>` (the qwen-style reasoning close the
                // model substitutes) is the same noise. The other
                // DROP_TOKENS (turn close, end-of-turn, file separator)
                // are likewise structural — drop and stay in Text.
                t if t == CHANNEL_CLOSE || t == THINK_CLOSE || DROP_TOKENS.contains(&t) => {
                    State::Text
                }
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

    fn step_in_tool_response(&mut self, is_finish: bool) -> bool {
        // Drop the echoed tool-result block through its close marker.
        if let Some(pos) = self.buf.find(TOOL_RESPONSE_CLOSE) {
            self.buf.drain(..pos + TOOL_RESPONSE_CLOSE.len());
            self.state = State::Text;
            return true;
        }
        // Hold for the close unless a partial close at the tail could still
        // complete; at EOF drop whatever remains.
        if is_finish {
            self.buf.clear();
            self.state = State::Text;
        }
        false
    }

    fn step_in_turn_header(&mut self, is_finish: bool) -> bool {
        // `<|turn>role\n` — drop the whole header line.
        if let Some(nl) = self.buf.find('\n') {
            self.buf.drain(..nl + 1);
            self.state = State::Text;
            return true;
        }
        if is_finish {
            self.buf.clear();
            self.state = State::Text;
        }
        false
    }

    fn step_in_channel(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        // A reasoning block opened with `<|channel>` or `<think>` closes
        // with whichever close marker the model emits first — it mixes
        // the native `<channel|>` and the qwen-style `</think>`.
        let close = [CHANNEL_CLOSE, THINK_CLOSE]
            .iter()
            .filter_map(|m| self.buf.find(m).map(|p| (p, *m)))
            .min_by_key(|(p, _)| *p);
        if let Some((pos, marker)) = close {
            if pos > 0 {
                out.push(ParserEvent::ThinkDelta(self.buf[..pos].to_owned()));
            }
            self.buf.drain(..pos + marker.len());
            self.state = State::Text;
            return true;
        }
        let total = self.buf.len();
        let emit_upto = if is_finish {
            total
        } else {
            ambiguous_tail_start(&self.buf, &[CHANNEL_CLOSE, THINK_CLOSE])
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
        // Body shape is `[call:]NAME<opener>…` where `<opener>` is the
        // native `{` or the python-call `(`. `call:` is optional — the
        // model drops it in several open-marker variants. Hold only while
        // the buffer could still complete to the `call:` prefix.
        let after_prefix = if self.buf.starts_with(CALL_PREFIX) {
            CALL_PREFIX.len()
        } else if CALL_PREFIX.starts_with(self.buf.as_str()) {
            return false;
        } else {
            0
        };
        let rest = &self.buf[after_prefix..];
        let brace = rest.find('{');
        let paren = rest.find('(');
        let opener = match (brace, paren) {
            (Some(b), Some(p)) if p < b => Some((b'(', p)),
            (Some(b), _) => Some((b'{', b)),
            (None, Some(p)) => Some((b'(', p)),
            (None, None) => None,
        };
        // No-args call: the close marker arrives before any opener.
        if let Some(c) = rest.find(TOOL_CALL_CLOSE) {
            if opener.is_none_or(|(_, o)| c < o) {
                let name = rest[..c].trim().to_owned();
                self.buf.drain(..after_prefix + c);
                self.emit_call_head(out, name, "{}".to_owned());
                self.state = State::AwaitingClose;
                return true;
            }
        }
        let Some((kind, opos)) = opener else {
            // No opener and no close yet — wait for more bytes. finish()
            // flushes a header still stuck at EOF.
            return false;
        };
        let name = rest[..opos].trim().to_owned();
        if kind == b'{' {
            // Native args body — hand off to the depth-tracking scanner.
            self.buf.drain(..after_prefix + opos + 1);
            self.current_name = name.clone();
            self.args_body.clear();
            self.args_depth = 1;
            self.args_in_string = false;
            out.push(ParserEvent::ToolCallOpen {
                index: self.next_index,
                name,
            });
            self.state = State::InArgs;
            return true;
        }
        // Python-call body `(…)`: need the matching `)` in the buffer.
        // Hold until it lands — the body is short (one decode chunk in
        // practice, and the non-stream path delivers the whole reply).
        let inner_start = after_prefix + opos;
        let Some(close_rel) = matching_paren(&self.buf[inner_start..]) else {
            return false;
        };
        let inner = self.buf[inner_start + 1..inner_start + close_rel].to_owned();
        let json =
            serde_json::to_string(&parse_python_args(&inner)).unwrap_or_else(|_| "{}".to_owned());
        self.buf.drain(..inner_start + close_rel + 1);
        self.emit_call_head(out, name, json);
        self.state = State::AwaitingClose;
        true
    }

    /// Emit `ToolCallOpen` + the complete `ToolCallArgumentsDelta` for a
    /// call whose args were parsed in one shot (no-args or python-call
    /// body). The matching `ToolCallClose` is emitted by
    /// [`Self::step_awaiting_close`] once the close marker lands.
    fn emit_call_head(&mut self, out: &mut Vec<ParserEvent>, name: String, json: String) {
        out.push(ParserEvent::ToolCallOpen {
            index: self.next_index,
            name,
        });
        out.push(ParserEvent::ToolCallArgumentsDelta {
            index: self.next_index,
            arguments: json,
        });
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
        // The model closes a call with the native `<tool_call|>`, the
        // Hermes-style `</tool_call>`, or (for `<|tool_code|>`-opened
        // calls) `</tool_code>`.
        let close = [TOOL_CALL_CLOSE, HERMES_CALL_CLOSE, TOOL_CODE_CLOSE]
            .iter()
            .filter_map(|m| self.buf.find(m).map(|p| (p, *m)))
            .min_by_key(|(p, _)| *p);
        if let Some((pos, marker)) = close {
            self.buf.drain(..pos + marker.len());
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

    fn step_in_tool_code(&mut self, out: &mut Vec<ParserEvent>, is_finish: bool) -> bool {
        // A `<|tool_code|>` block closes with `</tool_code>` (agent-prose
        // convention) OR the native `<tool_call|>` (the model mixes them).
        let close = [TOOL_CODE_CLOSE, TOOL_CALL_CLOSE]
            .iter()
            .filter_map(|m| self.buf.find(m).map(|p| (p, *m)))
            .min_by_key(|(p, _)| *p);
        if let Some((pos, marker)) = close {
            let block = self.buf[..pos].to_owned();
            self.buf.drain(..pos + marker.len());
            self.state = State::Text;
            self.emit_tool_code_block(out, &block);
            return true;
        }
        // No close marker yet. Hold for more input unless this is EOF, in
        // which case parse what we have (the model may have stopped before
        // the close, or the close was dropped).
        if is_finish {
            let block = std::mem::take(&mut self.buf);
            self.state = State::Text;
            self.emit_tool_code_block(out, &block);
            return true;
        }
        false
    }

    /// Turn a buffered `<|tool_code|>` block body into a tool call, or
    /// surface it verbatim if it can't be structured.
    fn emit_tool_code_block(&mut self, out: &mut Vec<ParserEvent>, block: &str) {
        if let Some((name, args)) = parse_tool_code_block(block) {
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
            return;
        }
        emit_text(out, &format!("{TOOL_CODE_OPEN}{block}"));
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

/// Parse a `<|tool_code|>` block body into `(name, json_arguments)`.
///
/// Accepts the Gemini-CLI agent-prose convention:
/// ```text
/// NAME
/// {"arg":"value"}
/// ```
/// The first identifier-shaped token is the function name; the first
/// JSON object after it is the arguments. Tolerates stuttered open
/// markers and a ```` ```tool_code ```` markdown fence the model may
/// have re-emitted inside the block. Returns `None` when there's no
/// identifier name or no parseable JSON object — the caller then
/// surfaces the block verbatim rather than fabricating a call.
fn parse_tool_code_block(block: &str) -> Option<(String, String)> {
    let mut s = block.trim();
    // Strip stuttered open markers + markdown fences the model repeats.
    loop {
        let trimmed = s.trim_start();
        let next = trimmed
            .strip_prefix(TOOL_CODE_OPEN)
            .or_else(|| trimmed.strip_prefix("```tool_code"))
            .or_else(|| trimmed.strip_prefix("```"));
        match next {
            Some(rest) => s = rest,
            None => break,
        }
    }
    let s = s.trim();
    // JSON / native object body: `NAME { … }`.
    if let Some(brace) = s.find('{') {
        if let Some(name) = first_identifier(&s[..brace]) {
            let mut stream = serde_json::Deserializer::from_str(&s[brace..]).into_iter::<Value>();
            if let Some(Ok(v @ Value::Object(_))) = stream.next() {
                return Some((name, serde_json::to_string(&v).ok()?));
            }
        }
    }
    // Python-call body: `NAME(arg=value, …)`.
    parse_python_call(s)
}

/// Parse a python-style call `NAME(k=v, …)` into `(name, json_args)`.
/// gemma4 emits this Gemini-CLI convention instead of its native
/// `call:NAME{…}` for a sizeable fraction of calls.
///
/// Used only as the *fallback* for a `<|tool_code|>` block, which is
/// ambiguous — it may wrap genuine code (`print('hi')`) rather than a
/// tool call. To avoid fabricating a call from a code snippet, a
/// non-empty arg list must yield at least one *named* (`k=v`) argument;
/// a purely positional body is treated as code and left to leak.
/// Returns `None` when there's no identifier name, no balanced `(…)`, or
/// the body is positional-only.
fn parse_python_call(s: &str) -> Option<(String, String)> {
    let s = s.trim();
    let open = s.find('(')?;
    let name = first_identifier(&s[..open])?;
    let close_rel = matching_paren(&s[open..])?;
    let inner = &s[open + 1..open + close_rel];
    let args = parse_python_args(inner);
    let n_named = args.as_object().map_or(0, Map::len);
    if !inner.trim().is_empty() && n_named == 0 {
        return None;
    }
    Some((name, serde_json::to_string(&args).ok()?))
}

/// Byte index of the `)` matching the `(` at byte 0 of `s`, or `None` if
/// `s` doesn't start with `(` or has no balanced close. String literals
/// (`"…"` / `'…'`, with backslash escapes) are skipped so a paren inside
/// a value doesn't miscount.
fn matching_paren(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.first() != Some(&b'(') {
        return None;
    }
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'"' | b'\'' => quote = Some(c),
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            },
        }
        i += 1;
    }
    None
}

/// Parse a python argument list body (between the outer parens) into a
/// JSON object. Positional args (no `=`) are skipped — a schema-bound
/// tool call needs named args. Values use python literals: `"s"`/`'s'`
/// → string, `True`/`False` → bool, `None` → null, JSON literals
/// (numbers, `[...]`, `{...}`) parsed as-is, else a bare-token string.
fn parse_python_args(inner: &str) -> Value {
    let mut obj = Map::new();
    for part in split_top_level(inner, b',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some(eq) = split_top_level(part, b'=').next().map(|k| k.len()) else {
            continue;
        };
        if eq >= part.len() {
            continue; // positional arg, no `=`
        }
        let key = part[..eq].trim().trim_matches(['"', '\'']).to_owned();
        if key.is_empty() {
            continue;
        }
        obj.insert(key, parse_py_value(part[eq + 1..].trim()));
    }
    Value::Object(obj)
}

/// Split `s` on top-level occurrences of `sep` — not inside `"…"`/`'…'`
/// strings or `(`/`[`/`{` brackets. Always yields at least one segment.
fn split_top_level(s: &str, sep: u8) -> impl Iterator<Item = &str> {
    let b = s.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'"' | b'\'' => quote = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                _ if c == sep && depth == 0 => {
                    parts.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            },
        }
        i += 1;
    }
    parts.push(&s[start..]);
    parts.into_iter()
}

/// Parse one python literal value into JSON.
fn parse_py_value(s: &str) -> Value {
    let s = s.trim();
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        return Value::String(s[1..s.len() - 1].to_owned());
    }
    match s {
        "True" => return Value::Bool(true),
        "False" => return Value::Bool(false),
        "None" | "null" => return Value::Null,
        _ => {}
    }
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return v;
    }
    Value::String(s.to_owned())
}

/// First `[A-Za-z0-9_]+` run in `s`, or `None` if there is none.
fn first_identifier(s: &str) -> Option<String> {
    let id: String = s
        .chars()
        .skip_while(|c| !(c.is_ascii_alphanumeric() || *c == '_'))
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (!id.is_empty()).then_some(id)
}

/// Characters a garbled gemma scaffolding marker collapses to.
fn is_scaffold_char(c: char) -> bool {
    c.is_ascii_lowercase() || matches!(c, '<' | '>' | '|' | '*' | '/') || c.is_whitespace()
}

const SCAFFOLD_WORDS: &[&str] = &[
    "turn", "channel", "thought", "model", "system", "user", "assistant",
];

/// True if `line` is a corrupted echo of the gemma generation-prompt
/// scaffolding (`<|turn>model`, `<|channel>thought`, `<channel|>`, …)
/// rather than real reply content. Tight signal: short, drawn only from
/// ASCII-lowercase + marker chars + whitespace (real replies carry
/// capitals / digits / punctuation), and either carries a marker char
/// alongside a scaffolding word or is the bare `thought` echo.
fn is_scaffolding_line(line: &str) -> bool {
    let s = line.trim();
    if s.is_empty() || s.len() > 40 {
        return false;
    }
    if !s.chars().all(is_scaffold_char) {
        return false;
    }
    let has_marker = s.contains(['<', '>', '|']);
    let has_word = SCAFFOLD_WORDS.iter().any(|w| s.contains(w));
    // The bare `thought` echo: `thought` then only marker/whitespace
    // (so `thoughts`, `thoughtful` — real words — are NOT scaffolding).
    let bare_thought = s.strip_prefix("thought").is_some_and(|rest| {
        rest.chars()
            .all(|c| matches!(c, '<' | '>' | '|' | '*' | '/') || c.is_whitespace())
    });
    (has_marker && has_word) || bare_thought
}

/// True if a partial (newline-less) leading run could still become a
/// scaffolding line, so the echo guard should wait for more bytes
/// rather than emit it as content. Only waits when the partial already
/// shows scaffolding intent (a marker char, or a `thought` prefix) —
/// real lowercase prose flows immediately.
fn could_extend_to_scaffolding_line(partial: &str) -> bool {
    let s = partial.trim_start();
    if s.is_empty() {
        return true;
    }
    if s.len() > 40 || !s.chars().all(is_scaffold_char) {
        return false;
    }
    s.contains(['<', '>', '|']) || "thought".starts_with(s) || s.starts_with("thought")
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
        if self.pending_thought_echo {
            // Still resolving the forced-thought echo prefix; hold the
            // buffer rather than letting drain() emit a partial echo
            // (`thought`) as text. finish() forces a decision at EOF.
            return Vec::new();
        }
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
            State::InChannel
            | State::Text
            | State::InToolCode
            | State::InToolResponse
            | State::InTurnHeader => {
                // Already drained on is_finish=true above (InToolCode's
                // EOF leak/parse + InTurnHeader's / InToolResponse's EOF
                // drop are handled inside their step fns).
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

    fn tool_call_of(events: &[ParserEvent]) -> Option<(String, String)> {
        let name = events.iter().find_map(|e| match e {
            ParserEvent::ToolCallOpen { name, .. } => Some(name.clone()),
            _ => None,
        })?;
        let args = events.iter().find_map(|e| match e {
            ParserEvent::ToolCallArgumentsDelta { arguments, .. } => Some(arguments.clone()),
            _ => None,
        })?;
        Some((name, args))
    }

    fn text_of(events: &[ParserEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn tool_code_block_json_body() {
        // The shape a Gemini-CLI-style agent provokes from gemma.
        let events = collect("<|tool_code|>bash\n{\"command\": \"ls -R\"}\n</tool_code>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "bash");
        assert_eq!(args, r#"{"command":"ls -R"}"#);
        assert!(text_of(&events).is_empty(), "no text leak");
    }

    #[test]
    fn tool_code_block_stuttered_open() {
        // Models often double the open marker, with a leading space.
        let events = collect(" <|tool_code|> <|tool_code|>bash\n{\"command\": \"ls -R\"}\n</tool_code>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "bash");
        assert_eq!(args, r#"{"command":"ls -R"}"#);
    }

    #[test]
    fn tool_code_named_tool() {
        let events =
            collect("<|tool_code|>run_shell_command\n{\"command\":\"ls\"}\n</tool_code>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "run_shell_command");
        assert_eq!(args, r#"{"command":"ls"}"#);
    }

    // The leniency corpus below is six real gemma-4-31B captures (Q4_0 and
    // Q8_0, pp2tp2): the model wobbles the open marker and the body form
    // but always closes with `<tool_call|>`. Each must yield a structured
    // tool call with no text leak.

    #[test]
    fn lenient_canonical_native_regression() {
        let events =
            collect("<|tool_call>call:run_shell{comando:<|\"|>ls -F<|\"|>}<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "run_shell");
        assert_eq!(args, r#"{"comando":"ls -F"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_missing_call_prefix() {
        let events =
            collect("<|tool_call>get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "get_weather");
        assert_eq!(args, r#"{"city":"Paris"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_tool_def_open() {
        let events =
            collect("<|tool>run_shell{comando:<|\"|>find . -name OTB<|\"|>}<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "run_shell");
        assert_eq!(args, r#"{"comando":"find . -name OTB"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_tool_def_open_with_call_prefix() {
        let events =
            collect("<|tool>call:run_shell{comando:<|\"|>ls<|\"|>}<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "run_shell");
        assert_eq!(args, r#"{"comando":"ls"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_bare_call_open() {
        let events =
            collect("<|call:run_shell{comando:<|\"|>ls -F<|\"|>}<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "run_shell");
        assert_eq!(args, r#"{"comando":"ls -F"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_tool_code_python_body() {
        let events = collect("<|tool_code|>get_weather(city=\"Tokyo\")<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "get_weather");
        assert_eq!(args, r#"{"city":"Tokyo"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_tool_call_open_python_body() {
        let events = collect("<|tool_call>get_weather(city=\"Paris\")<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "get_weather");
        assert_eq!(args, r#"{"city":"Paris"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_symmetric_pipe_open_with_doubled_echo() {
        // Real pi capture (Q8_0, thinking on): symmetric-pipe open
        // `<|tool_call|>`, a clean body, then an echoed `<|tool_response|>`
        // + duplicate body before the real `<tool_call|>` close. The first
        // body parses; AwaitingClose skips the echo to the close.
        let events = collect(
            "<|tool_call|>call:bash{command:<|\"|>cd OTB-FRONT && git log -1 -p<|\"|>}<|tool_response|>call:bash{command:<|\"|>cd OTB-FRONT && git log -1 -p<|\"|>}<tool_call|>",
        );
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "bash");
        assert_eq!(args, r#"{"command":"cd OTB-FRONT && git log -1 -p"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_hermes_markers_python_body() {
        // The model sometimes substitutes the Hermes `<tool_call>…
        // </tool_call>` pair for its native markers.
        let events = collect("<tool_call>get_weather(city=\"Paris\")</tool_call>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "get_weather");
        assert_eq!(args, r#"{"city":"Paris"}"#);
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_python_body_multi_arg_types() {
        let events = collect(
            "<|tool_code|>edit(path=\"a.rs\", line=12, dry_run=True, tags=[\"x\",\"y\"])<tool_call|>",
        );
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "edit");
        let v: Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["path"], "a.rs");
        assert_eq!(v["line"], 12);
        assert_eq!(v["dry_run"], true);
        assert_eq!(v["tags"], serde_json::json!(["x", "y"]));
    }

    #[test]
    fn lenient_no_args_call() {
        let events = collect("<|tool_call>call:list_dir<tool_call|>");
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "list_dir");
        assert_eq!(args, "{}");
        assert!(text_of(&events).is_empty(), "no text leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_tool_response_echo_dropped() {
        // The model regurgitates the injected tool result before its next
        // turn; the block must not leak into content.
        let events = collect(
            "<|tool_response>response:run_shell{value:<|\"|>commit abc123<|\"|>}<tool_response|>Now I'll inspect it.",
        );
        assert_eq!(text_of(&events), "Now I'll inspect it.");
        assert!(tool_call_of(&events).is_none());
    }

    #[test]
    fn lenient_tool_response_echo_then_call() {
        let events = collect(
            "<|tool_response>response:ls{value:<|\"|>a.txt<|\"|>}<tool_response|><|tool_call>call:run_shell{comando:<|\"|>cat a.txt<|\"|>}<tool_call|>",
        );
        let (name, args) = tool_call_of(&events).expect("tool call after echo");
        assert_eq!(name, "run_shell");
        assert_eq!(args, r#"{"comando":"cat a.txt"}"#);
        assert!(text_of(&events).is_empty(), "no leak: {:?}", text_of(&events));
    }

    #[test]
    fn lenient_think_tags_route_to_reasoning() {
        // The model substitutes qwen-style <think></think> for its native
        // channel; the empty block must not leak into content.
        let events = collect("<think>\n</think>The answer is 42.");
        assert_eq!(text_of(&events), "The answer is 42.");
        assert!(
            !text_of(&events).contains("think"),
            "think markers leaked: {:?}",
            text_of(&events)
        );
    }

    #[test]
    fn lenient_brace_inside_string_value_preserved() {
        // A `{`/`}` inside a value must not
        // break depth tracking. flambeau delimits by <|"|>, so it holds.
        let events = collect(
            "<|tool_call>call:write{code:<|\"|>fn f() { x }<|\"|>}<tool_call|>",
        );
        let (name, args) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "write");
        let v: Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["code"], "fn f() { x }");
    }

    #[test]
    fn tool_code_no_close_at_eof_still_parses() {
        // Model stopped before emitting </tool_code>; finish() should
        // still lift the call from what it has.
        let events = collect("<|tool_code|>bash\n{\"command\":\"ls\"}");
        let (name, _) = tool_call_of(&events).expect("tool call");
        assert_eq!(name, "bash");
    }

    #[test]
    fn tool_code_unparseable_leaks_verbatim() {
        // No JSON object → can't structure → surface verbatim, no fake
        // tool call.
        let events = collect("<|tool_code|>python\nprint('hi')\n</tool_code>");
        assert!(tool_call_of(&events).is_none());
        assert!(
            text_of(&events).contains("print('hi')"),
            "block should surface: {events:#?}"
        );
    }

    #[test]
    fn tool_code_streaming_char_by_char() {
        let input = "<|tool_code|>bash\n{\"command\":\"ls -R\"}\n</tool_code>";
        let one_shot = collect(input);
        let mut p = Gemma4ToolCallParser::new();
        let mut out: Vec<ParserEvent> = Vec::new();
        for ch in input.chars() {
            out.extend(p.push(&ch.to_string()));
        }
        out.extend(p.finish());
        assert_eq!(ParserEvent::coalesce(out), one_shot);
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

    fn echo_text(input: &str) -> String {
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut events = p.push(input);
        events.extend(p.finish());
        ParserEvent::coalesce(events)
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn with_pending_channel_close_garbled_close_variants() {
        // Live gemma4-26B shapes: `thought` + a garbled `<channel|>`
        // (`**`, `>`, bare) + newline, then the reply.
        assert_eq!(
            echo_text("thought\nThe three primary colors are red."),
            "The three primary colors are red."
        );
        assert_eq!(
            echo_text("thought**\nThe boiling point is 100C."),
            "The boiling point is 100C."
        );
        assert_eq!(echo_text("thought>\nHola."), "Hola.");
        assert_eq!(
            echo_text("thought\n<channel|>The answer is Paris."),
            "The answer is Paris."
        );
    }

    #[test]
    fn with_pending_channel_close_real_thought_word_preserved() {
        // A genuine reply that merely starts with `thought…` must NOT be
        // stripped — divergence at the first alphanumeric after `thought`.
        assert_eq!(
            echo_text("thoughts are mental processes."),
            "thoughts are mental processes."
        );
        assert_eq!(
            echo_text("thought experiments are useful."),
            "thought experiments are useful."
        );
    }

    #[test]
    fn structural_tokens_dropped_anywhere() {
        // Clean gemma special tokens are never content — dropped mid /
        // end of response too (not just at the start).
        let t = |s: &str| {
            let mut p = Gemma4ToolCallParser::new();
            let mut e = p.push(s);
            e.extend(p.finish());
            ParserEvent::coalesce(e)
                .iter()
                .filter_map(|x| match x {
                    ParserEvent::TextDelta(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<String>()
        };
        assert_eq!(t("c-a-t (cat)<|end_of_turn>"), "c-a-t (cat)");
        assert_eq!(t("answer here <|file_separator|>"), "answer here ");
        assert_eq!(t("<|turn>model\nThe reply."), "The reply.");
        assert_eq!(t("mid <turn|> response"), "mid  response");
    }

    #[test]
    fn with_pending_channel_close_turn_header_echo() {
        // The user-reported shape: `thought\n<|turn>model\n` then reply.
        assert_eq!(
            echo_text("thought\n<|turn>model\nThe capital is Paris."),
            "The capital is Paris."
        );
        // Garbled `<|channel>thought` merged: `<channelthought>`.
        assert_eq!(
            echo_text("<channelthought>\nHello! How can I help you today?"),
            "Hello! How can I help you today?"
        );
        // Bare `<|turn>model` line.
        assert_eq!(echo_text("<|turn>model\nApple."), "Apple.");
    }

    #[test]
    fn with_pending_channel_close_scaffolding_keeps_real_lowercase() {
        // Real lowercase content that merely contains a scaffold word or
        // an angle bracket must NOT be stripped.
        assert_eq!(echo_text("the model is large."), "the model is large.");
        assert_eq!(echo_text("use <br> for line breaks."), "use <br> for line breaks.");
        // A reply that is the single word the user asked for.
        assert_eq!(echo_text("thoughts"), "thoughts");
    }

    #[test]
    fn with_pending_channel_close_garbled_streaming_char_by_char() {
        let input = "thought**\nThe boiling point is 100C.";
        let mut p = Gemma4ToolCallParser::with_pending_channel_close();
        let mut out: Vec<ParserEvent> = Vec::new();
        for ch in input.chars() {
            out.extend(p.push(&ch.to_string()));
        }
        out.extend(p.finish());
        let text: String = ParserEvent::coalesce(out)
            .iter()
            .filter_map(|e| match e {
                ParserEvent::TextDelta(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "The boiling point is 100C.");
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
