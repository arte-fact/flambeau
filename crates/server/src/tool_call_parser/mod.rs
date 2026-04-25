//! Tool-call parser trait + format-dispatch.
//!
//! The parser consumes the **decoded text stream** (what the tokenizer
//! produced from sampled tokens) and emits a stream of structured events
//! that the chat-completions handler (T2.5) assembles into
//! `tool_calls[]` — for the non-streaming path — and the SSE producer
//! (T3) translates into `delta.tool_calls[]` chunks.
//!
//! The core design goal (see ROADMAP-V2-TOOL-CALLING-AND-MCP §T2): the
//! state machine's shape mirrors XGrammar's "structural tag" primitive —
//! `Outside` (free text) vs `Inside { schema }` (constrained JSON body).
//! Today we run it post-hoc over the decoded text; under T5, the same
//! states drive an `llguidance` mask generator that constrains logits
//! directly. Keeping the shape compatible from day one avoids a rewrite
//! when T5 lands.
//!
//! Two formats exist in the Qwen family and we expose a trait so the
//! format-specific logic is one file each:
//!
//! - **Hermes-JSON** (`hermes.rs`): `<tool_call>\n{...JSON...}\n</tool_call>`.
//!   Qwen3 / Qwen3.5 / Qwen3.6 with the Qwen-official (HuggingFace-pushed)
//!   chat template.
//! - **Qwen3-Coder XML** (`qwen3_coder.rs`): nested XML with
//!   `<function=name>…<parameter=k>v</parameter>…</function>` inside a
//!   `<tool_call>`. Qwen3-Coder family and — per the T1.3 parity-cert
//!   finding — the Unsloth "UD" Qwen3.6 quants on this rig.
//!
//! Both parsers conform to [`ToolCallParser`]. Selection at request
//! time goes through [`dispatcher`], which honours the `tool_call_format`
//! request field (`"hermes" | "qwen3_coder" | "auto"`) and falls back to
//! an architecture-based default when the caller says `"auto"` or omits
//! it.
//!
//! The `arguments` field emitted by [`ParserEvent::ToolCallArgumentsDelta`]
//! / collected across the turn is always a JSON-encoded *string* — never
//! an object — so the server wire format stays stable. This guards
//! llama.cpp #20198.

use anyhow::{anyhow, Result};

pub mod hermes;
pub mod qwen3_coder;

/// Events emitted as the decoder-text stream is consumed.
///
/// The stream is append-only: text emitted earlier in a turn is never
/// retracted. The parser MUST satisfy the "buffer-before-emit"
/// discipline — an ambiguous prefix (e.g. a bare `<` in free text)
/// stays buffered until the following characters resolve the ambiguity.
/// Shipping a `TextDelta("<tool_cal")` that later becomes a tool-call
/// open is the class of bug practitioners reported across llama.cpp,
/// vLLM, and SGLang — don't do it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ParserEvent {
    /// Free-text content to surface in the assistant message's `content`
    /// field. May be empty (no-op).
    TextDelta(String),
    /// `<think>…</think>` content, emitted so future `reasoning_content`
    /// support can surface it. Today the server discards it.
    ThinkDelta(String),
    /// Start of a tool call. `index` is unique within one assistant turn
    /// (0-based, monotonically increasing). `name` is the function name.
    /// Some parsers (Hermes) know the name at open time; Coder-XML
    /// discovers it inside the body — both are legal so long as the
    /// server's assembler tolerates `name` being emitted on Open OR
    /// first ArgumentsDelta.
    ToolCallOpen { index: u32, name: String },
    /// A chunk of the arguments-JSON string for the tool call at `index`.
    /// Chunks concatenate to the full arguments string. The parser must
    /// re-serialise object-shaped payloads (Coder-XML's parameter list)
    /// into a JSON string before emitting here.
    ToolCallArgumentsDelta { index: u32, arguments: String },
    /// End of the tool call at `index`. After this the parser is back
    /// in free-text mode until the next `<tool_call>`.
    ToolCallClose { index: u32 },
}

impl ParserEvent {
    /// Concatenate adjacent text-like events (TextDelta+TextDelta,
    /// ThinkDelta+ThinkDelta, ArgumentsDelta for same index +
    /// ArgumentsDelta). Makes events comparable regardless of how a
    /// stream was chunked. Used by fixture tests (T2.4) and by the
    /// non-streaming assembler (T2.5).
    pub fn coalesce(events: Vec<ParserEvent>) -> Vec<ParserEvent> {
        let mut out: Vec<ParserEvent> = Vec::with_capacity(events.len());
        for e in events {
            match (out.last_mut(), &e) {
                (
                    Some(ParserEvent::TextDelta(acc)),
                    ParserEvent::TextDelta(s),
                ) => acc.push_str(s),
                (
                    Some(ParserEvent::ThinkDelta(acc)),
                    ParserEvent::ThinkDelta(s),
                ) => acc.push_str(s),
                (
                    Some(ParserEvent::ToolCallArgumentsDelta {
                        index: a_idx,
                        arguments: acc,
                    }),
                    ParserEvent::ToolCallArgumentsDelta {
                        index: b_idx,
                        arguments: s,
                    },
                ) if a_idx == b_idx => acc.push_str(s),
                _ => out.push(e),
            }
        }
        out
    }
}

/// T2.5 assembler: walk a (coalesced) parser-event stream and split it
/// into the assistant `content` string and the `tool_calls[]` list that
/// go into the OpenAI response body.
///
/// - `TextDelta` chunks concatenate into `content`.
/// - `ThinkDelta` is discarded today (reasoning_content is V3 scope).
/// - Each `ToolCallOpen` + its `ToolCallArgumentsDelta`s + matching
///   `ToolCallClose` build one [`crate::api::ToolCall`].
/// - Close events whose matching Open never fired still produce a
///   `ToolCall` with an empty name — imperfect, but preserves parser
///   events rather than dropping them.
pub fn split_events(
    events: Vec<ParserEvent>,
) -> (String, Vec<crate::api::ToolCall>) {
    use crate::api::{FunctionCall, ToolCall};
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut pending: std::collections::HashMap<u32, (String, String)> =
        std::collections::HashMap::new();

    for e in events {
        match e {
            ParserEvent::TextDelta(s) => content.push_str(&s),
            ParserEvent::ThinkDelta(_) => {}
            ParserEvent::ToolCallOpen { index, name } => {
                pending
                    .entry(index)
                    .and_modify(|e| e.0 = name.clone())
                    .or_insert((name, String::new()));
            }
            ParserEvent::ToolCallArgumentsDelta { index, arguments } => {
                pending
                    .entry(index)
                    .or_insert_with(|| (String::new(), String::new()))
                    .1
                    .push_str(&arguments);
            }
            ParserEvent::ToolCallClose { index } => {
                let (name, args) = pending.remove(&index).unwrap_or_default();
                tool_calls.push(ToolCall {
                    id: format!("call_{}", index),
                    kind: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: args,
                    },
                });
            }
        }
    }
    (content, tool_calls)
}

/// A streaming-capable tool-call parser.
///
/// Implementations must be deterministic: feeding the same bytes split
/// into different chunks (e.g. `push("<tool_call>\n")` vs
/// `push("<tool_"); push("call>\n");`) must produce the same total
/// event stream. The T2.4 fixture corpus exercises this invariant via
/// randomised chunk splits.
///
/// Malformed input is NEVER a panic. Garbage JSON in a tool-call body,
/// a missing `</tool_call>`, or a `<tool_call>` inside a `<think>`
/// block — each parser should degrade gracefully: surface the run of
/// text as `TextDelta`, log a warning (upstream of the parser), and
/// continue. Dropping the assistant turn is worse than surfacing
/// imperfect text.
pub trait ToolCallParser: Send {
    /// Feed a chunk of decoder-produced text. Returns the events
    /// produced, in order. May return an empty `Vec` when the chunk is
    /// entirely inside a buffered ambiguous prefix.
    fn push(&mut self, chunk: &str) -> Vec<ParserEvent>;

    /// Flush on decode-loop exit. Returns any events buffered behind
    /// the buffer-before-emit guard (e.g. trailing text that never
    /// resolved to a tool-call). Idempotent after the first call.
    fn finish(&mut self) -> Vec<ParserEvent>;
}

/// Format tag — the on-the-wire tool-call shape the parser expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Hermes-style JSON: `<tool_call>\n{"name":...,"arguments":...}\n</tool_call>`.
    Hermes,
    /// Qwen3-Coder XML: `<tool_call><function=name><parameter=k>v</parameter>…</function></tool_call>`.
    QwenCoder,
}

impl ToolCallFormat {
    /// Parse `"hermes" | "qwen3_coder"`. Other strings — including
    /// `"auto"` — are not accepted here; use [`dispatcher`] to resolve
    /// `"auto"` against a model architecture.
    pub fn from_explicit(s: &str) -> Result<Self> {
        match s {
            "hermes" => Ok(Self::Hermes),
            "qwen3_coder" | "qwen3-coder" | "qwen_coder" => Ok(Self::QwenCoder),
            other => Err(anyhow!(
                "unknown tool_call_format: {other:?} (expected \"hermes\" or \"qwen3_coder\")"
            )),
        }
    }
}

/// Choose a parser format, honouring (in precedence order):
///
/// 1. An explicit `request_override` — `Some("hermes" | "qwen3_coder")`.
/// 2. `"auto"` or `None` — fall back to the architecture default.
///
/// Architecture defaults are deliberately conservative:
/// - `qwen35moe` (our V1 model arch) defaults to **`Hermes`**.
///   **Caveat**: the specific Qwen3.6 GGUF on the V1 rig (Unsloth's
///   `UD-Q8_K_XL` build) ships a **Coder-XML** chat template — see
///   `certs/chat_template/qwen35moe_tools/README.md`. Clients running
///   that GGUF should set `tool_call_format: "qwen3_coder"` explicitly.
///   A finer-grained auto-detection from the loaded GGUF template will
///   land as a follow-up once we reliably probe the template shape.
/// - any other arch → `Hermes` (safest default — the open-source
///   standard used across vLLM / SGLang / llama.cpp for non-Coder Qwens).
pub fn choose_format(
    request_override: Option<&str>,
    server_default: ToolCallFormat,
) -> Result<ToolCallFormat> {
    match request_override {
        Some("auto") | Some("") | None => Ok(server_default),
        Some(s) => ToolCallFormat::from_explicit(s),
    }
}

/// Peek at the GGUF-embedded chat template and decide whether the
/// model is trained to emit Hermes-JSON or Qwen3-Coder-XML
/// tool calls. The Unsloth "UD" Qwen3.6 GGUFs ship a Coder-XML
/// template even though `general.architecture` says `qwen35moe`,
/// so we can't rely on arch alone — but the template itself
/// contains the literal tag tokens that classify it.
///
/// Used by `flambeau serve` at startup to set
/// `ServerState.tool_call_format_default` once per process. Per-
/// request `tool_call_format` overrides are honoured first.
pub fn detect_format_from_template(template_src: &str) -> ToolCallFormat {
    // Coder-XML signature: literal tag instructions in the system
    // prompt branch. Hermes templates emit JSON-shaped guidance.
    if template_src.contains("<function=") || template_src.contains("<parameter=") {
        ToolCallFormat::QwenCoder
    } else {
        ToolCallFormat::Hermes
    }
}

/// Build a parser for the chosen format. Box-returning so callers can
/// store the parser as a trait object without knowing which concrete
/// type it is.
pub fn dispatcher(
    request_override: Option<&str>,
    server_default: ToolCallFormat,
) -> Result<Box<dyn ToolCallParser>> {
    match choose_format(request_override, server_default)? {
        ToolCallFormat::Hermes => Ok(Box::new(hermes::HermesJsonParser::new())),
        ToolCallFormat::QwenCoder => Ok(Box::new(qwen3_coder::QwenCoderXmlParser::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_hermes() {
        assert_eq!(
            choose_format(Some("hermes"), ToolCallFormat::QwenCoder).unwrap(),
            ToolCallFormat::Hermes
        );
    }

    #[test]
    fn explicit_qwen_coder() {
        assert_eq!(
            choose_format(Some("qwen3_coder"), ToolCallFormat::Hermes).unwrap(),
            ToolCallFormat::QwenCoder
        );
        assert_eq!(
            choose_format(Some("qwen3-coder"), ToolCallFormat::Hermes).unwrap(),
            ToolCallFormat::QwenCoder
        );
    }

    #[test]
    fn auto_and_none_use_server_default() {
        // None / "auto" / "" all fall through to the server default.
        assert_eq!(
            choose_format(None, ToolCallFormat::Hermes).unwrap(),
            ToolCallFormat::Hermes
        );
        assert_eq!(
            choose_format(Some("auto"), ToolCallFormat::QwenCoder).unwrap(),
            ToolCallFormat::QwenCoder
        );
        assert_eq!(
            choose_format(Some(""), ToolCallFormat::QwenCoder).unwrap(),
            ToolCallFormat::QwenCoder
        );
    }

    #[test]
    fn unknown_format_is_error() {
        assert!(choose_format(Some("qwen42"), ToolCallFormat::Hermes).is_err());
    }

    #[test]
    fn detect_format_qwen_coder_template() {
        // Unsloth UD Qwen3.6 chat template signature.
        let tpl = r#"
            ... <tool_call>
            <function=example>
            <parameter=key>
            value
            </parameter>
            </function>
            </tool_call> ...
        "#;
        assert_eq!(
            detect_format_from_template(tpl),
            ToolCallFormat::QwenCoder
        );
    }

    #[test]
    fn detect_format_hermes_template() {
        // Hermes-JSON template: no <function= / <parameter= literals.
        let tpl = r#"<tool_call>{"name":"example","arguments":{}}</tool_call>"#;
        assert_eq!(detect_format_from_template(tpl), ToolCallFormat::Hermes);
    }

    // ---- T2.5: split_events ----

    #[test]
    fn split_events_pure_text() {
        let evts = vec![
            ParserEvent::TextDelta("hello ".into()),
            ParserEvent::TextDelta("world".into()),
        ];
        let (content, tc) = split_events(ParserEvent::coalesce(evts));
        assert_eq!(content, "hello world");
        assert!(tc.is_empty());
    }

    #[test]
    fn split_events_discards_think() {
        let evts = vec![
            ParserEvent::ThinkDelta("secret reasoning".into()),
            ParserEvent::TextDelta("answer".into()),
        ];
        let (content, tc) = split_events(evts);
        assert_eq!(content, "answer");
        assert!(tc.is_empty(), "think must not leak into tool_calls");
    }

    #[test]
    fn split_events_single_tool_call() {
        let evts = vec![
            ParserEvent::TextDelta("calling: ".into()),
            ParserEvent::ToolCallOpen { index: 0, name: "f".into() },
            ParserEvent::ToolCallArgumentsDelta {
                index: 0,
                arguments: r#"{"x":1}"#.into(),
            },
            ParserEvent::ToolCallClose { index: 0 },
        ];
        let (content, tc) = split_events(evts);
        assert_eq!(content, "calling: ");
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "f");
        assert_eq!(tc[0].function.arguments, r#"{"x":1}"#);
        assert_eq!(tc[0].kind, "function");
        assert_eq!(tc[0].id, "call_0");
    }

    #[test]
    fn split_events_parallel_calls_preserve_order_and_indices() {
        let evts = vec![
            ParserEvent::ToolCallOpen { index: 0, name: "a".into() },
            ParserEvent::ToolCallArgumentsDelta {
                index: 0,
                arguments: r#"{"p":1}"#.into(),
            },
            ParserEvent::ToolCallClose { index: 0 },
            ParserEvent::ToolCallOpen { index: 1, name: "b".into() },
            ParserEvent::ToolCallArgumentsDelta {
                index: 1,
                arguments: r#"{"q":2}"#.into(),
            },
            ParserEvent::ToolCallClose { index: 1 },
        ];
        let (_, tc) = split_events(evts);
        assert_eq!(tc.len(), 2);
        assert_eq!(tc[0].function.name, "a");
        assert_eq!(tc[0].id, "call_0");
        assert_eq!(tc[1].function.name, "b");
        assert_eq!(tc[1].id, "call_1");
    }

    #[test]
    fn split_events_arguments_accumulate_across_deltas() {
        let evts = vec![
            ParserEvent::ToolCallOpen { index: 0, name: "f".into() },
            ParserEvent::ToolCallArgumentsDelta {
                index: 0,
                arguments: r#"{"x":"#.into(),
            },
            ParserEvent::ToolCallArgumentsDelta {
                index: 0,
                arguments: r#""hi"}"#.into(),
            },
            ParserEvent::ToolCallClose { index: 0 },
        ];
        let (_, tc) = split_events(evts);
        assert_eq!(tc[0].function.arguments, r#"{"x":"hi"}"#);
    }

    #[test]
    fn dispatcher_builds_parser() {
        // Both branches must construct without panicking. The passthrough
        // behaviour itself is covered in the per-parser test modules.
        let mut p = dispatcher(Some("hermes"), ToolCallFormat::QwenCoder).unwrap();
        let _ = p.push("hello");
        let _ = p.finish();
        let mut p = dispatcher(Some("qwen3_coder"), ToolCallFormat::Hermes).unwrap();
        let _ = p.push("hello");
        let _ = p.finish();
    }
}
