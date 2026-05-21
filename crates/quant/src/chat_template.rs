//! Chat-template renderer — piece 3 of 3.
//! GGUF embeds `tokenizer.chat_template` as a Jinja2 string (7816 bytes for
//! Qwen3.6 — macros, whitespace control, conditionals). `minijinja` renders
//! it against a `messages: [{role, content}, ...]` array plus a few flags
//! (`add_generation_prompt`, `enable_thinking`, ...) to produce the raw text
//! prompt the model sees.
//! The HTTP layer () constructs this from OpenAI `/v1/chat/completions`
//! payloads, renders, tokenizes (), runs the forward, and samples
//! (). Chat template is the glue between wire format and model input.

use anyhow::{anyhow, Context, Result};
use minijinja::value::{Kwargs, Value as MjValue};
use minijinja::{Environment, Error, ErrorKind};
use serde::{Deserialize, Serialize};

use crate::gguf::GgufFile;

/// Python-`json.dumps`-compatible compact JSON formatter: space after
/// `,` and `:`, no other whitespace. Matches what `nlohmann::json::dump`
/// (the C++ JSON lib llama.cpp uses, via its `tojson` Jinja filter)
/// emits by default. Byte-exact with llama.cpp is the T1.3 gate.
struct PythonCompactFormatter;

impl serde_json::ser::Formatter for PythonCompactFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Flambeau's custom `tojson` filter: uses `PythonCompactFormatter` so
/// the output is byte-identical to llama.cpp's `tojson` on the same
/// data. Shadows minijinja's stock `tojson` (which emits compact
/// no-space JSON). Preserves field order for `serde_json::Value` inputs
/// via the workspace's `preserve_order` feature.
fn tojson_python_style(
    value: &MjValue,
    indent: Option<MjValue>,
    args: Kwargs,
) -> Result<MjValue, Error> {
    let indent = match indent {
        Some(v) => Some(v),
        None => args.get("indent")?,
    };
    args.assert_all_used()?;
    let mut out = Vec::<u8>::new();
    let render_err = |err: serde_json::Error| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    };
    if let Some(indent) = indent {
        // Pretty-print path — matches minijinja's stock behaviour when an
        // `indent=N` is passed. Qwen3.6's template doesn't use it today
        // but keep API parity with the stock filter.
        let indent_n: usize = match bool::try_from(indent.clone()).ok() {
            Some(true) => 2,
            Some(false) => 0,
            None => usize::try_from(indent)
                .map_err(|_| Error::new(ErrorKind::InvalidOperation, "invalid indent"))?,
        };
        let spaces = " ".repeat(indent_n);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(spaces.as_bytes());
        let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
        serde::Serialize::serialize(value, &mut ser).map_err(render_err)?;
    } else {
        let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonCompactFormatter);
        serde::Serialize::serialize(value, &mut ser).map_err(render_err)?;
    }
    // SAFETY: serde_json always emits valid UTF-8.
    let s = unsafe { String::from_utf8_unchecked(out) };
    Ok(MjValue::from_safe_string(s))
}

/// One entry in an OpenAI-style `messages` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// "system" / "user" / "assistant" / "tool".
    pub role: String,
    /// Message text. Qwen's template handles strings; multi-modal content
    /// arrays (vision / video) are V2.
    pub content: String,
}

/// Rendered chat template: the raw text prompt to feed the tokenizer.
pub struct ChatTemplate {
    env: Environment<'static>,
    template_name: &'static str,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("template_name", &self.template_name)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Load `tokenizer.chat_template` from a GGUF and pre-compile it.
    /// # Errors
    /// - `anyhow` wrapping "tokenizer.chat_template missing" if the GGUF
    /// lacks the metadata key.
    /// - Any error [`from_string`] can return (minijinja parse failure).
    pub fn load_from_gguf(file: &GgufFile) -> Result<Self> {
        let tpl_str = file
            .metadata_str("tokenizer.chat_template")
            .context("tokenizer.chat_template missing from GGUF metadata")?
            .to_owned();
        Self::from_string(tpl_str)
    }

    /// Pre-compile a raw template string. Accepts owned String so we can
    /// stash it in the environment for the `'static` lifetime.
    /// # Errors
    /// `anyhow` wrapping `minijinja::Error` if the template fails to parse.
    pub fn from_string(tpl_str: String) -> Result<Self> {
        let mut env = Environment::new();
        // Install Python-compat string/list methods. Qwen's template uses
        // `.startswith`, `.split`, `.endswith`, etc. — minijinja proper
        // doesn't ship these; `minijinja_contrib::pycompat` does.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Shadow minijinja's built-in `tojson` with one that matches
        // llama.cpp's (nlohmann::json) output byte-for-byte: field order
        // preserved, `, ` / `: ` separators. See T1.3 parity cert.
        env.add_filter("tojson", tojson_python_style);
        // Leak the template string — it lives for the rest of the process
        // and Jinja needs a `'static` reference. One per model load, fine.
        let leaked: &'static str = Box::leak(tpl_str.into_boxed_str());
        env.add_template("chat", leaked)
            .map_err(|e| anyhow!("minijinja add_template: {e}"))?;
        Ok(Self {
            env,
            template_name: "chat",
        })
    }

    /// Render messages into the model prompt. `add_generation_prompt=true`
    /// appends the assistant header so the model completes into a new turn.
    /// Generic over the message type so callers (e.g. the HTTP server)
    /// can pass their request-body type directly rather than cloning into
    /// [`ChatMessage`] — see C5 in `RUST-PERF-CORRECTIONS.md`. The only
    /// requirement is that `M` serialises into something the template's
    /// `messages` binding understands (role + content fields, matching
    /// the OpenAI chat-completions schema).
    /// Short-hand for [`render_with_tools`] with no tools.
    /// # Errors
    /// `anyhow` wrapping a `minijinja::Error` if the template references an
    /// undefined variable, applies an unknown filter/method, or raises from
    /// inside a `{% raise %}` block.
    pub fn render<M: serde::Serialize>(
        &self,
        messages: &[M],
        add_generation_prompt: bool,
    ) -> Result<String> {
        // `enable_thinking=false` preserves the pre-2 behaviour (the
        // server's non-thinking default) for callers that don't care.
        self.render_with_tools::<M, ()>(messages, None, add_generation_prompt, Some(false))
    }

    /// Render messages + a tool list. Qwen3.5/3.6's GGUF-embedded Jinja
    /// template inspects `tools` (when non-empty) and renders tool
    /// definitions into the system prompt using Hermes-style formatting.
    /// Pass `None` or an empty slice for tool-free turns — identical to
    /// calling [`render`].
    /// `enable_thinking`:
    /// - `Some(false)` — bind `enable_thinking=false` in Jinja context.
    /// Qwen3.6's template renders a closed `<think>\n\n</think>\n\n`
    /// after the generation-prompt assistant header, keeping the
    /// model in non-thinking mode (Qwen's recommended stable-agent
    /// default).
    /// - `Some(true)` — bind `enable_thinking=true`. The template opens
    /// a `<think>\n` block for reasoning generation.
    /// - `None` — leave the variable unbound. The template's
    /// `enable_thinking is defined` check is false, which takes the
    /// open-`<think>` branch. This is what llama.cpp's
    /// `test-chat-template` does when no `enable_thinking` is passed
    /// in the JSON input, and is the behaviour this matches byte-for-byte.
    ///
    /// # Errors
    /// Same as [`render`].
    pub fn render_with_tools<M: serde::Serialize, T: serde::Serialize>(
        &self,
        messages: &[M],
        tools: Option<&[T]>,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let tpl = self
            .env
            .get_template(self.template_name)
            .map_err(|e| anyhow!("minijinja get_template: {e}"))?;
        // Template branches on `tools` being truthy. Empty-list sentinel
        // → template's `{% if tools %}` takes no-tools branch (byte-exact
        // with the pre-2 hard-coded behaviour).
        let tools_value = match tools {
            Some(t) if !t.is_empty() => MjValue::from_serialize(t),
            _ => MjValue::from_serialize(Vec::<()>::new()),
        };
        // Qwen3.6's chat template calls `tool_call.arguments | tojson`
        // inside a `{% if tool_call.arguments is mapping %}` guard. Our
        // wire contract (and OpenAI's) keeps `arguments` as a JSON-encoded
        // string, not an object. If we pass the string through as-is the
        // `is mapping` test fails and no `<parameter=...>` block is
        // emitted — a silent drop of parameters in prior-turn tool calls.
        // llama.cpp's `common_chat_msgs_parse_oaicompat` fixes this by
        // parsing `arguments` string → object before handing off to the
        // template. We mirror it: build an MjValue for the messages
        // through `normalise_messages_for_template`, which walks any
        // `tool_calls[].function.arguments` string and replaces it with
        // its parsed JSON value (if parsable).
        let messages_value = normalise_messages_for_template(messages)?;
        let mut ctx = std::collections::BTreeMap::<&str, MjValue>::new();
        ctx.insert("messages", messages_value);
        ctx.insert("tools", tools_value);
        ctx.insert(
            "add_generation_prompt",
            MjValue::from_serialize(add_generation_prompt),
        );
        if let Some(thinking) = enable_thinking {
            ctx.insert("enable_thinking", MjValue::from_serialize(thinking));
        }
        let rendered = tpl
            .render(MjValue::from_serialize(&ctx))
            .map_err(|e| anyhow!("minijinja render: {e}"))?;
        Ok(rendered)
    }
}

/// Walk messages and convert `tool_calls[].function.arguments` from
/// its OpenAI-wire string form (JSON-encoded) into a parsed object when
/// the template will need to iterate it. Non-JSON-parsable argument
/// strings are left as-is (the template's `is mapping` test then drops
/// them, same as llama.cpp, which is the least-surprising behaviour).
/// The conversion is serialise-roundtrip (message → JSON value → mutate
/// → MjValue) rather than in-place; this is the cheap, obvious
/// implementation and runs once per request, not per token.
fn normalise_messages_for_template<M: serde::Serialize>(messages: &[M]) -> Result<MjValue> {
    // Serialise to a mutable JSON array so we can patch the
    // `arguments` field. `serde_json::Value` preserves field order with
    // the workspace's `preserve_order` feature, so we don't reshuffle
    // anything the template sees.
    let mut j: serde_json::Value = serde_json::to_value(messages).context("serialise messages")?;
    if let Some(arr) = j.as_array_mut() {
        for msg in arr.iter_mut() {
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
                continue;
            };
            for tc in tool_calls.iter_mut() {
                let Some(func) = tc.get_mut("function").and_then(|v| v.as_object_mut()) else {
                    continue;
                };
                if let Some(args) = func.get_mut("arguments") {
                    if let Some(s) = args.as_str() {
                        // Try to parse the JSON string. If it fails
                        // (model emitted junk, or a non-JSON blob),
                        // leave it as a string — the template's
                        // `is mapping` check then silently omits params,
                        // matching llama.cpp's behaviour.
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                            *args = parsed;
                        }
                    }
                }
            }
        }
    }
    Ok(MjValue::from_serialize(j))
}
