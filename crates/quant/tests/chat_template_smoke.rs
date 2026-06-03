//! chat-template smoke + structural parity.
//! llama.cpp's apply_chat_template on Qwen3.6 produces a string with
//! `<|im_start|>{role}\n{content}<|im_end|>\n` turns. We check that our
//! minijinja render:
//! 1. Produces non-empty output.
//! 2. Contains the expected role/content markup for each message.
//! 3. With `add_generation_prompt=true`, ends with an open assistant turn
//!    (`<|im_start|>assistant\n`) — so the model decodes into it.
//!
//! Full llama.cpp byte-exact parity is a larger follow-up (requires
//! extracting the exact whitespace + macro-rendering llama.cpp does; the
//! Qwen template has conditional branches for tools/images/thinking that
//! we only exercise in the "happy path" here).

use flambeau_quant::{ChatMessage, ChatTemplate, GgufFile};

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn render_system_user_assistant_turn() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping chat-template smoke");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tpl = ChatTemplate::load_from_gguf(&gguf).expect("load chat template");

    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a helpful assistant.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        },
    ];
    let rendered = tpl
        .render(&messages, /*add_generation_prompt=*/ true)
        .expect("render");

    eprintln!("=== rendered ===\n{rendered}\n================");
    assert!(!rendered.is_empty(), "empty render");
    assert!(
        rendered.contains("<|im_start|>system"),
        "missing system header: {rendered:?}"
    );
    assert!(
        rendered.contains("You are a helpful assistant."),
        "system content dropped: {rendered:?}"
    );
    assert!(
        rendered.contains("<|im_start|>user"),
        "missing user header: {rendered:?}"
    );
    assert!(
        rendered.contains("Hello"),
        "user content dropped: {rendered:?}"
    );
    // Generation prompt opens an assistant turn. Qwen3 emits an assistant
    // header + an empty <think>...</think> marker when enable_thinking=false
    // (the non-thinking variant of its chat template). Either is valid open.
    assert!(
        rendered.contains("<|im_start|>assistant"),
        "assistant header missing: {rendered:?}"
    );
    // No closing `<|im_end|>` on the final turn (model completes into it).
    assert!(
        !rendered.trim_end().ends_with("<|im_end|>"),
        "final turn unexpectedly closed: {rendered:?}"
    );
}

#[test]
fn render_without_generation_prompt_closes_last_turn() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tpl = ChatTemplate::load_from_gguf(&gguf).expect("load chat template");

    let messages = vec![
        ChatMessage {
            role: "user".into(),
            content: "Ping".into(),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Pong".into(),
        },
    ];
    let rendered = tpl
        .render(&messages, /*add_generation_prompt=*/ false)
        .expect("render");

    assert!(rendered.contains("Pong"), "assistant content dropped");
    // No open assistant turn appended when generation_prompt=false.
    assert!(
        !rendered.trim_end().ends_with("<|im_start|>assistant"),
        "unexpected open assistant turn: {rendered:?}"
    );
}

/// T1.2: Qwen3.6 `<tool_call>` / `</tool_call>` are textual tags, not
/// single vocab tokens — they tokenise as normal BPE sequences and must
/// round-trip cleanly through encode/decode. If the vocab-scan in
/// `tokenizer::load_from_gguf` ever auto-registers them as specials by
/// mistake, this test surfaces it.
#[test]
fn tool_call_tags_roundtrip_through_tokenizer() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tok = flambeau_quant::load_from_gguf(&gguf).expect("load tokenizer");

    let fixtures = [
        "<tool_call>",
        "</tool_call>",
        "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":\"{}\"}\n</tool_call>",
        "prefix text <tool_call>{\"name\":\"f\",\"arguments\":\"{}\"}</tool_call> suffix",
        // Qwen3.6 can emit think + tool_call together (llama.cpp #21118
        // guards the newline-less variant here):
        "<think>reasoning</think><tool_call>{\"name\":\"f\",\"arguments\":\"{}\"}</tool_call>",
    ];
    for f in fixtures {
        let ids = tok.encode(f).expect("encode");
        let back = tok.decode(&ids).expect("decode");
        assert_eq!(
            back,
            f,
            "tool-call tag round-trip failed: original={f:?}, decoded={back:?}, n_ids={}",
            ids.len()
        );
    }
}

/// T1.2: `render_with_tools` threads a non-empty `tools[]` through the
/// Jinja context. Uses a hand-rolled minimal template so the test does
/// not need a GGUF file.
#[test]
fn render_with_tools_passes_into_context() {
    use serde_json::json;
    let tpl_src = r#"
{%- for m in messages -%}
[{{ m.role }}] {{ m.content }}
{% endfor -%}
{%- if tools -%}
TOOLS: {% for t in tools %}{{ t.function.name }} {% endfor %}
{%- else -%}
NO_TOOLS
{%- endif -%}
"#;
    let tpl = ChatTemplate::from_string(tpl_src.to_owned()).expect("parse template");
    let msgs = vec![ChatMessage {
        role: "user".into(),
        content: "hi".into(),
    }];
    // Empty tools list → template takes the no-tools branch.
    let r_none = tpl
        .render_with_tools::<_, serde_json::Value>(&msgs, None, false, None)
        .expect("render no tools");
    assert!(r_none.contains("NO_TOOLS"), "no-tools branch: {r_none:?}");

    // Non-empty list → template sees the list.
    let tools = vec![json!({"type":"function","function":{"name":"get_weather"}})];
    let r_some = tpl
        .render_with_tools(&msgs, Some(&tools), false, None)
        .expect("render with tools");
    assert!(
        r_some.contains("TOOLS: get_weather"),
        "tools branch: {r_some:?}"
    );

    // Back-compat: render() delegates to render_with_tools with None, so
    // the pre-2 empty-tools behaviour is preserved byte-for-byte.
    let r_compat = tpl.render(&msgs, false).expect("render compat");
    assert_eq!(r_compat, r_none);
}

#[test]
fn can_feed_render_into_tokenizer() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tpl = ChatTemplate::load_from_gguf(&gguf).expect("load chat template");
    let tok = flambeau_quant::load_from_gguf(&gguf).expect("load tokenizer");

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "Hello".into(),
    }];
    let rendered = tpl.render(&messages, true).expect("render");
    let ids = tok.encode(&rendered).expect("encode");
    eprintln!("rendered {} bytes → {} tokens", rendered.len(), ids.len());
    assert!(!ids.is_empty());
    // Round-trip — decode should come back to the rendered text (no lossy
    // special-token handling).
    let back = tok.decode(&ids).expect("decode");
    assert_eq!(back, rendered, "tokenize-detokenize not round-trip");
}
