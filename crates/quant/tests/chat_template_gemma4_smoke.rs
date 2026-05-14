//! Gemma 4 chat-template smoke. Loads the GGUF-embedded Jinja
//! template, renders a system+user turn, asserts the expected
//! `<|turn>...<turn|>` markup is present and `add_generation_prompt`
//! ends with an open assistant turn. Skipped when the GGUF is absent.

use flambeau_quant::{ChatMessage, ChatTemplate, GgufFile};

const MODELS: &[&str] = &[
    "/artefact/models/gemma-4-E4B-it-Q4_0.gguf",
    "/artefact/models/gemma-4-31B-it-Q4_0.gguf",
    "/artefact/models/gemma-4-26B-A4B-it-Q8_0.gguf",
];

fn first_present() -> Option<std::path::PathBuf> {
    MODELS
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
}

#[test]
fn render_with_tools_emits_tool_block() {
    let Some(path) = first_present() else {
        eprintln!("skipping — no gemma4 GGUF found in /artefact/models");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tpl = ChatTemplate::load_from_gguf(&gguf).expect("load chat template");

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "What's the weather in Paris?".into(),
    }];
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    })];
    let rendered = tpl
        .render_with_tools::<ChatMessage, _>(&messages, Some(&tools), true, Some(false))
        .expect("render with tools");

    eprintln!("=== rendered ({} bytes) ===\n{rendered}\n=========================", rendered.len());
    assert!(rendered.contains("<|tool>"), "missing tool open marker");
    assert!(rendered.contains("<tool|>"), "missing tool close marker");
    assert!(
        rendered.contains("declaration:get_weather"),
        "missing tool declaration"
    );
    assert!(
        rendered.contains("location"),
        "missing parameter name"
    );
}

#[test]
fn render_assistant_tool_call_then_tool_response() {
    let Some(path) = first_present() else {
        eprintln!("skipping — no gemma4 GGUF found in /artefact/models");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tpl = ChatTemplate::load_from_gguf(&gguf).expect("load chat template");

    // Mixed history: user → assistant-with-tool_call → tool-response → user.
    let messages = vec![
        serde_json::json!({"role": "user", "content": "Weather in Paris?"}),
        serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
                }
            }]
        }),
        serde_json::json!({
            "role": "tool",
            "name": "get_weather",
            "content": "{\"temp\": 18, \"conditions\": \"sunny\"}"
        }),
        serde_json::json!({"role": "user", "content": "Thanks!"}),
    ];
    let tools: Vec<serde_json::Value> = vec![];
    let rendered = tpl
        .render_with_tools::<serde_json::Value, serde_json::Value>(
            &messages,
            Some(&tools),
            true,
            Some(false),
        )
        .expect("render mixed history");

    eprintln!("=== rendered ({} bytes) ===\n{rendered}\n=========================", rendered.len());
    assert!(
        rendered.contains("<|tool_call>call:get_weather"),
        "missing tool_call marker for emitted call"
    );
}

#[test]
fn render_system_user_assistant_turn() {
    let Some(path) = first_present() else {
        eprintln!("skipping — no gemma4 GGUF found in /artefact/models");
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

    eprintln!("=== rendered ({} bytes) ===\n{rendered}\n=========================", rendered.len());
    assert!(!rendered.is_empty(), "empty render");
    assert!(
        rendered.contains("<|turn>system"),
        "missing system turn header: {rendered:?}"
    );
    assert!(
        rendered.contains("You are a helpful assistant."),
        "system content dropped"
    );
    assert!(
        rendered.contains("<|turn>user"),
        "missing user turn header"
    );
    assert!(
        rendered.contains("Hello"),
        "user content dropped"
    );
    assert!(
        rendered.trim_end().ends_with("<|turn>model"),
        "expected open assistant turn at end, got: {rendered:?}"
    );
}
