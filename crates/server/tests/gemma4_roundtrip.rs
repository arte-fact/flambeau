//! End-to-end gemma4 chat-template render → parse round-trip.
//!
//! 1. Loads gemma4's GGUF-embedded Jinja chat template via
//!    `flambeau_quant::ChatTemplate`.
//! 2. Renders a conversation containing an assistant turn with one
//!    tool call (matching the wire format the model would emit at
//!    decode time).
//! 3. Feeds the rendered assistant turn through
//!    `flambeau_server::tool_call_parser::gemma4::Gemma4ToolCallParser`
//!    and asserts the call name + arguments are recovered.
//!
//! Skipped when no gemma4 GGUF is present in `/artefact/models`.

use flambeau_quant::{ChatTemplate, GgufFile};
use flambeau_server::tool_call_parser::{
    gemma4::Gemma4ToolCallParser, ParserEvent, ToolCallParser,
};

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
fn render_then_parse_recovers_tool_call() {
    let Some(path) = first_present() else {
        eprintln!("skipping — no gemma4 GGUF in /artefact/models");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tpl = ChatTemplate::load_from_gguf(&gguf).expect("load chat template");

    // Build a history with the assistant emitting a tool call. The
    // chat template renders the call into the gemma4 wire format
    // (`<|tool_call>call:NAME{...}<tool_call|>`).
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
    ];
    let tools: Vec<serde_json::Value> = vec![];
    let rendered = tpl
        .render_with_tools::<serde_json::Value, serde_json::Value>(
            &messages,
            Some(&tools),
            /*add_generation_prompt=*/ false,
            /*enable_thinking=*/ Some(false),
        )
        .expect("render");

    // Locate the assistant turn (everything between `<|turn>model\n`
    // and the next `<turn|>`). The parser is what runs over the
    // model's *output* during decode — i.e. the same byte sequence the
    // template emitted for the prior assistant turn.
    let assistant_open = "<|turn>model\n";
    let pos = rendered
        .find(assistant_open)
        .expect("missing model turn header");
    let after_header = &rendered[pos + assistant_open.len()..];
    let close_pos = after_header.find("<turn|>").expect("missing turn close");
    let assistant_body = &after_header[..close_pos];
    eprintln!(
        "=== assistant body ({} bytes) ===\n{}\n================",
        assistant_body.len(),
        assistant_body
    );

    // Feed through the parser.
    let mut parser = Gemma4ToolCallParser::new();
    let mut events = parser.push(assistant_body);
    events.extend(parser.finish());
    let events = ParserEvent::coalesce(events);
    eprintln!("=== parsed events ===\n{events:#?}");

    let name = events
        .iter()
        .find_map(|e| match e {
            ParserEvent::ToolCallOpen { name, .. } => Some(name.clone()),
            _ => None,
        })
        .expect("no ToolCallOpen emitted");
    assert_eq!(name, "get_weather", "unexpected call name");

    let args = events
        .iter()
        .find_map(|e| match e {
            ParserEvent::ToolCallArgumentsDelta { arguments, .. } => Some(arguments.clone()),
            _ => None,
        })
        .expect("no ToolCallArgumentsDelta emitted");
    let parsed: serde_json::Value = serde_json::from_str(&args).expect("args JSON");
    let obj = parsed.as_object().expect("args object");
    assert_eq!(
        obj.get("location"),
        Some(&serde_json::Value::String("Paris".into())),
        "location lost"
    );
    assert_eq!(
        obj.get("unit"),
        Some(&serde_json::Value::String("celsius".into())),
        "unit lost"
    );

    assert!(
        events
            .iter()
            .any(|e| matches!(e, ParserEvent::ToolCallClose { index: 0 })),
        "no ToolCallClose(0)"
    );
}
