//! Hermes-style `<tool_call>\n{JSON}\n</tool_call>` parser.
//! **T2.1 skeleton**: this module ships with a passthrough impl that
//! treats the full decoder stream as free text (single `TextDelta`
//! events). The real state machine lands in T2.2 (happy-path Hermes
//! JSON) + T2.3 (buffer-before-emit + `<think>` interaction).

use super::{ParserEvent, ToolCallParser};

/// Hermes parser. Currently a passthrough (T2.1 skeleton).
#[derive(Debug, Default)]
pub struct HermesJsonParser {
    // T2.2 will replace this with the `{Text, MaybeThinkOpen, InThink,
    // MaybeToolOpen, InToolBody, MaybeToolClose}` state machine + its
    // buffered-prefix scratch.
    _placeholder: (),
}

impl HermesJsonParser {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ToolCallParser for HermesJsonParser {
    fn push(&mut self, chunk: &str) -> Vec<ParserEvent> {
        if chunk.is_empty() {
            Vec::new()
        } else {
            vec![ParserEvent::TextDelta(chunk.to_owned())]
        }
    }

    fn finish(&mut self) -> Vec<ParserEvent> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_emits_text_chunks() {
        let mut p = HermesJsonParser::new();
        assert_eq!(
            p.push("hello "),
            vec![ParserEvent::TextDelta("hello ".into())]
        );
        assert_eq!(
            p.push("world"),
            vec![ParserEvent::TextDelta("world".into())]
        );
        assert!(p.finish().is_empty());
    }

    #[test]
    fn empty_push_is_noop() {
        let mut p = HermesJsonParser::new();
        assert!(p.push("").is_empty());
    }
}
