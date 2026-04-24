//! T2.4 — Fixture-driven regression corpus for `QwenCoderXmlParser`.
//!
//! Walks `crates/server/tests/tool_call_fixtures/*.txt` and, for each
//! fixture, asserts that the coalesced event stream the parser produces
//! matches the committed `*.expected.jsonl`. Runs the input twice —
//! once as one `push()` and once char-by-char — and requires
//! **identical coalesced output** from both. Streaming equivalence is
//! the load-bearing invariant; this is where we catch it if we break
//! it.
//!
//! New bugs land as new fixture pairs — see
//! `crates/server/tests/tool_call_fixtures/README.md`.

use std::fs;
use std::path::{Path, PathBuf};

use flambeau_server::tool_call_parser::qwen3_coder::QwenCoderXmlParser;
use flambeau_server::tool_call_parser::{ParserEvent, ToolCallParser};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tool_call_fixtures")
}

fn parse_expected(path: &Path) -> Vec<ParserEvent> {
    let body = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("parse {path:?} line {line:?}: {e}"))
        })
        .collect()
}

fn run_one_shot(input: &str) -> Vec<ParserEvent> {
    let mut p = QwenCoderXmlParser::new();
    let mut evts = p.push(input);
    evts.extend(p.finish());
    ParserEvent::coalesce(evts)
}

fn run_char_by_char(input: &str) -> Vec<ParserEvent> {
    let mut p = QwenCoderXmlParser::new();
    let mut out = Vec::new();
    let mut scratch = [0u8; 4];
    for ch in input.chars() {
        let slice = ch.encode_utf8(&mut scratch);
        out.extend(p.push(slice));
    }
    out.extend(p.finish());
    ParserEvent::coalesce(out)
}

#[test]
fn fixtures_match_committed_expectations() {
    let dir = fixtures_dir();
    let mut fixtures: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("open {dir:?}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("txt"))
        .collect();
    fixtures.sort();
    assert!(
        !fixtures.is_empty(),
        "no .txt fixtures found under {dir:?}"
    );

    let mut failed: Vec<String> = Vec::new();
    for fixture in &fixtures {
        let name = fixture
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        let input = fs::read_to_string(fixture)
            .unwrap_or_else(|e| panic!("read {fixture:?}: {e}"));
        let expected_path = fixture.with_extension("expected.jsonl");
        if !expected_path.exists() {
            failed.push(format!(
                "{name}: missing {expected_path:?} (run with --nocapture and commit the \
                 observed output if it's correct — see README.md)"
            ));
            continue;
        }
        let expected = parse_expected(&expected_path);

        let one_shot = run_one_shot(&input);
        if one_shot != expected {
            failed.push(format!(
                "{name} one_shot mismatch:\n  expected: {expected:#?}\n  got:      {one_shot:#?}"
            ));
            continue;
        }

        let chunked = run_char_by_char(&input);
        if chunked != one_shot {
            failed.push(format!(
                "{name} streaming-vs-one-shot mismatch:\n  one_shot: {one_shot:#?}\n  chunked:  {chunked:#?}"
            ));
            continue;
        }

        eprintln!("ok   {name}  ({} events)", one_shot.len());
    }

    if !failed.is_empty() {
        panic!(
            "{} / {} fixtures failed:\n\n{}",
            failed.len(),
            fixtures.len(),
            failed.join("\n\n---\n\n")
        );
    }
}
