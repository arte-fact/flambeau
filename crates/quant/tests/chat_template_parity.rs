//! T1.3 — Chat-template parity cert vs `llama.cpp --jinja`.
//!
//! For a fixed set of (messages, tools, add_generation_prompt) inputs,
//! the prompt string flambeau's Jinja render produces must be
//! byte-identical to `llama.cpp`'s `test-chat-template` render on the
//! same template extracted from the same GGUF.
//!
//! Both implementations share `minijinja`-family semantics but differ in
//! edge cases (pycompat method coverage, whitespace control flags,
//! default filter behaviour) — byte-level parity is the real gate.
//!
//! Activated by two environment variables:
//!   FLAMBEAU_QWEN3_GGUF                    → path to a Qwen3.6 GGUF
//!   FLAMBEAU_LLAMACPP_TEST_CHAT_TEMPLATE   → path to llama.cpp's test binary
//!
//! With either unset the test prints a skip message and passes — the cert
//! machinery still ships, but isn't a hard gate on environments that
//! don't have llama.cpp built.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use flambeau_quant::{ChatTemplate, GgufFile};

fn env_path(name: &str) -> Option<PathBuf> {
    let p = std::env::var(name).ok().map(PathBuf::from)?;
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn fixtures_dir() -> PathBuf {
    // `CARGO_MANIFEST_DIR` points at `crates/quant`; workspace root is two
    // levels up.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("certs/chat_template/qwen35moe_tools/fixtures")
}

/// Minimal representation of the fixture JSON schema (shared with
/// llama.cpp's `test-chat-template`). We deserialise only what flambeau
/// needs to render; `bos_token`/`eos_token` pass through but aren't used
/// by Qwen3.6's template (no `{{ bos_token }}` reference).
#[derive(serde::Deserialize)]
struct Fixture {
    messages: Vec<serde_json::Value>,
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    #[serde(default = "default_add_gen_prompt")]
    add_generation_prompt: bool,
}

fn default_add_gen_prompt() -> bool {
    true
}

#[test]
fn chat_template_byte_parity_vs_llamacpp() {
    let Some(gguf_path) = env_path("FLAMBEAU_QWEN3_GGUF") else {
        eprintln!(
            "FLAMBEAU_QWEN3_GGUF unset — skipping. See \
             certs/chat_template/qwen35moe_tools/README.md"
        );
        return;
    };
    let Some(test_chat_bin) = env_path("FLAMBEAU_LLAMACPP_TEST_CHAT_TEMPLATE") else {
        eprintln!(
            "FLAMBEAU_LLAMACPP_TEST_CHAT_TEMPLATE unset — skipping. See \
             certs/chat_template/qwen35moe_tools/README.md"
        );
        return;
    };

    // Extract the GGUF's Jinja template to a tmp file so llama.cpp can
    // consume it. flambeau loads it directly via GgufFile::metadata_str.
    let gguf = GgufFile::open(&gguf_path).expect("open GGUF");
    let tpl_src = gguf
        .metadata_str("tokenizer.chat_template")
        .expect("tokenizer.chat_template missing from GGUF")
        .to_owned();
    let tmpdir = tempdir();
    let tpl_path = tmpdir.join("template.jinja");
    fs::write(&tpl_path, &tpl_src).expect("write tmp template");

    let tpl = ChatTemplate::from_string(tpl_src).expect("parse template");

    let fixtures = fixtures_dir();
    let mut entries: Vec<PathBuf> = fs::read_dir(&fixtures)
        .expect("open fixtures/")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no fixtures found in {fixtures:?}");

    let mut failures: Vec<String> = Vec::new();
    for fixture_path in &entries {
        let name = fixture_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        let body = fs::read_to_string(fixture_path).expect("read fixture");
        let fixture: Fixture = serde_json::from_str(&body).unwrap_or_else(|e| {
            panic!("parse {name}: {e}")
        });

        // Reference render via llama.cpp.
        let ref_out = tmpdir.join(format!("{name}.expected.txt"));
        let status = Command::new(&test_chat_bin)
            .arg(&tpl_path)
            .args(["--json", fixture_path.to_str().unwrap()])
            .args(["--output", ref_out.to_str().unwrap()])
            .status()
            .expect("spawn test-chat-template");
        assert!(status.success(), "test-chat-template failed on {name}");
        let expected = fs::read_to_string(&ref_out).expect("read expected");

        // Flambeau render. Fixtures don't set `enable_thinking`, so we
        // pass `None` — matches llama.cpp's behaviour when the flag is
        // absent from the JSON input (template takes the open-`<think>`
        // branch via the `is defined` check).
        let got = tpl
            .render_with_tools(
                &fixture.messages,
                if fixture.tools.is_empty() {
                    None
                } else {
                    Some(fixture.tools.as_slice())
                },
                fixture.add_generation_prompt,
                None,
            )
            .unwrap_or_else(|e| panic!("flambeau render {name}: {e}"));

        if got != expected {
            failures.push(diff_report(name, &expected, &got));
        } else {
            eprintln!("ok   {name}");
        }
    }

    if !failures.is_empty() {
        panic!(
            "chat-template parity failed on {}/{} fixtures:\n\n{}",
            failures.len(),
            entries.len(),
            failures.join("\n\n---\n\n")
        );
    }
}

fn diff_report(name: &str, expected: &str, got: &str) -> String {
    // Line-by-line diff, with a byte-count and first-divergence index
    // for quick triage.
    let mut out = format!(
        "FAIL {name}  (expected {} bytes, got {} bytes)\n",
        expected.len(),
        got.len()
    );
    let first_diff = expected
        .bytes()
        .zip(got.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| expected.len().min(got.len()));
    out.push_str(&format!("first byte diff @ {first_diff}\n"));
    let ctx = 40usize;
    let lo = first_diff.saturating_sub(ctx);
    let hi_e = (first_diff + ctx).min(expected.len());
    let hi_g = (first_diff + ctx).min(got.len());
    out.push_str(&format!(
        "  expected: …{:?}…\n",
        &expected[lo..hi_e]
    ));
    out.push_str(&format!("  got:      …{:?}…\n", &got[lo..hi_g]));
    out
}

/// Minimum-viable tmpdir that persists for the test run. Uses
/// `env::temp_dir()` + a unique suffix; cleanup on process exit is OS
/// responsibility — we don't pull `tempfile` just for this test.
fn tempdir() -> PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let path = base.join(format!("flambeau-chat-template-parity-{pid}-{nanos}"));
    fs::create_dir_all(&path).expect("create tmpdir");
    path
}
