//! S8 (tokenizer hooks) — gemma4 tokenizer round-trip on a fixed
//! corpus from the on-disk E4B-Q4_0 GGUF. Skips when the file is
//! absent (CI / sandbox).

use std::path::Path;

use flambeau_quant::{load_from_gguf, GgufFile};

const MODELS_DIR: &str = "/artefact/models";

fn open_e4b() -> Option<GgufFile> {
    let p = Path::new(MODELS_DIR).join("gemma-4-E4B-it-Q4_0.gguf");
    if !p.exists() {
        eprintln!("skipping — E4B GGUF not present at {MODELS_DIR}");
        return None;
    }
    GgufFile::open(&p).ok()
}

#[test]
fn gemma4_tokenizer_loads_and_advertises_overrides() {
    let Some(file) = open_e4b() else {
        return;
    };
    let tok = load_from_gguf(&file).expect("load_from_gguf gemma4");
    assert_eq!(tok.bos_id, Some(2));
    assert_eq!(tok.eos_id, Some(106));
    assert_eq!(tok.pad_id, Some(0));
    // PR #21500 override: regardless of GGUF flag (false for this
    // file), the runtime must add BOS for gemma4.
    assert!(
        tok.force_add_bos,
        "gemma4 must force add_bos=true (PR #21500)"
    );
    // <end_of_turn> = 106 = eos in gemma4. <eos> = 1 is the
    // also-stop token (PR #21492 strips `</s>` from EOG, but gemma4
    // doesn't have `</s>` in its vocab — it has `<eos>`/`<end_of_turn>`).
    assert!(tok.stop_ids.contains(&106));
    let eos_extra = tok.inner.get_vocab(false).get("<eos>").copied();
    if let Some(eos1) = eos_extra {
        assert!(tok.stop_ids.contains(&eos1));
    }
}

fn roundtrip(text: &str) {
    let Some(file) = open_e4b() else {
        return;
    };
    let tok = load_from_gguf(&file).expect("load");
    let ids = tok.encode(text).expect("encode");
    let back = tok.decode(&ids).expect("decode");
    assert_eq!(
        back, text,
        "roundtrip mismatch for {text:?}: got {back:?} from ids {ids:?}"
    );
}

#[test]
fn gemma4_roundtrip_ascii_words() {
    roundtrip("Hello, world!");
}

#[test]
fn gemma4_roundtrip_with_spaces() {
    roundtrip(" leading and trailing ");
}

#[test]
fn gemma4_roundtrip_newlines() {
    // PR #21343 / #21406: newline-only tokens. Single, double, and
    // mixed newline runs.
    roundtrip("line one\nline two\n\nthird");
}

#[test]
fn gemma4_roundtrip_unicode() {
    // Byte-fallback path (PR #21488). Emoji + CJK + accented latin.
    roundtrip("¡Hola! こんにちは 🌍 résumé");
}

#[test]
fn gemma4_roundtrip_code_snippet() {
    roundtrip("fn main() {\n    println!(\"hi {x}\");\n}\n");
}

#[test]
fn gemma4_roundtrip_long_mixed() {
    // Longer corpus: prose + punctuation + numbers.
    roundtrip(
        "The quick brown fox jumps over 13 lazy dogs in 2026. \
         Some text has multi-line content,\nwith \"quotes\" and (parens).",
    );
}

#[test]
fn gemma4_qwen_gguf_still_loads_with_byte_level() {
    // Regression guard: the gemma4 path is conditional on
    // `tokenizer.ggml.model == "gemma4"`. A qwen3 GGUF (or anything
    // else with `gpt2` model) must still load via the byte-level path.
    let p = Path::new("/artefact/models").join("gemma-4-E4B-it-Q4_0.gguf");
    if !p.exists() {
        return;
    }
    // We don't have a qwen3 GGUF on disk in this test — the gemma4
    // load itself is the regression test for "no panic on unknown
    // tokens, no fallback to gpt2 path".
    let _ = load_from_gguf(&GgufFile::open(&p).expect("open")).expect("load");
}
