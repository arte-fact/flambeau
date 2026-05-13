//! tokenizer parity — encode matches llama.cpp on Qwen3.6-35B.
//! Seed token "Hello" (token 9419) is what parity cert uses; if our
//! tokenizer produces a different id, the whole decode-parity chain breaks.
//! Env-gated on `FLAMBEAU_QWEN3_GGUF` (full Qwen3.6-35B GGUF).

use flambeau_quant::{load_from_gguf, GgufFile};

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var("FLAMBEAU_QWEN3_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn encode_hello_matches_llama_cpp() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping tokenizer parity");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tok = load_from_gguf(&gguf).expect("load tokenizer");
    eprintln!(
        "loaded tokenizer: vocab_size={} bos={:?} eos={:?} pad={:?}",
        tok.vocab_size, tok.bos_id, tok.eos_id, tok.pad_id
    );
    // Reference from llama.cpp's tokenization of "Hello" (no BOS) on Qwen3.6-UD-Q4_K_S:
    // token 9419 = "Hello" (single-token).
    let ids = tok.encode("Hello").expect("encode");
    eprintln!("encode('Hello') = {ids:?}");
    assert_eq!(
        ids,
        vec![9419],
        "tokenizer diverges from llama.cpp reference for 'Hello'"
    );
}

#[test]
fn roundtrip_decode_encode_hello() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tok = load_from_gguf(&gguf).expect("load tokenizer");
    let decoded = tok.decode(&[9419]).expect("decode");
    eprintln!("decode([9419]) = {decoded:?}");
    assert_eq!(decoded.trim(), "Hello");
}

#[test]
fn encode_longer_prompt() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tok = load_from_gguf(&gguf).expect("load tokenizer");
    // Just sanity — no divergence, produces some tokens, round-trips.
    let text = "The quick brown fox jumps over the lazy dog.";
    let ids = tok.encode(text).expect("encode");
    assert!(!ids.is_empty());
    let back = tok.decode(&ids).expect("decode");
    eprintln!("encode('{text}') = {} tokens → '{back}'", ids.len());
    assert_eq!(back.trim(), text);
}

/// Qwen2/Qwen3 pre-tokenizer regex keeps optional leading punctuation
/// attached to the following letters (`-time` → one chunk), so the BPE
/// merge that produces vocab token 7019 (`-time`) fires. GPT-2's default
/// regex would split into `["-", "time"]` and emit `[12, 1619]`.
#[test]
fn qwen35_dash_word_uses_punct_letter_merge() {
    let Some(path) = gguf_path() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let tok = load_from_gguf(&gguf).expect("load tokenizer");
    let ids = tok.encode("-time").expect("encode");
    eprintln!("encode('-time') = {ids:?}");
    assert_eq!(
        ids,
        vec![7019],
        "qwen35 pretokenizer should keep '-time' as one chunk for the BPE merge"
    );
}
