//! Sanity: does our encoder recognise Qwen's special tokens as single ids?

use flambeau_quant::{load_from_gguf, GgufFile};

#[test]
fn special_tokens_encode_as_single_id() {
    let Some(path) = std::env::var("FLAMBEAU_QWEN3_GGUF").ok() else {
        eprintln!("skip — FLAMBEAU_QWEN3_GGUF unset");
        return;
    };
    let gguf = GgufFile::open(&path).unwrap();
    let tok = load_from_gguf(&gguf).unwrap();

    for s in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"] {
        let ids = tok.encode(s).unwrap();
        eprintln!("encode({s:?}) = {ids:?}");
    }
    eprintln!("stop_ids = {:?}", tok.stop_ids);
}
