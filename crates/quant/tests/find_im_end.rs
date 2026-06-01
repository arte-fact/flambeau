//! Debug: find im_end / im_start token IDs.

use flambeau_quant::gguf::{GgufFile, Value};

#[test]
fn dump_special_tokens() {
    let Some(path) = std::env::var("FLAMBEAU_QWEN3_GGUF").ok() else {
        eprintln!("FLAMBEAU_QWEN3_GGUF unset — skipping");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open GGUF");
    let arr = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .unwrap()
        .as_array()
        .unwrap();
    for (i, v) in arr.iter().enumerate() {
        if let Value::String(s) = v {
            if s.contains("im_end") || s.contains("im_start") || s.contains("endoftext") {
                eprintln!("id={i} token={s:?}");
            }
        }
    }
    eprintln!("total vocab entries: {}", arr.len());
}
