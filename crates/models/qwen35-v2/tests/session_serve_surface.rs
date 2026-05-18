//! Smokes the `Session<A>` serve-adapter surface (prefill_logits +
//! one-token-logits + reset_kv) on Qwen3.5-9B-Q4_1 / SD. The methods
//! plug `Session<A>` into the server's `ModelDriver` slot.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

#[test]
fn session_prefill_logits_then_decode_then_reset() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let mut session =
        Session::<Qwen35V2>::new(file, Topology::SingleDevice { device: 0 })
            .expect("Session<Qwen35V2> SD");

    let prompt: Vec<u32> = vec![1, 100, 200, 300];
    let mut prefill_logits = Vec::new();
    session
        .forward_prefill_logits(&prompt, 0, &mut prefill_logits)
        .expect("prefill_logits");
    assert!(!prefill_logits.is_empty(), "prefill returned empty logits");
    assert!(
        prefill_logits.iter().all(|x| x.is_finite()),
        "prefill logits contain non-finite"
    );

    let mut decode_logits = Vec::new();
    session
        .forward_one_token_logits(42, prompt.len(), &mut decode_logits)
        .expect("decode_logits");
    assert_eq!(decode_logits.len(), prefill_logits.len(), "vocab mismatch");
    assert!(
        decode_logits.iter().all(|x| x.is_finite()),
        "decode logits contain non-finite"
    );

    session.reset_kv().expect("reset_kv");

    let mut after_reset = Vec::new();
    session
        .forward_one_token_logits(1, 0, &mut after_reset)
        .expect("forward post-reset");
    assert_eq!(after_reset.len(), prefill_logits.len());

    session.dispose().expect("dispose");
}
