//! Sanity-check `forward_prefill_logits` on a chat-template-rendered
//! prompt. Compares argmax of the v2 SD prefill vs the model's
//! legacy SD prefill via the existing flambeau-qwen3-moe path.
//! Used to root-cause the v2 serve degeneration.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::{ChatTemplate, GgufFile};
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

#[test]
fn session_chat_prefill_argmax_coherent() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }
    let file = GgufFile::open(&path).expect("open gguf");
    let tokenizer = flambeau_quant::load_from_gguf(&file).expect("tokenizer");
    let tpl = ChatTemplate::load_from_gguf(&file).expect("chat template");

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": "Reply with exactly: hello world"
    })];
    let tools: Vec<serde_json::Value> = vec![];
    let rendered = tpl
        .render_with_tools::<serde_json::Value, serde_json::Value>(
            &messages,
            Some(&tools),
            true,
            Some(false),
        )
        .expect("render");
    eprintln!("rendered prompt ({} bytes):\n{rendered}", rendered.len());
    let prompt_ids = tokenizer.encode(&rendered).expect("tokenize");
    eprintln!("tokenized: {} ids", prompt_ids.len());

    let mut session = Session::<Qwen35V2>::new(file, Topology::SingleDevice { device: 0 })
        .expect("Session SD");

    let mut logits = Vec::new();
    session
        .forward_prefill_logits(&prompt_ids, 0, &mut logits)
        .expect("prefill");
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let top5: Vec<(usize, f32)> = {
        let mut v: Vec<_> = logits.iter().copied().enumerate().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v.into_iter().take(5).collect()
    };
    eprintln!("argmax = {argmax}");
    eprintln!("top-5: {top5:?}");
    let decoded = tokenizer
        .decode(&[argmax as u32])
        .unwrap_or_else(|_| String::from("<decode-err>"));
    eprintln!("argmax decoded: {decoded:?}");

    session.dispose().expect("dispose");
    assert!(logits.iter().all(|x| x.is_finite()));

    // Diagnostic — also run the same prefill in two halves: tokens[..H],
    // dispose, fresh session, tokens[..H+rest]. If chained-single-token
    // forward is order-independent, the final-token argmax should
    // match. (Today's chained path is the v1 implementation; the real
    // batched-prefill kernel is in P9-OUT.)
    let file2 = GgufFile::open(&path).expect("open gguf again");
    let mut s2 = Session::<Qwen35V2>::new(file2, Topology::SingleDevice { device: 0 })
        .expect("Session SD #2");
    let mut logits2 = Vec::new();
    s2.forward_prefill_logits(&prompt_ids, 0, &mut logits2)
        .expect("prefill #2");
    let argmax2 = logits2
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("argmax across fresh session #2: {argmax2}");
    s2.dispose().expect("dispose #2");
    assert_eq!(argmax, argmax2, "Sessions diverged on identical input");
}
