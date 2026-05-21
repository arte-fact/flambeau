//! Qwen3.5-9B-Q4_1 hybrid (PP=2 of TP=2 = pp2tp2) smoke through
//! `Session<Qwen35V2>`. 4 GPUs total. Per memory feedback the link
//! {2,3} is faulty, so the TP-inner ring must avoid pairing 2 with 3:
//! stage0 = {0,2}, stage1 = {1,3} — each stage's AR happens across a
//! healthy pair (0↔2 and 1↔3); peer-copy crosses stages (2→1 etc.)
//! but that's a one-shot host hop, not AR.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";
const EXPECTED_ARGMAX: usize = 5328;

#[test]
fn session_qwen35_9b_pp2tp2_argmax_matches_sd() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    // qwen35-9B has 32 layers; even split per stage.
    let mut session = Session::<Qwen35V2>::new(
        file,
        Topology::Hybrid {
            stages: vec![vec![0, 2], vec![1, 3]],
            layer_split: Some(vec![16, 16]),
        },
    )
    .expect("Session<Qwen35V2> pp2tp2");

    session.forward_one_token(1, 0).expect("pp2tp2 forward");
    let logits = session.logits().to_vec();
    session.dispose().expect("pp2tp2 dispose");

    let mut finite = true;
    for &l in &logits {
        if !l.is_finite() {
            finite = false;
            break;
        }
    }
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("Session<Qwen35V2> pp2tp2: argmax={argmax} finite={finite}");
    assert!(finite, "qwen35-v2 pp2tp2 logits contain NaN/Inf");
    assert_eq!(
        argmax, EXPECTED_ARGMAX,
        "pp2tp2 argmax {argmax} != SD-expected {EXPECTED_ARGMAX}"
    );
}
