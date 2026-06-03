//! Qwen3.5-27B-Q4_0 tp_size=2 smoke. 15 GB weights → ~7.5 GB per
//! rank, comfortably fits 16 GB MI50s on hip:0,1.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-27B-Q4_0.gguf";

#[test]
fn session_qwen35_27b_q4_0_tp_size_2_runs() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    let mut session = Session::<Qwen35V2>::new(
        file,
        Topology::Tp {
            devices: vec![0, 1],
        },
        None,
        1,
        1,
        None,
        flambeau_forward::KvLayout::F16Contig,
    )
    .expect("Session<Qwen35V2> TP=2");

    session.forward_one_token(1, 0).expect("TP=2 forward");
    let logits = session.logits().to_vec();
    session.dispose().expect("dispose");

    let mut finite = true;
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &l in &logits {
        if !l.is_finite() {
            finite = false;
        }
        if l < min {
            min = l;
        }
        if l > max {
            max = l;
        }
    }
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("Qwen3.5-27B-Q4_0 TP=2: min={min:.4} max={max:.4} argmax={argmax} finite={finite}");
    assert!(finite, "27B TP=2 logits contain NaN/Inf");
    assert!(min < max, "logits constant");
}
