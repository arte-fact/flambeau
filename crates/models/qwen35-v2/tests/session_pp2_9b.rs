//! Qwen3.5-9B-Q4_1 pipeline-parallel (PP=2) smoke through
//! `Session<Qwen35V2>`. PP routes one layer-range per stage; no AR.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";
const EXPECTED_ARGMAX: usize = 5328;

#[test]
fn session_qwen35_9b_pp_size_2_argmax_matches_sd() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    // qwen35-9B has 32 layers; even split per rank.
    let mut session = Session::<Qwen35V2>::new(
        file,
        Topology::Pp {
            devices: vec![0, 1],
            layer_split: Some(vec![16, 16]),
        },
        flambeau_forward::LaunchParams {
            ctx_cap: None,
            prefill_ubatch: 1,
            max_slots: 1,
            paged_kv_pages: None,
            kv_layout: flambeau_forward::KvLayout::F16Contig,
        },
    )
    .expect("Session<Qwen35V2> PP=2");

    session.forward_one_token(1, 0).expect("PP=2 forward");
    let logits = session.logits().to_vec();
    session.dispose().expect("PP=2 dispose");

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
    eprintln!("Session<Qwen35V2> PP=2: argmax={argmax} finite={finite}");
    assert!(finite, "qwen35-v2 PP=2 logits contain NaN/Inf");
    assert_eq!(
        argmax, EXPECTED_ARGMAX,
        "PP=2 argmax {argmax} != SD-expected {EXPECTED_ARGMAX}"
    );
}
