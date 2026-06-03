//! Qwen3.5-9B-Q4_1 tp_size=2 through `Session<Qwen35V2>`. qwen35 is
//! hybrid (8 full-attn + 24 GDN layers); the GDN layers route through
//! `GdnTpMode::KReplicated` per the arch's rep_outer requirement.
//!
//! Single Session per test process (cargo test runs each test file as
//! its own binary, fresh process). The expected argmax is hard-coded
//! from the SD smoke run; the test asserts TP=2 produces finite
//! logits with the same argmax. Cross-session leakage in a single
//! process is documented separately and not exercised here.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

/// argmax predicted by the qwen35-v2 SD smoke at token=1, position=0
/// — Q-gate-correct, GDN-state-zeroed. Stable across processes.
const EXPECTED_ARGMAX: usize = 5328;

#[test]
fn session_qwen35_9b_tp_size_2_argmax_matches_sd() {
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
        flambeau_forward::LaunchParams {
            ctx_cap: None,
            prefill_ubatch: 1,
            max_slots: 1,
            paged_kv_pages: None,
            kv_layout: flambeau_forward::KvLayout::F16Contig,
        },
    )
    .expect("Session<Qwen35V2> TP=2");

    session.forward_one_token(1, 0).expect("TP=2 forward");
    let logits = session.logits().to_vec();
    session.dispose().expect("TP=2 dispose");

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
    eprintln!(
        "Session<Qwen35V2> TP=2: logits[0..4]={:?} argmax={argmax} finite={finite}",
        &logits[..4.min(logits.len())]
    );
    assert!(finite, "qwen35-v2 TP=2 logits contain NaN/Inf");
    assert_eq!(
        argmax, EXPECTED_ARGMAX,
        "TP=2 argmax {argmax} != SD-expected {EXPECTED_ARGMAX}"
    );
}
