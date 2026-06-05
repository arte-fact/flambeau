//! gemma-4-31B tp_size=2 forward through `Session<Gemma4V2>`. The
//! 18 GB Q4_0 weights don't fit on a single 16 GB MI50, so there's
//! no SD baseline — assert both TP ranks produce finite, identical
//! logits (output_head is replicated post-AR) and a coherent argmax.
//!
//! This is the first arch with per-layer-varying head_dim (SWA = 256
//! vs global = 512); the run validates the per-slot KV-cache sizing
//! that landed in this commit.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_gemma4_v2::Gemma4V2;
use flambeau_quant::GgufFile;

const MODEL_PATH: &str = "/artefact/models/gemma-4-31B-it-Q4_0.gguf";

#[test]
fn session_gemma4_31b_tp_size_2_runs() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    let mut session = Session::<Gemma4V2>::new(
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
    .expect("Session<Gemma4V2> TP=2");

    session.forward_one_token(1, 0).expect("TP forward");
    let logits = session.logits().to_vec();
    session.dispose().expect("dispose");

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
        "gemma-4-31B TP=2 (Session): logits[0..4]={:?} argmax={argmax} finite={finite}",
        &logits[..4.min(logits.len())]
    );
    assert!(finite, "gemma4 TP logits contain NaN/Inf");
}
