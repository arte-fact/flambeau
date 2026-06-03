//! Qwen3.5-9B-Q3_K_S native-K-quant smoke. Exercises the
//! `QuantWeight { ptr, dtype, n_elems }` struct shape with K-quant
//! dtypes (Q3_K + Q4_K superblocks). Pre-refactor this would have
//! gone through the dequant-to-Q8_0 fallback; native path now means
//! the K-quant bytes ride straight to the device and the HIP MMVQ
//! kernel reads them as-is.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q3_K_S.gguf";

#[test]
fn session_qwen35_9b_q3_k_s_sd_finite_logits() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    let mut session = Session::<Qwen35V2>::new(
        file,
        Topology::SingleDevice { device: 0 },
        flambeau_forward::LaunchParams {
            ctx_cap: None,
            prefill_ubatch: 1,
            max_slots: 1,
            paged_kv_pages: None,
            kv_layout: flambeau_forward::KvLayout::F16Contig,
        },
    )
    .expect("Session<Qwen35V2> SD");
    session.forward_one_token(1, 0).expect("SD forward");
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
    eprintln!("qwen35-9B-Q3_K_S SD: min={min:.4} max={max:.4} argmax={argmax} finite={finite}");
    assert!(finite, "Q3_K_S logits contain NaN/Inf");
    assert!(
        min < max,
        "logits are constant — model not actually computing"
    );
}
