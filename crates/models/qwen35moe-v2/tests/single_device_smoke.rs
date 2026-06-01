//! Qwen3.6-35B-A3B-Q3_K_S forward smoke on a single 16 GB MI50.
//! Tight VRAM (15 GB on disk + scratch) — skips if OOM or model
//! file not present. Asserts finite + non-constant logits.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35moe_v2::Qwen35MoeV2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.6-35B-A3B-Q3_K_S.gguf";

#[test]
fn session_qwen36_a3b_q3_k_s_sd_finite_logits() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    let mut session = match Session::<Qwen35MoeV2>::new(
        file,
        Topology::SingleDevice { device: 0 },
        None,
        16,
        1,
    ) {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("alloc") || msg.contains("out of memory") || msg.contains("OOM") {
                eprintln!("SKIP: SD load OOM on 16 GB MI50: {msg}");
                return;
            }
            panic!("Session<Qwen35MoeV2> SD load failed: {msg}");
        }
    };

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
    eprintln!(
        "qwen36-35B-A3B-Q3_K_S SD: min={min:.4} max={max:.4} argmax={argmax} finite={finite}"
    );
    assert!(finite, "logits contain NaN/Inf");
    assert!(min < max, "logits are constant");
}
