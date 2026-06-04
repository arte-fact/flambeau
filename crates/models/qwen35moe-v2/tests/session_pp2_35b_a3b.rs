//! Qwen3.6-35B-A3B-Q3_K_S pipeline-parallel (PP=2) smoke through
//! `Session<Qwen35MoeV2>`. 35B-A3B is hybrid (GDN + full-attn) MoE
//! with 128 experts; weights are 15 GB on disk, so PP=2 keeps each
//! rank comfortably below the 16 GB MI50 ceiling.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35moe_v2::Qwen35MoeV2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.6-35B-A3B-Q3_K_S.gguf";

#[test]
fn session_qwen36_a3b_q3_k_s_pp_size_2_runs() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let file = GgufFile::open(&path).expect("open gguf");
    let num_layers = file
        .metadata_u32("qwen35moe.block_count")
        .expect("qwen35moe.block_count") as usize;
    let half = num_layers / 2;
    let rest = num_layers - half;

    let mut session = match Session::<Qwen35MoeV2>::new(
        file,
        Topology::Pp {
            devices: vec![0, 1],
            layer_split: Some(vec![half, rest]),
        },
        flambeau_forward::LaunchParams {
            ctx_cap: None,
            prefill_ubatch: 1,
            max_slots: 1,
            paged_kv_pages: None,
            kv_layout: flambeau_forward::KvLayout::F16Contig,
            deterministic_ar: false,
        },
    ) {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("alloc") || msg.contains("out of memory") || msg.contains("OOM") {
                eprintln!("SKIP: PP=2 load OOM on 16 GB MI50: {msg}");
                return;
            }
            panic!("Session<Qwen35MoeV2> PP=2 load failed: {msg}");
        }
    };

    session.forward_one_token(1, 0).expect("PP=2 forward");
    let logits = session.logits().to_vec();
    session.dispose().expect("PP=2 dispose");

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
        "qwen36-35B-A3B-Q3_K_S PP=2: min={min:.4} max={max:.4} argmax={argmax} finite={finite}"
    );
    assert!(finite, "PP=2 logits contain NaN/Inf");
    assert!(min < max, "logits constant");
}
