//! pp_size=2 × tp_size=2 parity vs SD on Qwen3-Embedding-0.6B through
//! `flambeau_forward::Session<Qwen3V2>`. Per-stage AR + inter-stage
//! peer_buffer handoff are all internal to the orchestrator.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen3_v2::{Qwen3V2, Qwen3V2Config};

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

#[test]
fn session_pp2_tp2_matches_single_device() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let baseline: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (SD)");
        let mut s = Session::<Qwen3V2>::new(file, Topology::SingleDevice { device: 0 })
            .expect("Session SD");
        s.forward_one_token(1, 0).expect("SD forward");
        let l = s.logits().to_vec();
        s.dispose().expect("SD dispose");
        l
    };

    let num_layers = {
        let file = GgufFile::open(&path).expect("open gguf (probe)");
        Qwen3V2Config::from_gguf(&file)
            .expect("config probe")
            .num_layers
    };
    let split = num_layers / 2;
    let layer_split = vec![split, num_layers - split];

    let hyb_logits: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (Hybrid)");
        let mut s = Session::<Qwen3V2>::new(
            file,
            Topology::Hybrid {
                stages: vec![vec![0, 0], vec![0, 0]],
                layer_split: Some(layer_split),
            },
        )
        .expect("Session Hybrid");
        s.forward_one_token(1, 0).expect("Hybrid forward");
        let l = s.logits().to_vec();
        s.dispose().expect("Hybrid dispose");
        l
    };

    let mut max_abs_diff = 0.0_f32;
    let mut worst = 0;
    for (i, (&a, &b)) in hyb_logits.iter().zip(baseline.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            worst = i;
        }
    }
    let baseline_argmax = baseline
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let hyb_argmax = hyb_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "Session Hybrid pp2tp2 vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst} \
         argmax SD={baseline_argmax} Hyb={hyb_argmax}"
    );
    assert_eq!(hyb_argmax, baseline_argmax);
    assert!(
        max_abs_diff < 5e-2,
        "Session Hybrid logits diverge from SD by {max_abs_diff:.6} at idx {worst}"
    );
}
