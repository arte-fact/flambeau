//! tp_size=2 parity vs SD on Qwen3-Embedding-0.6B through
//! `flambeau_forward::Session<Qwen3V2>`. AR reduction-tree order
//! differs from SD → tolerate ≤ 5e-2 abs diff; argmax must match.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen3_v2::Qwen3V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

#[test]
fn session_tp_size_2_matches_single_device() {
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

    let tp_logits: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (TP)");
        let mut s = Session::<Qwen3V2>::new(
            file,
            Topology::Tp {
                devices: vec![0, 0],
            },
        )
        .expect("Session TP");
        s.forward_one_token(1, 0).expect("TP forward");
        let l = s.logits().to_vec();
        s.dispose().expect("TP dispose");
        l
    };

    let mut max_abs_diff = 0.0_f32;
    let mut worst = 0;
    for (i, (&a, &b)) in tp_logits.iter().zip(baseline.iter()).enumerate() {
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
    let tp_argmax = tp_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "Session TP vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst} \
         argmax SD={baseline_argmax} TP={tp_argmax}"
    );
    assert_eq!(tp_argmax, baseline_argmax, "argmax differs SD vs TP");
    assert!(
        max_abs_diff < 5e-2,
        "Session TP logits diverge from SD by {max_abs_diff:.6} at idx {worst}"
    );
}
