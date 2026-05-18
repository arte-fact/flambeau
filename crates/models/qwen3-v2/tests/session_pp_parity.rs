//! pp_size=2 parity vs SD on Qwen3-Embedding-0.6B, driven through
//! `flambeau_forward::Session<Qwen3V2>`. Confirms the persistent-worker
//! topology orchestrator collapses what `pp_parity.rs` had to spell
//! out by hand. Host F16 roundtrip is lossless → bit-equal logits.

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen3_v2::Qwen3V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

#[test]
fn session_pp_size_2_matches_single_device() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let baseline_logits: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (SD)");
        let mut session = Session::<Qwen3V2>::new(file, Topology::SingleDevice { device: 0 })
            .expect("Session SD");
        session.forward_one_token(1, 0).expect("SD forward");
        let logits = session.logits().to_vec();
        session.dispose().expect("SD dispose");
        logits
    };

    let num_layers = {
        let file = GgufFile::open(&path).expect("open gguf (count)");
        let m = flambeau_qwen3_v2::load_from_gguf(
            &file,
            &flambeau_backend_hip::HipDevice::new(0).expect("hip dev"),
        )
        .expect("load count");
        let n = m.config.num_layers;
        let mut m = m;
        m.dispose(&flambeau_backend_hip::HipDevice::new(0).unwrap())
            .ok();
        n
    };
    let split = num_layers / 2;
    eprintln!("PP layout: rank 0 = layers [0..{split}); rank 1 = layers [{split}..{num_layers}).");

    let pp_logits: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (PP)");
        let topology = Topology::Pp {
            devices: vec![0, 0],
            layer_split: Some(vec![split, num_layers - split]),
        };
        let mut session = Session::<Qwen3V2>::new(file, topology).expect("Session PP");
        session.forward_one_token(1, 0).expect("PP forward");
        let logits = session.logits().to_vec();
        session.dispose().expect("PP dispose");
        logits
    };

    assert_eq!(
        pp_logits.len(),
        baseline_logits.len(),
        "logits len mismatch"
    );
    let mut max_abs_diff = 0.0_f32;
    let mut worst_idx = 0;
    for (i, (&a, &b)) in pp_logits.iter().zip(baseline_logits.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            worst_idx = i;
        }
    }
    let sd_argmax = baseline_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let pp_argmax = pp_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "Session PP vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst_idx} \
         argmax SD={sd_argmax} PP={pp_argmax}"
    );
    assert_eq!(pp_argmax, sd_argmax, "argmax differs SD vs PP");
    assert_eq!(
        max_abs_diff, 0.0,
        "Session PP logits diverge from SD by {max_abs_diff:.6} at idx {worst_idx}"
    );
}
