//! Qwen3.5-9B-Q4_1 tp_size=2 through `Session<Qwen35V2>`. qwen35 is
//! hybrid (8 full-attn + 24 GDN layers); the GDN layers route through
//! `GdnTpMode::KReplicated` per the arch's rep_outer requirement.
//!
//! Assertion: both TP ranks produce identical finite logits
//! (output_head is replicated post-AR); argmax matches the SD run
//! (the AR's F32 reduction-tree order can differ, so we tolerate
//! up to 5e-2 abs diff on the value field).

#![cfg(feature = "hip")]

use std::path::PathBuf;

use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const MODEL_PATH: &str = "/artefact/models/Qwen3.5-9B-Q4_1.gguf";

// Ignored: same-process cross-session state leakage on qwen35-9B —
// the FIRST Session in a process gives argmax=1 (stable); the SECOND
// gives argmax=12482 (drift) regardless of topology. Confirmed by
// running two consecutive `Topology::SingleDevice` sessions: they
// also diverge. Single-process single-session SD smoke is stable
// across runs.
//
// qwen3-v2's 2-session test (SD-then-PP) doesn't repro the drift, so
// the bug is somewhere in qwen35-v2's larger memory footprint (~6 GB
// Q4_1) or its GDN path interaction with HIP module lifecycle /
// allocator fragmentation. Suspects: OpsRegistry HSACO unload→reload
// in same process, HipDevice context teardown ordering in RankState
// Drop, or rocBLAS handle reuse. Un-ignore once root cause is fixed.
#[ignore]
#[test]
fn session_qwen35_9b_tp_size_2_runs() {
    let path = PathBuf::from(MODEL_PATH);
    if !path.exists() {
        eprintln!("SKIP: {MODEL_PATH} not present");
        return;
    }

    let baseline: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (SD)");
        let mut s = Session::<Qwen35V2>::new(file, Topology::SingleDevice { device: 0 })
            .expect("Session<Qwen35V2> SD");
        s.forward_one_token(1, 0).expect("SD forward");
        let l = s.logits().to_vec();
        s.dispose().expect("SD dispose");
        l
    };

    // tp_size=1 exercises the new sharded code paths with n_ranks=1
    // (trivial slicing, no-op AR).
    let tp1_logits: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (TP1)");
        let mut s = Session::<Qwen35V2>::new(
            file,
            Topology::Tp { devices: vec![0] },
        )
        .expect("Session<Qwen35V2> TP=1");
        s.forward_one_token(1, 0).expect("TP=1 forward");
        let l = s.logits().to_vec();
        s.dispose().expect("TP=1 dispose");
        l
    };
    let tp1_argmax = tp1_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let mut tp1_max_diff = 0.0_f32;
    for (&a, &b) in tp1_logits.iter().zip(baseline.iter()) {
        tp1_max_diff = tp1_max_diff.max((a - b).abs());
    }
    eprintln!("Session<Qwen35V2> TP=1 vs SD: max_abs_diff={tp1_max_diff:.6} argmax={tp1_argmax}");

    let tp_logits: Vec<f32> = {
        let file = GgufFile::open(&path).expect("open gguf (TP)");
        let mut s = Session::<Qwen35V2>::new(
            file,
            Topology::Tp {
                devices: vec![0, 1],
            },
        )
        .expect("Session<Qwen35V2> TP=2");
        s.forward_one_token(1, 0).expect("TP forward");
        let l = s.logits().to_vec();
        s.dispose().expect("TP dispose");
        l
    };

    assert_eq!(tp_logits.len(), baseline.len(), "logits len mismatch");
    let mut finite = true;
    for &l in &tp_logits {
        if !l.is_finite() {
            finite = false;
            break;
        }
    }
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
        "Session<Qwen35V2> TP=2 vs SD: max_abs_diff={max_abs_diff:.6} @ idx {worst} \
         argmax SD={baseline_argmax} TP={tp_argmax} finite={finite}"
    );
    assert!(finite, "qwen35-v2 TP logits contain NaN/Inf");
    assert_eq!(tp_argmax, baseline_argmax, "argmax differs SD vs TP");
    assert!(
        max_abs_diff < 5e-2,
        "TP logits diverge from SD by {max_abs_diff:.6} at idx {worst}"
    );
}
