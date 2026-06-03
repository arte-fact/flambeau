//! Phase K4b — `Session::forward_mixed` smoke on Qwen3.5-9B-Q4_1.
//!
//! Loads the model via the production `Session<Qwen35V2>::new` path
//! (one worker thread per rank), fires `Session::forward_mixed` once
//! with K prefill rows for slot 0 + N decode rows for slots 1..=N,
//! and asserts the emitted logits are finite + non-constant +
//! correctly N+1 rows.
//!
//! Env-gated by `FLAMBEAU_MIXED_MICROBENCH_GGUF` (same env as the K3b
//! microbench — pass the GGUF path or the test skips).

#![cfg(feature = "hip")]

use flambeau_backend_hip::device_count;
use flambeau_forward::{Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const K_PREFILL: usize = 8;
const N_DECODE: usize = 3;
const CTX_CAP: usize = 4096;
const PREFILL_UBATCH: usize = 32; // unused in the mixed path; required by Session::new

fn maybe_skip() -> Option<String> {
    let path = std::env::var("FLAMBEAU_MIXED_MICROBENCH_GGUF").ok()?;
    if device_count().ok()? < 1 {
        eprintln!("[skip] no HIP device");
        return None;
    }
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skip] GGUF not at {path}");
        return None;
    }
    Some(path)
}

#[test]
fn session_forward_mixed_smoke() {
    let Some(path) = maybe_skip() else {
        return;
    };

    let file = GgufFile::open(&path).expect("open gguf");
    let topology = Topology::SingleDevice { device: 0 };
    let max_slots = N_DECODE + 1;
    let mut session = Session::<Qwen35V2>::new(
        file,
        topology,
        Some(CTX_CAP),
        PREFILL_UBATCH.max(K_PREFILL + N_DECODE),
        max_slots,
        None,
        flambeau_forward::KvLayout::F16Contig,
    )
    .expect("Session<Qwen35V2>::new");

    let n_total = K_PREFILL + N_DECODE;
    let tokens: Vec<u32> = (0..n_total as u32).map(|i| (i + 1) % 50).collect();
    let mut positions: Vec<usize> = (0..K_PREFILL).collect();
    positions.extend(K_PREFILL..K_PREFILL + N_DECODE);
    let mut slot_ids = vec![0usize; K_PREFILL];
    slot_ids.extend(1..=N_DECODE);

    session
        .forward_mixed(&tokens, &positions, &slot_ids, K_PREFILL)
        .expect("Session::forward_mixed");

    // Logits buffer is row-major `[(N + 1), vocab]`. Each row must be
    // finite + non-constant.
    let vocab = session.vocab_size() / (N_DECODE + 1);
    assert!(vocab > 0, "vocab_size came back 0");
    for row_i in 0..=N_DECODE {
        let row = session.mixed_logits_row(row_i, vocab);
        assert_eq!(row.len(), vocab, "row {row_i} len mismatch");
        for (j, &l) in row.iter().enumerate() {
            assert!(
                l.is_finite(),
                "row {row_i} logit[{j}] = {l} not finite"
            );
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = row.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            max - min > 1e-3,
            "row {row_i}: max-min = {} (degenerate)",
            max - min
        );
    }
    eprintln!(
        "K4b smoke: forward_mixed produced {} logit rows, vocab={vocab}, all finite + non-constant",
        N_DECODE + 1
    );
    session.dispose().expect("session dispose");
}
