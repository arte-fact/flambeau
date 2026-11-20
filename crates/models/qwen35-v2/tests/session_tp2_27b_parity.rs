//! Qwen3.8-27B TP-2 vs PP-2 parity. The 27B is the qwen35 geometry where
//! TP-2 shards K heads (48 V / 16 K heads, `GdnTpMode::FullShard`), so the
//! V heads each rank owns must follow the kernel's `h % H_kv` pairing.
//! PP-2 keeps every head on one device and is the reference.
//!
//! PP-2 decodes greedily; TP-2 is forced along the PP-2 tokens, so every
//! position compares the two topologies on the same context. Per position:
//! KL(PP‖TP) ≤ `MAX_KL`; log-probability gap ≤ `MAX_LOGPROB_GAP` on tokens
//! PP gives at least `MIN_P` to; same argmax wherever PP's top-1 / top-2
//! margin is at least `MIN_MARGIN`. TP/PP reduction-order noise stays well
//! inside these bounds (same spread as PP q8 KV against PP f16 KV); the
//! mis-sharded V heads fail all three from the first position.
//!
//! Model path: `$FLAMBEAU_27B_GGUF`, default `Qwen3.8-27B-UD-Q4_K_M.gguf` in
//! the test models directory (`flambeau_test_models::models_dir`). Skips when
//! absent.

#![cfg(feature = "hip")]

use flambeau_forward::{KvLayout, LaunchParams, Session, Topology};
use flambeau_quant::GgufFile;
use flambeau_qwen35_v2::Qwen35V2;

const PROMPTS: &[&str] = &[
    "Write a complete Python module implementing an LRU cache class with get, put, delete and a __len__ method, using only the standard library. Return only code.",
    "A train leaves city A at 60 km/h. Two hours later a second train leaves A on the same track at 90 km/h. How long after the second train departs does it catch the first, and how far from A? Show the steps.",
    "Explain to a curious teenager why the sky is blue during the day and red at sunset, in about 250 words.",
    "Write the opening paragraph of a mystery novel set in a lighthouse.",
    "List ten tips for keeping houseplants alive, one sentence each.",
];
const STEPS: usize = 32;
const MAX_KL: f64 = 0.05;
const MAX_LOGPROB_GAP: f32 = 0.6;
const MIN_P: f32 = 0.01;
const MIN_MARGIN: f32 = 0.5;


fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
    row.iter().map(|&v| v - lse).collect()
}

/// Log-softmax rows per prompt (prefill row, then one per decode step) and
/// the tokens fed back. `forced` replays a reference trajectory instead of
/// the session's own greedy picks.
fn run(
    path: &std::path::Path,
    topology: Topology,
    prompts: &[Vec<u32>],
    forced: Option<&[Vec<u32>]>,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<u32>>) {
    let file = GgufFile::open(path).expect("open gguf");
    let mut session = Session::<Qwen35V2>::new(
        file,
        topology,
        LaunchParams {
            ctx_cap: Some(4096),
            prefill_ubatch: 512,
            max_slots: 1,
            paged_kv_pages: None,
            kv_layout: KvLayout::F16Contig,
            mtp_max_draft: 0,
            decode_graph: true,
        },
    )
    .expect("Session<Qwen35V2>");
    let mut rows_all = Vec::new();
    let mut fed_all = Vec::new();
    let mut logits = Vec::new();
    for (pi, ids) in prompts.iter().enumerate() {
        session.reset_kv().expect("reset");
        session.forward_prefill_logits(ids, 0, &mut logits).expect("prefill");
        let mut rows = vec![log_softmax(&logits)];
        let mut fed = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            let tok = match forced {
                Some(f) => f[pi][step],
                None => argmax(&logits),
            };
            session
                .forward_one_token_logits(tok, ids.len() + step, &mut logits)
                .expect("decode");
            rows.push(log_softmax(&logits));
            fed.push(tok);
        }
        rows_all.push(rows);
        fed_all.push(fed);
    }
    session.dispose().expect("dispose");
    (rows_all, fed_all)
}

/// Violations of the parity bounds at one position, empty when it passes.
fn check(pp: &[f32], tp: &[f32]) -> Vec<String> {
    let mut bad = Vec::new();
    let kl: f64 = pp
        .iter()
        .zip(tp)
        .map(|(&p, &t)| (p as f64).exp() * (p as f64 - t as f64))
        .sum();
    if kl > MAX_KL {
        bad.push(format!("KL {kl:.4}"));
    }
    let ln_min_p = MIN_P.ln();
    let gap = pp
        .iter()
        .zip(tp)
        .filter(|(&p, _)| p >= ln_min_p)
        .map(|(&p, &t)| (p - t).abs())
        .fold(0.0f32, f32::max);
    if gap > MAX_LOGPROB_GAP {
        bad.push(format!("logprob gap {gap:.3}"));
    }
    let mut order: Vec<usize> = (0..pp.len()).collect();
    order.select_nth_unstable_by(1, |&a, &b| pp[b].total_cmp(&pp[a]));
    let (mut top1, mut top2) = (order[0], order[1]);
    if pp[top2] > pp[top1] {
        std::mem::swap(&mut top1, &mut top2);
    }
    if pp[top1] - pp[top2] >= MIN_MARGIN && argmax(tp) as usize != top1 {
        bad.push(format!("argmax {} != PP {top1}", argmax(tp)));
    }
    bad
}

#[test]
fn qwen38_27b_tp2_matches_pp2() {
    let Some(path) = flambeau_test_models::model_from_env(
        "qwen38_27b_tp2_matches_pp2",
        "FLAMBEAU_27B_GGUF",
        "Qwen3.8-27B-UD-Q4_K_M.gguf",
    ) else {
        return;
    };
    let file = GgufFile::open(&path).expect("open gguf");
    let tokenizer = flambeau_quant::tokenizer::load_from_gguf(&file).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tokenizer.encode(p).expect("encode"))
        .collect();
    drop(file);

    let n_layers = 64usize;
    let (pp, trajectory) = run(
        &path,
        Topology::Pp {
            devices: vec![0, 1],
            layer_split: Some(vec![n_layers / 2, n_layers / 2]),
        },
        &prompts,
        None,
    );
    let (tp, _) = run(&path, Topology::Tp { devices: vec![0, 1] }, &prompts, Some(&trajectory));

    let mut failures = Vec::new();
    for (i, (pp_rows, tp_rows)) in pp.iter().zip(&tp).enumerate() {
        let mut n_bad = 0usize;
        for (step, (p, t)) in pp_rows.iter().zip(tp_rows).enumerate() {
            let bad = check(p, t);
            if !bad.is_empty() {
                n_bad += 1;
                failures.push(format!("prompt {i} step {step}: {}", bad.join(", ")));
            }
        }
        eprintln!("prompt {i}: {n_bad}/{} positions out of bounds", pp_rows.len());
    }
    assert!(failures.is_empty(), "TP-2 diverges from PP-2: {failures:#?}");
}
