//! TP-2a — `ShardedForwardOneTokenScratchTp` allocation smoke.
//!
//! Allocates the TP scratch on a 4× HIP cluster, verifies per-rank
//! byte budget, then disposes. GPU-gated; skips cleanly without HIP.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture: only allocations + dispose; the unsafe inside dispose \
              is documented in tp.rs."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipCluster};
use flambeau_qwen3_moe::config::{AttentionFamily, RopeSpec};
use flambeau_qwen3_moe::Qwen3MoEConfig;
use flambeau_qwen3_moe::forward::ShardedForwardOneTokenScratchTp;
use flambeau_runtime::RankId;

fn qwen35_27b_cfg() -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        arch: "qwen35".into(),
        family: AttentionFamily::Hybrid,
        hidden_size: 5120,
        vocab_size: 248320,
        num_layers: 64,
        num_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        context_length: 32768,
        rms_norm_eps: 1e-6,
        rope: RopeSpec {
            freq_base: 1_000_000.0,
            rotated_dims: 256,
            sections: None,
        },
        num_experts: 0,
        num_experts_per_tok: 1,
        moe_intermediate_size: 17408, // dense FFN width on qwen35
        shared_expert_intermediate_size: None,
        full_attention_interval: Some(4),
        gdn: None,
        tied_lm_head: false,
    }
}

#[test]
fn tp_scratch_allocates_and_disposes() -> Result<()> {
    match device_count() {
        Ok(n) if n >= 4 => (),
        Ok(n) => {
            eprintln!("[skip] need >= 4 HIP devices for tp scratch smoke (have {n})");
            return Ok(());
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return Ok(());
        }
    }

    let cfg = qwen35_27b_cfg();
    let cluster = HipCluster::new(&[0, 1, 2, 3])?;
    let scratch = ShardedForwardOneTokenScratchTp::new(&cfg, &cluster)?;

    // Invariant 1: every rank holds the same rank-level allocation
    // (hidden_a + hidden_b + 2 partials = 4 × hidden × 2 bytes = 4·H·2).
    let expected_per_rank = 4 * cfg.hidden_size * 2;
    for (r, rs) in scratch.per_rank.iter().enumerate() {
        assert_eq!(
            rs.allocated_bytes(),
            expected_per_rank,
            "rank {r}: rank-level alloc {} != expected {expected_per_rank}",
            rs.allocated_bytes()
        );
    }

    // Invariant 2: head_rank defaults to 0 — only that rank carries
    // OutputHeadScratch.
    assert_eq!(scratch.head_rank, RankId(0));
    for (r, rs) in scratch.per_rank.iter().enumerate() {
        if r == 0 {
            assert!(rs.output_head.is_some(), "rank 0 should hold OutputHeadScratch");
        } else {
            assert!(
                rs.output_head.is_none(),
                "rank {r} should not hold OutputHeadScratch (V1 head_rank=0)"
            );
        }
    }

    // Invariant 3: total rank-level bytes == ranks × per-rank.
    assert_eq!(
        scratch.rank_level_bytes(),
        cluster.ranks() * expected_per_rank
    );

    eprintln!(
        "TP-2a scratch (Qwen3.5-27B): rank-level = {:.2} MiB/rank x {} ranks = {:.2} MiB total",
        expected_per_rank as f64 / 1024.0 / 1024.0,
        cluster.ranks(),
        scratch.rank_level_bytes() as f64 / 1024.0 / 1024.0
    );

    scratch.dispose(&cluster)?;
    Ok(())
}

#[test]
fn tp_scratch_rejects_out_of_range_head_rank() -> Result<()> {
    match device_count() {
        Ok(n) if n >= 1 => (),
        Ok(_) => {
            eprintln!("[skip] no HIP devices");
            return Ok(());
        }
        Err(e) => {
            eprintln!("[skip] HIP unavailable: {e}");
            return Ok(());
        }
    }
    let cfg = qwen35_27b_cfg();
    let cluster = HipCluster::new(&[0])?;
    let res = ShardedForwardOneTokenScratchTp::new_with_head_rank(&cfg, &cluster, RankId(7));
    match res {
        Ok(scratch) => {
            scratch.dispose(&cluster)?;
            panic!("head_rank=7 on 1-rank cluster should have failed");
        }
        Err(e) => assert!(
            e.to_string().contains("out of range"),
            "expected 'out of range' diagnostic, got: {e}"
        ),
    }
    Ok(())
}
