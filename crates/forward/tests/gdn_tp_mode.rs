#![cfg(feature = "hip")]

use flambeau_forward::ctx::GdnDims;
use flambeau_forward::loader::{gdn_tp_mode_for, per_rank_gdn_dims, GdnTpMode};

fn qwen36_27b() -> GdnDims {
    // 24 v-heads / 16 k-heads → ratio 1.5 (non-integer → FullShard at TP>1).
    GdnDims {
        d_inner: 3072,
        num_v_heads: 24,
        num_k_heads: 16,
        head_k_dim: 128,
        head_v_dim: 128,
        conv_channels: 7168,
        conv_kernel: 4,
    }
}

fn qwen35_9b() -> GdnDims {
    // 32 v-heads / 16 k-heads → ratio 2 (integer → KReplicated at TP>1).
    GdnDims {
        d_inner: 4096,
        num_v_heads: 32,
        num_k_heads: 16,
        head_k_dim: 128,
        head_v_dim: 128,
        conv_channels: 8192,
        conv_kernel: 4,
    }
}

fn qwen36_35b_a3b() -> GdnDims {
    // Same 32/16 ratio as the 9B → KReplicated.
    qwen35_9b()
}

#[test]
fn sd_picks_kreplicated_irrespective_of_geometry() {
    for g in [qwen36_27b(), qwen35_9b()] {
        assert_eq!(gdn_tp_mode_for(g, 1), GdnTpMode::KReplicated);
    }
}

#[test]
fn qwen36_27b_picks_fullshard_at_tp2() {
    assert_eq!(gdn_tp_mode_for(qwen36_27b(), 2), GdnTpMode::FullShard);
}

#[test]
fn qwen36_27b_picks_fullshard_at_tp4() {
    // 24 v / 4 = 6 ; 6 % 16 = 6 ≠ 0 → FullShard.
    assert_eq!(gdn_tp_mode_for(qwen36_27b(), 4), GdnTpMode::FullShard);
}

#[test]
fn qwen35_9b_picks_kreplicated_at_tp2() {
    // 32 v / 2 = 16 ; 16 % 16 = 0 → KReplicated.
    assert_eq!(gdn_tp_mode_for(qwen35_9b(), 2), GdnTpMode::KReplicated);
}

#[test]
fn qwen35_9b_falls_back_to_fullshard_at_tp4() {
    // 32 v / 4 = 8 ; 8 % 16 = 8 ≠ 0 → FullShard. Documents the
    // geometry gate's behavior: KReplicated requires local_v ≥
    // num_k_heads as a multiple. tp4 on the 9B drops below it.
    assert_eq!(gdn_tp_mode_for(qwen35_9b(), 4), GdnTpMode::FullShard);
}

#[test]
fn qwen36_35b_a3b_picks_kreplicated_at_tp2() {
    assert_eq!(
        gdn_tp_mode_for(qwen36_35b_a3b(), 2),
        GdnTpMode::KReplicated
    );
}

#[test]
fn per_rank_gdn_dims_sd_is_identity() {
    let g = qwen35_9b();
    let out = per_rank_gdn_dims(g, 1);
    assert_eq!(out.num_v_heads, 32);
    assert_eq!(out.num_k_heads, 16);
    assert_eq!(out.d_inner, 4096);
    assert_eq!(out.conv_channels, 8192);
}

#[test]
fn per_rank_gdn_dims_kreplicated_shards_v_keeps_k() {
    let g = qwen35_9b();
    let out = per_rank_gdn_dims(g, 2);
    assert_eq!(out.num_v_heads, 16); // sharded
    assert_eq!(out.num_k_heads, 16); // replicated
    assert_eq!(out.d_inner, 16 * 128); // local v × head_v_dim
    assert_eq!(out.conv_channels, 2 * 16 * 128 + 16 * 128);
}

#[test]
fn per_rank_gdn_dims_fullshard_divides_both() {
    let g = qwen36_27b();
    let out = per_rank_gdn_dims(g, 2);
    assert_eq!(out.num_v_heads, 12); // sharded
    assert_eq!(out.num_k_heads, 8); // sharded
    assert_eq!(out.d_inner, 12 * 128);
    assert_eq!(out.conv_channels, 2 * 8 * 128 + 12 * 128);
}
