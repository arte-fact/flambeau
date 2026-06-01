#![cfg(feature = "hip")]

use flambeau_forward::ctx::GdnDims;
use flambeau_forward::loader::ShardMode;
use flambeau_forward::{
    per_layer_kv_widths, scratch_config_for, KvLayerShape, MoeShape, ScratchShape,
};

struct HybridShape {
    n_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    full_attn_interval: usize,
}

impl KvLayerShape for HybridShape {
    fn num_layers(&self) -> usize {
        self.n_layers
    }
    fn kv_width_at(&self, li: usize, n_ranks: usize) -> usize {
        if (li + 1) % self.full_attn_interval == 0 {
            (self.kv_heads / n_ranks) * self.head_dim
        } else {
            0
        }
    }
}

#[test]
fn per_layer_kv_widths_zeroes_recurrent_layers_and_shards() {
    let shape = HybridShape {
        n_layers: 8,
        kv_heads: 4,
        head_dim: 256,
        full_attn_interval: 4,
    };
    let widths = per_layer_kv_widths(&shape, 2);
    assert_eq!(widths.len(), 8);
    assert_eq!(widths, vec![0, 0, 0, 512, 0, 0, 0, 512]);
}

#[test]
fn per_layer_kv_widths_uniform_n_ranks_1() {
    struct Uniform {
        n: usize,
        w: usize,
    }
    impl KvLayerShape for Uniform {
        fn num_layers(&self) -> usize {
            self.n
        }
        fn kv_width_at(&self, _li: usize, n_ranks: usize) -> usize {
            self.w / n_ranks
        }
    }
    let widths = per_layer_kv_widths(&Uniform { n: 4, w: 1024 }, 1);
    assert_eq!(widths, vec![1024; 4]);
}

// ---- ScratchShape golden-diff tests ---------------------------------
//
// These assert that `scratch_config_for(..., None)` returns byte-identical
// values to the pre-Phase-2 hand-written `Arch::scratch_config` bodies.
// If the trait shape ever changes a derivation, these tests catch it
// before any arch crate sees the regression.

/// Synthetic qwen35-v2-style dense hybrid (e.g. Qwen3.6-27B). Mirrors
/// the math at qwen35-v2/src/arch.rs HEAD~ (commit c5622ec, pre-builder).
struct QwenDenseShape {
    n_layers: usize,
    hidden: usize,
    intermediate: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    context_length: usize,
    full_attn_interval: usize,
    gdn: GdnDims,
}

impl KvLayerShape for QwenDenseShape {
    fn num_layers(&self) -> usize {
        self.n_layers
    }
    fn kv_width_at(&self, li: usize, n_ranks: usize) -> usize {
        if (li + 1) % self.full_attn_interval != 0 {
            0
        } else {
            (self.n_kv_heads / n_ranks) * self.head_dim
        }
    }
}

impl ScratchShape for QwenDenseShape {
    fn hidden(&self) -> usize {
        self.hidden
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_seq_len(&self) -> usize {
        self.context_length
    }
    fn intermediate_per_rank(&self, n_ranks: usize) -> usize {
        self.intermediate / n_ranks
    }
    fn q_width_per_rank(&self, n_ranks: usize) -> usize {
        (self.n_heads / n_ranks) * self.head_dim
    }
    fn moe_per_rank(&self, _n_ranks: usize) -> Option<MoeShape> {
        None
    }
    fn gdn_per_rank(&self, _n_ranks: usize) -> Option<GdnDims> {
        Some(self.gdn)
    }
    fn attn_q_gated(&self) -> bool {
        true
    }
}

#[test]
fn scratch_config_for_qwen_dense_matches_handwritten() {
    let shape = QwenDenseShape {
        n_layers: 64,
        hidden: 5120,
        intermediate: 17408,
        n_heads: 24,
        n_kv_heads: 4,
        head_dim: 256,
        vocab: 248320,
        context_length: 262144,
        full_attn_interval: 4,
        gdn: GdnDims {
            d_inner: 6144,
            num_v_heads: 48,
            num_k_heads: 16,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_channels: 10240,
            conv_kernel: 4,
        },
    };
    let cfg = scratch_config_for(
        &shape,
        ShardMode::Tp {
            rank: 0,
            n_ranks: 2,
        },
        512,
        1,
        None,
    );
    assert_eq!(cfg.hidden, 5120);
    assert_eq!(cfg.intermediate, 17408 / 2);
    assert_eq!(cfg.q_width, (24 / 2) * 256);
    assert_eq!(cfg.kv_width, (4 / 2) * 256);
    assert_eq!(cfg.vocab, 248320);
    assert_eq!(cfg.max_seq_len, 262144);
    assert_eq!(cfg.num_layers, 64);
    assert_eq!(cfg.max_experts, 0);
    assert_eq!(cfg.max_experts_per_tok, 0);
    assert_eq!(cfg.shared_intermediate, 0);
    assert!(cfg.attn_q_gated);
    assert!(cfg.gdn.is_some());
    assert_eq!(cfg.per_layer_embd, 0);
    assert_eq!(cfg.max_prefill_tokens, 512);
    assert_eq!(cfg.max_slots, 1);
    let widths = cfg.per_layer_kv_widths.as_ref().unwrap();
    assert_eq!(widths.len(), 64);
    let expected_w = (4 / 2) * 256;
    for li in 0..64 {
        let is_full_attn = (li + 1) % 4 == 0;
        assert_eq!(widths[li], if is_full_attn { expected_w } else { 0 });
    }
}

/// Synthetic qwen35moe-v2 (Qwen3.6-35B-A3B).
struct QwenMoeShape {
    n_layers: usize,
    hidden: usize,
    expert_intermediate: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    context_length: usize,
    full_attn_interval: usize,
    num_experts: usize,
    experts_per_tok: usize,
    shared_expert_intermediate: usize,
    gdn: GdnDims,
}

impl KvLayerShape for QwenMoeShape {
    fn num_layers(&self) -> usize {
        self.n_layers
    }
    fn kv_width_at(&self, li: usize, n_ranks: usize) -> usize {
        if (li + 1) % self.full_attn_interval != 0 {
            0
        } else {
            (self.n_kv_heads / n_ranks) * self.head_dim
        }
    }
}

impl ScratchShape for QwenMoeShape {
    fn hidden(&self) -> usize {
        self.hidden
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_seq_len(&self) -> usize {
        self.context_length
    }
    fn intermediate_per_rank(&self, n_ranks: usize) -> usize {
        self.expert_intermediate / n_ranks
    }
    fn q_width_per_rank(&self, n_ranks: usize) -> usize {
        (self.n_heads / n_ranks) * self.head_dim
    }
    fn moe_per_rank(&self, _n_ranks: usize) -> Option<MoeShape> {
        Some(MoeShape {
            num_experts: self.num_experts,
            experts_per_tok: self.experts_per_tok,
            shared_intermediate_per_rank: self.shared_expert_intermediate,
        })
    }
    fn gdn_per_rank(&self, _n_ranks: usize) -> Option<GdnDims> {
        Some(self.gdn)
    }
    fn attn_q_gated(&self) -> bool {
        true
    }
}

#[test]
fn scratch_config_for_qwen_moe_matches_handwritten() {
    let shape = QwenMoeShape {
        n_layers: 48,
        hidden: 4096,
        expert_intermediate: 1536,
        n_heads: 32,
        n_kv_heads: 4,
        head_dim: 128,
        vocab: 152064,
        context_length: 32768,
        full_attn_interval: 4,
        num_experts: 128,
        experts_per_tok: 8,
        shared_expert_intermediate: 512,
        gdn: GdnDims {
            d_inner: 4096,
            num_v_heads: 32,
            num_k_heads: 16,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_channels: 8192,
            conv_kernel: 4,
        },
    };
    let cfg = scratch_config_for(
        &shape,
        ShardMode::Tp {
            rank: 0,
            n_ranks: 2,
        },
        512,
        1,
        None,
    );
    assert_eq!(cfg.intermediate, 1536 / 2);
    assert_eq!(cfg.max_experts, 128);
    assert_eq!(cfg.max_experts_per_tok, 8);
    assert_eq!(cfg.shared_intermediate, 512);
    assert!(cfg.attn_q_gated);
    let widths = cfg.per_layer_kv_widths.as_ref().unwrap();
    assert_eq!(widths.len(), 48);
    assert_eq!(widths[3], (4 / 2) * 128); // L3 is full-attn (3+1 % 4 == 0)
    assert_eq!(widths[0], 0); // L0 is GDN
}

/// Synthetic gemma4-v2 (per-layer attn alternation, optional MoE,
/// optional per-layer embd). Most demanding shape — validates the
/// trait holds beyond the qwen pair.
struct Gemma4Shape {
    n_layers: usize,
    hidden: usize,
    intermediate: usize,
    num_heads: usize,
    num_kv_heads: Vec<usize>,
    per_layer_head_dim: Vec<usize>,
    vocab: usize,
    context_length: usize,
    moe: Option<(usize, usize, usize)>, // (num_experts, experts_per_tok, moe_intermediate)
    per_layer_embd: usize,
}

impl KvLayerShape for Gemma4Shape {
    fn num_layers(&self) -> usize {
        self.n_layers
    }
    fn kv_width_at(&self, li: usize, n_ranks: usize) -> usize {
        (self.num_kv_heads[li] / n_ranks) * self.per_layer_head_dim[li]
    }
}

impl ScratchShape for Gemma4Shape {
    fn hidden(&self) -> usize {
        self.hidden
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_seq_len(&self) -> usize {
        self.context_length
    }
    fn intermediate_per_rank(&self, n_ranks: usize) -> usize {
        match self.moe {
            Some((_, _, moe_i)) => moe_i / n_ranks,
            None => self.intermediate / n_ranks,
        }
    }
    fn q_width_per_rank(&self, n_ranks: usize) -> usize {
        self.per_layer_head_dim
            .iter()
            .map(|&hd| (self.num_heads / n_ranks) * hd)
            .max()
            .unwrap_or(0)
    }
    fn moe_per_rank(&self, n_ranks: usize) -> Option<MoeShape> {
        self.moe.map(|(n, k, _)| MoeShape {
            num_experts: n,
            experts_per_tok: k,
            shared_intermediate_per_rank: self.intermediate / n_ranks,
        })
    }
    fn gdn_per_rank(&self, _n_ranks: usize) -> Option<GdnDims> {
        None
    }
    fn attn_q_gated(&self) -> bool {
        false
    }
    fn per_layer_embd(&self) -> usize {
        self.per_layer_embd
    }
}

#[test]
fn scratch_config_for_gemma4_per_layer_max_widths() {
    // 4 layers: SWA (head_dim 256), global (head_dim 256), SWA, global.
    // Same head_dim per layer here; q_width = (num_heads/n_ranks)*256.
    let shape = Gemma4Shape {
        n_layers: 4,
        hidden: 2048,
        intermediate: 8192,
        num_heads: 8,
        num_kv_heads: vec![4, 4, 4, 4],
        per_layer_head_dim: vec![256, 256, 256, 256],
        vocab: 256000,
        context_length: 8192,
        moe: None,
        per_layer_embd: 0,
    };
    let cfg = scratch_config_for(
        &shape,
        ShardMode::Tp {
            rank: 0,
            n_ranks: 2,
        },
        512,
        1,
        None,
    );
    assert_eq!(cfg.q_width, (8 / 2) * 256);
    assert_eq!(cfg.kv_width, (4 / 2) * 256);
    assert_eq!(cfg.intermediate, 8192 / 2);
    assert_eq!(cfg.max_experts, 0);
    assert!(!cfg.attn_q_gated);
    assert!(cfg.gdn.is_none());
    let widths = cfg.per_layer_kv_widths.as_ref().unwrap();
    assert_eq!(widths, &vec![(4 / 2) * 256; 4]);
}

#[test]
fn scratch_config_for_gemma4_per_layer_head_dim_alternation() {
    // Real gemma4 SWA vs global: head_dim_swa=256, head_dim_global=256
    // (currently identical on gemma-4-31B). To exercise the max-over-
    // layers logic, use mismatched dims.
    let shape = Gemma4Shape {
        n_layers: 4,
        hidden: 2048,
        intermediate: 8192,
        num_heads: 8,
        num_kv_heads: vec![2, 4, 2, 4],
        per_layer_head_dim: vec![128, 256, 128, 256],
        vocab: 256000,
        context_length: 8192,
        moe: None,
        per_layer_embd: 0,
    };
    let cfg = scratch_config_for(&shape, ShardMode::Replicated, 512, 1, None);
    // q_width = max over layers = num_heads * max(head_dim) = 8 * 256
    assert_eq!(cfg.q_width, 8 * 256);
    // kv_width = max over per_layer_kv = max(num_kv_heads[li] * head_dim[li])
    // = max(2*128, 4*256, 2*128, 4*256) = 1024
    assert_eq!(cfg.kv_width, 4 * 256);
    let widths = cfg.per_layer_kv_widths.as_ref().unwrap();
    assert_eq!(widths, &vec![2 * 128, 4 * 256, 2 * 128, 4 * 256]);
}
