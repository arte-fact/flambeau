#![cfg(feature = "hip")]

use flambeau_forward::{per_layer_kv_widths, KvLayerShape};

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
