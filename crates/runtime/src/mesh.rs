//! Multi-device topology: `Mesh<N>`, rank ids, collective ops.
//! A `Mesh` is `N` ordered compute ranks cooperating on one model. `Mesh<1>`
//! is the degenerate single-GPU case — every downstream component passes
//! through the same trait surface as `Mesh<4>`, no `if N == 1` branching.
//! The collective ops (`AllReduce`, `AllGather`, `AllToAll`, `Broadcast`) are
//! op-trait-shaped so the dispatch story is identical to `QMatMul`: one
//! contract per op, many impls (host-bounce CPU reference here, RCCL in
//! `backend-hip`, later NCCL on CUDA).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RankId(pub u32);

impl fmt::Display for RankId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rank{}", self.0)
    }
}

/// A mesh describes the size and per-rank identity of a collective group.
/// V1 ships a single flat mesh. Multi-dimensional sub-meshes (TP × PP, etc.)
/// are a V2 concern; the trait shape here must not assume flatness beyond
/// `rank_count`.
pub trait Mesh: Send + Sync + 'static {
    /// Number of ranks cooperating.
    fn rank_count(&self) -> u32;

    /// A backend-specific label, used in error messages ("hip-mesh", "cpu-ref").
    fn backend(&self) -> &'static str;
}

/// Reduction operator for `AllReduce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

/// Dtype tag for collectives — intentionally narrow: Qwen3.6 V1 only needs
/// F32 (residuals, router logits) and F16 (activations, experts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveDType {
    F32,
    F16,
}

impl CollectiveDType {
    pub fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Shared cfg for collective launches.
#[derive(Debug, Clone, Copy)]
pub struct CollectiveCfg {
    pub elem_count: usize,
    pub dtype: CollectiveDType,
    pub op: ReduceOp,
}

impl CollectiveCfg {
    pub fn new(elem_count: usize, dtype: CollectiveDType, op: ReduceOp) -> Self {
        Self { elem_count, dtype, op }
    }

    pub fn buffer_bytes(&self) -> usize {
        self.elem_count * self.dtype.bytes()
    }
}

/// Maps each transformer layer index to the rank that owns its weights +
/// KV / GDN state. Constructed once at load time by `Qwen3MoEShardedModel`.
/// V1 ships the contiguous-block policy: rank `r` owns layers
/// `[r * per, (r+1) * per)` for `per = ceil(num_layers / num_ranks)`. The
/// pattern minimises cross-rank hops per decoded token to exactly
/// `num_ranks - 1` (one hand-off per stage boundary).
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    layer_to_rank: Vec<RankId>,
    num_ranks: u32,
}

impl LayerAssignment {
    /// Contiguous-block assignment: layer `il` lives on rank `il / per`
    /// where `per = ceil(num_layers / num_ranks)`. Last rank gets the
    /// remainder if the division isn't even; for Qwen3.6 (40 layers ÷ 4
    /// ranks) the split is exactly 10/10/10/10.
    pub fn contiguous(num_layers: usize, num_ranks: u32) -> Self {
        assert!(num_ranks >= 1, "num_ranks must be >= 1");
        assert!(num_layers >= num_ranks as usize, "num_layers must be >= num_ranks");
        let r = num_ranks as usize;
        let per = num_layers.div_ceil(r);
        let layer_to_rank = (0..num_layers)
            .map(|il| RankId((il / per).min(r - 1) as u32))
            .collect();
        Self {
            layer_to_rank,
            num_ranks,
        }
    }

    /// Per-rank layer counts explicitly. Layers are assigned contiguously
    /// to ranks in order: rank 0 gets `counts[0]` layers, rank 1 gets
    /// `counts[1]` layers, and so on. Useful for PP load-balancing when
    /// the last rank carries additional non-layer work (output norm + LM
    /// head) and should receive fewer transformer layers to compensate.
    pub fn from_counts(counts: &[u32]) -> Self {
        let num_ranks = counts.len() as u32;
        let num_layers: usize = counts.iter().map(|&c| c as usize).sum();
        assert!(num_ranks >= 1, "need >= 1 rank");
        assert!(num_layers >= 1, "need >= 1 layer total");
        let mut layer_to_rank = Vec::with_capacity(num_layers);
        for (rank, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                layer_to_rank.push(RankId(rank as u32));
            }
        }
        Self { layer_to_rank, num_ranks }
    }

    pub fn num_layers(&self) -> usize {
        self.layer_to_rank.len()
    }

    pub fn num_ranks(&self) -> u32 {
        self.num_ranks
    }

    pub fn rank_for(&self, il: usize) -> RankId {
        self.layer_to_rank[il]
    }

    /// List of layer indices owned by `rank`, in ascending order.
    pub fn layers_on(&self, rank: RankId) -> Vec<usize> {
        (0..self.layer_to_rank.len())
            .filter(|&il| self.layer_to_rank[il] == rank)
            .collect()
    }

    /// `true` iff rank 0 — the rank that owns `token_embd` and runs the
    /// embedding gather at the start of the pipeline.
    pub fn is_first(&self, rank: RankId) -> bool {
        rank.0 == 0
    }

    /// `true` iff the LAST rank in the mesh — the rank that owns
    /// `output_norm` + `output` (or the tied `token_embd`) and runs the
    /// LM head at the end of the pipeline.
    pub fn is_last(&self, rank: RankId) -> bool {
        rank.0 == self.num_ranks - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_40_layers_4_ranks_is_10_each() {
        let a = LayerAssignment::contiguous(40, 4);
        assert_eq!(a.num_layers(), 40);
        assert_eq!(a.num_ranks(), 4);
        for r in 0..4 {
            assert_eq!(a.layers_on(RankId(r)).len(), 10);
        }
        // First layer on rank 0, last on rank 3.
        assert_eq!(a.rank_for(0), RankId(0));
        assert_eq!(a.rank_for(39), RankId(3));
        // Boundary: layers 9 and 10 live on different ranks.
        assert_eq!(a.rank_for(9), RankId(0));
        assert_eq!(a.rank_for(10), RankId(1));

        assert!(a.is_first(RankId(0)));
        assert!(!a.is_first(RankId(1)));
        assert!(a.is_last(RankId(3)));
        assert!(!a.is_last(RankId(0)));
    }

    #[test]
    fn contiguous_degenerate_single_rank() {
        let a = LayerAssignment::contiguous(40, 1);
        assert!((0..40).all(|il| a.rank_for(il) == RankId(0)));
        assert!(a.is_first(RankId(0)));
        assert!(a.is_last(RankId(0)));
    }

    #[test]
    fn contiguous_uneven_split_41_layers_4_ranks_last_gets_remainder() {
        // per = ceil(41/4) = 11 → ranks 0..=2 get 11 each, rank 3 gets the
        // remaining 8. Every layer maps to some valid rank in [0, 4).
        let a = LayerAssignment::contiguous(41, 4);
        assert_eq!(a.layers_on(RankId(0)).len(), 11);
        assert_eq!(a.layers_on(RankId(1)).len(), 11);
        assert_eq!(a.layers_on(RankId(2)).len(), 11);
        assert_eq!(a.layers_on(RankId(3)).len(), 8);
    }
}
