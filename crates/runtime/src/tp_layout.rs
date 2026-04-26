//! TP-1a — generic weight-sharding layout for tensor parallelism.
//!
//! [`WeightLayout`] encodes how a single weight tensor is distributed
//! across a TP mesh of `world` ranks. The three variants follow standard
//! Megatron-LM nomenclature:
//!
//! - [`WeightLayout::Replicated`] — every rank holds the full tensor.
//!   Used for 1-D parameters (norms, biases that aren't on a sharded
//!   projection's output dim) and for global tensors small enough to
//!   trade memory for AllReduce-free dispatch (token-embd in V1; later
//!   vocab-shardable).
//! - [`WeightLayout::ColParallel`] — split along the *output* dimension.
//!   Each rank computes `Y_r = X · W_r^T` on the full input and emits
//!   `1/world` of the output rows. No AllReduce required at this stage.
//! - [`WeightLayout::RowParallel`] — split along the *input* dimension.
//!   Each rank computes `Y_r = X_r · W_r^T` on its input slice and emits
//!   a *partial* output. AllReduce-sum on the partial buffer reconstructs
//!   the full output (this is where `BarP2pAllReduce` lives in TP-2).
//!
//! "Output dim" / "input dim" are framework conventions; the concrete
//! tensor axis is recorded in `dim` so the slicing code (TP-1b) knows
//! which axis to cut. For `[rows, cols]` matmul weights stored as
//! `[output_dim=rows, input_dim=cols]`:
//! - ColParallel splits dim 0 (rows = output features).
//! - RowParallel splits dim 1 (cols = input features).
//! For 1-D tensors that *are* sharded (biases on a ColParallel projection
//! output), use ColParallel with `dim = 0`.
//!
//! Model-family-specific tensor-name → layout maps live in the model
//! crate (e.g. `crates/models/qwen3-moe/src/tp_layout.rs`); this module
//! only owns the enum + helpers.

use std::fmt;

/// How a single weight tensor is distributed across the TP mesh.
///
/// Construct via the safe constructors ([`WeightLayout::col_parallel`],
/// [`WeightLayout::row_parallel`], [`WeightLayout::replicated`]) which
/// validate `world >= 1` and reject the meaningless `world = 0` case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Full tensor on every rank. Per-rank bytes = full tensor bytes.
    Replicated,
    /// Sharded along `dim`, output-dim-style. Per-rank bytes =
    /// `total / world` (when divisible).
    ColParallel { world: u32, dim: usize },
    /// Sharded along `dim`, input-dim-style. Per-rank bytes =
    /// `total / world` (when divisible). Caller is responsible for
    /// AllReduce-summing partials produced by RowParallel matmuls.
    RowParallel { world: u32, dim: usize },
    /// **TP-4a** — head-aware permutation slicing for fused-QKV
    /// tensors (GDN's `attn_qkv` and `ssm_conv1d`).
    ///
    /// Source tensor outer dim is `[V_part | K_part | Q_part]` with
    /// `V_part = num_v_heads · head_v_dim`,
    /// `K_part = num_k_heads · head_k_dim`,
    /// `Q_part = num_k_heads · head_k_dim` (Q and K share head shape
    /// in GDN). Flat ColParallel along the outer dim cuts across these
    /// boundaries; this variant slices each sub-slab independently
    /// and re-concatenates per-rank as `[V_local | K_local | Q_local]`.
    ///
    /// Divisibility: `num_v_heads % world == 0` and
    /// `num_k_heads % world == 0`. The slicing function in
    /// `qwen3-moe::tp_slice` validates these at apply time.
    FusedQkvParallel {
        world: u32,
        num_v_heads: u32,
        num_k_heads: u32,
        head_v_dim: u32,
        head_k_dim: u32,
    },
}

impl WeightLayout {
    /// Replicated layout (world is implied by the mesh).
    pub fn replicated() -> Self {
        WeightLayout::Replicated
    }

    /// Build a [`WeightLayout::ColParallel`] after sanity-checking
    /// `world >= 1`.
    pub fn col_parallel(world: u32, dim: usize) -> Self {
        assert!(world >= 1, "world must be >= 1 (got {world})");
        WeightLayout::ColParallel { world, dim }
    }

    /// Build a [`WeightLayout::RowParallel`] after sanity-checking
    /// `world >= 1`.
    pub fn row_parallel(world: u32, dim: usize) -> Self {
        assert!(world >= 1, "world must be >= 1 (got {world})");
        WeightLayout::RowParallel { world, dim }
    }

    /// `true` iff this layout shards the tensor (i.e., per-rank bytes
    /// are smaller than full bytes).
    pub fn is_sharded(self) -> bool {
        matches!(
            self,
            WeightLayout::ColParallel { .. }
                | WeightLayout::RowParallel { .. }
                | WeightLayout::FusedQkvParallel { .. }
        )
    }

    /// `true` iff this layout requires an AllReduce on the *output* of
    /// a matmul that consumes this weight. RowParallel requires AR;
    /// ColParallel, FusedQkvParallel, and Replicated do not (the matmul
    /// output has the per-rank head subset; AR happens later if at all).
    pub fn requires_output_all_reduce(self) -> bool {
        matches!(self, WeightLayout::RowParallel { .. })
    }

    /// Mesh size this layout was built for. `Replicated` is mesh-size-
    /// agnostic at the type level, so this returns `None` in that case
    /// and the caller carries the mesh size externally.
    pub fn world(self) -> Option<u32> {
        match self {
            WeightLayout::Replicated => None,
            WeightLayout::ColParallel { world, .. }
            | WeightLayout::RowParallel { world, .. }
            | WeightLayout::FusedQkvParallel { world, .. } => Some(world),
        }
    }

    /// Axis the layout shards. `Replicated` and `FusedQkvParallel`
    /// (which permutes rather than slicing a single contiguous range)
    /// return `None`.
    pub fn dim(self) -> Option<usize> {
        match self {
            WeightLayout::Replicated | WeightLayout::FusedQkvParallel { .. } => None,
            WeightLayout::ColParallel { dim, .. } | WeightLayout::RowParallel { dim, .. } => {
                Some(dim)
            }
        }
    }

    /// Per-rank length along the sharded axis when the full axis has
    /// `total_along_dim` elements. For [`WeightLayout::Replicated`]
    /// (no shard axis) this returns `total_along_dim` unchanged.
    ///
    /// # Errors
    /// [`LayoutError::Indivisible`] if `total_along_dim % world != 0`.
    /// Sharded layouts must divide cleanly so each rank's slice lands
    /// on a power-of-two ggml block boundary for quantised tensors
    /// (Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 all use 32-element blocks; uniform
    /// world ≤ 8 satisfies this when the unsharded dim is multiple of
    /// 256, which all V1/V2 target shapes are).
    pub fn shard_size(self, total_along_dim: usize) -> Result<usize, LayoutError> {
        match self {
            WeightLayout::Replicated => Ok(total_along_dim),
            WeightLayout::ColParallel { world, .. }
            | WeightLayout::RowParallel { world, .. }
            | WeightLayout::FusedQkvParallel { world, .. } => {
                if total_along_dim % (world as usize) != 0 {
                    Err(LayoutError::Indivisible {
                        total: total_along_dim,
                        world,
                    })
                } else {
                    Ok(total_along_dim / world as usize)
                }
            }
        }
    }

    /// Byte offset of rank `r`'s slice along the sharded axis, given the
    /// elements-per-shard-step `bytes_per_unit` (typically the row stride
    /// for ColParallel, the column stride for RowParallel). For
    /// [`WeightLayout::Replicated`] the offset is always `0`.
    ///
    /// # Errors
    /// [`LayoutError::RankOutOfRange`] if `r >= world`.
    /// [`LayoutError::Indivisible`] propagated from
    /// [`Self::shard_size`].
    pub fn shard_byte_offset(
        self,
        rank: u32,
        total_along_dim: usize,
        bytes_per_unit: usize,
    ) -> Result<usize, LayoutError> {
        match self {
            WeightLayout::Replicated => Ok(0),
            WeightLayout::ColParallel { world, .. } | WeightLayout::RowParallel { world, .. } => {
                if rank >= world {
                    return Err(LayoutError::RankOutOfRange { rank, world });
                }
                let shard = self.shard_size(total_along_dim)?;
                Ok((rank as usize) * shard * bytes_per_unit)
            }
            WeightLayout::FusedQkvParallel { .. } => {
                // Not a single contiguous range — slicing is a
                // permutation. The qwen3-moe::tp_slice path computes
                // V/K/Q sub-slab offsets directly; this method has no
                // single-offset answer to give.
                Err(LayoutError::FusedQkvNoSingleOffset)
            }
        }
    }
}

impl fmt::Display for WeightLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightLayout::Replicated => write!(f, "Replicated"),
            WeightLayout::ColParallel { world, dim } => {
                write!(f, "ColParallel{{world={world},dim={dim}}}")
            }
            WeightLayout::RowParallel { world, dim } => {
                write!(f, "RowParallel{{world={world},dim={dim}}}")
            }
            WeightLayout::FusedQkvParallel {
                world,
                num_v_heads,
                num_k_heads,
                head_v_dim,
                head_k_dim,
            } => write!(
                f,
                "FusedQkvParallel{{world={world},vH={num_v_heads}/{head_v_dim},kH={num_k_heads}/{head_k_dim}}}"
            ),
        }
    }
}

/// Errors produced by [`WeightLayout`] helpers.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    #[error("layout shard {total} along dim with world={world} is not divisible")]
    Indivisible { total: usize, world: u32 },
    #[error("rank {rank} is out of range for world={world}")]
    RankOutOfRange { rank: u32, world: u32 },
    #[error(
        "FusedQkvParallel slicing is a permutation, not a contiguous range; \
         use the slicer in qwen3-moe::tp_slice for V/K/Q sub-slab offsets"
    )]
    FusedQkvNoSingleOffset,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_parallel_divisible_shard_size() {
        let l = WeightLayout::col_parallel(4, 0);
        assert_eq!(l.shard_size(8192).unwrap(), 2048);
        assert_eq!(l.world(), Some(4));
        assert_eq!(l.dim(), Some(0));
        assert!(l.is_sharded());
        assert!(!l.requires_output_all_reduce());
    }

    #[test]
    fn row_parallel_requires_ar() {
        let l = WeightLayout::row_parallel(4, 1);
        assert!(l.is_sharded());
        assert!(l.requires_output_all_reduce());
    }

    #[test]
    fn replicated_is_world_agnostic() {
        let l = WeightLayout::replicated();
        assert_eq!(l.world(), None);
        assert_eq!(l.dim(), None);
        assert!(!l.is_sharded());
        // shard_size of replicated returns the full size.
        assert_eq!(l.shard_size(5120).unwrap(), 5120);
        assert_eq!(l.shard_byte_offset(99, 5120, 2).unwrap(), 0);
    }

    #[test]
    fn indivisible_returns_error() {
        let l = WeightLayout::col_parallel(4, 0);
        assert_eq!(
            l.shard_size(13),
            Err(LayoutError::Indivisible { total: 13, world: 4 })
        );
    }

    #[test]
    fn shard_byte_offset_per_rank() {
        // Qwen3.5-27B attn_q: nQ*head_dim = 64*128 = 8192 rows.
        // World=4: each rank holds 2048 rows. Row stride = hidden=5120
        // F16 = 10240 bytes (caveat: weights are usually quantised — use
        // the quantised row stride at the call site; this test uses raw
        // F16 stride for clarity).
        let l = WeightLayout::col_parallel(4, 0);
        let row_stride = 5120 * 2;
        assert_eq!(l.shard_byte_offset(0, 8192, row_stride).unwrap(), 0);
        assert_eq!(l.shard_byte_offset(1, 8192, row_stride).unwrap(), 2048 * row_stride);
        assert_eq!(l.shard_byte_offset(2, 8192, row_stride).unwrap(), 4096 * row_stride);
        assert_eq!(l.shard_byte_offset(3, 8192, row_stride).unwrap(), 6144 * row_stride);
    }

    #[test]
    fn rank_out_of_range() {
        let l = WeightLayout::col_parallel(4, 0);
        assert_eq!(
            l.shard_byte_offset(4, 8192, 1),
            Err(LayoutError::RankOutOfRange { rank: 4, world: 4 })
        );
    }

    #[test]
    fn display_formats() {
        assert_eq!(WeightLayout::replicated().to_string(), "Replicated");
        assert_eq!(
            WeightLayout::col_parallel(4, 0).to_string(),
            "ColParallel{world=4,dim=0}"
        );
        assert_eq!(
            WeightLayout::row_parallel(2, 1).to_string(),
            "RowParallel{world=2,dim=1}"
        );
    }
}
