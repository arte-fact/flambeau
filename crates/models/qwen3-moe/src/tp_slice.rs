//! TP-1b — host-side weight slicing for TP-sharded uploads.
//!
//! Given a [`WeightLayout`] and a rank, produce the byte slice that
//! belongs to that rank. The actual H2D upload is the caller's
//! responsibility (TP-1c stitches this into the model loader).
//!
//! ## Slicing semantics
//!
//! GGUF stores 2-D tensors as `dims = [outer, inner]` with the outer
//! dim *outermost* and the inner dim *contiguous*. Per-row stride is
//! `(inner / block_size) × type_size` bytes. Quantised dtypes pack
//! `block_size` elements (32 for Q*_0/Q*_1/Q8_0, 256 for K-quants)
//! into a single header-prefixed block.
//!
//! - [`WeightLayout::Replicated`] — return the full mmap slice.
//! - [`WeightLayout::ColParallel`] with `dim = 0` — return a contiguous
//!   row range `[rank·outer/world, (rank+1)·outer/world)`. Borrowed
//!   from the mmap (zero-copy).
//! - [`WeightLayout::RowParallel`] with `dim = 1` — return a packed
//!   per-rank buffer: for each row, copy the column subrange
//!   `[rank·inner/world, (rank+1)·inner/world)`. Owned because the
//!   slice isn't contiguous in the mmap.
//!
//! Other (`dim` value, layout) combinations are rejected as
//! [`SliceError::UnsupportedAxis`] — keeps the surface honest while
//! the model surface is small. TP-4 (MoE 3-D tensors) will extend
//! the dim list when needed.
//!
//! ## Quantised-tensor block alignment
//!
//! ColParallel: rows are block-aligned by construction; any multiple-
//! of-`world` row count is also a multiple of the block alignment, so
//! per-rank slice always starts on a row boundary which is also a
//! block boundary.
//!
//! RowParallel: per-rank inner length is `inner / world`. We require
//! `(inner / world) % block_size == 0` so the per-rank slice ends on
//! a block boundary. For Qwen3.5-27B at world=4: `q_width=8192/4=2048
//! % 32 == 0` and `intermediate=27648/4=6912 % 32 == 0`. ✓

use std::borrow::Cow;

use anyhow::{anyhow, Context, Result};
use flambeau_quant::GgufFile;
use flambeau_runtime::WeightLayout;
use thiserror::Error;

/// Slicing error — distinct from `anyhow` so callers can match on
/// "unsupported axis" vs "block misalignment".
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SliceError {
    #[error(
        "TP slicing along dim {dim} for layout {layout} is not implemented \
         (only ColParallel{{dim=0}} and RowParallel{{dim=1}} are wired)"
    )]
    UnsupportedAxis { layout: String, dim: usize },
    #[error(
        "tensor `{name}` outer dim {outer} is not divisible by world {world}"
    )]
    OuterIndivisible { name: String, outer: usize, world: u32 },
    #[error(
        "tensor `{name}` inner dim {inner} is not divisible by world {world}"
    )]
    InnerIndivisible { name: String, inner: usize, world: u32 },
    #[error(
        "tensor `{name}` per-rank inner length {per_rank_inner} is not a \
         multiple of dtype block_size {block_size}"
    )]
    InnerBlockMisaligned {
        name: String,
        per_rank_inner: usize,
        block_size: usize,
    },
    #[error("tensor `{name}` is not 2-D (has {ndims} dims); TP slicing requires 2-D")]
    NotTwoDimensional { name: String, ndims: usize },
    #[error("rank {rank} is out of range for world {world}")]
    RankOutOfRange { rank: u32, world: u32 },
}

/// Bytes belonging to `rank` after applying `layout` to tensor `name`.
///
/// Borrowed (zero-copy) for Replicated and ColParallel; Owned for
/// RowParallel (the per-rank slice isn't contiguous in mmap, so we
/// pack on the host).
///
/// # Errors
/// - [`SliceError`] for layout / divisibility / alignment violations.
/// - Propagated `anyhow::Error` from the underlying `GgufFile` accessors
///   (unknown tensor, mmap truncated, etc.).
pub fn slice_for_tp<'a>(
    file: &'a GgufFile,
    name: &str,
    layout: WeightLayout,
    rank: u32,
) -> Result<Cow<'a, [u8]>> {
    match layout {
        WeightLayout::Replicated => {
            let raw = file
                .tensor_raw(name)
                .with_context(|| format!("tensor_raw `{name}`"))?;
            Ok(Cow::Borrowed(raw))
        }
        WeightLayout::ColParallel { world, dim: 0 } => {
            slice_col_parallel_dim0(file, name, world, rank)
        }
        WeightLayout::RowParallel { world, dim: 1 } => {
            slice_row_parallel_dim1(file, name, world, rank)
        }
        // **TP-4b** — MoE expert tensors are 3D `[n_experts, dim1, dim2]`.
        // Sharding within an expert means slicing dim 1 (ffn_gate/up_exps)
        // or dim 2 (ffn_down_exps).
        WeightLayout::ColParallel { world, dim: 1 } => {
            slice_col_parallel_dim1_3d(file, name, world, rank)
        }
        WeightLayout::RowParallel { world, dim: 2 } => {
            slice_row_parallel_dim2_3d(file, name, world, rank)
        }
        WeightLayout::FusedQkvParallel {
            world,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            kq_replicated,
        } => slice_fused_qkv_parallel(
            file,
            name,
            world,
            rank,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            kq_replicated,
        ),
        WeightLayout::ColParallel { dim, .. } => {
            Err(SliceError::UnsupportedAxis {
                layout: "ColParallel".into(),
                dim,
            }
            .into())
        }
        WeightLayout::RowParallel { dim, .. } => {
            Err(SliceError::UnsupportedAxis {
                layout: "RowParallel".into(),
                dim,
            }
            .into())
        }
    }
}

/// Per-rank byte length under `layout` for `name` — useful for
/// allocation sizing and for the cert artefact without materialising
/// the slice.
pub fn slice_bytes_for_tp(file: &GgufFile, name: &str, layout: WeightLayout) -> Result<usize> {
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    let total = info.size_in_bytes() as usize;
    match layout {
        WeightLayout::Replicated => Ok(total),
        WeightLayout::ColParallel { world, .. } | WeightLayout::RowParallel { world, .. } => {
            if total % (world as usize) != 0 {
                return Err(anyhow!(
                    "tensor `{name}` total bytes {total} not divisible by world {world}"
                ));
            }
            Ok(total / world as usize)
        }
        WeightLayout::FusedQkvParallel {
            world,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            kq_replicated,
        } => {
            if !kq_replicated {
                if total % (world as usize) != 0 {
                    return Err(anyhow!(
                        "tensor `{name}` total bytes {total} not divisible by world {world}"
                    ));
                }
                return Ok(total / world as usize);
            }
            // kq_replicated: per-rank rows = (V_part / world) + 2·K_part.
            let v_part_rows = (num_v_heads as usize) * (head_v_dim as usize);
            let k_part_rows = (num_k_heads as usize) * (head_k_dim as usize);
            if v_part_rows % (world as usize) != 0 {
                return Err(anyhow!(
                    "tensor `{name}` V_part rows {v_part_rows} not divisible by world {world}"
                ));
            }
            let outer_full = v_part_rows + 2 * k_part_rows;
            // Derive per-rank bytes from row count × bytes-per-row.
            let row_bytes = total
                .checked_div(outer_full)
                .ok_or_else(|| anyhow!("tensor `{name}` outer_full=0"))?;
            if total != outer_full * row_bytes {
                return Err(anyhow!(
                    "tensor `{name}` total bytes {total} not a multiple of outer_full {outer_full} (row_bytes derivation failed)"
                ));
            }
            let v_local_rows = v_part_rows / (world as usize);
            let per_rank_rows = v_local_rows + 2 * k_part_rows;
            Ok(per_rank_rows * row_bytes)
        }
    }
}

fn slice_col_parallel_dim0<'a>(
    file: &'a GgufFile,
    name: &str,
    world: u32,
    rank: u32,
) -> Result<Cow<'a, [u8]>> {
    if rank >= world {
        return Err(SliceError::RankOutOfRange { rank, world }.into());
    }
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    if info.dims.is_empty() || info.dims.len() > 2 {
        return Err(SliceError::NotTwoDimensional {
            name: name.into(),
            ndims: info.dims.len(),
        }
        .into());
    }
    let outer = info.dims[0] as usize;
    if outer % (world as usize) != 0 {
        return Err(SliceError::OuterIndivisible {
            name: name.into(),
            outer,
            world,
        }
        .into());
    }
    let rows_per_rank = (outer / world as usize) as u64;
    let row_start = (rank as u64) * rows_per_rank;

    // **TP-4a-i3** — 1-D path for per-head 1-D tensors (GDN `ssm_a`,
    // `ssm_dt_bias`, shape `[num_v_heads]`). Slice the full tensor
    // bytes directly: row_count elements × type_size bytes/element.
    if info.dims.len() == 1 {
        let bytes_per_elem = info.type_size() as usize;
        let byte_start = (row_start as usize) * bytes_per_elem;
        let byte_len = (rows_per_rank as usize) * bytes_per_elem;
        let full = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw `{name}` (1-D) rank={rank}"))?;
        if byte_start + byte_len > full.len() {
            return Err(SliceError::OuterIndivisible {
                name: name.into(),
                outer,
                world,
            }
            .into());
        }
        return Ok(Cow::Borrowed(&full[byte_start..byte_start + byte_len]));
    }

    let raw = file
        .tensor_row_range_raw(name, row_start, rows_per_rank)
        .with_context(|| format!("tensor_row_range_raw `{name}` rank={rank}"))?;
    Ok(Cow::Borrowed(raw))
}

/// **TP-4b** — slice a 3D MoE tensor `[n_experts, dim1, dim2]` along
/// dim 1 (ColParallel). For each expert, the rank takes a contiguous
/// row range `[r·dim1/world, (r+1)·dim1/world)` of length `dim2`.
/// Used for `ffn_gate_exps` and `ffn_up_exps` which have shape
/// `[n_experts, moe_intermediate, hidden]` — sharding the per-expert
/// intermediate slab.
fn slice_col_parallel_dim1_3d<'a>(
    file: &'a GgufFile,
    name: &str,
    world: u32,
    rank: u32,
) -> Result<Cow<'a, [u8]>> {
    if rank >= world {
        return Err(SliceError::RankOutOfRange { rank, world }.into());
    }
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    if info.dims.len() != 3 {
        return Err(SliceError::NotTwoDimensional {
            name: name.into(),
            ndims: info.dims.len(),
        }
        .into());
    }
    let n_experts = info.dims[0] as usize;
    let dim1 = info.dims[1] as usize;
    let dim2 = info.dims[2] as usize;
    if dim1 % (world as usize) != 0 {
        return Err(SliceError::OuterIndivisible {
            name: name.into(),
            outer: dim1,
            world,
        }
        .into());
    }
    let block_size = info.block_size() as usize;
    if dim2 % block_size != 0 {
        return Err(SliceError::InnerBlockMisaligned {
            name: name.into(),
            per_rank_inner: dim2,
            block_size,
        }
        .into());
    }
    let type_size = info.type_size() as usize;
    let row_bytes = (dim2 / block_size) * type_size;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;

    let dim1_local = dim1 / (world as usize);
    let rank_row_start = (rank as usize) * dim1_local;
    let expert_slab_bytes = dim1 * row_bytes;
    let total_local_bytes = n_experts * dim1_local * row_bytes;
    let mut packed = Vec::with_capacity(total_local_bytes);
    for e in 0..n_experts {
        let expert_offset = e * expert_slab_bytes;
        let rank_offset = expert_offset + rank_row_start * row_bytes;
        let rank_end = rank_offset + dim1_local * row_bytes;
        if rank_end > raw.len() {
            return Err(anyhow!(
                "tensor `{name}` mmap {} < expected {} (expert {e})",
                raw.len(),
                rank_end
            ));
        }
        packed.extend_from_slice(&raw[rank_offset..rank_end]);
    }
    debug_assert_eq!(packed.len(), total_local_bytes);
    Ok(Cow::Owned(packed))
}

/// **TP-4b** — slice a 3D MoE tensor `[n_experts, dim1, dim2]` along
/// dim 2 (RowParallel). For each expert × each row, the rank takes a
/// contiguous column range `[r·dim2/world, (r+1)·dim2/world)`.
/// Used for `ffn_down_exps` `[n_experts, hidden, moe_intermediate]` —
/// each expert's per-row inner slice corresponds to the input dim of
/// the down projection.
///
/// Block alignment: `dim2 / world` must be a multiple of the dtype's
/// `block_size` (Q4_*/Q5_*/Q8_0 = 32; K-quants = 256).
fn slice_row_parallel_dim2_3d<'a>(
    file: &'a GgufFile,
    name: &str,
    world: u32,
    rank: u32,
) -> Result<Cow<'a, [u8]>> {
    if rank >= world {
        return Err(SliceError::RankOutOfRange { rank, world }.into());
    }
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    if info.dims.len() != 3 {
        return Err(SliceError::NotTwoDimensional {
            name: name.into(),
            ndims: info.dims.len(),
        }
        .into());
    }
    let n_experts = info.dims[0] as usize;
    let dim1 = info.dims[1] as usize;
    let dim2 = info.dims[2] as usize;
    if dim2 % (world as usize) != 0 {
        return Err(SliceError::InnerIndivisible {
            name: name.into(),
            inner: dim2,
            world,
        }
        .into());
    }
    let block_size = info.block_size() as usize;
    let dim2_local = dim2 / (world as usize);
    if dim2_local % block_size != 0 {
        return Err(SliceError::InnerBlockMisaligned {
            name: name.into(),
            per_rank_inner: dim2_local,
            block_size,
        }
        .into());
    }
    let type_size = info.type_size() as usize;
    let full_row_bytes = (dim2 / block_size) * type_size;
    let local_row_bytes = (dim2_local / block_size) * type_size;
    let col_byte_offset = (rank as usize) * local_row_bytes;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;
    let total_local_bytes = n_experts * dim1 * local_row_bytes;
    let mut packed = Vec::with_capacity(total_local_bytes);
    for e in 0..n_experts {
        let expert_offset = e * dim1 * full_row_bytes;
        for row in 0..dim1 {
            let row_full_start = expert_offset + row * full_row_bytes + col_byte_offset;
            let row_full_end = row_full_start + local_row_bytes;
            if row_full_end > raw.len() {
                return Err(anyhow!(
                    "tensor `{name}` mmap {} < expected {} (expert {e} row {row})",
                    raw.len(),
                    row_full_end
                ));
            }
            packed.extend_from_slice(&raw[row_full_start..row_full_end]);
        }
    }
    debug_assert_eq!(packed.len(), total_local_bytes);
    Ok(Cow::Owned(packed))
}

/// **TP-4a** — head-aware fused-QKV permutation slicer.
///
/// Source tensor is `[outer, inner]` with outer dim
/// `outer = V_part_full + 2 · K_part_full` where:
///   - `V_part_full = num_v_heads · head_v_dim`
///   - `K_part_full = num_k_heads · head_k_dim`
///
/// (Q and K share the GDN head shape; total = 1·V + 1·K + 1·Q.)
///
/// Per rank, we want `[V_local | K_local | Q_local]` re-assembled
/// where each sub-slab is the rank's contiguous head slice. For
/// quantised dtypes the rows must already be block-aligned (any
/// 2-D ggml tensor is); the per-sub-slab row counts must each
/// individually divide cleanly by `world`.
fn slice_fused_qkv_parallel<'a>(
    file: &'a GgufFile,
    name: &str,
    world: u32,
    rank: u32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    head_k_dim: u32,
    kq_replicated: bool,
) -> Result<Cow<'a, [u8]>> {
    if rank >= world {
        return Err(SliceError::RankOutOfRange { rank, world }.into());
    }
    if num_v_heads % world != 0 {
        return Err(anyhow!(
            "FusedQkvParallel: num_v_heads {num_v_heads} not divisible by world {world}"
        ));
    }
    if !kq_replicated && num_k_heads % world != 0 {
        return Err(anyhow!(
            "FusedQkvParallel: num_k_heads {num_k_heads} not divisible by world {world}"
        ));
    }
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    if info.dims.len() != 2 {
        return Err(SliceError::NotTwoDimensional {
            name: name.into(),
            ndims: info.dims.len(),
        }
        .into());
    }
    let outer_full = info.dims[0] as usize;
    let v_part_full = (num_v_heads as usize) * (head_v_dim as usize);
    let k_part_full = (num_k_heads as usize) * (head_k_dim as usize);
    let expect_outer = v_part_full + 2 * k_part_full;
    if outer_full != expect_outer {
        return Err(anyhow!(
            "FusedQkvParallel: tensor `{name}` outer dim {outer_full} != \
             V_part({v_part_full}) + 2·K_part({k_part_full}) = {expect_outer}"
        ));
    }
    let inner = info.dims[1] as usize;
    let block_size = info.block_size() as usize;
    if inner % block_size != 0 {
        return Err(SliceError::InnerBlockMisaligned {
            name: name.into(),
            per_rank_inner: inner,
            block_size,
        }
        .into());
    }
    let type_size = info.type_size() as usize;
    let row_bytes = (inner / block_size) * type_size;

    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;
    if raw.len() < outer_full * row_bytes {
        return Err(anyhow!(
            "tensor `{name}` mmap {} < expected {}",
            raw.len(),
            outer_full * row_bytes
        ));
    }

    // **Bug 4 fix** — on-disk QKV layout is `[Q | K | V]`, NOT `[V | K | Q]`.
    // Verified empirically: world=1 is bit-correct vs llama.cpp+candle on
    // Qwen3.5-9B (argmax=11 for seed 9419), and at world=1 this slicer is
    // a no-op pass-through (all sub-slices sum to the whole tensor in
    // on-disk order). The forward kernel reads the per-rank slab as
    // `[Q@0 | K@local_qk | V@2*local_qk]` (gdn_tp.rs:289-293), which
    // implies the slab order — and therefore the on-disk order, since
    // world=1 is identity — is `[Q | K | V]`. The previous code labeled
    // sub-slabs as if on-disk were `[V | K | Q]`, so per-rank slabs at
    // world>1 were filled with bytes from the wrong on-disk regions.
    //
    // **TP-4d-i3** — `kq_replicated=true` keeps Q and K full per rank
    // (rep_outer arches qwen35moe / qwen36moe). Only V is split.
    let r = rank as usize;
    let v_local_rows = v_part_full / (world as usize);

    // On-disk row offsets (same regardless of kq_replicated):
    //   Q rows live at [0, k_part_full)               (Q has the same shape as K)
    //   K rows live at [k_part_full, 2*k_part_full)
    //   V rows live at [2*k_part_full, outer_full)
    let (q_rows, k_rows, q_offset_rows, k_offset_rows) = if kq_replicated {
        // K and Q full per rank.
        (k_part_full, k_part_full, 0, k_part_full)
    } else {
        let k_local_rows = k_part_full / (world as usize);
        (
            k_local_rows,
            k_local_rows,
            r * k_local_rows,
            k_part_full + r * k_local_rows,
        )
    };
    let v_offset_rows = 2 * k_part_full + r * v_local_rows;

    let q_bytes = q_rows * row_bytes;
    let k_bytes = k_rows * row_bytes;
    let v_bytes = v_local_rows * row_bytes;
    let total_local_bytes = q_bytes + k_bytes + v_bytes;

    // Pack per-rank slab in `[Q_local | K_local | V_local]` order — the
    // order the GDN forward kernel expects (gdn_tp.rs:289-293).
    let mut packed = Vec::with_capacity(total_local_bytes);
    packed
        .extend_from_slice(&raw[q_offset_rows * row_bytes..(q_offset_rows + q_rows) * row_bytes]);
    packed
        .extend_from_slice(&raw[k_offset_rows * row_bytes..(k_offset_rows + k_rows) * row_bytes]);
    packed.extend_from_slice(
        &raw[v_offset_rows * row_bytes..(v_offset_rows + v_local_rows) * row_bytes],
    );
    debug_assert_eq!(packed.len(), total_local_bytes);
    Ok(Cow::Owned(packed))
}

fn slice_row_parallel_dim1<'a>(
    file: &'a GgufFile,
    name: &str,
    world: u32,
    rank: u32,
) -> Result<Cow<'a, [u8]>> {
    if rank >= world {
        return Err(SliceError::RankOutOfRange { rank, world }.into());
    }
    let info = file
        .info(name)
        .with_context(|| format!("info `{name}`"))?;
    if info.dims.len() != 2 {
        return Err(SliceError::NotTwoDimensional {
            name: name.into(),
            ndims: info.dims.len(),
        }
        .into());
    }
    let outer = info.dims[0] as usize;
    let inner = info.dims[1] as usize;
    if inner % (world as usize) != 0 {
        return Err(SliceError::InnerIndivisible {
            name: name.into(),
            inner,
            world,
        }
        .into());
    }
    let block_size = info.block_size() as usize;
    let per_rank_inner = inner / world as usize;
    if per_rank_inner % block_size != 0 {
        return Err(SliceError::InnerBlockMisaligned {
            name: name.into(),
            per_rank_inner,
            block_size,
        }
        .into());
    }
    let type_size = info.type_size() as usize;
    let full_row_blocks = inner / block_size;
    let full_row_bytes = full_row_blocks * type_size;
    let per_rank_row_blocks = per_rank_inner / block_size;
    let per_rank_row_bytes = per_rank_row_blocks * type_size;
    let col_block_offset = (rank as usize) * per_rank_row_blocks;
    let col_byte_offset = col_block_offset * type_size;

    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw `{name}`"))?;
    if raw.len() < outer * full_row_bytes {
        return Err(anyhow!(
            "tensor `{name}` mmap {} < expected {}",
            raw.len(),
            outer * full_row_bytes
        ));
    }
    let mut packed = Vec::with_capacity(outer * per_rank_row_bytes);
    for row in 0..outer {
        let row_start = row * full_row_bytes + col_byte_offset;
        let row_end = row_start + per_rank_row_bytes;
        packed.extend_from_slice(&raw[row_start..row_end]);
    }
    Ok(Cow::Owned(packed))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Slicing logic is exercised end-to-end against the real Qwen3.5-27B
    // GGUF in `tests/tp_slice_qwen35_27b.rs` — that harness covers
    // ColParallel{dim=0}, RowParallel{dim=1}, and Replicated round-trips
    // including a byte-for-byte mmap-reconstruction check on RowParallel.
    // Pure unit-level checks here cover only the error-formatting surface.

    #[test]
    fn slice_error_display_outer_indivisible() {
        let err = SliceError::OuterIndivisible {
            name: "blk.0.attn_q.weight".into(),
            outer: 8192,
            world: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("8192"));
        assert!(msg.contains("world 3"));
    }

    #[test]
    fn slice_error_display_inner_block_misaligned() {
        let err = SliceError::InnerBlockMisaligned {
            name: "blk.0.ffn_down.weight".into(),
            per_rank_inner: 17,
            block_size: 32,
        };
        let msg = err.to_string();
        assert!(msg.contains("17"));
        assert!(msg.contains("32"));
    }
}
