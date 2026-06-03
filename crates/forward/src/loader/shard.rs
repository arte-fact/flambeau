//! Generic quant-weight upload + col/row sharded variants. Each
//! sharded helper falls back to dequant→Q8_0 for non-native dtypes.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::op::QDtype;
use flambeau_core::DevicePtr;
use flambeau_quant::{GgmlDType, GgufFile};

use crate::ctx::QuantWeight;

use super::primitives::{
    dtype_qmatmul_native, f32_to_q8_0_bytes, upload_bytes, upload_dequant_to_f16, upload_raw,
    wrap_quant,
};
use super::ShardMode;

/// Sharded-upload parameters. Used by `upload_col_sharded_quant` and
/// `upload_row_sharded_quant`. `n_rows` / `n_cols` describe the full
/// GGUF tensor; this rank gets the slice along whichever dim matches.
#[derive(Copy, Clone, Debug)]
pub struct ShardSpec {
    pub n_rows: usize,
    pub n_cols: usize,
    pub rank: usize,
    pub n_ranks: usize,
}

/// Upload a GGUF tensor as a `QuantWeight`. Bytes-as-is when
/// `dtype_qmatmul_native`; otherwise host dequant → re-quantise to
/// Q8_0 (the F16 / BF16 / F32 path).
pub fn upload_quant_weight(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let info = file
        .info(name)
        .with_context(|| format!("tensor info {name}"))?;
    if dtype_qmatmul_native(info.dtype) {
        let ptr = upload_raw(file, device, name, allocs)?;
        return wrap_quant(ptr, n_elems, info.dtype);
    }
    // Fallback: F16 / BF16 / F32 → host dequant → re-quantise Q8_0.
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != n_elems {
        bail!(
            "{name}: dequant produced {} elems, expected {n_elems}",
            f32_vec.len()
        );
    }
    let q8_0_bytes = f32_to_q8_0_bytes(name, &f32_vec)?;
    let ptr = upload_bytes(device, &q8_0_bytes, allocs)?;
    wrap_quant(ptr, n_elems, GgmlDType::Q8_0)
}

/// Upload a host-dequantised tensor as an F16 `QuantWeight`. The
/// returned weight dispatches through `ops.dense_gemv_f16_f16` (no
/// per-call activation re-quantise to Q8_1). Use for small dense
/// weights where the F16 path is faster than mmvq + quantise — e.g.
/// MoE router (`ffn_gate_inp`).
pub fn upload_router_f16(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let tensor = upload_dequant_to_f16(file, device, name, n_elems, allocs)?;
    Ok(QuantWeight {
        ptr: tensor.ptr,
        dtype: QDtype::F16,
        n_elems,
    })
}

/// Col-shard along GGUF dim-0 (output rows). Native dtypes ride a
/// contiguous byte slice per rank; F16 / BF16 / F32 fall back to
/// dequant → row-slice → Q8_0.
pub fn upload_col_sharded_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    spec: ShardSpec,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let ShardSpec {
        n_rows,
        n_cols,
        rank,
        n_ranks,
    } = spec;
    if n_rows % n_ranks != 0 {
        bail!("{name}: n_rows {n_rows} not divisible by n_ranks {n_ranks}");
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let rows_per_rank = n_rows / n_ranks;
    if dtype_qmatmul_native(info.dtype) {
        let block_size = info.dtype.block_size();
        let type_size = info.dtype.type_size();
        if n_cols % block_size != 0 {
            bail!("{name}: n_cols {n_cols} not divisible by block_size {block_size}");
        }
        let row_bytes = n_cols / block_size * type_size;
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw {name}"))?;
        let expected_total = n_rows * row_bytes;
        // `<`, not `!=`: qwen35's gated `attn_q` has 2× rows on disk
        // (Q rows then a sigmoid-gate slab); the dense-attn composite
        // reads only the Q-half via `n_rows = n_heads * head_dim`.
        if raw.len() < expected_total {
            bail!(
                "{name}: raw bytes {} < expected {expected_total} (n_rows={n_rows}, row_bytes={row_bytes})",
                raw.len()
            );
        }
        let start = rank * rows_per_rank * row_bytes;
        let end = start + rows_per_rank * row_bytes;
        let ptr = upload_bytes(device, &raw[start..end], allocs)?;
        wrap_quant(ptr, rows_per_rank * n_cols, info.dtype)
    } else {
        let f32_vec = file
            .dequantize_tensor(name)
            .with_context(|| format!("dequantize {name}"))?;
        if f32_vec.len() != n_rows * n_cols {
            bail!(
                "{name}: dequant produced {} elems, expected {}",
                f32_vec.len(),
                n_rows * n_cols
            );
        }
        let start = rank * rows_per_rank * n_cols;
        let end = start + rows_per_rank * n_cols;
        let bytes = f32_to_q8_0_bytes(name, &f32_vec[start..end])?;
        let ptr = upload_bytes(device, &bytes, allocs)?;
        wrap_quant(ptr, rows_per_rank * n_cols, GgmlDType::Q8_0)
    }
}

/// Row-shard along GGUF dim-1 (input cols). Same dequant-fallback
/// policy as `upload_col_sharded_quant`.
pub fn upload_row_sharded_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    spec: ShardSpec,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let ShardSpec {
        n_rows,
        n_cols,
        rank,
        n_ranks,
    } = spec;
    if n_cols % n_ranks != 0 {
        bail!("{name}: n_cols {n_cols} not divisible by n_ranks {n_ranks}");
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let cols_per_rank = n_cols / n_ranks;
    if dtype_qmatmul_native(info.dtype) {
        let block_size = info.dtype.block_size();
        let type_size = info.dtype.type_size();
        if cols_per_rank % block_size != 0 {
            bail!("{name}: cols_per_rank {cols_per_rank} not divisible by block_size {block_size}");
        }
        let row_bytes = n_cols / block_size * type_size;
        let half_row_bytes = cols_per_rank / block_size * type_size;
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw {name}"))?;
        let mut shard: Vec<u8> = Vec::with_capacity(n_rows * half_row_bytes);
        for r in 0..n_rows {
            let row_start = r * row_bytes;
            let col_offset = rank * half_row_bytes;
            shard.extend_from_slice(
                &raw[row_start + col_offset..row_start + col_offset + half_row_bytes],
            );
        }
        let ptr = upload_bytes(device, &shard, allocs)?;
        wrap_quant(ptr, n_rows * cols_per_rank, info.dtype)
    } else {
        let f32_vec = file
            .dequantize_tensor(name)
            .with_context(|| format!("dequantize {name}"))?;
        if f32_vec.len() != n_rows * n_cols {
            bail!(
                "{name}: dequant produced {} elems, expected {}",
                f32_vec.len(),
                n_rows * n_cols
            );
        }
        let col_offset = rank * cols_per_rank;
        let mut shard_f32: Vec<f32> = Vec::with_capacity(n_rows * cols_per_rank);
        for r in 0..n_rows {
            let row_start = r * n_cols;
            shard_f32.extend_from_slice(
                &f32_vec[row_start + col_offset..row_start + col_offset + cols_per_rank],
            );
        }
        let bytes = f32_to_q8_0_bytes(name, &shard_f32)?;
        let ptr = upload_bytes(device, &bytes, allocs)?;
        wrap_quant(ptr, n_rows * cols_per_rank, GgmlDType::Q8_0)
    }
}

/// Replicate or col-shard along dim-0 based on `shard`.
pub(super) fn upload_col(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    match shard {
        ShardMode::Replicated => upload_quant_weight(file, device, name, n_rows * n_cols, allocs),
        ShardMode::Tp { rank, n_ranks } => upload_col_sharded_quant(
            file,
            device,
            name,
            ShardSpec {
                n_rows,
                n_cols,
                rank,
                n_ranks,
            },
            allocs,
        ),
    }
}

/// Replicate or row-shard along dim-1 based on `shard`.
pub(super) fn upload_row(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    match shard {
        ShardMode::Replicated => upload_quant_weight(file, device, name, n_rows * n_cols, allocs),
        ShardMode::Tp { rank, n_ranks } => upload_row_sharded_quant(
            file,
            device,
            name,
            ShardSpec {
                n_rows,
                n_cols,
                rank,
                n_ranks,
            },
            allocs,
        ),
    }
}
