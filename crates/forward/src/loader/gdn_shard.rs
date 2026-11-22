//! GDN-specific TP shard helpers.
//!
//! The fused-QKV tensors (`attn_qkv`, `ssm_conv1d`) have a [Q | K | V]
//! row layout on disk: Q occupies rows `[0, qk_size)`, K rows
//! `[qk_size, 2·qk_size)`, V rows `[2·qk_size, 2·qk_size + v_size)`,
//! where `qk_size = num_k_heads · head_k_dim` and
//! `v_size = num_v_heads · head_v_dim`. Per-rank packing matches:
//! the kernel reads `[Q_local | K_local | V_local]` in that order.
//!
//! `kq_replicated = true` keeps Q and K full per rank; only V splits.
//! `kq_replicated = false` splits all three.

use anyhow::{bail, Context, Result};
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F32};
use flambeau_quant::{GgmlDType, GgufFile};

use crate::ctx::QuantWeight;

use super::primitives::{dtype_qmatmul_native, f32_to_q8_0_bytes, upload_bytes, wrap_quant};

/// Per-head dims for the fused GDN QKV slab.
#[derive(Copy, Clone, Debug)]
pub struct GdnHeadDims {
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
}

/// Sharding decision for the fused GDN QKV / conv1d slab. `v_runs` are the
/// rank's global V heads as `(first_head, n_heads)` runs, in local order
/// (see [`gdn_v_head_runs`]).
#[derive(Copy, Clone, Debug)]
pub struct GdnShardCtx<'a> {
    pub kq_replicated: bool,
    pub rank: usize,
    pub n_ranks: usize,
    pub v_runs: &'a [(usize, usize)],
}

/// Global V heads owned by TP rank `rank`, as `(first_head, n_heads)` runs
/// in the rank's local head order.
///
/// The state-step kernel pairs local V head `h` with local K head
/// `h / n_rep` when `rep_inner_layout` and `h % H_kv` otherwise. A
/// contiguous V slice satisfies the first pairing, and the second whenever
/// K is replicated. When K is sharded under the modulo pairing (qwen35 27B:
/// 48 V / 16 K heads on 2 ranks), global V head `j` reads K head `j % nk`,
/// so the rank owns every `j` whose `j % nk` falls in its K range; in
/// ascending order the local index then satisfies `h % nk_local ==
/// j % nk - first_k`.
pub fn gdn_v_head_runs(
    num_v_heads: usize,
    num_k_heads: usize,
    kq_replicated: bool,
    rep_inner_layout: bool,
    rank: usize,
    n_ranks: usize,
) -> Result<Vec<(usize, usize)>> {
    if num_v_heads % n_ranks != 0 {
        bail!("gdn: num_v_heads {num_v_heads} not divisible by n_ranks {n_ranks}");
    }
    let v_local = num_v_heads / n_ranks;
    if n_ranks == 1 || kq_replicated || rep_inner_layout {
        return Ok(vec![(rank * v_local, v_local)]);
    }
    if num_k_heads % n_ranks != 0 || num_v_heads % num_k_heads != 0 {
        bail!(
            "gdn: cannot shard {num_v_heads} V / {num_k_heads} K heads over {n_ranks} ranks \
             with modulo head pairing"
        );
    }
    let k_local = num_k_heads / n_ranks;
    Ok((0..num_v_heads / num_k_heads)
        .map(|g| (g * num_k_heads + rank * k_local, k_local))
        .collect())
}

/// Row ranges (in rows of `rows_per_head`) covered by `runs`, offset by `base`.
fn head_row_ranges(
    runs: &[(usize, usize)],
    rows_per_head: usize,
    base: usize,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    runs.iter()
        .map(move |&(h, n)| (base + h * rows_per_head, n * rows_per_head))
}

/// Pack the per-rank [Q | K | V] byte slab. `row_bytes` is the
/// on-disk row stride (inner dim × bytes-per-element, accounting
/// for quant block packing). Returns `(packed_bytes, per_rank_rows)`.
pub(super) fn pack_gdn_qkv_slab(
    name: &str,
    raw: &[u8],
    row_bytes: usize,
    heads: GdnHeadDims,
    shard: GdnShardCtx,
) -> Result<(Vec<u8>, usize)> {
    let GdnHeadDims {
        num_v_heads,
        num_k_heads,
        head_v_dim,
        head_k_dim,
    } = heads;
    let GdnShardCtx {
        kq_replicated,
        rank,
        n_ranks,
        v_runs,
    } = shard;
    let v_part_full = num_v_heads * head_v_dim;
    let k_part_full = num_k_heads * head_k_dim;
    let outer_full = v_part_full + 2 * k_part_full;
    if raw.len() < outer_full * row_bytes {
        bail!(
            "{name}: raw bytes {} < expected {}",
            raw.len(),
            outer_full * row_bytes
        );
    }
    if v_part_full % n_ranks != 0 {
        bail!("{name}: v_part {v_part_full} not divisible by n_ranks {n_ranks}");
    }
    if !kq_replicated && k_part_full % n_ranks != 0 {
        bail!("{name}: k_part {k_part_full} not divisible by n_ranks {n_ranks} (FullShard)");
    }
    let v_local = v_part_full / n_ranks;
    let (q_rows, k_rows, q_off, k_off) = if kq_replicated {
        (k_part_full, k_part_full, 0, k_part_full)
    } else {
        let k_local = k_part_full / n_ranks;
        (
            k_local,
            k_local,
            rank * k_local,
            k_part_full + rank * k_local,
        )
    };
    let per_rank_rows = q_rows + k_rows + v_local;
    let mut packed = Vec::with_capacity(per_rank_rows * row_bytes);
    packed.extend_from_slice(&raw[q_off * row_bytes..(q_off + q_rows) * row_bytes]);
    packed.extend_from_slice(&raw[k_off * row_bytes..(k_off + k_rows) * row_bytes]);
    for (start, len) in head_row_ranges(v_runs, head_v_dim, 2 * k_part_full) {
        packed.extend_from_slice(&raw[start * row_bytes..(start + len) * row_bytes]);
    }
    if packed.len() != per_rank_rows * row_bytes {
        bail!("{name}: V head runs cover {} rows, expected {v_local}", packed.len() / row_bytes - q_rows - k_rows);
    }
    Ok((packed, per_rank_rows))
}

/// Upload a `[conv_channels, hidden]` GDN fused-QKV quant weight,
/// sharded per `kq_replicated`. Native dtypes ride bytes-as-is;
/// F16 / BF16 / F32 fall back to dequant → Q8_0 on host.
pub fn upload_gdn_fused_qkv_quant(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    heads: GdnHeadDims,
    hidden: usize,
    shard: GdnShardCtx<'_>,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let GdnHeadDims {
        num_v_heads,
        num_k_heads,
        head_v_dim,
        head_k_dim,
    } = heads;
    let GdnShardCtx {
        kq_replicated,
        rank,
        n_ranks,
        v_runs,
    } = shard;
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    if dtype_qmatmul_native(info.dtype) {
        let block_size = info.dtype.block_size();
        let type_size = info.dtype.type_size();
        if hidden % block_size != 0 {
            bail!("{name}: hidden {hidden} not block_size {block_size} aligned");
        }
        let row_bytes = (hidden / block_size) * type_size;
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw {name}"))?;
        let (packed, per_rank_rows) = pack_gdn_qkv_slab(name, raw, row_bytes, heads, shard)?;
        let ptr = upload_bytes(device, &packed, allocs)?;
        wrap_quant(ptr, per_rank_rows * hidden, info.dtype)
    } else {
        let v_part_full = num_v_heads * head_v_dim;
        let k_part_full = num_k_heads * head_k_dim;
        let outer_full = v_part_full + 2 * k_part_full;
        let f32_vec = file
            .dequantize_tensor(name)
            .with_context(|| format!("dequantize {name}"))?;
        if f32_vec.len() != outer_full * hidden {
            bail!(
                "{name}: dequant produced {} elems, expected {}",
                f32_vec.len(),
                outer_full * hidden
            );
        }
        if v_part_full % n_ranks != 0 {
            bail!("{name}: v_part {v_part_full} not divisible by n_ranks {n_ranks}");
        }
        if !kq_replicated && k_part_full % n_ranks != 0 {
            bail!("{name}: k_part {k_part_full} not divisible by n_ranks {n_ranks} (FullShard)");
        }
        let v_local = v_part_full / n_ranks;
        let (q_rows, k_rows, q_off, k_off) = if kq_replicated {
            (k_part_full, k_part_full, 0, k_part_full)
        } else {
            let k_local = k_part_full / n_ranks;
            (
                k_local,
                k_local,
                rank * k_local,
                k_part_full + rank * k_local,
            )
        };
        let per_rank_rows = q_rows + k_rows + v_local;
        let mut packed_f32: Vec<f32> = Vec::with_capacity(per_rank_rows * hidden);
        packed_f32.extend_from_slice(&f32_vec[q_off * hidden..(q_off + q_rows) * hidden]);
        packed_f32.extend_from_slice(&f32_vec[k_off * hidden..(k_off + k_rows) * hidden]);
        for (start, len) in head_row_ranges(v_runs, head_v_dim, 2 * k_part_full) {
            packed_f32.extend_from_slice(&f32_vec[start * hidden..(start + len) * hidden]);
        }
        if packed_f32.len() != per_rank_rows * hidden {
            bail!("{name}: V head runs do not cover {v_local} rows");
        }
        let bytes = f32_to_q8_0_bytes(name, &packed_f32)?;
        let ptr = upload_bytes(device, &bytes, allocs)?;
        wrap_quant(ptr, per_rank_rows * hidden, GgmlDType::Q8_0)
    }
}

/// Upload a `[conv_channels, conv_kernel]` F32 GDN fused-QKV tensor
/// (`ssm_conv1d`), sharded per `kq_replicated`.
pub fn upload_gdn_fused_qkv_f32(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    heads: GdnHeadDims,
    conv_kernel: usize,
    shard: GdnShardCtx<'_>,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    let row_bytes = conv_kernel * 4;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let (packed, per_rank_rows) = pack_gdn_qkv_slab(name, raw, row_bytes, heads, shard)?;
    let bytes = packed.len();
    let ptr = device.alloc(bytes).context("alloc gdn conv1d shard")?;
    let stream = device.default_stream();
    // SAFETY: `ptr` owns `bytes`; `packed` is `bytes` host F32 bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(packed.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync gdn conv1d upload")?;
    allocs.push((ptr, bytes));
    let n_elems = per_rank_rows * conv_kernel;
    Ok(unsafe { Tensor::<F32>::from_raw(ptr, n_elems) })
}

/// Upload the per-head entries of a 1D F32 tensor (`ssm_dt.bias`, `ssm_a`)
/// for the V heads in `head_runs`, in run order.
pub fn upload_f32_heads(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    full_len: usize,
    head_runs: &[(usize, usize)],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != full_len {
        bail!(
            "{name}: dequant produced {} elems, expected {full_len}",
            f32_vec.len()
        );
    }
    let mut picked: Vec<f32> = Vec::new();
    for (start, len) in head_row_ranges(head_runs, 1, 0) {
        picked.extend_from_slice(&f32_vec[start..start + len]);
    }
    let bytes = picked.len() * 4;
    let ptr = device.alloc(bytes).context("alloc F32 head shard")?;
    let stream = device.default_stream();
    // SAFETY: `ptr` owns `bytes`; `picked` is `bytes` host F32 bytes and
    // outlives the synchronize below.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(picked.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync F32 head shard upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F32>::from_raw(ptr, picked.len()) })
}

/// Upload the rows of a `[n_heads * rows_per_head, n_cols]` quant weight
/// belonging to the V heads in `head_runs` (`attn_gate`: `rows_per_head =
/// head_v_dim`; `ssm_alpha` / `ssm_beta`: 1).
pub fn upload_head_rows_quant(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    n_cols: usize,
    rows_per_head: usize,
    head_runs: &[(usize, usize)],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let n_rows: usize = head_runs.iter().map(|&(_, n)| n * rows_per_head).sum();
    if dtype_qmatmul_native(info.dtype) {
        let block_size = info.dtype.block_size();
        if n_cols % block_size != 0 {
            bail!("{name}: n_cols {n_cols} not divisible by block_size {block_size}");
        }
        let row_bytes = n_cols / block_size * info.dtype.type_size();
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw {name}"))?;
        let mut packed = Vec::with_capacity(n_rows * row_bytes);
        for (start, len) in head_row_ranges(head_runs, rows_per_head, 0) {
            let range = start * row_bytes..(start + len) * row_bytes;
            if range.end > raw.len() {
                bail!("{name}: head rows end at byte {} > {}", range.end, raw.len());
            }
            packed.extend_from_slice(&raw[range]);
        }
        let ptr = upload_bytes(device, &packed, allocs)?;
        wrap_quant(ptr, n_rows * n_cols, info.dtype)
    } else {
        let f32_vec = file
            .dequantize_tensor(name)
            .with_context(|| format!("dequantize {name}"))?;
        let mut packed: Vec<f32> = Vec::with_capacity(n_rows * n_cols);
        for (start, len) in head_row_ranges(head_runs, rows_per_head, 0) {
            let range = start * n_cols..(start + len) * n_cols;
            if range.end > f32_vec.len() {
                bail!("{name}: head rows end at elem {} > {}", range.end, f32_vec.len());
            }
            packed.extend_from_slice(&f32_vec[range]);
        }
        let bytes = f32_to_q8_0_bytes(name, &packed)?;
        let ptr = upload_bytes(device, &bytes, allocs)?;
        wrap_quant(ptr, n_rows * n_cols, GgmlDType::Q8_0)
    }
}

/// Upload the columns of a `[n_rows, n_heads * cols_per_head]` quant weight
/// belonging to the V heads in `head_runs` (`ssm_out`, row-sharded along
/// `d_inner`). Each run must cover whole quant blocks.
pub fn upload_head_cols_quant(
    file: &GgufFile,
    device: &impl Device,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    cols_per_head: usize,
    head_runs: &[(usize, usize)],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let cols_local: usize = head_runs.iter().map(|&(_, n)| n * cols_per_head).sum();
    if dtype_qmatmul_native(info.dtype) {
        let block_size = info.dtype.block_size();
        let type_size = info.dtype.type_size();
        if n_cols % block_size != 0 {
            bail!("{name}: n_cols {n_cols} not divisible by block_size {block_size}");
        }
        let row_bytes = n_cols / block_size * type_size;
        let byte_runs: Vec<(usize, usize)> = head_row_ranges(head_runs, cols_per_head, 0)
            .map(|(start, len)| {
                if start % block_size != 0 || len % block_size != 0 {
                    bail!(
                        "{name}: V head run [{start}, {}) splits a {block_size}-element quant block",
                        start + len
                    );
                }
                Ok((start / block_size * type_size, len / block_size * type_size))
            })
            .collect::<Result<_>>()?;
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw {name}"))?;
        if raw.len() < n_rows * row_bytes {
            bail!("{name}: raw bytes {} < expected {}", raw.len(), n_rows * row_bytes);
        }
        let mut packed: Vec<u8> = Vec::with_capacity(n_rows * cols_local / block_size * type_size);
        for r in 0..n_rows {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            for &(off, len) in &byte_runs {
                packed.extend_from_slice(&row[off..off + len]);
            }
        }
        let ptr = upload_bytes(device, &packed, allocs)?;
        wrap_quant(ptr, n_rows * cols_local, info.dtype)
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
        let mut packed: Vec<f32> = Vec::with_capacity(n_rows * cols_local);
        for r in 0..n_rows {
            let row = &f32_vec[r * n_cols..(r + 1) * n_cols];
            for (start, len) in head_row_ranges(head_runs, cols_per_head, 0) {
                packed.extend_from_slice(&row[start..start + len]);
            }
        }
        let bytes = f32_to_q8_0_bytes(name, &packed)?;
        let ptr = upload_bytes(device, &bytes, allocs)?;
        wrap_quant(ptr, n_rows * cols_local, GgmlDType::Q8_0)
    }
}
