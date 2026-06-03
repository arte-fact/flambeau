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
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F32};
use flambeau_quant::{GgmlDType, GgufFile};

use crate::ctx::QuantWeight;

use super::primitives::{dtype_qmatmul_native, f32_to_q8_0_bytes, upload_bytes, wrap_quant};

/// Pack the per-rank [Q | K | V] byte slab. `row_bytes` is the
/// on-disk row stride (inner dim × bytes-per-element, accounting
/// for quant block packing). Returns `(packed_bytes, per_rank_rows)`.
pub(super) fn pack_gdn_qkv_slab(
    name: &str,
    raw: &[u8],
    row_bytes: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim: usize,
    head_k_dim: usize,
    kq_replicated: bool,
    rank: usize,
    n_ranks: usize,
) -> Result<(Vec<u8>, usize)> {
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
    let v_off = 2 * k_part_full + rank * v_local;
    let per_rank_rows = q_rows + k_rows + v_local;
    let mut packed = Vec::with_capacity(per_rank_rows * row_bytes);
    packed.extend_from_slice(&raw[q_off * row_bytes..(q_off + q_rows) * row_bytes]);
    packed.extend_from_slice(&raw[k_off * row_bytes..(k_off + k_rows) * row_bytes]);
    packed.extend_from_slice(&raw[v_off * row_bytes..(v_off + v_local) * row_bytes]);
    Ok((packed, per_rank_rows))
}

/// Upload a `[conv_channels, hidden]` GDN fused-QKV quant weight,
/// sharded per `kq_replicated`. Native dtypes ride bytes-as-is;
/// F16 / BF16 / F32 fall back to dequant → Q8_0 on host.
pub fn upload_gdn_fused_qkv_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim: usize,
    head_k_dim: usize,
    hidden: usize,
    kq_replicated: bool,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
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
        let (packed, per_rank_rows) = pack_gdn_qkv_slab(
            name,
            raw,
            row_bytes,
            num_v_heads,
            num_k_heads,
            head_v_dim,
            head_k_dim,
            kq_replicated,
            rank,
            n_ranks,
        )?;
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
        let v_off = 2 * k_part_full + rank * v_local;
        let per_rank_rows = q_rows + k_rows + v_local;
        let mut packed_f32: Vec<f32> = Vec::with_capacity(per_rank_rows * hidden);
        packed_f32.extend_from_slice(&f32_vec[q_off * hidden..(q_off + q_rows) * hidden]);
        packed_f32.extend_from_slice(&f32_vec[k_off * hidden..(k_off + k_rows) * hidden]);
        packed_f32.extend_from_slice(&f32_vec[v_off * hidden..(v_off + v_local) * hidden]);
        let bytes = f32_to_q8_0_bytes(name, &packed_f32)?;
        let ptr = upload_bytes(device, &bytes, allocs)?;
        wrap_quant(ptr, per_rank_rows * hidden, GgmlDType::Q8_0)
    }
}

/// Upload a `[conv_channels, conv_kernel]` F32 GDN fused-QKV tensor
/// (`ssm_conv1d`), sharded per `kq_replicated`.
pub fn upload_gdn_fused_qkv_f32(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    num_v_heads: usize,
    num_k_heads: usize,
    head_v_dim: usize,
    head_k_dim: usize,
    conv_kernel: usize,
    kq_replicated: bool,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    let row_bytes = conv_kernel * 4;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let (packed, per_rank_rows) = pack_gdn_qkv_slab(
        name,
        raw,
        row_bytes,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        head_k_dim,
        kq_replicated,
        rank,
        n_ranks,
    )?;
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

/// HtoD a contiguous shard of a 1D F32 tensor. `full_len` is the
/// on-disk element count; the rank receives `full_len / n_ranks`
/// elements starting at `rank * (full_len / n_ranks)`.
pub fn upload_f32_array_sharded(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    full_len: usize,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    if full_len % n_ranks != 0 {
        bail!("{name}: full_len {full_len} not divisible by n_ranks {n_ranks}");
    }
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != full_len {
        bail!(
            "{name}: dequant produced {} elems, expected {full_len}",
            f32_vec.len()
        );
    }
    let per_rank = full_len / n_ranks;
    let start = rank * per_rank;
    let slice = &f32_vec[start..start + per_rank];
    let bytes = per_rank * 4;
    let ptr = device.alloc(bytes).context("alloc F32 shard")?;
    let stream = device.default_stream();
    // SAFETY: `ptr` owns `bytes`; `slice` is `bytes` host F32 bytes.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(slice.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync F32 shard upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F32>::from_raw(ptr, per_rank) })
}
