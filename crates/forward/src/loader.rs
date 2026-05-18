//! Arch-agnostic GGUF → device helpers. Every helper appends
//! `(DevicePtr, bytes)` to a caller-supplied `&mut Vec` so one
//! disposer walks every alloc.
//!
//! Two layers of API:
//! - byte/tensor primitives (`upload_bytes`, `wrap_quant`,
//!   `upload_*_sharded_quant`) at the bottom — what the existing TP
//!   sharded paths use directly.
//! - **per-layer-kind composers** (`load_dense_attn_layer`,
//!   `load_dense_ffn_layer`, `load_gdn_layer`, `load_embedding`,
//!   `load_lm_head`) on top — what arch-specific model loaders call.
//!   Each composer takes a `ShardMode` and a spec struct; the model
//!   loader is just per-arch tensor-name + per-layer-dim glue.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::{CopyDirection, Device, DevicePtr};
use flambeau_model_ops::{Tensor, F16, F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
use flambeau_quant::{GgmlDType, GgufFile};
use half::f16;

use crate::ctx::QuantWeight;

pub fn upload_bytes(
    device: &HipDevice,
    bytes: &[u8],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let ptr = device.alloc(bytes.len()).context("alloc")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns bytes.len(); bytes is a host slice of the same len.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(bytes.as_ptr() as usize),
            bytes.len(),
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after upload")?;
    allocs.push((ptr, bytes.len()));
    Ok(ptr)
}

pub fn upload_f16_from_f32(
    device: &HipDevice,
    f32_vec: &[f32],
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let f16_vec: Vec<f16> = f32_vec.iter().map(|&v| f16::from_f32(v)).collect();
    let bytes = f16_vec.len() * 2;
    let ptr = device.alloc(bytes).context("alloc F16")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns bytes; f16_vec has the matching byte count.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(f16_vec.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F16>::from_raw(ptr, f16_vec.len()) })
}

/// `(block_size_elems, type_size_bytes)` for GGUF dtypes model-ops's
/// `qmatmul` supports.
pub fn quant_block_info(dtype: GgmlDType) -> Result<(usize, usize)> {
    Ok(match dtype {
        GgmlDType::Q4_0 => (32, 18),
        GgmlDType::Q4_1 => (32, 20),
        GgmlDType::Q5_0 => (32, 22),
        GgmlDType::Q5_1 => (32, 24),
        GgmlDType::Q8_0 => (32, 34),
        other => bail!(
            "quant_block_info: dtype {other:?} not supported by model-ops::qmatmul"
        ),
    })
}

pub fn wrap_quant(ptr: DevicePtr, n_elems: usize, dtype: GgmlDType) -> Result<QuantWeight> {
    Ok(match dtype {
        GgmlDType::Q4_0 => QuantWeight::Q4_0(unsafe { Tensor::<Q4_0>::from_raw(ptr, n_elems) }),
        GgmlDType::Q4_1 => QuantWeight::Q4_1(unsafe { Tensor::<Q4_1>::from_raw(ptr, n_elems) }),
        GgmlDType::Q5_0 => QuantWeight::Q5_0(unsafe { Tensor::<Q5_0>::from_raw(ptr, n_elems) }),
        GgmlDType::Q5_1 => QuantWeight::Q5_1(unsafe { Tensor::<Q5_1>::from_raw(ptr, n_elems) }),
        GgmlDType::Q8_0 => QuantWeight::Q8_0(unsafe { Tensor::<Q8_0>::from_raw(ptr, n_elems) }),
        other => bail!("wrap_quant: dtype {other:?} not supported by model-ops::qmatmul"),
    })
}

pub fn upload_raw(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<DevicePtr> {
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    upload_bytes(device, raw, allocs)
}

/// Host-dequant any tensor → cast to F16 → HtoD. Used for norm
/// weights (F32 in GGUF) and quantised embedding tables.
pub fn upload_dequant_to_f16(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    expected_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F16>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != expected_elems {
        bail!(
            "loader: {name} dequant produced {} elems, expected {expected_elems}",
            f32_vec.len()
        );
    }
    upload_f16_from_f32(device, &f32_vec, allocs)
}

/// Upload a GGUF tensor as a `QuantWeight`. If the tensor's dtype is
/// supported by model-ops's `qmatmul` set, upload bytes-as-is. Else
/// (K-quants / IQ / MXFP4 / F16 / etc.) dequant on host → re-quantise
/// to Q8_0 → upload. Same precedent as the V1 MXFP4 / Iq4_Xs paths.
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
    if matches!(
        info.dtype,
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0
    ) {
        let ptr = upload_raw(file, device, name, allocs)?;
        return wrap_quant(ptr, n_elems, info.dtype);
    }
    // Fallback: host dequant → re-quantise to Q8_0 → upload.
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != n_elems {
        bail!(
            "{name}: dequant produced {} elems, expected {n_elems}",
            f32_vec.len()
        );
    }
    if n_elems % 32 != 0 {
        bail!("{name}: Q8_0 fallback needs n_elems % 32 == 0 (got {n_elems})");
    }
    // The on-disk weight is `[rows, cols]`. The Q8_0 quantiser packs one
    // row at a time; `cols` is known to be a divisor of n_elems but we
    // don't have row-shape info here. Use a single contiguous quantise
    // (matches GGUF's per-block layout — each Q8_0 block is 32 elems,
    // and the row boundary is irrelevant to the matmul).
    let mut q8_0_bytes: Vec<u8> = Vec::with_capacity(n_elems / 32 * 34);
    let mut cursor = 0usize;
    while cursor < n_elems {
        flambeau_quant::quantize_k::quantize_row_q8_0(
            &f32_vec[cursor..cursor + 32],
            &mut q8_0_bytes,
        );
        cursor += 32;
    }
    let ptr = upload_bytes(device, &q8_0_bytes, allocs)?;
    wrap_quant(ptr, n_elems, GgmlDType::Q8_0)
}

/// Quantise an F32 buffer to Q8_0 (32-elem blocks). The buffer must
/// be a multiple of 32 elements long.
fn f32_to_q8_0_bytes(name: &str, f32_buf: &[f32]) -> Result<Vec<u8>> {
    if f32_buf.len() % 32 != 0 {
        bail!(
            "{name}: Q8_0 fallback needs len %% 32 == 0 (got {})",
            f32_buf.len()
        );
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(f32_buf.len() / 32 * 34);
    let mut cursor = 0usize;
    while cursor < f32_buf.len() {
        flambeau_quant::quantize_k::quantize_row_q8_0(
            &f32_buf[cursor..cursor + 32],
            &mut bytes,
        );
        cursor += 32;
    }
    Ok(bytes)
}

/// True when `dtype` rides bytes-as-is through `wrap_quant` (the
/// model-ops `qmatmul` supported set). False → caller must dequant
/// fallback to Q8_0.
fn dtype_qmatmul_native(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0
    )
}

/// Col-shard along GGUF dim-0 (output rows). For native-supported
/// dtypes the rank reads a contiguous byte slice; for K-quants /
/// IQ / MXFP4 the tensor is dequantised on host, row-sliced, and
/// re-quantised to Q8_0 — same fallback as `upload_quant_weight`.
#[allow(clippy::too_many_arguments)]
pub fn upload_col_sharded_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    if n_rows % n_ranks != 0 {
        bail!(
            "{name}: n_rows {n_rows} not divisible by n_ranks {n_ranks}"
        );
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let rows_per_rank = n_rows / n_ranks;
    if dtype_qmatmul_native(info.dtype) {
        let (block_size, type_size) = quant_block_info(info.dtype)?;
        if n_cols % block_size != 0 {
            bail!(
                "{name}: n_cols {n_cols} not divisible by block_size {block_size}"
            );
        }
        let row_bytes = n_cols / block_size * type_size;
        let raw = file
            .tensor_raw(name)
            .with_context(|| format!("tensor_raw {name}"))?;
        let expected_total = n_rows * row_bytes;
        // `>=`, not `==`: qwen35's `attn_q` is [2·n_heads·head_dim, hidden]
        // (Q rows then a sigmoid-gate slab). The shared dense-attn
        // loader interprets `n_rows = n_heads · head_dim` and reads
        // only the Q-half, matching the Replicated path's behavior.
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
#[allow(clippy::too_many_arguments)]
pub fn upload_row_sharded_quant(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    rank: usize,
    n_ranks: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<QuantWeight> {
    if n_cols % n_ranks != 0 {
        bail!("{name}: n_cols {n_cols} not divisible by n_ranks {n_ranks}");
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    let cols_per_rank = n_cols / n_ranks;
    if dtype_qmatmul_native(info.dtype) {
        let (block_size, type_size) = quant_block_info(info.dtype)?;
        if cols_per_rank % block_size != 0 {
            bail!(
                "{name}: cols_per_rank {cols_per_rank} not divisible by block_size {block_size}"
            );
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

// =============================================================================
// Per-layer-kind composers — used by arch-specific model loaders.
// =============================================================================

use crate::ctx::{
    Activation, AttnWeights, EmbeddingWeights, FfnWeights, GdnDims, GdnWeights, LmHeadWeights,
};

/// How a matmul weight gets uploaded.
#[derive(Clone, Copy, Debug)]
pub enum ShardMode {
    Replicated,
    Tp { rank: usize, n_ranks: usize },
}

impl ShardMode {
    pub fn n_ranks(self) -> usize {
        match self {
            ShardMode::Replicated => 1,
            ShardMode::Tp { n_ranks, .. } => n_ranks,
        }
    }
}

/// Upload a `[n_rows, n_cols]` quant weight either bytes-as-is
/// (Replicated) or col-sharded along GGUF dim-0 (Tp).
fn upload_col(
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
            file, device, name, n_rows, n_cols, rank, n_ranks, allocs,
        ),
    }
}

/// Upload a `[n_rows, n_cols]` quant weight either bytes-as-is
/// (Replicated) or row-sharded along GGUF dim-1 (Tp).
fn upload_row(
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
            file, device, name, n_rows, n_cols, rank, n_ranks, allocs,
        ),
    }
}

// ---- GDN TP shard helpers ----------------------------------------------------
//
// The fused-QKV tensors (`attn_qkv`, `ssm_conv1d`) have a [Q | K | V]
// row layout on disk: Q occupies rows `[0, qk_size)`, K rows
// `[qk_size, 2·qk_size)`, V rows `[2·qk_size, 2·qk_size + v_size)`,
// where `qk_size = num_k_heads · head_k_dim` and
// `v_size = num_v_heads · head_v_dim`. Per-rank packing matches: the
// kernel reads `[Q_local | K_local | V_local]` in that order.
//
// `kq_replicated = true` keeps Q and K full per rank; only V splits
// `kq_replicated = false` splits all three.

/// Compute the per-rank packed [Q | K | V] byte slab. `row_bytes` is
/// the on-disk row stride (inner dim × bytes-per-element, accounting
/// for quant block packing). Returns the packed bytes + per-rank row
/// count.
#[allow(clippy::too_many_arguments)]
fn pack_gdn_qkv_slab(
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
        bail!(
            "{name}: k_part {k_part_full} not divisible by n_ranks {n_ranks} (FullShard)"
        );
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
/// sharded per `kq_replicated`. The on-disk dtype is supported by
/// model-ops's `qmatmul`; this helper does not dequantise.
#[allow(clippy::too_many_arguments)]
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
    let (block_size, type_size) = quant_block_info(info.dtype)?;
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
}

/// Upload a `[conv_channels, conv_kernel]` F32 GDN fused-QKV tensor
/// (`ssm_conv1d`), sharded per `kq_replicated`.
#[allow(clippy::too_many_arguments)]
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

/// HtoD a host-dequantised tensor as `Tensor<F32>` (no quantise step).
/// Used for the F32 scalars some arches store on disk (GDN's
/// `ssm_dt_bias`, `ssm_a`, `ssm_conv1d`).
pub fn upload_f32_tensor(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    expected_elems: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Tensor<F32>> {
    let f32_vec = file
        .dequantize_tensor(name)
        .with_context(|| format!("dequantize {name}"))?;
    if f32_vec.len() != expected_elems {
        bail!(
            "{name}: dequant produced {} elems, expected {expected_elems}",
            f32_vec.len()
        );
    }
    let bytes = f32_vec.len() * 4;
    let ptr = device.alloc(bytes).context("alloc F32")?;
    let stream = device.default_stream();
    // SAFETY: ptr owns `bytes`; f32_vec has the matching byte count.
    unsafe {
        device.memcpy_async(
            stream,
            CopyDirection::HostToDevice,
            ptr,
            DevicePtr(f32_vec.as_ptr() as usize),
            bytes,
        )?;
    }
    flambeau_core::Stream::synchronize(stream).context("sync after F32 upload")?;
    allocs.push((ptr, bytes));
    Ok(unsafe { Tensor::<F32>::from_raw(ptr, expected_elems) })
}

// ---- Embedding ---------------------------------------------------------------

pub struct EmbeddingSpec<'a> {
    pub token_embd_name: &'a str,
    pub vocab_size: usize,
    pub hidden: usize,
    /// Gemma4 sets `Some(sqrt(hidden))`; qwen leaves `None`.
    pub post_scale: Option<f32>,
}

/// Always replicated — every rank holds the full embedding table.
pub fn load_embedding(
    file: &GgufFile,
    device: &HipDevice,
    spec: &EmbeddingSpec,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<EmbeddingWeights> {
    let token_embd = upload_dequant_to_f16(
        file,
        device,
        spec.token_embd_name,
        spec.vocab_size * spec.hidden,
        allocs,
    )?;
    Ok(EmbeddingWeights {
        token_embd,
        vocab_size: spec.vocab_size,
        hidden: spec.hidden,
        post_scale: spec.post_scale,
    })
}

// ---- LM head -----------------------------------------------------------------

pub struct LmHeadSpec<'a> {
    pub output_norm_name: &'a str,
    /// If `tied`, the loader uploads `token_embd_name`'s bytes again
    /// as the LM-head matmul weight.
    pub lm_head_name: &'a str,
    pub vocab_size: usize,
    pub hidden: usize,
    pub rms_eps: f32,
    /// Gemma4: `Some(30.0)`. Qwen / others: `None`.
    pub final_logit_softcap: Option<f32>,
}

pub fn load_lm_head(
    file: &GgufFile,
    device: &HipDevice,
    spec: &LmHeadSpec,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<LmHeadWeights> {
    let output_norm = upload_dequant_to_f16(
        file,
        device,
        spec.output_norm_name,
        spec.hidden,
        allocs,
    )?;
    let lm_head = upload_quant_weight(
        file,
        device,
        spec.lm_head_name,
        spec.vocab_size * spec.hidden,
        allocs,
    )?;
    Ok(LmHeadWeights {
        output_norm,
        lm_head,
        final_logit_softcap: spec.final_logit_softcap,
        vocab_size: spec.vocab_size,
        hidden: spec.hidden,
        rms_eps: spec.rms_eps,
    })
}

// ---- Dense attention layer ---------------------------------------------------

pub struct DenseAttnLayerSpec<'a> {
    pub attn_norm_name: &'a str,
    pub attn_q_name: &'a str,
    pub attn_k_name: &'a str,
    /// `None` for gemma4-style "V from K" layers (no attn_v on disk).
    pub attn_v_name: Option<&'a str>,
    pub attn_output_name: &'a str,
    pub attn_q_norm_name: Option<&'a str>,
    pub attn_k_norm_name: Option<&'a str>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub rotated_dims: usize,
    pub rope_theta: f32,
    pub window_size: i32,
    pub rms_eps: f32,
    pub softmax_scale: Option<f32>,
}

/// Builds per-rank `AttnWeights`. Under `ShardMode::Tp`, `n_heads` /
/// `n_kv_heads` are divided by `n_ranks`; weights are col-sharded
/// (Q/K/V) or row-sharded (output_proj) accordingly. Replicated norms
/// are always full.
pub fn load_dense_attn_layer(
    file: &GgufFile,
    device: &HipDevice,
    spec: &DenseAttnLayerSpec,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<AttnWeights> {
    let n_ranks = shard.n_ranks();
    if spec.n_heads % n_ranks != 0 {
        bail!(
            "dense_attn: n_heads {} not divisible by n_ranks {n_ranks}",
            spec.n_heads
        );
    }
    if spec.n_kv_heads % n_ranks != 0 {
        bail!(
            "dense_attn: n_kv_heads {} not divisible by n_ranks {n_ranks}",
            spec.n_kv_heads
        );
    }

    let q_width = spec.n_heads * spec.head_dim;
    let kv_width = spec.n_kv_heads * spec.head_dim;
    let n_heads_local = spec.n_heads / n_ranks;
    let n_kv_heads_local = spec.n_kv_heads / n_ranks;

    let attn_norm = upload_dequant_to_f16(file, device, spec.attn_norm_name, spec.hidden, allocs)?;
    let attn_q = upload_col(
        file,
        device,
        spec.attn_q_name,
        q_width,
        spec.hidden,
        shard,
        allocs,
    )?;
    let attn_k = upload_col(
        file,
        device,
        spec.attn_k_name,
        kv_width,
        spec.hidden,
        shard,
        allocs,
    )?;
    let attn_v = if let Some(name) = spec.attn_v_name {
        Some(upload_col(file, device, name, kv_width, spec.hidden, shard, allocs)?)
    } else {
        None
    };
    let attn_output = upload_row(
        file,
        device,
        spec.attn_output_name,
        spec.hidden,
        q_width,
        shard,
        allocs,
    )?;
    let attn_q_norm = spec
        .attn_q_norm_name
        .map(|n| upload_dequant_to_f16(file, device, n, spec.head_dim, allocs))
        .transpose()
        .ok()
        .flatten();
    let attn_k_norm = spec
        .attn_k_norm_name
        .map(|n| upload_dequant_to_f16(file, device, n, spec.head_dim, allocs))
        .transpose()
        .ok()
        .flatten();

    Ok(AttnWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        attn_q_norm,
        attn_k_norm,
        n_heads: n_heads_local,
        n_kv_heads: n_kv_heads_local,
        head_dim: spec.head_dim,
        rotated_dims: spec.rotated_dims,
        rope_theta: spec.rope_theta,
        window_size: spec.window_size,
        rms_eps: spec.rms_eps,
        softmax_scale: spec.softmax_scale,
    })
}

// ---- Dense FFN layer ---------------------------------------------------------

pub struct DenseFfnLayerSpec<'a> {
    pub ffn_norm_name: &'a str,
    pub ffn_gate_name: &'a str,
    pub ffn_up_name: &'a str,
    pub ffn_down_name: &'a str,
    pub hidden: usize,
    pub intermediate: usize,
    pub activation: Activation,
    pub rms_eps: f32,
}

pub fn load_dense_ffn_layer(
    file: &GgufFile,
    device: &HipDevice,
    spec: &DenseFfnLayerSpec,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<FfnWeights> {
    let n_ranks = shard.n_ranks();
    if spec.intermediate % n_ranks != 0 {
        bail!(
            "dense_ffn: intermediate {} not divisible by n_ranks {n_ranks}",
            spec.intermediate
        );
    }
    let ffn_norm = upload_dequant_to_f16(file, device, spec.ffn_norm_name, spec.hidden, allocs)?;
    let ffn_gate = upload_col(
        file,
        device,
        spec.ffn_gate_name,
        spec.intermediate,
        spec.hidden,
        shard,
        allocs,
    )?;
    let ffn_up = upload_col(
        file,
        device,
        spec.ffn_up_name,
        spec.intermediate,
        spec.hidden,
        shard,
        allocs,
    )?;
    let ffn_down = upload_row(
        file,
        device,
        spec.ffn_down_name,
        spec.hidden,
        spec.intermediate,
        shard,
        allocs,
    )?;
    Ok(FfnWeights {
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        activation: spec.activation,
        rms_eps: spec.rms_eps,
    })
}

// ---- GDN layer ---------------------------------------------------------------

/// How the loader shards a GDN layer's K vs V heads across TP ranks.
/// Independent from `rep_inner_layout` (the block-kernel K-head
/// repeat math); this enum only controls upload-time tensor slicing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnTpMode {
    /// Shard both K and V heads uniformly along the head axis.
    /// Geometry gate: `num_k_heads % n_ranks == 0`
    ///   ∧ `num_v_heads % n_ranks == 0`.
    /// Pairs with `rep_inner_layout = true` on qwen3-Next-family
    /// arches.
    FullShard,
    /// Replicate the full `num_k_heads × head_k_dim` slab of Q and K
    /// channels on every rank; shard only the V slab.
    /// Geometry gate: `num_v_heads % n_ranks == 0`
    ///   ∧ `(num_v_heads / n_ranks) % num_k_heads == 0`
    /// (the per-rank kernel computes `n_rep = local_num_v_heads /
    /// num_k_heads` as a clean integer). Pairs with
    /// `rep_inner_layout = false` on qwen35 / qwen35moe / qwen36moe.
    KReplicated,
}

pub struct GdnLayerSpec<'a> {
    pub attn_norm_name: &'a str,
    pub attn_qkv_name: &'a str,
    pub attn_gate_name: &'a str,
    pub ssm_alpha_name: &'a str,
    pub ssm_beta_name: &'a str,
    pub ssm_out_name: &'a str,
    pub ssm_dt_bias_name: &'a str,
    pub ssm_a_name: &'a str,
    pub ssm_conv1d_name: &'a str,
    pub ssm_norm_name: &'a str,
    pub hidden: usize,
    /// Per-rank dims when `shard` is `Tp` — caller has already divided
    /// `num_v_heads` (always) and `num_k_heads` (only under `FullShard`)
    /// by `n_ranks` before constructing the spec. `conv_channels` and
    /// `d_inner` are likewise per-rank.
    pub dims: GdnDims,
    pub rms_eps: f32,
    pub rep_inner_layout: bool,
    /// TP shard policy. Ignored under `ShardMode::Replicated`.
    pub tp_mode: GdnTpMode,
}

/// Replicated or TP-sharded GDN layer load. Under
/// `ShardMode::Tp { rank, n_ranks }` the caller must have set
/// `spec.dims` to the **per-rank** values consistent with
/// `spec.tp_mode` (see `GdnTpMode`). The on-disk Q|K|V geometry of
/// `attn_qkv` / `ssm_conv1d` is interpreted by this loader.
pub fn load_gdn_layer(
    file: &GgufFile,
    device: &HipDevice,
    spec: &GdnLayerSpec,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<GdnWeights> {
    let g = spec.dims;
    let (rank, n_ranks, kq_replicated) = match shard {
        ShardMode::Replicated => (0usize, 1usize, false),
        ShardMode::Tp { rank, n_ranks } => {
            let kq_rep = matches!(spec.tp_mode, GdnTpMode::KReplicated);
            // Geometry gates derived from per-rank `g`. Caller already
            // divided num_v_heads (always) and num_k_heads (FullShard).
            let global_num_v_heads = g.num_v_heads * n_ranks;
            if global_num_v_heads % n_ranks != 0 {
                bail!(
                    "load_gdn_layer: global num_v_heads {global_num_v_heads} not divisible by n_ranks {n_ranks}"
                );
            }
            match spec.tp_mode {
                GdnTpMode::FullShard => {
                    let global_num_k_heads = g.num_k_heads * n_ranks;
                    if global_num_k_heads % n_ranks != 0 {
                        bail!(
                            "load_gdn_layer FullShard: global num_k_heads {global_num_k_heads} not divisible by n_ranks {n_ranks}"
                        );
                    }
                }
                GdnTpMode::KReplicated => {
                    if g.num_v_heads % g.num_k_heads != 0 {
                        bail!(
                            "load_gdn_layer KReplicated: per-rank num_v_heads {} not divisible by num_k_heads {} (n_rep must be a clean integer; pick FullShard or change n_ranks)",
                            g.num_v_heads,
                            g.num_k_heads
                        );
                    }
                }
            }
            (rank, n_ranks, kq_rep)
        }
    };

    // Norm weights replicated regardless of mode.
    let attn_norm =
        upload_dequant_to_f16(file, device, spec.attn_norm_name, spec.hidden, allocs)?;
    let ssm_norm_w =
        upload_dequant_to_f16(file, device, spec.ssm_norm_name, g.head_v_dim, allocs)?;

    // Global K-head count for the fused-QKV unpacking: under
    // KReplicated `g.num_k_heads` is already the global count; under
    // FullShard `g.num_k_heads = global/n_ranks`, so re-scale.
    let global_num_k_heads = if kq_replicated {
        g.num_k_heads
    } else {
        g.num_k_heads * n_ranks
    };
    let global_num_v_heads = g.num_v_heads * n_ranks;

    let (attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out, ssm_dt_bias, ssm_a, ssm_conv1d) =
        if matches!(shard, ShardMode::Replicated) {
            (
                upload_quant_weight(
                    file,
                    device,
                    spec.attn_qkv_name,
                    g.conv_channels * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.attn_gate_name,
                    g.d_inner * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.ssm_alpha_name,
                    g.num_v_heads * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.ssm_beta_name,
                    g.num_v_heads * spec.hidden,
                    allocs,
                )?,
                upload_quant_weight(
                    file,
                    device,
                    spec.ssm_out_name,
                    spec.hidden * g.d_inner,
                    allocs,
                )?,
                upload_f32_tensor(file, device, spec.ssm_dt_bias_name, g.num_v_heads, allocs)?,
                upload_f32_tensor(file, device, spec.ssm_a_name, g.num_v_heads, allocs)?,
                upload_f32_tensor(
                    file,
                    device,
                    spec.ssm_conv1d_name,
                    g.conv_kernel * g.conv_channels,
                    allocs,
                )?,
            )
        } else {
            (
                upload_gdn_fused_qkv_quant(
                    file,
                    device,
                    spec.attn_qkv_name,
                    global_num_v_heads,
                    global_num_k_heads,
                    g.head_v_dim,
                    g.head_k_dim,
                    spec.hidden,
                    kq_replicated,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                // attn_gate [d_inner, hidden] — row-shard along d_inner
                // (per-rank d_inner already in spec.dims.d_inner).
                upload_col_sharded_quant(
                    file,
                    device,
                    spec.attn_gate_name,
                    g.d_inner * n_ranks,
                    spec.hidden,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                // ssm_alpha [num_v_heads, hidden] — V-shard along outer.
                upload_col_sharded_quant(
                    file,
                    device,
                    spec.ssm_alpha_name,
                    global_num_v_heads,
                    spec.hidden,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                upload_col_sharded_quant(
                    file,
                    device,
                    spec.ssm_beta_name,
                    global_num_v_heads,
                    spec.hidden,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                // ssm_out [hidden, d_inner] — col-shard along d_inner
                // (the row-parallel partial-hidden output gets AR-summed
                // by the composite's hook).
                upload_row_sharded_quant(
                    file,
                    device,
                    spec.ssm_out_name,
                    spec.hidden,
                    g.d_inner * n_ranks,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                upload_f32_array_sharded(
                    file,
                    device,
                    spec.ssm_dt_bias_name,
                    global_num_v_heads,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                upload_f32_array_sharded(
                    file,
                    device,
                    spec.ssm_a_name,
                    global_num_v_heads,
                    rank,
                    n_ranks,
                    allocs,
                )?,
                upload_gdn_fused_qkv_f32(
                    file,
                    device,
                    spec.ssm_conv1d_name,
                    global_num_v_heads,
                    global_num_k_heads,
                    g.head_v_dim,
                    g.head_k_dim,
                    g.conv_kernel,
                    kq_replicated,
                    rank,
                    n_ranks,
                    allocs,
                )?,
            )
        };

    Ok(GdnWeights {
        attn_norm,
        attn_qkv,
        attn_gate,
        ssm_alpha,
        ssm_beta,
        ssm_out,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d,
        ssm_norm_w,
        dims: g,
        rms_eps: spec.rms_eps,
        rep_inner_layout: spec.rep_inner_layout,
    })
}
