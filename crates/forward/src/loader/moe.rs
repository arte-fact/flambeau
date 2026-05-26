//! MoE-specific helpers. Real-GGUF MoE stores all experts in one
//! stacked tensor (`[n_experts, dim_a, dim_b]`). Splitting it into
//! per-expert handles via `n_experts` separate hipMalloc calls is
//! slow + fragmenting; instead we upload the whole stacked tensor as
//! one buffer and return `Vec<QuantWeight>` views into it.

use anyhow::{bail, Context, Result};
use flambeau_backend_hip::HipDevice;
use flambeau_core::op::QDtype;
use flambeau_core::DevicePtr;
use flambeau_quant::GgufFile;

use crate::ctx::QuantWeight;

use super::primitives::{dtype_qmatmul_native, ggml_to_qdtype, upload_bytes};
use super::ShardMode;

/// Upload a `[n_experts, dim_a, dim_b]` stacked-expert quant tensor
/// as one buffer and return `n_experts` `QuantWeight` views, each
/// pointing at its expert's `dim_a * dim_b` element slice.
///
/// Native-quant only (Q4_0..Q8_0 + K-quants + IQ). F16/BF16/F32
/// MoE expert tensors are unusual; if a model ships them, add a
/// dequant→Q8_0 fallback the same way `upload_quant_weight` does.
pub fn upload_moe_experts_stacked(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_experts: usize,
    dim_a: usize,
    dim_b: usize,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Vec<QuantWeight>> {
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    if !dtype_qmatmul_native(info.dtype) {
        bail!(
            "{name}: dtype {:?} not native — MoE stacked-expert dequant fallback not implemented",
            info.dtype
        );
    }
    let block_size = info.dtype.block_size();
    let type_size = info.dtype.type_size();
    let elems_per_expert = dim_a * dim_b;
    if elems_per_expert % block_size != 0 {
        bail!(
            "{name}: per-expert elems {elems_per_expert} not divisible by block_size {block_size}"
        );
    }
    let bytes_per_expert = (elems_per_expert / block_size) * type_size;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let expected = n_experts * bytes_per_expert;
    if raw.len() < expected {
        bail!(
            "{name}: raw bytes {} < expected {} ({n_experts} experts × {bytes_per_expert} B)",
            raw.len(),
            expected
        );
    }
    let base_ptr = upload_bytes(device, &raw[..expected], allocs)?;
    let qd = ggml_to_qdtype(info.dtype)?;
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        experts.push(QuantWeight {
            ptr: base_ptr.offset_bytes(e * bytes_per_expert),
            dtype: qd,
            n_elems: elems_per_expert,
        });
    }
    let _: QDtype = qd; // silence unused-warning if qd ever goes unused
    Ok(experts)
}

/// Col-shard variant of [`upload_moe_experts_stacked`]. Splits the
/// stacked tensor along the OUTER dimension `dim_a` (e.g.
/// `intermediate` for gate/up). Each rank gets a contiguous slice of
/// `dim_a / n_ranks` rows of each expert's `[dim_a, dim_b]` slab,
/// uploaded as one tightly-packed buffer; returns per-expert
/// `QuantWeight` views with `local_dim_a * dim_b` elements each.
pub fn upload_moe_experts_stacked_col_sharded(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_experts: usize,
    dim_a: usize,
    dim_b: usize,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Vec<QuantWeight>> {
    let (rank, n_ranks) = match shard {
        ShardMode::Replicated => (0_usize, 1_usize),
        ShardMode::Tp { rank, n_ranks } => (rank, n_ranks),
    };
    if dim_a % n_ranks != 0 {
        bail!("{name}: dim_a {dim_a} not divisible by n_ranks {n_ranks}");
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    if !dtype_qmatmul_native(info.dtype) {
        bail!(
            "{name}: dtype {:?} not native — MoE sharded-stacked dequant fallback not implemented",
            info.dtype
        );
    }
    let block_size = info.dtype.block_size();
    let type_size = info.dtype.type_size();
    if dim_b % block_size != 0 {
        bail!(
            "{name}: dim_b {dim_b} not divisible by block_size {block_size} — col-shard requires \
             clean inner-dim alignment"
        );
    }
    let local_dim_a = dim_a / n_ranks;
    let row_bytes = (dim_b / block_size) * type_size;
    let full_per_expert = dim_a * row_bytes;
    let local_per_expert = local_dim_a * row_bytes;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let expected = n_experts * full_per_expert;
    if raw.len() < expected {
        bail!(
            "{name}: raw bytes {} < expected {} ({n_experts} experts × {full_per_expert} B)",
            raw.len(),
            expected
        );
    }
    let mut host: Vec<u8> = Vec::with_capacity(n_experts * local_per_expert);
    for e in 0..n_experts {
        let off = e * full_per_expert + rank * local_per_expert;
        host.extend_from_slice(&raw[off..off + local_per_expert]);
    }
    let base_ptr = upload_bytes(device, &host, allocs)?;
    let qd = ggml_to_qdtype(info.dtype)?;
    let elems_per_expert = local_dim_a * dim_b;
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        experts.push(QuantWeight {
            ptr: base_ptr.offset_bytes(e * local_per_expert),
            dtype: qd,
            n_elems: elems_per_expert,
        });
    }
    Ok(experts)
}

/// Upload a fused gate+up MoE expert tensor `[n_experts, 2*inter, hidden]`
/// (gemma4 MoE layout) as two SEPARATE stacked buffers — one for gate
/// and one for up — each `[n_experts, inter, hidden]`. The indexed
/// MoE kernels expect each expert's rows to be contiguous within its
/// own stacked tensor, so we cannot keep the fused on-disk layout and
/// just offset; we must compact at load time.
///
/// Native-quant only. F16/BF16/F32 fused gate+up isn't a real-world
/// shape so we bail rather than add a dequant fallback.
pub fn upload_moe_experts_fused_gate_up_stacked(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_experts: usize,
    inter: usize,
    hidden: usize,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<(Vec<QuantWeight>, Vec<QuantWeight>)> {
    let (rank, n_ranks) = match shard {
        ShardMode::Replicated => (0_usize, 1_usize),
        ShardMode::Tp { rank, n_ranks } => (rank, n_ranks),
    };
    if inter % n_ranks != 0 {
        bail!("{name}: inter {inter} not divisible by n_ranks {n_ranks}");
    }
    let local_inter = inter / n_ranks;
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    if !dtype_qmatmul_native(info.dtype) {
        bail!(
            "{name}: dtype {:?} not native — fused gate+up dequant fallback not implemented",
            info.dtype
        );
    }
    let block_size = info.dtype.block_size();
    let type_size = info.dtype.type_size();
    if hidden % block_size != 0 {
        bail!("{name}: hidden {hidden} not divisible by block_size {block_size}");
    }
    let row_bytes = (hidden / block_size) * type_size;
    let full_half_per_expert = inter * row_bytes;
    let full_per_expert = 2 * full_half_per_expert;
    let local_half_per_expert = local_inter * row_bytes;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let expected = n_experts * full_per_expert;
    if raw.len() < expected {
        bail!(
            "{name}: raw bytes {} < expected {expected} ({n_experts} × 2 × {inter} × {row_bytes})",
            raw.len()
        );
    }
    let mut gate_buf: Vec<u8> = Vec::with_capacity(n_experts * local_half_per_expert);
    let mut up_buf: Vec<u8> = Vec::with_capacity(n_experts * local_half_per_expert);
    let rank_row_offset = rank * local_inter * row_bytes;
    for e in 0..n_experts {
        let base = e * full_per_expert;
        let gate_lo = base + rank_row_offset;
        gate_buf.extend_from_slice(&raw[gate_lo..gate_lo + local_half_per_expert]);
        let up_lo = base + full_half_per_expert + rank_row_offset;
        up_buf.extend_from_slice(&raw[up_lo..up_lo + local_half_per_expert]);
    }
    let gate_ptr = upload_bytes(device, &gate_buf, allocs)?;
    let up_ptr = upload_bytes(device, &up_buf, allocs)?;
    let qd = ggml_to_qdtype(info.dtype)?;
    let elems_per_expert = local_inter * hidden;
    let gate: Vec<QuantWeight> = (0..n_experts)
        .map(|e| QuantWeight {
            ptr: gate_ptr.offset_bytes(e * local_half_per_expert),
            dtype: qd,
            n_elems: elems_per_expert,
        })
        .collect();
    let up: Vec<QuantWeight> = (0..n_experts)
        .map(|e| QuantWeight {
            ptr: up_ptr.offset_bytes(e * local_half_per_expert),
            dtype: qd,
            n_elems: elems_per_expert,
        })
        .collect();
    Ok((gate, up))
}

/// Row-shard variant of [`upload_moe_experts_stacked`]. Splits the
/// stacked tensor along the INNER dimension `dim_b` (e.g.
/// `intermediate` for ffn_down where layout is `[hidden,
/// intermediate]`). Each rank's slice of one expert's row is a
/// non-contiguous segment in the raw bytes; copies per (expert, row)
/// into a tightly-packed host buffer, then uploads. Returns per-expert
/// `QuantWeight` views with `dim_a * local_dim_b` elements each.
pub fn upload_moe_experts_stacked_row_sharded(
    file: &GgufFile,
    device: &HipDevice,
    name: &str,
    n_experts: usize,
    dim_a: usize,
    dim_b: usize,
    shard: ShardMode,
    allocs: &mut Vec<(DevicePtr, usize)>,
) -> Result<Vec<QuantWeight>> {
    let (rank, n_ranks) = match shard {
        ShardMode::Replicated => (0_usize, 1_usize),
        ShardMode::Tp { rank, n_ranks } => (rank, n_ranks),
    };
    if dim_b % n_ranks != 0 {
        bail!("{name}: dim_b {dim_b} not divisible by n_ranks {n_ranks}");
    }
    let info = file.info(name).with_context(|| format!("info {name}"))?;
    if !dtype_qmatmul_native(info.dtype) {
        bail!(
            "{name}: dtype {:?} not native — MoE sharded-stacked dequant fallback not implemented",
            info.dtype
        );
    }
    let block_size = info.dtype.block_size();
    let type_size = info.dtype.type_size();
    let local_dim_b = dim_b / n_ranks;
    if local_dim_b % block_size != 0 {
        bail!(
            "{name}: local_dim_b {local_dim_b} (= {dim_b}/{n_ranks}) not divisible by block_size \
             {block_size}"
        );
    }
    let full_row_bytes = (dim_b / block_size) * type_size;
    let local_row_bytes = (local_dim_b / block_size) * type_size;
    let full_per_expert = dim_a * full_row_bytes;
    let local_per_expert = dim_a * local_row_bytes;
    let raw = file
        .tensor_raw(name)
        .with_context(|| format!("tensor_raw {name}"))?;
    let expected = n_experts * full_per_expert;
    if raw.len() < expected {
        bail!(
            "{name}: raw bytes {} < expected {} ({n_experts} experts × {full_per_expert} B)",
            raw.len(),
            expected
        );
    }
    let mut host: Vec<u8> = Vec::with_capacity(n_experts * local_per_expert);
    for e in 0..n_experts {
        for r in 0..dim_a {
            let src = e * full_per_expert + r * full_row_bytes + rank * local_row_bytes;
            host.extend_from_slice(&raw[src..src + local_row_bytes]);
        }
    }
    let base_ptr = upload_bytes(device, &host, allocs)?;
    let qd = ggml_to_qdtype(info.dtype)?;
    let elems_per_expert = dim_a * local_dim_b;
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        experts.push(QuantWeight {
            ptr: base_ptr.offset_bytes(e * local_per_expert),
            dtype: qd,
            n_elems: elems_per_expert,
        });
    }
    Ok(experts)
}
