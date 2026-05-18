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
