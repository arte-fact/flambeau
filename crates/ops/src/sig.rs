//! Op-signature parameter aggregates.
//!
//! Every type in this module is a `Copy + Clone + Debug` POD whose only
//! purpose is to group recurring kernel-launch arguments by semantic
//! identity (launch context, buffers, shape, knobs). They contain no
//! methods, no invariants, and no behaviour — pass them by value, they
//! are pointer-sized stack aggregates.
//!
//! See `doc/CLIPPY_STRUCT_REFACTOR_PLAN.md` for the phased migration.

use crate::OpsRegistry;
use flambeau_backend_hip::HipStream;
use flambeau_core::device::DevicePtr;

/// Launch context handed to every `crates/ops/src/hip/*` free-fn wrapper.
/// Both fields are short-lived borrows shared across every kernel launch
/// in a forward pass — the registry owns the loaded modules, the stream
/// is the queue the launch enqueues onto.
#[derive(Copy, Clone)]
pub struct OpCtx<'a> {
    pub reg: &'a OpsRegistry,
    pub stream: &'a HipStream,
}

// --- matmul / mmvq families -------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct MmvqBuffers {
    pub weights: DevicePtr,
    pub act_q8_1: DevicePtr,
    pub dst: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqGateUpBuffers {
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub act_q8_1: DevicePtr,
    pub gate_out: DevicePtr,
    pub up_out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct MatmulShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqShape {
    pub n_rows: usize,
    pub k: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqBatchShape {
    pub n_rows: usize,
    pub k: usize,
    pub n_slots: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqGateUpShape {
    pub n_rows_gate: usize,
    pub n_rows_up: usize,
    pub k: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct MmvqGateUpBatchShape {
    pub n_rows_gate: usize,
    pub n_rows_up: usize,
    pub k: usize,
    pub n_slots: usize,
}

// --- attention family -------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct AttnBuffers {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub out: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnDecodeShape {
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_tokens_kv: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnPrefillShape {
    pub n_q_tokens: usize,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub n_k_tokens: usize,
    pub q_offset: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AttnKnobs {
    pub scale: f32,
    pub window_size: i32,
}

// --- norm / fused-norm family ----------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct NormBuffers {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub output: DevicePtr,
}

#[derive(Copy, Clone, Debug)]
pub struct NormResidual {
    pub residual_in: DevicePtr,
    pub residual_out: DevicePtr,
    pub residual_scale: f32,
}
