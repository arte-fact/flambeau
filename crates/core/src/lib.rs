//! flambeau-core — device-independent traits: Tensor, DType, Shape, Op contracts,
//! DispatchTable, Registry, CorrectnessCert.
//!
//! V1.0: empty stubs with the intended trait shape sketched in doc/ARCHITECTURE.md.
//! See `doc/ROADMAP-V1-QWEN36-GFX906.md` step V1.0 for what lands here first.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod device;
pub mod kernel_limits;
pub mod op;

pub use device::{
    CopyDirection, Device, DeviceError, DevicePtr, DeviceResult, Stream,
};
pub use kernel_limits::{MOE_SORT_MAX_EXPERTS, TOPK_MAX_EXPERTS};
pub use op::{
    DirectCallKernel, KernelDescriptor, KernelImpl, Op, OpContract, QDtype, QMatMul, QMatMulCfg,
    QMatMulInput, QMatMulOutput, RmsNorm, RmsNormCfg, RmsNormInput, RmsNormOutput, SwiGLU,
    SwiGLUCfg, SwiGLUInput, SwiGLUOutput, Tolerance,
};
