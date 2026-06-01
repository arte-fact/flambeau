//! Lightweight handle to a model weight tensor on device.

use flambeau_core::op::QDtype;
use flambeau_core::DevicePtr;

/// Borrowed handle to one model weight tensor on device. The block
/// holds these instead of owning the underlying alloc — the caller
/// (typically the model crate) owns the buffer and decides its
/// lifetime.
#[derive(Copy, Clone)]
pub struct WeightHandle {
    pub ptr: DevicePtr,
    pub dtype: QDtype,
    /// `[out_rows, in_cols]` for matmul weights. Norm weights are 1-D;
    /// callers store their length in the owning block's shape config.
    pub dims: [usize; 2],
}
