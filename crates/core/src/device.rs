//! Device / Stream trait surface.
//! Flambeau is device-generic at the trait level: `Device` and `Stream` are the
//! only surface models touch. Concrete `HipDevice` + `HipStream` live in
//! `flambeau-backend-hip`; `CudaDevice` + `CudaStream` will live in
//! `flambeau-backend-cuda` (V2).
//! Design notes:
//! - Allocation returns a `DevicePtr` newtype — opaque to consumers. Backends
//!   cast to their real pointer type internally.
//! - Streams are **explicit**: every kernel launch / memcpy / collective takes
//!   `&Stream`. No hidden `hipDeviceSynchronize` except at session boundaries
//!   (architectural rule 7 from CLAUDE.md).
//! - `DeviceError` is the common failure type. Backends wrap their native
//!   error codes (hipError_t, ncclResult_t) behind this with a string context.

use std::fmt;

use thiserror::Error;

/// Opaque device-side pointer. Backends produce these from their allocators
/// and interpret them in kernel launch / memcpy calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePtr(pub usize);

impl DevicePtr {
    pub const NULL: Self = Self(0);
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
    pub fn as_usize(self) -> usize {
        self.0
    }
    pub fn offset_bytes(self, n: usize) -> Self {
        Self(self.0 + n)
    }
}

impl fmt::Display for DevicePtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dev:{:#x}", self.0)
    }
}

/// Direction for host↔device memcpy.
#[derive(Debug, Clone, Copy)]
pub enum CopyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("{backend} device error ({code}): {message}")]
    Backend {
        backend: &'static str,
        code: i32,
        message: String,
    },

    #[error("allocation of {bytes} bytes failed on {backend} device {device}: {reason}")]
    Alloc {
        backend: &'static str,
        device: i32,
        bytes: usize,
        reason: String,
    },

    #[error("invalid device id {requested}: only {count} device(s) present")]
    InvalidDeviceId { requested: i32, count: i32 },

    #[error("no {backend} devices present on this host")]
    NoDevices { backend: &'static str },

    #[error("backend not built (rebuild with feature {feature:?})")]
    BackendNotBuilt { feature: &'static str },
}

pub type DeviceResult<T> = std::result::Result<T, DeviceError>;

/// A compute stream — ordered queue of GPU work. Launches issued on the same
/// stream run in issue order; cross-stream ordering requires events.
pub trait Stream: Send + Sync {
    /// Backend-specific raw handle (hipStream_t, cudaStream_t, ...).
    /// Kernel-launch code in backend crates casts back to the concrete type.
    fn raw_handle(&self) -> usize;

    /// Block the calling thread until every launch on this stream has retired.
    /// # Errors
    /// Returns `DeviceError::Backend` if the underlying stream-sync call
    /// (e.g. `hipStreamSynchronize`) reports a non-success status.
    fn synchronize(&self) -> DeviceResult<()>;
}

/// A compute device (one GPU). `Device` is the root of the trait hierarchy —
/// everything else (allocators, streams, kernel impls) flows from a concrete
/// device type.
pub trait Device: Send + Sync + 'static {
    type Stream: Stream;

    /// Backend identifier (`"hip"`, `"cuda"`, `"cpu"`). Used in error messages
    /// and dispatch keys.
    fn backend(&self) -> &'static str;

    /// Numeric device id on this host (0, 1, ...). `Mesh<N>` uses this to
    /// identify ranks.
    fn id(&self) -> i32;

    /// Stream created at device construction. Convenience for one-stream work;
    /// multi-stream users call `new_stream`.
    fn default_stream(&self) -> &Self::Stream;

    /// Create an additional stream.
    /// # Errors
    /// Returns `DeviceError::Backend` if the backend stream-create call fails.
    fn new_stream(&self) -> DeviceResult<Self::Stream>;

    /// Allocate `bytes` bytes of device memory. The returned pointer is owned
    /// by the caller and must be freed by `dealloc` on this same device.
    /// # Errors
    /// Returns `DeviceError::Alloc` if the backend allocator fails (typically
    /// out-of-memory). Zero-byte allocations are infallible and return a
    /// `DevicePtr::NULL` sentinel.
    fn alloc(&self, bytes: usize) -> DeviceResult<DevicePtr>;

    /// Free a pointer previously returned by `alloc` on this device.
    /// # Safety
    /// The pointer must have been returned by a successful `alloc` on **this**
    /// device, and no outstanding work on any stream may reference it.
    /// # Errors
    /// Returns `DeviceError::Backend` if the backend free call fails.
    /// `DevicePtr::NULL` is accepted and returns `Ok(())`.
    unsafe fn dealloc(&self, ptr: DevicePtr, bytes: usize) -> DeviceResult<()>;

    /// Copy `bytes` bytes across the H↔D boundary on `stream`. The caller is
    /// responsible for ensuring the host buffer lives until the stream
    /// synchronises for `HostToDevice` / `DeviceToHost` transfers.
    /// # Safety
    /// Both endpoints must be valid for the declared direction and size.
    /// # Errors
    /// Returns `DeviceError::Backend` if the backend async-memcpy enqueue
    /// fails. Zero-byte copies are infallible.
    unsafe fn memcpy_async(
        &self,
        stream: &Self::Stream,
        dir: CopyDirection,
        dst: DevicePtr,
        src: DevicePtr,
        bytes: usize,
    ) -> DeviceResult<()>;

    /// Block until **all** streams on this device have retired. Session
    /// boundaries only; hot paths must use `Stream::synchronize`.
    /// # Errors
    /// Returns `DeviceError::Backend` if the device-sync call fails.
    fn synchronize(&self) -> DeviceResult<()>;
}
