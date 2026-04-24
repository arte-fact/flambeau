//! `HipDevice` and `HipStream` — safe wrappers around `libamdhip64`.
//!
//! One `HipDevice` ≈ one GPU. Internally it owns:
//! - a default stream (every implicit launch goes here unless the caller
//!   creates another stream);
//! - a device id used by `hipSetDevice` before any operation.
//!
//! The device is pinned to its id — `HipDevice::new(0)` sets the context to
//! device 0 and expects every subsequent operation on that instance to happen
//! on that device. Callers holding devices for multiple GPUs must call
//! `bind()` (or its equivalent `hipSetDevice`) before issuing work on a
//! device that isn't the current HIP context.

use std::os::raw::c_int;
use std::ptr;

use flambeau_core::{CopyDirection, Device, DeviceError, DevicePtr, DeviceResult, Stream};

use crate::sys::{
    self, error_string, hipFree, hipGetDevice, hipGetDeviceCount, hipMalloc, hipMemcpyAsync,
    hipMemcpyKind, hipSetDevice, hipStreamCreate, hipStreamCreateWithFlags, hipStreamDestroy,
    hipStreamSynchronize,
    hipStream_t, HIP_SUCCESS,
};

const BACKEND: &str = "hip";

fn check(code: c_int, ctx: &'static str) -> DeviceResult<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(DeviceError::Backend {
            backend: BACKEND,
            code,
            message: format!("{ctx}: {}", error_string(code)),
        })
    }
}

/// Number of HIP devices visible to this process.
pub fn device_count() -> DeviceResult<i32> {
    let mut count: c_int = 0;
    // SAFETY: `hipGetDeviceCount` writes an `int` through the out-pointer and
    // reads nothing from it. `&mut count` is valid for writes of `sizeof(int)`.
    let code = unsafe { hipGetDeviceCount(&raw mut count) };
    check(code, "hipGetDeviceCount")?;
    Ok(count)
}

/// Bind the current thread's HIP context to `device_id`.
pub fn bind(device_id: i32) -> DeviceResult<()> {
    // SAFETY: `hipSetDevice` takes an `int` by value and touches no caller memory.
    // A negative or out-of-range id is returned as an error via the return code.
    check(unsafe { hipSetDevice(device_id) }, "hipSetDevice")
}

/// Return the HIP context's current device id.
pub fn current_device() -> DeviceResult<i32> {
    let mut id: c_int = -1;
    // SAFETY: `hipGetDevice` writes an `int` through the out-pointer and reads
    // nothing from it. `&mut id` is valid for writes of `sizeof(int)`.
    check(unsafe { hipGetDevice(&raw mut id) }, "hipGetDevice")?;
    Ok(id)
}

/// A HIP stream. Drop destroys the underlying `hipStream_t`.
pub struct HipStream {
    ptr: hipStream_t,
    device_id: i32,
}

impl std::fmt::Debug for HipStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipStream")
            .field("ptr", &(self.ptr as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: `hipStream_t` is an opaque driver handle with no Rust-side aliasing.
// HIP streams are documented as safe to pass between threads (operations
// serialise within a stream driver-side). `HipStream` owns its handle and
// destroys it on drop, so there is no cross-thread double-free risk.
unsafe impl Send for HipStream {}
// SAFETY: See `Send`. `&HipStream` only lets other threads read the handle
// pointer and call `hipStreamSynchronize` / submit work — both are thread-safe
// per the HIP runtime contract.
unsafe impl Sync for HipStream {}

impl HipStream {
    /// Create a new stream on `device_id`. Caller must have `bind(device_id)`
    /// in effect for the current thread.
    ///
    /// NB: `hipStreamCreate` creates a **blocking** stream (serialises with
    /// the null stream). For truly-concurrent streams on the same device
    /// use [`Self::new_non_blocking`].
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut s: hipStream_t = ptr::null_mut();
        // SAFETY: `hipStreamCreate` writes a stream handle through the
        // out-pointer. `&mut s` is valid for writes of a `hipStream_t`.
        check(unsafe { hipStreamCreate(&raw mut s) }, "hipStreamCreate")?;
        Ok(Self {
            ptr: s,
            device_id,
        })
    }

    /// V2.25.g — create a non-blocking stream (`hipStreamNonBlocking` = 1).
    /// These streams do NOT serialise with the null stream and can run
    /// concurrently with each other on the same device, subject to
    /// occupancy. Used by `HipCluster::reserve_aux_streams` so the
    /// V2.25.d async ubatch pipeline truly overlaps lanes on the same
    /// device.
    pub fn new_non_blocking(device_id: i32) -> DeviceResult<Self> {
        let mut s: hipStream_t = ptr::null_mut();
        const HIP_STREAM_NON_BLOCKING: std::os::raw::c_uint = 1;
        // SAFETY: `hipStreamCreateWithFlags` writes an opaque handle.
        check(
            unsafe { hipStreamCreateWithFlags(&raw mut s, HIP_STREAM_NON_BLOCKING) },
            "hipStreamCreateWithFlags",
        )?;
        Ok(Self { ptr: s, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn raw(&self) -> hipStream_t {
        self.ptr
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was returned by `hipStreamCreate` in `new()`
            // and is not shared with any other owning type (streams are Send
            // but never Clone). The null check above guards against a partially-
            // constructed instance. Destroy errors are non-actionable at drop.
            let _ = unsafe { hipStreamDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Stream for HipStream {
    fn raw_handle(&self) -> usize {
        self.ptr as usize
    }

    fn synchronize(&self) -> DeviceResult<()> {
        // SAFETY: `self.ptr` is a live stream handle (owned, non-null once
        // constructed; drop sets it back to null but no method can observe that).
        check(
            unsafe { hipStreamSynchronize(self.ptr) },
            "hipStreamSynchronize",
        )
    }
}

/// V2.25.b — HIP event for cross-stream DAG scheduling. Used by the async
/// peer-copy pipeline in `HipCluster::peer_copy_via_host_async`.
///
/// Created with `hipEventDisableTiming` — we never call `hipEventElapsedTime`,
/// just `hipEventRecord` / `hipStreamWaitEvent`. Drop destroys the handle.
pub struct HipEvent {
    ptr: crate::sys::hipEvent_t,
    device_id: i32,
}

impl std::fmt::Debug for HipEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipEvent")
            .field("ptr", &(self.ptr as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: hipEvent_t is an opaque driver handle. Events are thread-safe
// per the HIP runtime contract; record/wait ops serialise at the driver.
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    /// Create a new timing-disabled event on `device_id`. Caller must have
    /// `bind(device_id)` in effect.
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut e: crate::sys::hipEvent_t = ptr::null_mut();
        // SAFETY: `hipEventCreateWithFlags` writes an opaque handle through
        // the out-pointer. `&mut e` is valid for a `hipEvent_t`.
        check(
            unsafe {
                crate::sys::hipEventCreateWithFlags(
                    &raw mut e,
                    crate::sys::hipEventDisableTiming,
                )
            },
            "hipEventCreateWithFlags",
        )?;
        Ok(Self { ptr: e, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn raw(&self) -> crate::sys::hipEvent_t {
        self.ptr
    }

    /// Record this event on `stream`. After previous work on `stream`
    /// completes, the event transitions to the recorded state.
    pub fn record(&self, stream: &HipStream) -> DeviceResult<()> {
        // SAFETY: both handles are live (owned by Self / caller).
        check(
            unsafe { crate::sys::hipEventRecord(self.ptr, stream.raw()) },
            "hipEventRecord",
        )
    }

    /// Insert a wait on `stream`: subsequent work enqueued on `stream`
    /// won't start until the event transitions to recorded. Non-blocking
    /// on the host — the wait is driver-side.
    pub fn stream_wait(&self, stream: &HipStream) -> DeviceResult<()> {
        // SAFETY: both handles are live. flags=0 for standard semantics.
        check(
            unsafe { crate::sys::hipStreamWaitEvent(stream.raw(), self.ptr, 0) },
            "hipStreamWaitEvent",
        )
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was returned by hipEventCreateWithFlags and
            // is not aliased (events are owned, not Clone).
            let _ = unsafe { crate::sys::hipEventDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// A HIP device. Holds a device id and a default stream.
///
/// Construction calls `hipSetDevice` once, but there is no guarantee that the
/// process's HIP context stays on this device across calls — callers driving
/// multiple GPUs from one thread must `HipDevice::bind()` before operations.
#[derive(Debug)]
pub struct HipDevice {
    id: i32,
    default_stream: HipStream,
}

impl HipDevice {
    /// Create a new device wrapper for HIP device `id`. Sets the current
    /// HIP context to `id` and creates a default stream on it.
    pub fn new(id: i32) -> DeviceResult<Self> {
        let count = device_count()?;
        if count <= 0 {
            return Err(DeviceError::NoDevices { backend: BACKEND });
        }
        if id < 0 || id >= count {
            return Err(DeviceError::InvalidDeviceId {
                requested: id,
                count,
            });
        }
        bind(id)?;
        let default_stream = HipStream::new(id)?;
        Ok(Self { id, default_stream })
    }

    /// Bind the current thread's HIP context to this device. Required
    /// before alloc/free/stream ops when multiple devices are in use.
    pub fn bind(&self) -> DeviceResult<()> {
        bind(self.id)
    }
}

impl Device for HipDevice {
    type Stream = HipStream;

    fn backend(&self) -> &'static str {
        BACKEND
    }

    fn id(&self) -> i32 {
        self.id
    }

    fn default_stream(&self) -> &Self::Stream {
        &self.default_stream
    }

    fn new_stream(&self) -> DeviceResult<Self::Stream> {
        self.bind()?;
        HipStream::new(self.id)
    }

    fn alloc(&self, bytes: usize) -> DeviceResult<DevicePtr> {
        if bytes == 0 {
            return Ok(DevicePtr::NULL);
        }
        self.bind()?;
        let mut p: *mut std::os::raw::c_void = ptr::null_mut();
        // SAFETY: `hipMalloc` writes a pointer through the out-pointer and
        // reads nothing from it. `&mut p` is valid for writes of `sizeof(void*)`.
        // The returned device pointer is owned by `DevicePtr`.
        let code = unsafe { hipMalloc(&raw mut p, bytes) };
        if code != HIP_SUCCESS {
            return Err(DeviceError::Alloc {
                backend: BACKEND,
                device: self.id,
                bytes,
                reason: error_string(code),
            });
        }
        Ok(DevicePtr(p as usize))
    }

    unsafe fn dealloc(&self, ptr: DevicePtr, _bytes: usize) -> DeviceResult<()> {
        if ptr.is_null() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller's contract on `Device::dealloc` is that `ptr` was
        // returned by a prior `alloc` on a compatible device and is not
        // aliased. The null guard above handles `DevicePtr::NULL`.
        check(unsafe { hipFree(ptr.0 as *mut _) }, "hipFree")
    }

    unsafe fn memcpy_async(
        &self,
        stream: &Self::Stream,
        dir: CopyDirection,
        dst: DevicePtr,
        src: DevicePtr,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        let kind = match dir {
            CopyDirection::HostToDevice => hipMemcpyKind::HostToDevice,
            CopyDirection::DeviceToHost => hipMemcpyKind::DeviceToHost,
            CopyDirection::DeviceToDevice => hipMemcpyKind::DeviceToDevice,
        };
        // SAFETY: caller's contract on `Device::memcpy_async` is that `src`
        // and `dst` are each valid for reads/writes of `bytes` in their
        // respective address spaces per `dir`, and that neither is aliased
        // by another pending op on the same stream. `stream.raw()` is a
        // live handle owned by the caller's `HipStream`.
        let code = unsafe {
            hipMemcpyAsync(
                dst.0 as *mut _,
                src.0 as *const _,
                bytes,
                kind,
                stream.raw(),
            )
        };
        check(code, "hipMemcpyAsync")
    }

    fn synchronize(&self) -> DeviceResult<()> {
        self.bind()?;
        check(
            // SAFETY: `hipDeviceSynchronize` takes no arguments and operates
            // on the thread-current HIP context, which `self.bind()` just set.
            unsafe { sys::hipDeviceSynchronize() },
            "hipDeviceSynchronize",
        )
    }
}
