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
    hipMemcpyKind, hipSetDevice, hipStreamCreate, hipStreamDestroy, hipStreamSynchronize,
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
    let code = unsafe { hipGetDeviceCount(&mut count as *mut _) };
    check(code, "hipGetDeviceCount")?;
    Ok(count as i32)
}

/// Bind the current thread's HIP context to `device_id`.
pub fn bind(device_id: i32) -> DeviceResult<()> {
    // SAFETY: `hipSetDevice` takes an `int` by value and touches no caller memory.
    // A negative or out-of-range id is returned as an error via the return code.
    check(unsafe { hipSetDevice(device_id as c_int) }, "hipSetDevice")
}

/// Return the HIP context's current device id.
pub fn current_device() -> DeviceResult<i32> {
    let mut id: c_int = -1;
    // SAFETY: `hipGetDevice` writes an `int` through the out-pointer and reads
    // nothing from it. `&mut id` is valid for writes of `sizeof(int)`.
    check(unsafe { hipGetDevice(&mut id as *mut _) }, "hipGetDevice")?;
    Ok(id as i32)
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
    /// Create a new non-blocking stream on `device_id`. Caller must have
    /// `bind(device_id)` in effect for the current thread.
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut s: hipStream_t = ptr::null_mut();
        // SAFETY: `hipStreamCreate` writes a stream handle through the
        // out-pointer. `&mut s` is valid for writes of a `hipStream_t`.
        check(unsafe { hipStreamCreate(&mut s as *mut _) }, "hipStreamCreate")?;
        Ok(Self {
            ptr: s,
            device_id,
        })
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
        let code = unsafe { hipMalloc(&mut p as *mut _, bytes) };
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
