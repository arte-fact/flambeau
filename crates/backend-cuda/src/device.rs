//! `CudaDevice` / `CudaStream` / `CudaEvent` — safe wrappers over the CUDA
//! driver API.
//!
//! `CudaDevice` retains the device's primary context; `bind()` sets it current
//! on the calling thread. Multi-GPU callers `bind()` before issuing work on a
//! device whose context isn't current.

use std::ptr;
use std::sync::Once;

use flambeau_core::{CopyDirection, Device, DeviceError, DevicePtr, DeviceResult, Event, Stream};

use crate::sys::{
    cuCtxSetCurrent, cuCtxSynchronize, cuDeviceGet, cuDeviceGetCount,
    cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxRetain, cuEventCreate, cuEventDestroy,
    cuEventElapsedTime, cuEventRecord, cuEventSynchronize, cuInit, cuMemAlloc, cuMemFree,
    cuMemcpyAsync, cuMemcpyDtoDAsync, cuMemcpyDtoHAsync, cuMemcpyHtoDAsync, cuStreamCreate,
    cuStreamDestroy,
    cuStreamSynchronize, cuStreamWaitEvent, error_string, CUcontext, CUdevice, CUdeviceptr,
    CUevent, CUstream, CUDA_SUCCESS, CU_EVENT_DEFAULT, CU_EVENT_DISABLE_TIMING, CU_STREAM_DEFAULT,
    CU_STREAM_NON_BLOCKING,
};

const BACKEND: &str = "cuda";

fn check(code: std::os::raw::c_int, ctx: &'static str) -> DeviceResult<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DeviceError::Backend {
            backend: BACKEND,
            code,
            message: format!("{ctx}: {}", error_string(code)),
        })
    }
}

static CUDA_INIT: Once = Once::new();

/// `cuInit(0)` exactly once per process. Every other driver call requires it.
fn ensure_init() -> DeviceResult<()> {
    let mut init_code = CUDA_SUCCESS;
    CUDA_INIT.call_once(|| {
        // SAFETY: `cuInit` takes flags by value and touches no caller memory.
        init_code = unsafe { cuInit(0) };
    });
    check(init_code, "cuInit")
}

/// Number of CUDA devices visible to this process.
pub fn device_count() -> DeviceResult<i32> {
    ensure_init()?;
    let mut count: std::os::raw::c_int = 0;
    // SAFETY: writes an int through the out-pointer, reads nothing.
    check(unsafe { cuDeviceGetCount(&raw mut count) }, "cuDeviceGetCount")?;
    Ok(count)
}

/// A CUDA stream. Drop destroys the underlying `CUstream`.
pub struct CudaStream {
    ptr: CUstream,
    device_id: i32,
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("ptr", &(self.ptr as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: `CUstream` is an opaque driver handle; CUDA streams are documented
// as thread-safe to submit to. `CudaStream` owns its handle and destroys it on
// drop (never Clone), so there is no cross-thread double-free.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a blocking stream on the current context. Caller must have the
    /// owning device's context current (`CudaDevice::bind`).
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut s: CUstream = ptr::null_mut();
        // SAFETY: writes a stream handle through the out-pointer.
        check(unsafe { cuStreamCreate(&raw mut s, CU_STREAM_DEFAULT) }, "cuStreamCreate")?;
        Ok(Self { ptr: s, device_id })
    }

    /// Create a non-blocking stream (does not serialise with the NULL stream).
    pub fn new_non_blocking(device_id: i32) -> DeviceResult<Self> {
        let mut s: CUstream = ptr::null_mut();
        // SAFETY: writes a stream handle through the out-pointer.
        check(
            unsafe { cuStreamCreate(&raw mut s, CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate(non-blocking)",
        )?;
        Ok(Self { ptr: s, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn raw(&self) -> CUstream {
        self.ptr
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was returned by `cuStreamCreate` and is not
            // aliased (streams are Send but never Clone).
            let _ = unsafe { cuStreamDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Stream for CudaStream {
    fn raw_handle(&self) -> usize {
        self.ptr as usize
    }

    fn synchronize(&self) -> DeviceResult<()> {
        // SAFETY: `self.ptr` is a live stream handle.
        check(unsafe { cuStreamSynchronize(self.ptr) }, "cuStreamSynchronize")
    }
}

/// A CUDA event for cross-stream ordering. Created timing-disabled by default
/// (`CU_EVENT_DISABLE_TIMING`); the timing constructor enables
/// `cuEventElapsedTime`. Drop destroys the handle.
pub struct CudaEvent {
    ptr: CUevent,
    device_id: i32,
}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEvent")
            .field("ptr", &(self.ptr as usize))
            .field("device_id", &self.device_id)
            .finish()
    }
}

// SAFETY: `CUevent` is an opaque driver handle; record/wait are thread-safe.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Timing-disabled ordering event. Caller must have the device's context
    /// current.
    pub fn new(device_id: i32) -> DeviceResult<Self> {
        let mut e: CUevent = ptr::null_mut();
        // SAFETY: writes an event handle through the out-pointer.
        check(
            unsafe { cuEventCreate(&raw mut e, CU_EVENT_DISABLE_TIMING) },
            "cuEventCreate",
        )?;
        Ok(Self { ptr: e, device_id })
    }

    /// Timing-enabled event (for `elapsed_ms_since`).
    pub fn new_timing(device_id: i32) -> DeviceResult<Self> {
        let mut e: CUevent = ptr::null_mut();
        // SAFETY: writes an event handle through the out-pointer.
        check(
            unsafe { cuEventCreate(&raw mut e, CU_EVENT_DEFAULT) },
            "cuEventCreate(timing)",
        )?;
        Ok(Self { ptr: e, device_id })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` was returned by `cuEventCreate`, not aliased.
            let _ = unsafe { cuEventDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Event<CudaStream> for CudaEvent {
    fn record(&self, stream: &CudaStream) -> DeviceResult<()> {
        // SAFETY: both handles are live (owned by Self / caller).
        check(unsafe { cuEventRecord(self.ptr, stream.raw()) }, "cuEventRecord")
    }

    fn stream_wait(&self, stream: &CudaStream) -> DeviceResult<()> {
        // SAFETY: both handles live; flags=0 for standard semantics.
        check(
            unsafe { cuStreamWaitEvent(stream.raw(), self.ptr, 0) },
            "cuStreamWaitEvent",
        )
    }

    fn synchronize(&self) -> DeviceResult<()> {
        // SAFETY: handle owned by Self.
        check(unsafe { cuEventSynchronize(self.ptr) }, "cuEventSynchronize")
    }

    fn elapsed_ms_since(&self, start: &Self) -> DeviceResult<f32> {
        let mut ms: f32 = 0.0;
        // SAFETY: ms is a stack scalar; both event handles live + timing-enabled.
        check(
            unsafe { cuEventElapsedTime(&raw mut ms, start.ptr, self.ptr) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

/// A CUDA device. Holds the device ordinal, its retained primary context, and
/// a default stream. `bind()` sets the primary context current on the calling
/// thread.
#[derive(Debug)]
pub struct CudaDevice {
    id: i32,
    device: CUdevice,
    ctx: CUcontext,
    // `Option` so `Drop` can destroy the stream while the context is still
    // current+retained, before releasing the primary context.
    default_stream: Option<CudaStream>,
}

// SAFETY: `CUcontext` is an opaque driver handle. The device is bound per
// operation via `cuCtxSetCurrent`; primary-context handles are valid across
// threads. `CudaDevice` owns its retain and releases it on drop.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Create a device wrapper for CUDA device `id`: retain its primary
    /// context, set it current, and create a default stream.
    pub fn new(id: i32) -> DeviceResult<Self> {
        let count = device_count()?;
        if count <= 0 {
            return Err(DeviceError::NoDevices { backend: BACKEND });
        }
        if id < 0 || id >= count {
            return Err(DeviceError::InvalidDeviceId { requested: id, count });
        }
        let mut device: CUdevice = 0;
        // SAFETY: writes a CUdevice ordinal through the out-pointer.
        check(unsafe { cuDeviceGet(&raw mut device, id) }, "cuDeviceGet")?;
        let mut ctx: CUcontext = ptr::null_mut();
        // SAFETY: writes the primary context handle through the out-pointer;
        // ref-counted, released in Drop.
        check(
            unsafe { cuDevicePrimaryCtxRetain(&raw mut ctx, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        // SAFETY: `ctx` is the just-retained primary context.
        check(unsafe { cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;
        let default_stream = CudaStream::new(id)?;
        Ok(Self { id, device, ctx, default_stream: Some(default_stream) })
    }

    /// Set this device's primary context current on the calling thread.
    /// Required before alloc/free/stream/launch when multiple devices are used.
    pub fn bind(&self) -> DeviceResult<()> {
        // SAFETY: `self.ctx` is the retained primary context, valid for the
        // lifetime of `self`.
        check(unsafe { cuCtxSetCurrent(self.ctx) }, "cuCtxSetCurrent")
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // Bind first so the stream's `cuStreamDestroy` targets this context,
        // destroy the stream, then release the primary-context reference.
        // SAFETY: `self.ctx`/`self.device` are live until this drop completes.
        let _ = unsafe { cuCtxSetCurrent(self.ctx) };
        drop(self.default_stream.take());
        let _ = unsafe { cuDevicePrimaryCtxRelease(self.device) };
        self.ctx = ptr::null_mut();
    }
}

impl Device for CudaDevice {
    type Stream = CudaStream;
    type Event = CudaEvent;

    fn backend(&self) -> &'static str {
        BACKEND
    }

    fn id(&self) -> i32 {
        self.id
    }

    fn default_stream(&self) -> &Self::Stream {
        self.default_stream
            .as_ref()
            .expect("default_stream present until drop")
    }

    fn new_stream(&self) -> DeviceResult<Self::Stream> {
        self.bind()?;
        CudaStream::new(self.id)
    }

    fn alloc(&self, bytes: usize) -> DeviceResult<DevicePtr> {
        if bytes == 0 {
            return Ok(DevicePtr::NULL);
        }
        self.bind()?;
        let mut dptr: CUdeviceptr = 0;
        // SAFETY: writes a device pointer through the out-pointer; the result
        // is owned by the returned `DevicePtr`.
        let code = unsafe { cuMemAlloc(&raw mut dptr, bytes) };
        if code != CUDA_SUCCESS {
            return Err(DeviceError::Alloc {
                backend: BACKEND,
                device: self.id,
                bytes,
                reason: error_string(code),
            });
        }
        Ok(DevicePtr(dptr as usize))
    }

    unsafe fn dealloc(&self, ptr: DevicePtr, _bytes: usize) -> DeviceResult<()> {
        if ptr.is_null() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller's `Device::dealloc` contract — `ptr` came from a prior
        // `alloc` on this device and is not aliased. Null handled above.
        check(unsafe { cuMemFree(ptr.0 as CUdeviceptr) }, "cuMemFree")
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
        // SAFETY: caller's `Device::memcpy_async` contract — `src`/`dst` are
        // valid for `bytes` in their respective address spaces per `dir`, not
        // aliased by another pending op on `stream`. The driver API encodes
        // direction in the function name; a `DevicePtr` holds a device address
        // (cast to `CUdeviceptr`) or a host address (cast to `*mut c_void`)
        // per `dir`.
        let s = stream.raw();
        let code = unsafe {
            match dir {
                CopyDirection::HostToDevice => {
                    cuMemcpyHtoDAsync(dst.0 as CUdeviceptr, src.0 as *const _, bytes, s)
                }
                CopyDirection::DeviceToHost => {
                    cuMemcpyDtoHAsync(dst.0 as *mut _, src.0 as CUdeviceptr, bytes, s)
                }
                CopyDirection::DeviceToDevice => {
                    cuMemcpyDtoDAsync(dst.0 as CUdeviceptr, src.0 as CUdeviceptr, bytes, s)
                }
            }
        };
        check(code, "cuMemcpyAsync")
    }

    fn synchronize(&self) -> DeviceResult<()> {
        self.bind()?;
        // SAFETY: synchronises the current context, which `bind` just set.
        check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")
    }

    fn new_event(&self) -> DeviceResult<Self::Event> {
        self.bind()?;
        CudaEvent::new(self.id)
    }

    fn new_timing_event(&self) -> DeviceResult<Self::Event> {
        self.bind()?;
        CudaEvent::new_timing(self.id)
    }

    fn bind(&self) -> DeviceResult<()> {
        CudaDevice::bind(self)
    }

    unsafe fn memcpy_peer_async(
        &self,
        stream: &Self::Stream,
        dst: DevicePtr,
        _dst_device_id: i32,
        src: DevicePtr,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        // PUSH from this device: bind the source context so `stream`'s context
        // is current; UVA routes the copy to `dst`'s device by pointer.
        self.bind()?;
        // SAFETY: caller's `memcpy_peer_async` contract — `dst` valid for
        // `bytes` on its device, `src` valid for `bytes` on this device,
        // unaliased on `stream`; `stream.raw()` belongs to this (source) device.
        check(
            unsafe { cuMemcpyAsync(dst.0 as CUdeviceptr, src.0 as CUdeviceptr, bytes, stream.raw()) },
            "cuMemcpyAsync(peer push)",
        )
    }

    unsafe fn memcpy_peer_in_async(
        &self,
        stream: &Self::Stream,
        dst: DevicePtr,
        src: DevicePtr,
        _src_device_id: i32,
        bytes: usize,
    ) -> DeviceResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        // PULL to this device: bind the destination context so `stream`'s
        // context is current; UVA routes the copy from `src`'s device.
        self.bind()?;
        // SAFETY: caller's `memcpy_peer_in_async` contract — `dst` valid for
        // `bytes` on this device, `src` valid for `bytes` on its device,
        // unaliased on `stream`; `stream.raw()` belongs to this (dest) device.
        check(
            unsafe { cuMemcpyAsync(dst.0 as CUdeviceptr, src.0 as CUdeviceptr, bytes, stream.raw()) },
            "cuMemcpyAsync(peer pull)",
        )
    }
}
