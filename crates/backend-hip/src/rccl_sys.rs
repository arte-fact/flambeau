//! Minimal FFI to RCCL (`librccl`). Hand-written — see `sys.rs` for the
//! HIP-runtime equivalent and the same rationale.

#![allow(
    non_camel_case_types,
    non_snake_case,
    dead_code,
    reason = "hand-written FFI bindings mirror C NCCL/RCCL symbol names; `dead_code` \
              covers symbols held for future collectives that are declared but not yet \
              wired."
)]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "RCCL FFI — every unsafe block wraps an ncclGroup* / ncclAllReduce* / \
              ncclCommInitAll call with comm / stream / buffer pointers validated at \
              HipMesh construction and owned for the communicators lifetime."
)]

use std::os::raw::{c_char, c_int, c_void};

use crate::sys::hipStream_t;

pub const NCCL_UNIQUE_ID_BYTES: usize = 128;
pub const NCCL_SUCCESS: c_int = 0;

pub type ncclComm_t = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [c_char; NCCL_UNIQUE_ID_BYTES],
}

impl Default for ncclUniqueId {
    fn default() -> Self {
        Self {
            internal: [0; NCCL_UNIQUE_ID_BYTES],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ncclRedOp_t {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ncclDataType_t {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

extern "C" {
    pub fn ncclGetErrorString(result: c_int) -> *const c_char;

    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> c_int;
    pub fn ncclCommInitAll(comms: *mut ncclComm_t, ndev: c_int, devlist: *const c_int) -> c_int;
    pub fn ncclCommInitRank(
        newcomm: *mut ncclComm_t,
        nranks: c_int,
        comm_id: ncclUniqueId,
        rank: c_int,
    ) -> c_int;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> c_int;

    pub fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: hipStream_t,
    ) -> c_int;

    pub fn ncclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: hipStream_t,
    ) -> c_int;

    pub fn ncclBroadcast(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: c_int,
        comm: ncclComm_t,
        stream: hipStream_t,
    ) -> c_int;

    pub fn ncclSend(
        sendbuff: *const c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: c_int,
        comm: ncclComm_t,
        stream: hipStream_t,
    ) -> c_int;
    pub fn ncclRecv(
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: c_int,
        comm: ncclComm_t,
        stream: hipStream_t,
    ) -> c_int;

    pub fn ncclGroupStart() -> c_int;
    pub fn ncclGroupEnd() -> c_int;
}

pub fn nccl_error_string(code: c_int) -> String {
    if code == NCCL_SUCCESS {
        return "ncclSuccess".into();
    }
    unsafe {
        let p = ncclGetErrorString(code);
        if p.is_null() {
            return format!("ncclResult {code} (no string)");
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
