//! Hand-written FFI to NCCL (`libnccl`, `nccl*`): the collective surface the
//! CUDA `Cluster` needs — unique-id / comm init / all-reduce / destroy + error
//! text + the group API. Mirrors the `sys.rs` style (opaque handles, i32
//! results formatted via `ncclGetErrorString`). Linked only under the `nccl`
//! cargo feature; NCCL 2.28.x ABI.

use std::os::raw::{c_char, c_int, c_void};

/// `ncclResult_t::ncclSuccess`.
pub const NCCL_SUCCESS: c_int = 0;

/// `ncclComm_t` is `typedef struct ncclComm* ncclComm_t;` — opaque.
pub type NcclComm = *mut c_void;

/// `ncclUniqueId` is `struct { char internal[128]; }` (NCCL_UNIQUE_ID_BYTES).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NcclUniqueId {
    pub internal: [c_char; 128],
}

impl Default for NcclUniqueId {
    fn default() -> Self {
        NcclUniqueId { internal: [0; 128] }
    }
}

// `ncclDataType_t` (nccl.h). Only the reduction payloads the AR path issues.
pub const NCCL_FLOAT16: c_int = 6;
pub const NCCL_FLOAT32: c_int = 7;

// `ncclRedOp_t`. The forward AR path only ever sums.
pub const NCCL_SUM: c_int = 0;

#[link(name = "nccl")]
extern "C" {
    pub fn ncclGetErrorString(result: c_int) -> *const c_char;
    pub fn ncclGetVersion(version: *mut c_int) -> c_int;
    pub fn ncclGetUniqueId(unique_id: *mut NcclUniqueId) -> c_int;
    /// Single-process N-GPU init: `comms[i]` is rank `i` on `devlist[i]`.
    pub fn ncclCommInitAll(comms: *mut NcclComm, ndev: c_int, devlist: *const c_int) -> c_int;
    /// One rank of a world bootstrapped from a shared `ncclUniqueId`.
    pub fn ncclCommInitRank(comm: *mut NcclComm, nranks: c_int, commId: NcclUniqueId, rank: c_int) -> c_int;
    pub fn ncclCommDestroy(comm: NcclComm) -> c_int;
    pub fn ncclCommCount(comm: NcclComm, count: *mut c_int) -> c_int;
    pub fn ncclCommUserRank(comm: NcclComm, rank: *mut c_int) -> c_int;
    pub fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: c_int,
        op: c_int,
        comm: NcclComm,
        stream: *mut c_void,
    ) -> c_int;
    /// Aggregate the enclosed collective calls so multiple ranks issued from
    /// one thread do not deadlock (single-process multi-GPU idiom).
    pub fn ncclGroupStart() -> c_int;
    pub fn ncclGroupEnd() -> c_int;
}

/// Format an `ncclResult_t` as `"<text> (code N)"`.
pub fn nccl_error_text(code: c_int) -> String {
    // SAFETY: ncclGetErrorString returns a static NUL-terminated string for any
    // result code.
    let s = unsafe {
        let p = ncclGetErrorString(code);
        if p.is_null() {
            String::from("unknown")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    format!("{s} (code {code})")
}
