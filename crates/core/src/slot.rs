//! Graph-capture slot handles. A *slot* is a logical, backend-neutral handle
//! to an updateable parameter of a captured graph node — a scalar kernel arg
//! ([`ScalarSlot`]) or a memcpy node's pointer ([`MemcpySlot`]). The handle is
//! just a process-unique id; the per-backend graph-capture machinery
//! (`HipGraphExec`, the future `CudaGraphExec`) maps each id to a concrete
//! node + arg index.
//!
//! The id type lives in `core` so the portable op surface (`ops::Ops`, whose
//! capture-aware methods take slots) and both backends' graph executors share
//! one definition — a CUDA op impl must not import a HIP type to name a slot.

use std::sync::atomic::{AtomicU32, Ordering};

// ScalarSlot and MemcpySlot draw from one counter: type-distinct at compile
// time, but a shared namespace keeps every id unique across the process.
static NEXT_SLOT_ID: AtomicU32 = AtomicU32::new(1);

/// A logical handle to an updateable scalar kernel parameter. Allocated via
/// [`ScalarSlot::new`] and tagged onto a kernel arg at capture time; after
/// capture the backend graph executor resolves it to a `(node, arg_index)`
/// pair so the scalar can be rewritten per replay. Slots from one graph
/// executor are meaningless to another — reset by using a fresh executor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScalarSlot {
    id: u32,
}

/// A logical handle to an updateable memcpy node (typically its dst pointer —
/// src + count are usually fixed, as in KV-cache append). Allocated via
/// [`MemcpySlot::new`] and tagged at capture time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemcpySlot {
    id: u32,
}

impl ScalarSlot {
    /// Allocate a fresh slot id. Cheap — one atomic fetch-add.
    pub fn new() -> Self {
        Self {
            id: NEXT_SLOT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }
}

impl Default for ScalarSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl MemcpySlot {
    /// Allocate a fresh slot id. Cheap — one atomic fetch-add.
    pub fn new() -> Self {
        Self {
            id: NEXT_SLOT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }
}

impl Default for MemcpySlot {
    fn default() -> Self {
        Self::new()
    }
}
