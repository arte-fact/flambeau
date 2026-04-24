//! V2.26.a-i3 — thread-local capture-state recorder.
//!
//! While a capture is active on a thread, every `HipKernel::launch` call
//! appends a record to the thread-local state. `KernelArgs::push_slot`
//! attaches a logical [`ScalarSlot`] tag to a pushed arg so post-capture
//! we can correlate graph kernel nodes (enumerated via `hipGraphGetNodes`)
//! with the logical slots a caller declared.
//!
//! The result is a `SlotMap` that maps each user-declared slot to a
//! (kernel_node_idx, arg_index, arity) triple — enough for
//! [`crate::HipGraphExec::set_slot`] to build a fresh `kernelParams`
//! pointer array and call `hipGraphExecKernelNodeSetParams`.
//!
//! Scope of i3 is infra + a small unit test. Forward-path integration
//! lands in V2.26.a-i4 (`pos`-bearing kernels in `forward_layer_prefill`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// A logical handle to an updateable scalar kernel parameter. Obtained
/// via [`ScalarSlot::new`] and passed to [`super::module::KernelArgs::push_slot`]
/// at capture time to tag the arg as updateable.
///
/// The numeric id is process-global and monotonically increasing. Scopes
/// can reset by using a fresh HipGraphExec — slots from an old exec are
/// meaningless to a new one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScalarSlot {
    id: u32,
}

static NEXT_SLOT_ID: AtomicU32 = AtomicU32::new(1);

impl ScalarSlot {
    /// Allocate a fresh slot id. Cheap — just an atomic fetch-add.
    pub fn new() -> Self {
        let id = NEXT_SLOT_ID.fetch_add(1, Ordering::Relaxed);
        Self { id }
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

/// Per-launch record captured during an active graph capture. Indexes
/// in `tagged_slots` are into this launch's own `KernelArgs` slots.
#[derive(Debug)]
pub(crate) struct LaunchRecord {
    /// Total number of kernel arguments this launch pushed (used later
    /// to size the fresh pointer array we hand to
    /// `hipGraphExecKernelNodeSetParams`).
    pub arity: usize,
    /// Slots the launch tagged via `push_slot`. Each entry is
    /// `(slot, arg_index_within_this_launch)`.
    pub tagged_slots: Vec<(ScalarSlot, usize)>,
}

#[derive(Debug, Default)]
pub(crate) struct CaptureState {
    pub launches: Vec<LaunchRecord>,
}

thread_local! {
    static CAPTURE_STATE: RefCell<Option<CaptureState>> = const { RefCell::new(None) };
}

/// Enable capture-state recording on the current thread. Returned
/// guard disables it in Drop.
///
/// Nested captures are not supported — calling this while another scope
/// is active is a bug and panics.
pub(crate) struct CaptureScope {
    _no_send: std::marker::PhantomData<*const ()>,
}

impl CaptureScope {
    pub fn begin() -> Self {
        CAPTURE_STATE.with(|s| {
            let mut slot = s.borrow_mut();
            assert!(
                slot.is_none(),
                "nested HipGraphExec::capture on the same thread is unsupported"
            );
            *slot = Some(CaptureState::default());
        });
        Self {
            _no_send: std::marker::PhantomData,
        }
    }

    /// Consume the scope, return the collected launches. Replaces
    /// the thread-local with `None` (fresh state for the next capture).
    pub fn end(self) -> CaptureState {
        let taken = CAPTURE_STATE.with(|s| s.borrow_mut().take());
        // Drop the guard without its Drop impl re-clearing the slot.
        std::mem::forget(self);
        taken.expect("CaptureScope::end called with no active capture")
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        // Only reached if `end` wasn't called — e.g. panic unwind.
        // Clear state so the next capture starts fresh.
        CAPTURE_STATE.with(|s| *s.borrow_mut() = None);
    }
}

/// Called by `HipKernel::launch` before submitting a launch to HIP. If
/// capture is active, appends a [`LaunchRecord`] to the current state.
/// No-op when not capturing. `arity` is the total number of kernel args
/// pushed; `tagged` is the subset that were pushed via `push_slot`.
pub(crate) fn record_launch(arity: usize, tagged: &[(ScalarSlot, usize)]) {
    CAPTURE_STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.launches.push(LaunchRecord {
                arity,
                tagged_slots: tagged.to_vec(),
            });
        }
    });
}

/// Post-capture map from a [`ScalarSlot`] to the exact kernel-node +
/// arg index the slot was tagged at. Built by zipping recorded launches
/// with kernel-node handles in dispatch order.
#[derive(Debug, Default)]
pub struct SlotMap {
    /// slot -> (kernel_node_idx, arg_index, launch_arity)
    entries: HashMap<ScalarSlot, SlotBinding>,
}

#[derive(Clone, Copy, Debug)]
pub struct SlotBinding {
    pub kernel_node_idx: usize,
    pub arg_index: usize,
    pub arity: usize,
}

impl SlotMap {
    pub fn get(&self, slot: ScalarSlot) -> Option<&SlotBinding> {
        self.entries.get(&slot)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build a SlotMap by zipping recorded launches 1:1 with kernel
    /// node count. Fails if the counts don't match — indicates the
    /// capture closure did something the recorder didn't see (e.g.
    /// issued a kernel via a path that bypasses `HipKernel::launch`).
    pub(crate) fn from_recorder(
        launches: &[LaunchRecord],
        kernel_node_count: usize,
    ) -> Result<Self, String> {
        if launches.len() != kernel_node_count {
            return Err(format!(
                "slot-map build: recorded {} launches but captured graph has {} kernel nodes — \
                 launch-recorder saw fewer (bypassed launch path?) or more (untracked driver kernels)",
                launches.len(),
                kernel_node_count
            ));
        }
        let mut entries: HashMap<ScalarSlot, SlotBinding> = HashMap::new();
        for (node_idx, launch) in launches.iter().enumerate() {
            for &(slot, arg_idx) in &launch.tagged_slots {
                let binding = SlotBinding {
                    kernel_node_idx: node_idx,
                    arg_index: arg_idx,
                    arity: launch.arity,
                };
                if entries.insert(slot, binding).is_some() {
                    return Err(format!(
                        "slot {:?} declared at multiple launch sites in one capture — \
                         use a fresh slot per launch",
                        slot
                    ));
                }
            }
        }
        Ok(Self { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_ids_are_monotonic() {
        let a = ScalarSlot::new();
        let b = ScalarSlot::new();
        assert!(b.id() > a.id(), "{} !> {}", b.id(), a.id());
    }

    #[test]
    fn capture_scope_toggles_thread_local() {
        // No active state to start.
        let noop = CAPTURE_STATE.with(|s| s.borrow().is_some());
        assert!(!noop);
        let scope = CaptureScope::begin();
        let active = CAPTURE_STATE.with(|s| s.borrow().is_some());
        assert!(active);
        let state = scope.end();
        let cleaned = CAPTURE_STATE.with(|s| s.borrow().is_some());
        assert!(!cleaned);
        assert!(state.launches.is_empty());
    }

    #[test]
    fn record_launch_appends_only_during_capture() {
        // Without capture, record is a no-op.
        let slot = ScalarSlot::new();
        record_launch(3, &[(slot, 1)]);
        let empty = CAPTURE_STATE.with(|s| s.borrow().is_none());
        assert!(empty);

        // With capture, each call appends.
        let scope = CaptureScope::begin();
        record_launch(4, &[(slot, 2)]);
        record_launch(2, &[]);
        let state = scope.end();
        assert_eq!(state.launches.len(), 2);
        assert_eq!(state.launches[0].arity, 4);
        assert_eq!(state.launches[0].tagged_slots, vec![(slot, 2)]);
        assert!(state.launches[1].tagged_slots.is_empty());
    }

    #[test]
    fn slot_map_from_recorder_zips_launches_with_nodes() {
        let slot_a = ScalarSlot::new();
        let slot_b = ScalarSlot::new();
        let launches = vec![
            LaunchRecord {
                arity: 4,
                tagged_slots: vec![(slot_a, 3)],
            },
            LaunchRecord {
                arity: 3,
                tagged_slots: vec![(slot_b, 2)],
            },
        ];
        let map = SlotMap::from_recorder(&launches, 2).expect("zip");
        let ba = map.get(slot_a).expect("slot_a present");
        assert_eq!(ba.kernel_node_idx, 0);
        assert_eq!(ba.arg_index, 3);
        assert_eq!(ba.arity, 4);
        let bb = map.get(slot_b).expect("slot_b present");
        assert_eq!(bb.kernel_node_idx, 1);
        assert_eq!(bb.arg_index, 2);
        assert_eq!(bb.arity, 3);
    }

    #[test]
    fn slot_map_count_mismatch_is_error() {
        let launches = vec![LaunchRecord {
            arity: 4,
            tagged_slots: vec![],
        }];
        assert!(SlotMap::from_recorder(&launches, 2).is_err());
    }
}
