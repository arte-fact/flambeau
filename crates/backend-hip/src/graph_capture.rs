//! 6.a-i3 — thread-local capture-state recorder.
//! While a capture is active on a thread, every `HipKernel::launch` call
//! appends a record to the thread-local state. `KernelArgs::push_slot`
//! attaches a logical [`ScalarSlot`] tag to a pushed arg so post-capture
//! we can correlate graph kernel nodes (enumerated via `hipGraphGetNodes`)
//! with the logical slots a caller declared.
//! The result is a `SlotMap` that maps each user-declared slot to a
//! (kernel_node_idx, arg_index, arity) triple — enough for
//! [`crate::HipGraphExec::set_slot`] to build a fresh `kernelParams`
//! pointer array and call `hipGraphExecKernelNodeSetParams`.
//! Scope of i3 is infra + a small unit test. Forward-path integration
//! lands in 6.a-i4 (`pos`-bearing kernels in `forward_layer_prefill`).

use std::cell::RefCell;
use std::collections::HashMap;

// The slot id types are backend-neutral (a process-unique id); they live in
// `core` so the portable op surface and both backends' graph executors share
// one definition. Re-exported here so existing `graph_capture::ScalarSlot` /
// `flambeau_backend_hip::ScalarSlot` paths keep resolving. The HIP-specific
// capture machinery (records, SlotMap) stays below.
pub use flambeau_core::{MemcpySlot, ScalarSlot};

/// Per-launch record captured during an active graph capture. Indexes
/// in `tagged_slots` are into this launch's own `KernelArgs` slots.
#[derive(Clone, Debug)]
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
    /// 6.a-i5b — memcpy issues observed during this capture, in
    /// dispatch order. Each entry records whether the caller tagged the
    /// memcpy with a [`MemcpySlot`] and the initial (dst, src, count,
    /// kind) so post-capture we can seed the exec's memcpy shadow.
    pub memcpys: Vec<MemcpyRecord>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MemcpyRecord {
    pub slot: Option<MemcpySlot>,
    pub dst: usize,
    pub src: usize,
    pub count: usize,
    pub kind: crate::sys::hipMemcpyKind,
}

thread_local! {
    static CAPTURE_STATE: RefCell<Option<CaptureState>> = const { RefCell::new(None) };
}

/// Enable capture-state recording on the current thread. Returned
/// guard disables it in Drop.
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

/// 6.a-i5b — called by `HipDevice::memcpy_async*` before submitting
/// a memcpy. If capture is active, appends a [`MemcpyRecord`] so
/// post-capture we can zip records with memcpy-type graph nodes.
/// `slot=None` means the caller isn't interested in updating this
/// memcpy across replays — we still record it so the dispatch-order
/// cursor stays aligned with the graph's memcpy nodes.
pub(crate) fn record_memcpy(
    slot: Option<MemcpySlot>,
    dst: usize,
    src: usize,
    count: usize,
    kind: crate::sys::hipMemcpyKind,
) {
    CAPTURE_STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut() {
            state.memcpys.push(MemcpyRecord {
                slot,
                dst,
                src,
                count,
                kind,
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
    /// 6.a-i5b — memcpy-slot bindings. memcpy_node_idx indexes into
    /// the exec's ordered list of memcpy-type graph nodes (SEPARATE
    /// from kernel_nodes — their indices are not interchangeable).
    memcpy_entries: HashMap<MemcpySlot, MemcpyBinding>,
}

#[derive(Clone, Copy, Debug)]
pub struct MemcpyBinding {
    pub memcpy_node_idx: usize,
    /// Initial params captured at record time; `set_memcpy_slot`
    /// updates a mutable shadow keyed by slot, so src/count/kind can
    /// survive dst-only updates.
    pub dst: usize,
    pub src: usize,
    pub count: usize,
    pub kind: crate::sys::hipMemcpyKind,
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

    /// Iterate bindings that target `kernel_node_idx`. Used by
    /// `HipGraphExec::capture` to size each node's shadow.
    pub fn bindings_for_node(&self, kernel_node_idx: usize) -> impl Iterator<Item = &SlotBinding> {
        self.entries
            .values()
            .filter(move |b| b.kernel_node_idx == kernel_node_idx)
    }

    /// Build a SlotMap by zipping recorded launches 1:1 with kernel
    /// node count, and memcpy records 1:1 with memcpy node count.
    /// Fails if either count is off — indicates the capture closure
    /// did something the recorder didn't see (e.g. issued a kernel via
    /// a path that bypasses `HipKernel::launch`, or a memcpy via a
    /// path that bypasses the `memcpy_async*` recorder).
    pub(crate) fn from_recorder(
        launches: &[LaunchRecord],
        kernel_node_count: usize,
        memcpys: &[MemcpyRecord],
        memcpy_node_count: usize,
    ) -> Result<Self, String> {
        if launches.len() != kernel_node_count {
            return Err(format!(
                "slot-map build: recorded {} launches but captured graph has {} kernel nodes — \
                 launch-recorder saw fewer (bypassed launch path?) or more (untracked driver kernels)",
                launches.len(),
                kernel_node_count
            ));
        }
        if memcpys.len() != memcpy_node_count {
            return Err(format!(
                "slot-map build: recorded {} memcpys but captured graph has {} memcpy nodes — \
                 recorder / graph out of sync (untagged memcpy path? unusual kind fold-up?)",
                memcpys.len(),
                memcpy_node_count
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
        let mut memcpy_entries: HashMap<MemcpySlot, MemcpyBinding> = HashMap::new();
        for (memcpy_node_idx, rec) in memcpys.iter().enumerate() {
            if let Some(slot) = rec.slot {
                let binding = MemcpyBinding {
                    memcpy_node_idx,
                    dst: rec.dst,
                    src: rec.src,
                    count: rec.count,
                    kind: rec.kind,
                };
                if memcpy_entries.insert(slot, binding).is_some() {
                    return Err(format!(
                        "memcpy slot {:?} declared at multiple sites — use a fresh slot",
                        slot
                    ));
                }
            }
        }
        Ok(Self {
            entries,
            memcpy_entries,
        })
    }

    pub fn get_memcpy(&self, slot: MemcpySlot) -> Option<&MemcpyBinding> {
        self.memcpy_entries.get(&slot)
    }

    pub fn memcpy_len(&self) -> usize {
        self.memcpy_entries.len()
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
        let map = SlotMap::from_recorder(&launches, 2, &[], 0).expect("zip");
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
        assert!(SlotMap::from_recorder(&launches, 2, &[], 0).is_err());
    }

    #[test]
    fn memcpy_slot_zips_into_slot_map() {
        let ms = MemcpySlot::new();
        let memcpys = vec![
            MemcpyRecord {
                slot: None,
                dst: 0x1000,
                src: 0x2000,
                count: 512,
                kind: crate::sys::hipMemcpyKind::DeviceToDevice,
            },
            MemcpyRecord {
                slot: Some(ms),
                dst: 0x3000,
                src: 0x4000,
                count: 256,
                kind: crate::sys::hipMemcpyKind::DeviceToDevice,
            },
        ];
        let map = SlotMap::from_recorder(&[], 0, &memcpys, 2).expect("zip");
        assert_eq!(map.memcpy_len(), 1);
        let binding = map.get_memcpy(ms).expect("ms bound");
        assert_eq!(binding.memcpy_node_idx, 1);
        assert_eq!(binding.dst, 0x3000);
        assert_eq!(binding.src, 0x4000);
        assert_eq!(binding.count, 256);
    }

    #[test]
    fn record_memcpy_appends_only_during_capture() {
        // Outside capture: no-op.
        record_memcpy(
            None,
            0x1000,
            0x2000,
            64,
            crate::sys::hipMemcpyKind::DeviceToDevice,
        );

        let scope = CaptureScope::begin();
        record_memcpy(
            None,
            0x1000,
            0x2000,
            64,
            crate::sys::hipMemcpyKind::DeviceToDevice,
        );
        record_memcpy(
            Some(MemcpySlot::new()),
            0x3000,
            0x4000,
            128,
            crate::sys::hipMemcpyKind::HostToDevice,
        );
        let state = scope.end();
        assert_eq!(state.memcpys.len(), 2);
        assert!(state.memcpys[0].slot.is_none());
        assert!(state.memcpys[1].slot.is_some());
        assert_eq!(state.memcpys[1].count, 128);
    }
}
