//! Env-gated intermediate-state dump for forward-path parity bisection.
//!
//! Insert `probe::f32(name, ptr, n, &device, stream)?;` (or `f16`)
//! after a kernel writes to `ptr`. With the `parity-probe` feature
//! off, both functions are inlined no-ops and disappear from release
//! builds. With the feature on, each call site has its own
//! environment-variable knob:
//!
//! ```text
//! FLAMBEAU_PROBE_<NAME_UPPER>=<idx>
//! ```
//!
//! where dots in `name` become underscores. `<idx>` is the zero-based
//! index of the call this run should dump (the rest stay silent).
//! Each `name` keeps an independent counter, so adding a probe under
//! a new `name` doesn't shift the others.

use crate::{HipDevice, HipStream};
use anyhow::Result;
use flambeau_core::DevicePtr;

#[cfg(not(feature = "parity-probe"))]
#[inline(always)]
pub fn f32(_: &'static str, _: DevicePtr, _: usize, _: &HipDevice, _: &HipStream) -> Result<()> {
    Ok(())
}

#[cfg(not(feature = "parity-probe"))]
#[inline(always)]
pub fn f16(_: &'static str, _: DevicePtr, _: usize, _: &HipDevice, _: &HipStream) -> Result<()> {
    Ok(())
}

#[cfg(feature = "parity-probe")]
pub use enabled::{f16, f32};

#[cfg(feature = "parity-probe")]
mod enabled {
    use super::*;
    use flambeau_core::{CopyDirection, Device, Stream};
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn env_target(name: &str) -> Option<usize> {
        let key = name
            .chars()
            .map(|c| {
                if c == '.' {
                    '_'
                } else {
                    c.to_ascii_uppercase()
                }
            })
            .collect::<String>();
        std::env::var(format!("FLAMBEAU_PROBE_{key}"))
            .ok()
            .and_then(|s| s.parse().ok())
    }

    fn next_idx(name: &'static str) -> usize {
        use std::sync::OnceLock;
        static COUNTERS: OnceLock<Mutex<HashMap<&'static str, usize>>> = OnceLock::new();
        let m = COUNTERS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut g = m.lock().expect("probe counter mutex");
        let e = g.entry(name).or_insert(0);
        let idx = *e;
        *e += 1;
        idx
    }

    pub fn f32(
        name: &'static str,
        ptr: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let Some(target) = env_target(name) else {
            return Ok(());
        };
        let idx = next_idx(name);
        if idx != target {
            return Ok(());
        }
        let n = n_elems.min(16);
        let mut host = vec![0f32; n];
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(host.as_mut_ptr() as usize),
                ptr,
                n * 4,
            )?;
        }
        Stream::synchronize(stream)?;
        eprintln!("[probe {name} idx={idx}] F32 head[0..{n}] = {host:?}");
        Ok(())
    }

    pub fn f16(
        name: &'static str,
        ptr: DevicePtr,
        n_elems: usize,
        device: &HipDevice,
        stream: &HipStream,
    ) -> Result<()> {
        let Some(target) = env_target(name) else {
            return Ok(());
        };
        let idx = next_idx(name);
        if idx != target {
            return Ok(());
        }
        let n = n_elems.min(16);
        let mut host = vec![0u16; n];
        unsafe {
            device.memcpy_async(
                stream,
                CopyDirection::DeviceToHost,
                DevicePtr(host.as_mut_ptr() as usize),
                ptr,
                n * 2,
            )?;
        }
        Stream::synchronize(stream)?;
        let view: Vec<f32> = host
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        eprintln!("[probe {name} idx={idx}] F16 head[0..{n}] = {view:?}");
        Ok(())
    }
}
