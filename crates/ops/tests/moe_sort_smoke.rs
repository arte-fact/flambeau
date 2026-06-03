//! smoke test for `moe_sort_by_expert`: verify that the 3-kernel
//! sort groups (token, slot) pair indices by expert_id correctly.

#![cfg(feature = "hip")]
#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "test fixture — every unsafe block is a kernel launch or memcpy_async \
              over host/device buffers that live for the bounded synchronize that \
              follows; per-site SAFETY comments would just repeat this."
)]

use anyhow::Result;
use flambeau_backend_hip::{device_count, HipDevice};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_ops::{moe, OpsRegistry};

fn dev_or_skip() -> Option<HipDevice> {
    if device_count().ok()? < 1 {
        return None;
    }
    let d = HipDevice::new(0).ok()?;
    d.bind().ok()?;
    Some(d)
}

fn upload_i32(dev: &HipDevice, data: &[i32]) -> DevicePtr {
    let bytes = std::mem::size_of_val(data);
    let d = dev.alloc(bytes).unwrap();
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d,
            DevicePtr(data.as_ptr() as usize),
            bytes,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    d
}

fn download_i32(dev: &HipDevice, p: DevicePtr, n: usize) -> Vec<i32> {
    let mut host = vec![0i32; n];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::DeviceToHost,
            DevicePtr(host.as_mut_ptr() as usize),
            p,
            n * 4,
        )
        .unwrap();
    }
    dev.default_stream().synchronize().unwrap();
    host
}

#[test]
fn moe_sort_by_expert_groups_pairs() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        eprintln!("skip — no HIP");
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;

    // 8 tokens × top_k=4 = 32 pairs, 6 experts, deterministic mix.
    let n_experts = 6;
    let total = 32;
    let expert_ids: Vec<i32> = (0..total)
        .map(|i| ((i * 7 + 3) % n_experts) as i32)
        .collect();

    // Device allocations.
    let d_ids = upload_i32(&dev, &expert_ids);
    let d_counts = dev.alloc(n_experts * 4)?;
    let d_offsets = dev.alloc((n_experts + 1) * 4)?;
    let d_cursors = dev.alloc(n_experts * 4)?;
    let d_sorted = dev.alloc(total * 4)?;

    // Zero counts + cursors.
    let zeros = vec![0i32; n_experts];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_counts,
            DevicePtr(zeros.as_ptr() as usize),
            n_experts * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    moe::moe_sort_by_expert(
        flambeau_ops::OpCtx {
            reg: &reg,
            stream: dev.default_stream(),
        },
        flambeau_ops::MoeSortBuffers {
            expert_ids: d_ids,
            counts: d_counts,
            offsets: d_offsets,
            cursors: d_cursors,
            sorted_pair_idx: d_sorted,
        },
        flambeau_ops::MoeSortShape { total, n_experts },
    )?;

    // Download results.
    let counts = download_i32(&dev, d_counts, n_experts);
    let offsets = download_i32(&dev, d_offsets, n_experts + 1);
    let sorted = download_i32(&dev, d_sorted, total);

    // CPU reference.
    let mut cpu_counts = vec![0i32; n_experts];
    for &e in &expert_ids {
        cpu_counts[e as usize] += 1;
    }
    assert_eq!(counts, cpu_counts, "histogram mismatch");

    let mut cpu_offsets = vec![0i32; n_experts + 1];
    for e in 0..n_experts {
        cpu_offsets[e + 1] = cpu_offsets[e] + cpu_counts[e];
    }
    assert_eq!(offsets, cpu_offsets, "offsets prefix-sum mismatch");
    assert_eq!(offsets[n_experts] as usize, total, "offsets[last] != total");

    // Verify sorted: pairs in [offsets[e], offsets[e+1]) all have expert_id == e.
    for e in 0..n_experts {
        let lo = offsets[e] as usize;
        let hi = offsets[e + 1] as usize;
        for (k, &pair_idx) in sorted[lo..hi].iter().enumerate() {
            let k_abs = lo + k;
            assert_eq!(
                expert_ids[pair_idx as usize], e as i32,
                "expert {e}: sorted[{k_abs}]={pair_idx} has expert_id {}",
                expert_ids[pair_idx as usize]
            );
        }
    }

    // Verify each original pair is present exactly once.
    let mut seen = vec![false; total];
    for &p in &sorted {
        assert!(p >= 0 && (p as usize) < total, "out of range");
        assert!(!seen[p as usize], "pair {p} listed twice");
        seen[p as usize] = true;
    }
    assert!(seen.iter().all(|&b| b), "some pairs missing");

    unsafe {
        dev.dealloc(d_ids, total * 4)?;
        dev.dealloc(d_counts, n_experts * 4)?;
        dev.dealloc(d_offsets, (n_experts + 1) * 4)?;
        dev.dealloc(d_cursors, n_experts * 4)?;
        dev.dealloc(d_sorted, total * 4)?;
    }
    Ok(())
}

#[test]
fn moe_sort_by_expert_qwen3_6_scale() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;

    // Qwen3.6-35B pp=512 scale: n_tokens=512, top_k=8, n_experts=256.
    let n_experts = 256;
    let n_tokens = 512;
    let top_k = 8;
    let total = n_tokens * top_k;

    // Deterministic-but-distributed assignments.
    let expert_ids: Vec<i32> = (0..total)
        .map(|i| ((i * 997 + 41) % n_experts) as i32)
        .collect();

    let d_ids = upload_i32(&dev, &expert_ids);
    let d_counts = dev.alloc(n_experts * 4)?;
    let d_offsets = dev.alloc((n_experts + 1) * 4)?;
    let d_cursors = dev.alloc(n_experts * 4)?;
    let d_sorted = dev.alloc(total * 4)?;

    let zeros = vec![0i32; n_experts];
    unsafe {
        dev.memcpy_async(
            dev.default_stream(),
            CopyDirection::HostToDevice,
            d_counts,
            DevicePtr(zeros.as_ptr() as usize),
            n_experts * 4,
        )?;
    }
    dev.default_stream().synchronize()?;

    moe::moe_sort_by_expert(
        flambeau_ops::OpCtx {
            reg: &reg,
            stream: dev.default_stream(),
        },
        flambeau_ops::MoeSortBuffers {
            expert_ids: d_ids,
            counts: d_counts,
            offsets: d_offsets,
            cursors: d_cursors,
            sorted_pair_idx: d_sorted,
        },
        flambeau_ops::MoeSortShape { total, n_experts },
    )?;

    let counts = download_i32(&dev, d_counts, n_experts);
    let offsets = download_i32(&dev, d_offsets, n_experts + 1);
    let sorted = download_i32(&dev, d_sorted, total);

    let mut cpu_counts = vec![0i32; n_experts];
    for &e in &expert_ids {
        cpu_counts[e as usize] += 1;
    }
    assert_eq!(counts, cpu_counts);
    assert_eq!(offsets[n_experts], total as i32);

    // Spot-check a few buckets.
    for e in [0usize, 42, 127, 255] {
        let lo = offsets[e] as usize;
        let hi = offsets[e + 1] as usize;
        for &p in &sorted[lo..hi] {
            assert_eq!(expert_ids[p as usize], e as i32);
        }
    }

    unsafe {
        dev.dealloc(d_ids, total * 4)?;
        dev.dealloc(d_counts, n_experts * 4)?;
        dev.dealloc(d_offsets, (n_experts + 1) * 4)?;
        dev.dealloc(d_cursors, n_experts * 4)?;
        dev.dealloc(d_sorted, total * 4)?;
    }
    Ok(())
}

#[test]
fn moe_sort_by_expert_padded_groups_in_multiples_of_8() -> Result<()> {
    let Some(dev) = dev_or_skip() else {
        eprintln!("skip — no HIP");
        return Ok(());
    };
    let reg = OpsRegistry::new(&dev)?;

    let n_experts = 6;
    let top_k = 4;
    let max_tokens = 8;
    let total = max_tokens * top_k; // 32 pairs

    // Pattern that gives uneven counts per expert:
    // expert_ids[i] = (i * 7 + 3) mod 6
    let expert_ids: Vec<i32> = (0..total)
        .map(|i| ((i * 7 + 3) % n_experts) as i32)
        .collect();

    let d_ids = upload_i32(&dev, &expert_ids);
    let d_counts = dev.alloc(n_experts * 4)?;
    let d_offsets = dev.alloc((n_experts + 1) * 4)?;
    let d_cursors = dev.alloc(n_experts * 4)?;
    let d_sorted = dev.alloc(total * 4)?;
    let d_padded_off = dev.alloc((n_experts + 1) * 4)?;
    // Upper bound: total + n_experts * 7 (worst-case padding), rounded up.
    let padded_cap = total + n_experts * 8;
    let d_sorted_padded = dev.alloc(padded_cap * 4)?;

    moe::moe_sort_by_expert_padded(
        flambeau_ops::OpCtx {
            reg: &reg,
            stream: dev.default_stream(),
        },
        flambeau_ops::MoeSortPaddedBuffers {
            expert_ids: d_ids,
            counts: d_counts,
            offsets: d_offsets,
            cursors: d_cursors,
            sorted_pair_idx: d_sorted,
            padded_offsets: d_padded_off,
            sorted_pair_idx_padded: d_sorted_padded,
        },
        flambeau_ops::MoeSortPaddedShape {
            total,
            n_experts,
            max_tokens,
            top_k,
        },
    )?;

    let counts = download_i32(&dev, d_counts, n_experts);
    let offsets = download_i32(&dev, d_offsets, n_experts + 1);
    let padded_off = download_i32(&dev, d_padded_off, n_experts + 1);
    let sorted = download_i32(&dev, d_sorted, total);
    let sorted_padded = download_i32(&dev, d_sorted_padded, padded_cap);

    // Verify padded_offsets prefix-sum = sum of ceil(counts[e]/8)*8.
    let mut cpu_padded = vec![0i32; n_experts + 1];
    for e in 0..n_experts {
        let padded_c = ((counts[e] + 7) & !7) as i32;
        cpu_padded[e + 1] = cpu_padded[e] + padded_c;
    }
    assert_eq!(padded_off, cpu_padded, "padded_offsets mismatch");

    // Verify each expert's padded range:
    // - First `counts[e]` entries match the unpadded sorted range.
    // - Remaining (counts..padded_count) entries repeat the last real pair.
    // - Each padded range size is a multiple of 8.
    for e in 0..n_experts {
        let real = counts[e] as usize;
        let padded = (padded_off[e + 1] - padded_off[e]) as usize;
        assert_eq!(
            padded % 8,
            0,
            "expert {e} padded count {padded} not multiple of 8"
        );
        let lo_u = offsets[e] as usize;
        let lo_p = padded_off[e] as usize;
        for i in 0..real {
            assert_eq!(
                sorted_padded[lo_p + i],
                sorted[lo_u + i],
                "expert {e} real slot {i}"
            );
        }
        if real > 0 {
            let last = sorted[lo_u + real - 1];
            for i in real..padded {
                assert_eq!(
                    sorted_padded[lo_p + i],
                    last,
                    "expert {e} padding slot {i} should repeat last real entry"
                );
            }
        }
    }

    unsafe {
        dev.dealloc(d_ids, total * 4)?;
        dev.dealloc(d_counts, n_experts * 4)?;
        dev.dealloc(d_offsets, (n_experts + 1) * 4)?;
        dev.dealloc(d_cursors, n_experts * 4)?;
        dev.dealloc(d_sorted, total * 4)?;
        dev.dealloc(d_padded_off, (n_experts + 1) * 4)?;
        dev.dealloc(d_sorted_padded, padded_cap * 4)?;
    }
    Ok(())
}
