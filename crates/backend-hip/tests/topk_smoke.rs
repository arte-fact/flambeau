//! TopK + softmax correctness test against a CPU reference.

use flambeau_backend_hip::{
    device_count, HipDevice, HipKernel, HipModule, KernelArgs, LaunchCfg,
};
use flambeau_core::{CopyDirection, Device, DevicePtr, Stream};
use flambeau_kernels_hip as kernels;

fn maybe_skip() -> bool {
    match device_count() {
        Ok(n) if n >= 1 => true,
        _ => {
            eprintln!("[skip] no HIP device");
            false
        }
    }
}

fn cpu_topk_softmax(
    logits: &[f32],
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut idxs = vec![0i32; n_tokens * k];
    let mut wts = vec![0f32; n_tokens * k];
    for t in 0..n_tokens {
        let row = &logits[t * n_experts..(t + 1) * n_experts];
        // Sort (value, index) by value desc with lower-index-wins tiebreak.
        let mut pairs: Vec<(f32, i32)> =
            row.iter().enumerate().map(|(i, &v)| (v, i as i32)).collect();
        pairs.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        // Softmax over top-k raw logits.
        let top: Vec<f32> = pairs[..k].iter().map(|p| p.0).collect();
        let m = top.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = top.iter().map(|v| (v - m).exp()).sum();
        for i in 0..k {
            idxs[t * k + i] = pairs[i].1;
            wts[t * k + i] = ((pairs[i].0 - m).exp()) / sum;
        }
    }
    (idxs, wts)
}

#[test]
fn topk_matches_cpu_reference() {
    if !maybe_skip() {
        return;
    }
    let dev = HipDevice::new(0).unwrap();
    dev.bind().unwrap();

    let kb = kernels::hsaco("topk_f32").unwrap();
    let module = HipModule::load(dev.id(), kb).unwrap();
    let kernel: HipKernel<'_> = module.kernel("flambeau_topk_softmax_f32").unwrap();

    let n_tokens = 64usize;
    let n_experts = 128usize;
    let k = 8usize;

    // Deterministic random logits.
    let logits: Vec<f32> = (0..n_tokens * n_experts)
        .map(|i| {
            let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B97F4A7C15) as i32;
            (x as f32) / (i32::MAX as f32)
        })
        .collect();

    let (ref_idx, ref_wts) = cpu_topk_softmax(&logits, n_tokens, n_experts, k);

    let d_logits = {
        let bytes = logits.len() * 4;
        let d = dev.alloc(bytes).unwrap();
        unsafe {
            dev.memcpy_async(
                dev.default_stream(),
                CopyDirection::HostToDevice,
                d,
                DevicePtr(logits.as_ptr() as usize),
                bytes,
            )
            .unwrap();
        }
        dev.default_stream().synchronize().unwrap();
        d
    };
    let d_idx = dev.alloc(n_tokens * k * 4).unwrap();
    let d_wts = dev.alloc(n_tokens * k * 4).unwrap();

    let stream = dev.default_stream();
    let n_tokens_i = n_tokens as i32;
    let n_experts_i = n_experts as i32;
    let k_i = k as i32;
    let d_l_ptr: u64 = d_logits.as_usize() as u64;
    let d_i_ptr: u64 = d_idx.as_usize() as u64;
    let d_w_ptr: u64 = d_wts.as_usize() as u64;
    let mut args = KernelArgs::new();
    args.push(&d_l_ptr);
    args.push(&d_i_ptr);
    args.push(&d_w_ptr);
    args.push(&n_tokens_i);
    args.push(&n_experts_i);
    args.push(&k_i);
    let cfg = LaunchCfg::one_d(n_tokens as u32, n_experts as u32);
    unsafe { kernel.launch(stream, cfg, args).unwrap() };
    stream.synchronize().unwrap();

    let mut got_idx = vec![0i32; n_tokens * k];
    let mut got_wts = vec![0f32; n_tokens * k];
    unsafe {
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(got_idx.as_mut_ptr() as usize),
            d_idx,
            n_tokens * k * 4,
        )
        .unwrap();
        dev.memcpy_async(
            stream,
            CopyDirection::DeviceToHost,
            DevicePtr(got_wts.as_mut_ptr() as usize),
            d_wts,
            n_tokens * k * 4,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    unsafe {
        dev.dealloc(d_logits, logits.len() * 4).unwrap();
        dev.dealloc(d_idx, n_tokens * k * 4).unwrap();
        dev.dealloc(d_wts, n_tokens * k * 4).unwrap();
    }

    for t in 0..n_tokens {
        for i in 0..k {
            let g = got_idx[t * k + i];
            let r = ref_idx[t * k + i];
            assert_eq!(
                g, r,
                "token {t} pos {i}: got idx {g}, want {r}"
            );
            let gw = got_wts[t * k + i];
            let rw = ref_wts[t * k + i];
            assert!(
                (gw - rw).abs() < 1e-5,
                "token {t} pos {i}: got w={gw} want w={rw}"
            );
        }
    }
}
