//! Top-K + softmax sampler primitive. Output probs are FULL-VOCAB
//! softmax (`exp(v*inv_temp - max_v) / sum_v exp(...)`) — NOT
//! renormalised over the kept K, so host-side top-p / min-p matches
//! full-V semantics. Caller renormalises when needed.
//! `k ≤ SAMPLER_K_OUT_MAX` (single-block 4096-candidate cap).

use anyhow::bail;
use flambeau_ops::Ops;

pub use flambeau_ops::hip::sampling::SAMPLER_K_OUT_MAX;

use crate::dtype::{F32, I32};
use crate::error::Result;
use crate::tensor::Tensor;

/// `inv_temp = 1.0 / temperature` (pass `1.0` when `temperature ≤ 0`).
pub fn topk_softmax_f32(
    logits: &Tensor<F32>,
    out_ids: &mut Tensor<I32>,
    out_probs: &mut Tensor<F32>,
    vocab: usize,
    k: usize,
    inv_temp: f32,
    ops: &impl Ops,
) -> Result<()> {
    if k == 0 {
        bail!("topk_softmax_f32: k must be >= 1");
    }
    if k > SAMPLER_K_OUT_MAX {
        bail!("topk_softmax_f32: k {k} > SAMPLER_K_OUT_MAX {SAMPLER_K_OUT_MAX}");
    }
    if logits.n_elems < vocab {
        bail!(
            "topk_softmax_f32: logits has {} F32 elems, need >= {vocab}",
            logits.n_elems
        );
    }
    if out_ids.n_elems < k {
        bail!(
            "topk_softmax_f32: out_ids has {} I32 elems, need >= {k}",
            out_ids.n_elems
        );
    }
    if out_probs.n_elems < k {
        bail!(
            "topk_softmax_f32: out_probs has {} F32 elems, need >= {k}",
            out_probs.n_elems
        );
    }
    ops.topk_softmax_f32(logits.ptr, out_ids.ptr, out_probs.ptr, vocab, k, inv_temp)
}

#[cfg(test)]
fn cpu_topk_softmax(logits: &[f32], k: usize, inv_temp: f32) -> (Vec<i32>, Vec<f32>) {
    let scaled: Vec<f32> = logits.iter().map(|&l| l * inv_temp).collect();
    let max_full = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_full: f32 = scaled.iter().map(|&v| (v - max_full).exp()).sum();

    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
    let ids: Vec<i32> = top.iter().map(|(i, _)| *i as i32).collect();
    let probs: Vec<f32> = top
        .iter()
        .map(|(_, v)| (v - max_full).exp() / sum_full)
        .collect();
    (ids, probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flambeau_ops::HipOps;
    use crate::testing::{
        alloc, assert_close_f32, download, free, test_device, test_ops_registry, upload,
    };
    use flambeau_core::Device;

    #[test]
    fn topk_softmax_f32_matches_cpu_reference() {
        const VOCAB: usize = 1024;
        const K: usize = 8;
        const INV_TEMP: f32 = 1.0 / 0.7;

        let device = test_device();
        device.bind().expect("device bind");
        let stream = device.default_stream();
        let reg = test_ops_registry(&device);
        let ops = HipOps::new(&reg, stream);

        let logits_host: Vec<f32> = (0..VOCAB)
            .map(|i| {
                let x = (i as f32) * 0.13_f32.sin() + (i as f32 * 0.07).cos() * 2.0;
                x - (VOCAB as f32) * 0.05
            })
            .collect();
        let (expected_ids, expected_probs) = cpu_topk_softmax(&logits_host, K, INV_TEMP);

        let (logits_t, logits_ptr) = upload::<F32, f32>(&device, &logits_host, logits_host.len());
        let (mut ids_t, ids_ptr) = alloc::<I32>(&device, K);
        let (mut probs_t, probs_ptr) = alloc::<F32>(&device, K);

        topk_softmax_f32(
            &logits_t,
            &mut ids_t,
            &mut probs_t,
            VOCAB,
            K,
            INV_TEMP,
            &ops,
        )
        .expect("topk_softmax_f32");

        let got_ids: Vec<i32> = download::<I32, i32>(&device, &ids_t);
        let got_probs: Vec<f32> = download::<F32, f32>(&device, &probs_t);

        assert_eq!(
            got_ids, expected_ids,
            "top-K IDs diverge (got {got_ids:?}, expected {expected_ids:?})"
        );
        assert_close_f32(&got_probs, &expected_probs, 1e-5, 1e-5);

        free(&device, logits_ptr, logits_t.bytes());
        free(&device, ids_ptr, ids_t.bytes());
        free(&device, probs_ptr, probs_t.bytes());
    }
}
