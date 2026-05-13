//! Small cross-cutting helpers used by every submodule under `forward/`.
//! Nothing here is on the tight inner loop — these are setup /
//! one-shot conversions that appear in both decode and prefill paths.

#![cfg(feature = "hip")]

use anyhow::{bail, Context, Result};
use flambeau_core::{DevicePtr, QDtype};
use flambeau_ops::hip::{
    qmatmul::{mmvq, qmatmul},
    HipStream, OpsRegistry,
};
use flambeau_quant::GgmlDType;

use flambeau_ops::hip::moe::{
    indexed_moe_mmvq_iq1_m, indexed_moe_mmvq_iq1_s, indexed_moe_mmvq_iq2_s,
    indexed_moe_mmvq_iq2_xs, indexed_moe_mmvq_iq2_xxs, indexed_moe_mmvq_iq3_s,
    indexed_moe_mmvq_iq3_xxs, indexed_moe_mmvq_iq4_nl, indexed_moe_mmvq_iq4_xs,
    indexed_moe_mmvq_q2_k, indexed_moe_mmvq_q3_k, indexed_moe_mmvq_q4_0, indexed_moe_mmvq_q4_1,
    indexed_moe_mmvq_q4_k_gate_up, indexed_moe_mmvq_q4_k_r2, indexed_moe_mmvq_q5_k,
    indexed_moe_mmvq_q6_k, indexed_moe_mmvq_q8_0,
};

use crate::weights::DeviceTensor;

/// K-quant super-block size. Q4_K / Q5_K / Q6_K weights pack 256 elements
/// per super-block; MoE weight dims `[n_experts, inter, hidden]` must be
/// divisible by `QK_K` on the `hidden` and `inter` axes so our indexed-MoE
/// kernels can walk super-blocks without a row-straddling tail.
pub(crate) const QK_K: usize = 256;

/// Assert MoE expert dtypes fall inside the supported set and that
/// `hidden` / `inter` are `QK_K`-aligned. Shared by decode and prefill.
/// Gate and up must agree in dtype and be one of `Q4_K / Q8_0 / Q4_0`.
/// Down may additionally be `Q6_K` (UD-Q4_K_S `ffn_down_exps` promotion).
/// `label` disambiguates decode vs prefill in error messages; pass
/// `"indexed-MoE"` for decode and `"indexed-MoE prefill"` for prefill to
/// preserve the existing text that downstream tests grep against.
/// # Errors
/// Returns an error if any dtype is outside the supported set or
/// `hidden` / `inter` are not multiples of `QK_K`.
pub(crate) fn validate_moe_dtypes(
    label: &str,
    gate_dt: GgmlDType,
    up_dt: GgmlDType,
    down_dt: GgmlDType,
    hidden: usize,
    inter: usize,
) -> Result<()> {
    if hidden % QK_K != 0 {
        bail!("MoE expects hidden={hidden} divisible by QK_K={QK_K}");
    }
    if inter % QK_K != 0 {
        bail!("MoE expects moe_intermediate_size={inter} divisible by QK_K={QK_K}");
    }
    let gate_up_ok = matches!(
        gate_dt,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q8_0
            | GgmlDType::Q4_0
            | GgmlDType::Iq4Nl
            | GgmlDType::Iq4Xs
            | GgmlDType::Iq3Xxs
            | GgmlDType::Iq3S
            | GgmlDType::Iq2Xxs
            | GgmlDType::Iq2Xs
            | GgmlDType::Iq2S
            | GgmlDType::Iq1S
            | GgmlDType::Iq1M
    );
    if !(gate_dt == up_dt && gate_up_ok) {
        bail!(
            "{label} gate/up dtypes must match and be Q2_K, Q3_K, Q4_K, Q8_0, Q4_0 or IQ family; got gate={gate_dt:?}, up={up_dt:?}"
        );
    }
    let down_ok = matches!(
        down_dt,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8_0
            | GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Iq4Nl
            | GgmlDType::Iq4Xs
            | GgmlDType::Iq3Xxs
            | GgmlDType::Iq3S
            | GgmlDType::Iq2Xxs
            | GgmlDType::Iq2Xs
            | GgmlDType::Iq2S
            | GgmlDType::Iq1S
            | GgmlDType::Iq1M
    );
    if !down_ok {
        bail!(
            "{label} ffn_down_exps must be Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q4_1 or IQ family; got {down_dt:?}"
        );
    }
    Ok(())
}

/// Run the MoE gate+up matmul for one dispatch shape. Parametrised on
/// `n_tokens` so both decode (n_tokens=1) and prefill share a single
/// implementation. Dispatches:
/// - Q4_K → fused `indexed_moe_mmvq_q4_k_gate_up` (1 kernel, 2 outputs)
/// - Q8_0 / Q4_0 → two separate `indexed_moe_mmvq_q{8_0,4_0}` launches
/// Caller must have validated `dtype` via [`validate_moe_dtypes`].
/// # Errors
/// Returns an error if the underlying kernel launch fails.
#[expect(
    clippy::too_many_arguments,
    reason = "thin parametric wrapper over a family of indexed-MoE kernels; flattens into \
              the caller's existing scratch-pointer flow so introducing a context struct \
              would just rewrap the same pointers."
)]
#[allow(dead_code)] // kept available; previously used by inline MoE TP prefill (now migrated to blocks)
pub(crate) fn run_indexed_moe_gate_up(
    ops: &OpsRegistry,
    stream: &HipStream,
    dtype: GgmlDType,
    w_gate: DevicePtr,
    w_up: DevicePtr,
    x_q8_1: DevicePtr,
    expert_ids: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    inter: usize,
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<()> {
    match dtype {
        GgmlDType::Q4K => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_q4_k_gate_up(
                ops, stream, w_gate, w_up, x_q8_1, expert_ids, gate_out, up_out, inter,
                n_tokens, top_k, nb,
            )
            .context("indexed_moe gate+up q4_k")
        }
        GgmlDType::Q8_0 => {
            let nb = hidden / 32;
            indexed_moe_mmvq_q8_0(
                ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens,
                top_k, nb,
            )
            .context("indexed_moe gate q8_0")?;
            indexed_moe_mmvq_q8_0(
                ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k,
                nb,
            )
            .context("indexed_moe up q8_0")
        }
        GgmlDType::Q4_0 => {
            // 3.b.1 — fused gate+up reads Q8_1 activation once per block
            // and produces both outputs. Halves launch count for Q4_0 MoE
            // decode (where the tile8 MMQ path does not fire at n_tokens<32).
            let nb = hidden / 32;
            flambeau_ops::hip::moe::indexed_moe_mmvq_q4_0_gate_up(
                ops, stream, w_gate, w_up, x_q8_1, expert_ids, gate_out, up_out,
                inter, n_tokens, top_k, nb,
            )
            .context("indexed_moe gate+up q4_0 fused")
        }
        GgmlDType::Q2K => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_q2_k(
                ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens,
                top_k, nb,
            )
            .context("indexed_moe gate q2_k")?;
            indexed_moe_mmvq_q2_k(
                ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb,
            )
            .context("indexed_moe up q2_k")
        }
        GgmlDType::Q3K => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_q3_k(
                ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens,
                top_k, nb,
            )
            .context("indexed_moe gate q3_k")?;
            indexed_moe_mmvq_q3_k(
                ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb,
            )
            .context("indexed_moe up q3_k")
        }
        //— full IQ family MoE expert support.
        // Same separate-gate / separate-up pattern as Q2_K / Q3_K (no fused
        // gate_up variant yet).
        GgmlDType::Iq4Xs => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq4_xs(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq4_xs")?;
            indexed_moe_mmvq_iq4_xs(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq4_xs")
        }
        GgmlDType::Iq4Nl => {
            let nb = hidden / 32;
            indexed_moe_mmvq_iq4_nl(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq4_nl")?;
            indexed_moe_mmvq_iq4_nl(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq4_nl")
        }
        GgmlDType::Iq3Xxs => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq3_xxs(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq3_xxs")?;
            indexed_moe_mmvq_iq3_xxs(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq3_xxs")
        }
        GgmlDType::Iq3S => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq3_s(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq3_s")?;
            indexed_moe_mmvq_iq3_s(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq3_s")
        }
        GgmlDType::Iq2Xxs => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq2_xxs(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq2_xxs")?;
            indexed_moe_mmvq_iq2_xxs(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq2_xxs")
        }
        GgmlDType::Iq2Xs => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq2_xs(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq2_xs")?;
            indexed_moe_mmvq_iq2_xs(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq2_xs")
        }
        GgmlDType::Iq2S => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq2_s(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq2_s")?;
            indexed_moe_mmvq_iq2_s(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq2_s")
        }
        GgmlDType::Iq1S => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq1_s(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq1_s")?;
            indexed_moe_mmvq_iq1_s(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq1_s")
        }
        GgmlDType::Iq1M => {
            let nb = hidden / QK_K;
            indexed_moe_mmvq_iq1_m(ops, stream, w_gate, x_q8_1, expert_ids, gate_out, inter, n_tokens, top_k, nb).context("indexed_moe gate iq1_m")?;
            indexed_moe_mmvq_iq1_m(ops, stream, w_up, x_q8_1, expert_ids, up_out, inter, n_tokens, top_k, nb).context("indexed_moe up iq1_m")
        }
        _ => bail!("run_indexed_moe_gate_up: unsupported gate dtype {dtype:?} (expected Q2_K / Q3_K / Q4_K / Q8_0 / Q4_0 / IQ4_XS / IQ4_NL / IQ3_XXS / IQ3_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ1_S / IQ1_M)"),
    }
}

/// Run the MoE down matmul for one dispatch shape. Parametrised on
/// `n_tokens_eff` and `top_k_inner` so callers that have already flattened
/// the `(n_tokens, top_k)` routing into per-expert effective tokens can
/// pass `(n_tokens * top_k, 1)` (decode + prefill down-step pattern), and
/// the standard prefill path can pass `(n_tokens, top_k)` directly.
/// Dispatches Q4_K (r2 variant), Q6_K, Q8_0, Q4_0.
/// # Errors
/// Returns an error if the underlying kernel launch fails or the dtype is
/// outside the supported set.
#[expect(
    clippy::too_many_arguments,
    reason = "thin parametric wrapper over a family of indexed-MoE kernels; flattens into \
              the caller's existing scratch-pointer flow so introducing a context struct \
              would just rewrap the same pointers."
)]
#[allow(dead_code)] // kept available; see run_indexed_moe_gate_up note
pub(crate) fn run_indexed_moe_down(
    ops: &OpsRegistry,
    stream: &HipStream,
    dtype: GgmlDType,
    w_down: DevicePtr,
    activated_q8_1: DevicePtr,
    expert_ids: DevicePtr,
    down_out: DevicePtr,
    hidden: usize,
    n_tokens_eff: usize,
    top_k_inner: usize,
    inter: usize,
) -> Result<()> {
    match dtype {
        GgmlDType::Q4K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q4_k_r2(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q4_k r2")
        }
        GgmlDType::Q5K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q5_k(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q5_k")
        }
        GgmlDType::Q6K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q6_k(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q6_k")
        }
        GgmlDType::Q8_0 => {
            let nb = inter / 32;
            indexed_moe_mmvq_q8_0(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q8_0")
        }
        GgmlDType::Q4_0 => {
            let nb = inter / 32;
            indexed_moe_mmvq_q4_0(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q4_0")
        }
        GgmlDType::Q4_1 => {
            // 5.a — Qwen-published Qwen3.6-35B-A3B-Q4_0 packs
            // ffn_down_exps as Q4_1 (gate/up are Q4_0). Same call shape as
            // Q4_0; only the per-block reconstruction differs.
            let nb = inter / 32;
            indexed_moe_mmvq_q4_1(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q4_1")
        }
        GgmlDType::Q2K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q2_k(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q2_k")
        }
        GgmlDType::Q3K => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_q3_k(
                ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden,
                n_tokens_eff, top_k_inner, nb,
            )
            .context("indexed_moe down q3_k")
        }
        //— IQ family.
        GgmlDType::Iq4Xs => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq4_xs(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq4_xs")
        }
        GgmlDType::Iq4Nl => {
            let nb = inter / 32;
            indexed_moe_mmvq_iq4_nl(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq4_nl")
        }
        GgmlDType::Iq3Xxs => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq3_xxs(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq3_xxs")
        }
        GgmlDType::Iq3S => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq3_s(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq3_s")
        }
        GgmlDType::Iq2Xxs => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq2_xxs(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq2_xxs")
        }
        GgmlDType::Iq2Xs => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq2_xs(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq2_xs")
        }
        GgmlDType::Iq2S => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq2_s(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq2_s")
        }
        GgmlDType::Iq1S => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq1_s(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq1_s")
        }
        GgmlDType::Iq1M => {
            let nb = inter / QK_K;
            indexed_moe_mmvq_iq1_m(ops, stream, w_down, activated_q8_1, expert_ids, down_out, hidden, n_tokens_eff, top_k_inner, nb).context("indexed_moe down iq1_m")
        }
        _ => bail!("run_indexed_moe_down: unsupported down dtype {dtype:?} (expected Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q4_1 / IQ family)"),
    }
}

/// Map our `GgmlDType` (from GGUF) to the `QDtype` the qmatmul dispatcher
/// uses. Only the dtypes our V1 kernels support are allowed here; everything
/// else is a load-time error.
/// # Errors
/// Returns an error if `dtype` is not in the V1 qmatmul dispatch set.
pub(super) fn qdtype_of(dtype: GgmlDType) -> Result<QDtype> {
    Ok(match dtype {
        GgmlDType::Q2K => QDtype::Q2_K,
        GgmlDType::Q3K => QDtype::Q3_K,
        GgmlDType::Q4K => QDtype::Q4_K,
        GgmlDType::Q5K => QDtype::Q5_K,
        GgmlDType::Q6K => QDtype::Q6_K,
        GgmlDType::Q8K => QDtype::Q8_K,
        GgmlDType::Q8_0 => QDtype::Q8_0,
        GgmlDType::Q4_1 => QDtype::Q4_1,
        // 1.b — UD-Q8_K_XL reserves F16 for i-matrix-flagged layers
        // (Qwen3.6-27B-UD-Q8_K_XL: all 48 attn_gate + 48 ssm_out + scattered
        // attn_q/k + ffn_gate/up/down). `mmvq()` special-cases F16 to skip
        // the dispatch table and call the direct F16×Q8_1 kernel.
        GgmlDType::F16 => QDtype::F16,
        // F32 — used by the MoE router weight (`ffn_gate_inp`) on
        // older Qwen3.x GGUFs that predate the F32→F16 loader-side
        // conversion. Only the dense_gemv router path consumes F32;
        // qmatmul itself rejects F32 at dispatch time.
        GgmlDType::F32 => QDtype::F32,
        // 3.a — Q4_0 and Q5_0 unblock Qwen3.6-35B-A3B-Q4_0.
        GgmlDType::Q4_0 => QDtype::Q4_0,
        GgmlDType::Q5_0 => QDtype::Q5_0,
        // 6.a — Q5_1 (llama.cpp parity; no Qwen3 model currently uses it
        // but unblocks any incoming GGUF mix).
        GgmlDType::Q5_1 => QDtype::Q5_1,
        // native IQ4 MMVQ kernels. Direct mmap → memcpy → kernel
        // (no host re-quant), llama.cpp-style. Used by UD-Q3_K_XL etc.
        GgmlDType::Iq4Nl => QDtype::IQ4_NL,
        GgmlDType::Iq4Xs => QDtype::IQ4_XS,
        // native IQ3 MMVQ kernels (codebook lookup, 256/512-entry
        // u32 grid). Covers UD-Q3_K_XL MoE expert tensors.
        GgmlDType::Iq3Xxs => QDtype::IQ3_XXS,
        GgmlDType::Iq3S => QDtype::IQ3_S,
        // full IQ2/IQ1 family native MMVQ kernels (codebook
        // lookup, 256..2048-entry u64 grids).
        GgmlDType::Iq2Xxs => QDtype::IQ2_XXS,
        GgmlDType::Iq2Xs => QDtype::IQ2_XS,
        GgmlDType::Iq2S => QDtype::IQ2_S,
        GgmlDType::Iq1S => QDtype::IQ1_S,
        GgmlDType::Iq1M => QDtype::IQ1_M,
        other => bail!("weight dtype {other:?} not supported by V1 qmatmul dispatch"),
    })
}

/// Pull the `(n_rows, k)` pair out of a weight tensor's GGUF dims.
/// `flambeau_quant::GgufFile` reverses the on-wire dim order at parse time,
/// so `dims` is **outermost-first**: for a 2D weight `[n_rows, k]` we have
/// `dims = [n_rows, k]`.
/// # Errors
/// Returns an error if the weight is not 2D.
pub(super) fn mat_shape(w: &DeviceTensor) -> Result<(usize, usize)> {
    if w.dims.len() != 2 {
        bail!(
            "expected a 2D weight tensor for `{}`, got dims {:?}",
            w.name,
            w.dims
        );
    }
    let n_rows = w.dims[0] as usize;
    let k = w.dims[1] as usize;
    Ok((n_rows, k))
}

/// Byte count per vocabulary row for a 2D weight `[vocab, hidden]` of the
/// given dtype.
/// # Errors
/// Returns an error if `hidden` is not a multiple of the dtype's block size.
pub(super) fn row_bytes_for_dtype(dtype: GgmlDType, hidden: usize) -> Result<usize> {
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    if block_size > 1 && hidden % block_size != 0 {
        bail!(
            "token_embd hidden {hidden} is not a multiple of block_size {block_size} for {dtype:?}"
        );
    }
    let n_blocks = hidden / block_size;
    Ok(n_blocks * type_size)
}

/// Run an MMVQ against a weight `DeviceTensor`, validating dims and
/// dispatching on dtype. Keeps forward bodies readable.
/// # Errors
/// Returns an error if the tensor shape doesn't match the caller's
/// `(expected_rows, expected_k)`, or if the mmvq dispatch fails.
pub(super) fn run_mmvq_from_tensor(
    ops: &OpsRegistry,
    stream: &HipStream,
    w: &DeviceTensor,
    act_q8_1: DevicePtr,
    dst: DevicePtr,
    expected_rows: usize,
    expected_k: usize,
    label: &str,
) -> Result<()> {
    let dtype = qdtype_of(w.dtype)?;
    let (rows, k) = mat_shape(w)?;
    if rows != expected_rows || k != expected_k {
        bail!(
            "{label} shape [{rows}, {k}] != expected [{expected_rows}, {expected_k}]"
        );
    }
    mmvq(ops, stream, w.ptr, act_q8_1, dst, rows, k, dtype)
        .with_context(|| format!("mmvq {label}"))
}

/// Prefill counterpart to [`run_mmvq_from_tensor`]. Dispatches on dtype and
/// routes through MMQ when the caller's recipe applies; callers that don't
/// exercise the MmqLdsX64 kernel can pass [`DevicePtr(0)`] for
/// `act_q8_1_mmq` (no access).
/// # Errors
/// Returns an error if the tensor shape doesn't match the caller's
/// `(expected_rows, expected_k)`, or if the qmatmul dispatch fails.
pub(super) fn run_qmatmul_from_tensor(
    ops: &OpsRegistry,
    stream: &HipStream,
    w: &DeviceTensor,
    act_q8_1: DevicePtr,
    act_q8_1_mmq: DevicePtr,
    dst: DevicePtr,
    m: usize,
    expected_k: usize,
    expected_rows: usize,
    label: &str,
) -> Result<()> {
    let dtype = qdtype_of(w.dtype)?;
    let (rows, k) = mat_shape(w)?;
    if rows != expected_rows || k != expected_k {
        bail!(
            "{label} shape [{rows}, {k}] != expected [{expected_rows}, {expected_k}]"
        );
    }
    qmatmul(ops, stream, w.ptr, act_q8_1, act_q8_1_mmq, dst, m, k, rows, dtype)
        .with_context(|| format!("qmatmul {label}"))
}
