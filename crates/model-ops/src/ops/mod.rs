//! One file per op. Naming: `{op}_{dtype_in}[_to_{dtype_out}]`. Flat
//! re-exports in `src/lib.rs`.

pub mod activation;
pub mod add;
pub mod attn_decode;
pub mod attn_decode_batched;
pub mod attn_decode_splitk;
// pub mod attn_decode_splitk_h2;
pub mod attn_prefill;
pub mod cast;
pub mod gated_attn;
pub mod attn_decode_q8_kv;
pub mod attn_decode_q8_kv_splitk;
pub mod attn_prefill_q8_kv;
pub mod kv_append;
pub mod kv_append_f16_to_q8;
pub mod moe_router;
pub mod qmatmul;
pub mod quantize;
pub mod rmsnorm;
pub mod rope;
pub mod scale;
pub mod softcap;
pub mod topk_softmax;
pub mod v_unit_norm;
