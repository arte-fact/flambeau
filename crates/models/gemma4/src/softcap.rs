//! Gemma 4 logit softcap. The implementation lives in
//! [`flambeau_blocks::apply_logit_softcap_f32`]; this module
//! re-exports it under the gemma4-flavoured name for source
//! compatibility with existing call sites.

#![cfg(feature = "hip")]

pub use flambeau_blocks::apply_logit_softcap_f32 as apply_logit_softcap;
