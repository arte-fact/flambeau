//! `Model` + `Session` trait re-exports + `LoadedModel` alias.
//!
//! Trait defs (`Model`, `Session`, `SessionContext`, `BatchSlot`) live
//! in `flambeau-server-core`; this module just re-exports for crate-local
//! callers and defines the `LoadedModel = Arc<dyn Model>` alias the
//! server's state machine carries around.

#![cfg(feature = "hip")]

pub use flambeau_server_core::{
    BatchSlot, Model, ReasoningMarkers, ReasoningStyle, Session, SessionContext,
};

pub type LoadedModel = std::sync::Arc<dyn Model>;
