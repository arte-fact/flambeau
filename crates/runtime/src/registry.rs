//! Model registry — maps GGUF `general.architecture` → registered
//! `Model` implementor. Built per-binary (typically once at CLI
//! startup), passed into the server entry point.
//!
//! V1 is intentionally a thin lookup. As R5 progresses the registry
//! grows construction methods (`create(...) -> Box<dyn Model>`)
//! covering session allocation per topology. For now the registry is
//! used to validate that the GGUF's arch is supported before the
//! existing model-load path runs.

use std::sync::Arc;

use crate::model::Model;

/// Per-binary model registry. Build once at startup, register every
/// `Model` implementor the binary supports, then look up by GGUF arch.
#[derive(Default)]
pub struct Registry {
    models: Vec<Arc<dyn Model>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one model. Calling `register` twice with overlapping
    /// `supported_archs` is allowed today (last-registered wins on
    /// `find`); future phases may grow this to error on conflict.
    pub fn register(&mut self, model: Arc<dyn Model>) {
        self.models.push(model);
    }

    /// Look up a model that handles the given GGUF arch string.
    /// Returns `None` if no registered model lists `arch` in its
    /// `supported_archs`.
    pub fn find(&self, arch: &str) -> Option<&Arc<dyn Model>> {
        self.models
            .iter()
            .rev()
            .find(|m| m.supported_archs().iter().any(|&a| a == arch))
    }

    /// `find` but errors with the registered arch list so an unknown
    /// GGUF gets a useful diagnostic before the load path bails on
    /// some downstream invariant.
    pub fn validate(&self, arch: &str) -> Result<&Arc<dyn Model>, RegistryError> {
        self.find(arch).ok_or_else(|| {
            let mut known: Vec<&str> = self
                .models
                .iter()
                .flat_map(|m| m.supported_archs().iter().copied())
                .collect();
            known.sort();
            known.dedup();
            RegistryError::UnknownArch {
                arch: arch.to_string(),
                known: known.iter().map(|s| (*s).to_string()).collect(),
            }
        })
    }

    /// Iterate every registered model (e.g. for boot-log dumps).
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Model>> {
        self.models.iter()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error(
        "GGUF general.architecture `{arch}` is not registered; \
         known archs: [{}]",
        known.join(", ")
    )]
    UnknownArch { arch: String, known: Vec<String> },
}
