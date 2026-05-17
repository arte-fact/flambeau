//! Op vocabulary — one file per op.
//!
//! Each module declared below provides a single `pub fn` (or a small
//! family of dtype-keyed `pub fn`s when a kernel is dtype-monomorphic
//! per function). Naming follows `{op}_{dtype_in}[_to_{dtype_out}]`
//! where the dtype dimension matters. See `CLAUDE.md` for the recipe.
//!
//! Modules are added here as ops land. The flat re-exports live in
//! `src/lib.rs` so consumers `use flambeau_model_ops::<op>;` without
//! the `ops::` prefix.

pub mod add;
pub mod cast;
pub mod rmsnorm;
pub mod scale;
