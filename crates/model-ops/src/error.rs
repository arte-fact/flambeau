//! Crate-local result alias.
//!
//! Ops return `anyhow::Result<()>` via this alias so error chains
//! propagate naturally from kernel launches and from the test-side
//! `assert_close` helpers. There is no crate-specific error enum yet
//! — add one only if a real consumer needs structured handling
//! beyond `is_err() / display`.

pub type Result<T> = anyhow::Result<T>;
pub type Error = anyhow::Error;
