//! flambeau-mcp-server — M-track, dev-only.
//!
//! Tools exposed to Claude via MCP protocol:
//!   flambeau_sweep, flambeau_matrix, flambeau_profile, flambeau_dispatch_ab,
//!   flambeau_inspect, flambeau_cert_diff, flambeau_tune_dry.
//!
//! Every finding lands as a committable artefact (cert, matrix snapshot,
//! dispatch row, PMC JSON) — no ephemeral live-tune state. Never part of
//! production deployment; `flambeau serve` in prod does not talk to this crate.
