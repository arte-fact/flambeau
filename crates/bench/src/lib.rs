//! flambeau-bench — sweep / matrix / cert subcommands.
//!
//! V1.2+: `sweep --arch <gfx906|sm_86> --op <id> --shapes-from <file>` runs the
//! correctness grid, emits `certs/<backend>/<arch>/<impl_id>.json` with PMC snapshot.
//! `matrix` produces the regression table per (model, prompt_len, tg_len, devices).
//! `cert-check` validates every dispatch row has a matching cert — build-time gate.
