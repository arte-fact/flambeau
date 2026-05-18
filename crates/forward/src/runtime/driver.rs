//! Bridge: `Session<A>` implements `flambeau_runtime::ModelDriver` so
//! it drops into the server's `Box<dyn ModelDriver>` slot alongside
//! the legacy gemma4 drivers. Greedy `forward_*` use the host argmax
//! of the cached logits; `*_logits` variants delegate to the public
//! `Session` methods.

use anyhow::Result;
use flambeau_runtime::ModelDriver;

use crate::runtime::{Arch, Session};

fn argmax(logits: &[f32]) -> u32 {
    let (mut best_i, mut best_v) = (0_u32, f32::NEG_INFINITY);
    for (i, &l) in logits.iter().enumerate() {
        if l > best_v {
            best_v = l;
            best_i = i as u32;
        }
    }
    best_i
}

impl<A: Arch> ModelDriver for Session<A> {
    fn forward_prefill(&mut self, tokens: &[u32], start_position: usize) -> Result<u32> {
        if tokens.is_empty() {
            anyhow::bail!("Session::forward_prefill: empty tokens");
        }
        for (i, &t) in tokens.iter().enumerate() {
            self.forward_one_token(t, start_position + i)?;
        }
        Ok(argmax(self.logits()))
    }

    fn forward_one_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        Session::forward_one_token(self, token_id, position)?;
        Ok(argmax(self.logits()))
    }

    fn forward_prefill_logits(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        Session::forward_prefill_logits(self, tokens, start_position, logits_out)
    }

    fn forward_one_token_logits(
        &mut self,
        token_id: u32,
        position: usize,
        logits_out: &mut Vec<f32>,
    ) -> Result<()> {
        Session::forward_one_token_logits(self, token_id, position, logits_out)
    }

    fn vocab_size(&self) -> usize {
        Session::vocab_size(self)
    }

    fn reset_kv(&mut self) -> Result<()> {
        Session::reset_kv(self)
    }

    fn dispose(&mut self) -> Result<()> {
        Session::dispose_in_place(self)
    }
}
