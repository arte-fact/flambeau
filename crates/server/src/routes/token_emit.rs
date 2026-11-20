//! Per-token acceptance shared by the streamed and non-streamed decode
//! loops: every generated token, whether from a plain step or from an MTP
//! cycle, goes through [`Emitter::accept`], which applies the stop rules and
//! hands newly decoded text to a [`TokenSink`].

use anyhow::Result;
use flambeau_runtime::json_schema::JsonConstraint;

/// Receives the response text as it is decoded.
pub(crate) trait TokenSink {
    /// `delta` is complete UTF-8 and never contains any part of a stop
    /// sequence. Return `false` to end generation (client gone).
    fn text(&mut self, delta: &str) -> bool;
}

/// Sink of the non-streamed path: the response is rebuilt from the token
/// ids by `finalise`, so the deltas are dropped.
pub(crate) struct Discard;

impl TokenSink for Discard {
    fn text(&mut self, _delta: &str) -> bool {
        true
    }
}

/// Sink of the SSE path: forwards each delta to the stream's emit callback.
pub(crate) struct Forward<'a> {
    pub emit: &'a mut dyn FnMut(&str) -> bool,
}

impl TokenSink for Forward<'_> {
    fn text(&mut self, delta: &str) -> bool {
        (self.emit)(delta)
    }
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Finish {
    /// Stop token, stop sequence, complete JSON value or client gone.
    Stop,
    /// `max_tokens` reached.
    Length,
}

impl Finish {
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
        }
    }
}

/// Incremental detokenization of the generated ids plus stop-sequence
/// scanning on the decoded text.
///
/// Only the still-open segment `generated[cursor..]` is re-decoded per
/// token. A trailing U+FFFD (multi-byte glyph split across BPE tokens) is
/// held until the next token completes it.
pub(crate) struct TextTracker {
    patterns: Vec<String>,
    longest: usize,
    cursor: usize,
    segment_emitted: usize,
    text: String,
}

impl TextTracker {
    pub(crate) fn new(patterns: Vec<String>) -> Self {
        let longest = patterns.iter().map(String::len).max().unwrap_or(0);
        Self {
            patterns,
            longest,
            cursor: 0,
            segment_emitted: 0,
            text: String::new(),
        }
    }

    /// Text decoded so far, cut before the first stop sequence.
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// Length of the longest suffix of the text that is a proper prefix of
    /// a stop sequence: bytes that may still turn into one.
    pub(crate) fn pending_stop_prefix(&self) -> usize {
        let mut held = 0;
        for p in &self.patterns {
            for k in (held + 1..p.len().min(self.text.len() + 1)).rev() {
                if p.is_char_boundary(k) && self.text.ends_with(&p[..k]) {
                    held = k;
                    break;
                }
            }
        }
        held
    }

    /// Decode what `generated` added since the last call, cutting the text
    /// before a stop sequence. Returns whether one was found.
    pub(crate) fn advance(
        &mut self,
        generated: &[u32],
        decode: &dyn Fn(&[u32]) -> Result<String>,
    ) -> Result<bool> {
        if generated.len() <= self.cursor {
            return Ok(false);
        }
        let raw = decode(&generated[self.cursor..])?;
        let safe = raw.trim_end_matches('\u{FFFD}');
        let mut delta = String::new();
        if safe.len() > self.segment_emitted {
            delta.push_str(&safe[self.segment_emitted..]);
            self.segment_emitted = safe.len();
        }
        if safe.len() == raw.len() {
            self.cursor = generated.len();
            self.segment_emitted = 0;
        }
        if delta.is_empty() || self.patterns.is_empty() {
            self.text.push_str(&delta);
            return Ok(false);
        }

        let before = self.text.len();
        self.text.push_str(&delta);
        let mut from = before.saturating_sub(self.longest.saturating_sub(1));
        while !self.text.is_char_boundary(from) {
            from -= 1;
        }
        let hit = self
            .patterns
            .iter()
            .filter_map(|p| self.text[from..].find(p.as_str()).map(|i| from + i))
            .min();
        if let Some(cut) = hit {
            self.text.truncate(cut);
        }
        Ok(hit.is_some())
    }
}

/// Applies the stop rules to each generated token, in order. Text that may
/// still be the start of a stop sequence is held back from the sink until
/// it either completes one (dropped) or cannot anymore (released); it is
/// flushed when generation ends for any other reason, and dropped when the
/// client is gone.
pub(crate) struct Emitter<'a> {
    pub generated: Vec<u32>,
    pub text: TextTracker,
    pub json: Option<JsonConstraint>,
    stop_ids: &'a [u32],
    max_tokens: usize,
    sink: &'a mut dyn TokenSink,
    emitted: usize,
}

impl<'a> Emitter<'a> {
    pub(crate) fn new(
        patterns: Vec<String>,
        json: Option<JsonConstraint>,
        stop_ids: &'a [u32],
        max_tokens: usize,
        sink: &'a mut dyn TokenSink,
    ) -> Self {
        Self {
            generated: Vec::with_capacity(max_tokens),
            text: TextTracker::new(patterns),
            json,
            stop_ids,
            max_tokens,
            sink,
            emitted: 0,
        }
    }

    /// Hand the sink the text up to byte `end` not yet emitted.
    fn emit_to(&mut self, end: usize) -> bool {
        if end <= self.emitted {
            return true;
        }
        let alive = self.sink.text(&self.text.text()[self.emitted..end]);
        self.emitted = end;
        alive
    }

    fn finish(&mut self, finish: Finish) -> Option<Finish> {
        self.emit_to(self.text.text().len());
        Some(finish)
    }

    /// Record `tok`. Returns `Some` when generation must end with it:
    /// stop token (kept in `generated`, never decoded), stop sequence,
    /// complete JSON value, client gone, or `max_tokens` reached.
    pub(crate) fn accept(
        &mut self,
        tok: u32,
        decode: &dyn Fn(&[u32]) -> Result<String>,
    ) -> Result<Option<Finish>> {
        self.generated.push(tok);
        if self.stop_ids.contains(&tok) {
            return Ok(self.finish(Finish::Stop));
        }
        let mut json_complete = false;
        if let Some(js) = self.json.as_mut() {
            if let Ok(piece) = decode(&[tok]) {
                let _ = js.feed_slice(piece.as_bytes());
            }
            json_complete = js.is_complete();
        }
        if self.text.advance(&self.generated, decode)? {
            tracing::info!(target: "server.completion.string_stop", "stop sequence in decoded text");
            return Ok(self.finish(Finish::Stop));
        }
        let releasable = self.text.text().len() - self.text.pending_stop_prefix();
        if !self.emit_to(releasable) {
            return Ok(Some(Finish::Stop));
        }
        if json_complete {
            return Ok(self.finish(Finish::Stop));
        }
        if self.generated.len() >= self.max_tokens {
            return Ok(self.finish(Finish::Length));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIECES: &[&str] = &["Hello", " world", "END", " tail", "!", "<stop>"];
    const STOP_TOKEN: u32 = 99;

    /// Token i decodes to PIECES[i]; tokens 10 and 11 are the two halves of
    /// "é" (0xC3 0xA9), as a byte-level BPE splits it.
    fn decode(ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &t in ids {
            match t {
                10 => bytes.push(0xC3),
                11 => bytes.push(0xA9),
                t => bytes.extend_from_slice(PIECES[t as usize].as_bytes()),
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    struct Collect(String, usize);
    impl TokenSink for Collect {
        fn text(&mut self, delta: &str) -> bool {
            self.0.push_str(delta);
            self.1 += 1;
            true
        }
    }

    /// Feed a token run the way the decode loop feeds an MTP cycle.
    fn feed(e: &mut Emitter<'_>, run: &[u32]) -> Option<Finish> {
        for &t in run {
            if let Some(f) = e.accept(t, &decode).unwrap() {
                return Some(f);
            }
        }
        None
    }

    #[test]
    fn stop_sequence_mid_cycle_cuts_text_and_drops_later_tokens() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = Emitter::new(vec!["END".into()], None, &stops, 100, &mut sink);
        let finish = feed(&mut e, &[0, 1, 2, 3, 4]);
        assert_eq!(finish, Some(Finish::Stop));
        assert_eq!(e.generated, vec![0, 1, 2]);
        assert_eq!(e.text.text(), "Hello world");
        drop(e);
        assert_eq!(sink.0, "Hello world");
    }

    fn emitter<'a>(
        patterns: &[&str],
        stops: &'a [u32],
        max_tokens: usize,
        sink: &'a mut Collect,
    ) -> Emitter<'a> {
        let patterns = patterns.iter().map(|p| (*p).to_owned()).collect();
        Emitter::new(patterns, None, stops, max_tokens, sink)
    }

    #[test]
    fn held_prefix_is_released_once_it_cannot_match() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = emitter(&["ENDX"], &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 2]), None);
        drop(e);
        assert_eq!(sink.0, "Hello world", "\"END\" may start \"ENDX\": held");
        let mut sink = Collect(String::new(), 0);
        let mut e = emitter(&["ENDX"], &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 2, 3]), None);
        drop(e);
        assert_eq!(sink.0, "Hello worldEND tail");
    }

    #[test]
    fn held_prefix_completing_a_stop_sequence_is_never_emitted() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = emitter(&["worldEND"], &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 2, 3]), Some(Finish::Stop));
        assert_eq!(e.text.text(), "Hello ");
        drop(e);
        assert_eq!(sink.0, "Hello ");
    }

    #[test]
    fn stop_sequence_across_three_tokens_is_never_emitted() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = emitter(&[" worldEND tail"], &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 2, 3, 4]), Some(Finish::Stop));
        assert_eq!(e.generated, vec![0, 1, 2, 3]);
        assert_eq!(e.text.text(), "Hello");
        drop(e);
        assert_eq!(sink.0, "Hello");
    }

    #[test]
    fn max_tokens_flushes_a_held_prefix() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = emitter(&["ENDX"], &stops, 3, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 2, 3]), Some(Finish::Length));
        drop(e);
        assert_eq!(sink.0, "Hello worldEND");
    }

    #[test]
    fn stop_token_flushes_a_held_prefix() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = emitter(&["ENDX"], &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 2, STOP_TOKEN]), Some(Finish::Stop));
        drop(e);
        assert_eq!(sink.0, "Hello worldEND");
    }

    #[test]
    fn stop_token_mid_cycle_is_kept_but_not_decoded() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = Emitter::new(Vec::new(), None, &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, STOP_TOKEN, 1]), Some(Finish::Stop));
        assert_eq!(e.generated, vec![0, STOP_TOKEN]);
        drop(e);
        assert_eq!(sink.0, "Hello");
    }

    #[test]
    fn max_tokens_mid_cycle_reports_length() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = Emitter::new(Vec::new(), None, &stops, 3, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1, 4, 1]), Some(Finish::Length));
        assert_eq!(e.generated, vec![0, 1, 4]);
        drop(e);
        assert_eq!(sink.0, "Hello world!");
    }

    #[test]
    fn split_glyph_is_held_until_complete() {
        let mut sink = Collect(String::new(), 0);
        let stops = [STOP_TOKEN];
        let mut e = Emitter::new(Vec::new(), None, &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 10]), None);
        assert_eq!(e.text.text(), "Hello");
        assert_eq!(feed(&mut e, &[11, 4]), None);
        assert_eq!(e.text.text(), "Helloé!");
        drop(e);
        assert_eq!(sink.0, "Helloé!");
    }

    /// Streamed (forwarding) and non-streamed (discarding) requests run the
    /// same `Emitter`: for every way a run can end they keep the same tokens
    /// (hence the same `completion_tokens`) and the streamed text is the
    /// tracked text, without the stop sequence.
    #[test]
    fn streamed_and_collected_runs_keep_the_same_tokens() {
        let runs: [(&[u32], usize, Finish); 3] = [
            (&[0, 1, STOP_TOKEN, 4], 100, Finish::Stop),
            (&[0, 1, 2, 3], 100, Finish::Stop),
            (&[0, 1, 4, 1], 3, Finish::Length),
        ];
        for (run, max_tokens, expected) in runs {
            let stops = [STOP_TOKEN];
            let mut discard = Discard;
            let mut collected = Emitter::new(vec!["END".into()], None, &stops, max_tokens, &mut discard);
            assert_eq!(feed(&mut collected, run), Some(expected));

            let mut sink = Collect(String::new(), 0);
            let mut streamed = Emitter::new(vec!["END".into()], None, &stops, max_tokens, &mut sink);
            assert_eq!(feed(&mut streamed, run), Some(expected));

            assert_eq!(collected.generated, streamed.generated, "run {run:?}");
            let text = streamed.text.text().to_owned();
            assert_eq!(collected.text.text(), text, "run {run:?}");
            drop(streamed);
            assert_eq!(sink.0, text, "run {run:?}");
            assert!(!sink.0.contains("END"), "run {run:?}");
        }
    }

    #[test]
    fn client_gone_ends_generation() {
        struct Gone;
        impl TokenSink for Gone {
            fn text(&mut self, _delta: &str) -> bool {
                false
            }
        }
        let mut sink = Gone;
        let stops = [STOP_TOKEN];
        let mut e = Emitter::new(Vec::new(), None, &stops, 100, &mut sink);
        assert_eq!(feed(&mut e, &[0, 1]), Some(Finish::Stop));
        assert_eq!(e.generated, vec![0]);
    }
}
