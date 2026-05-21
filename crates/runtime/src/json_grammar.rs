//! Minimal JSON state machine for **P0.1** response_format=json_object.
//! Tracks a streaming JSON value as bytes are appended to it. After each
//! token the model proposes, the sampler decodes the candidate's bytes,
//! attempts to advance the state, and rejects (zeros out) candidates
//! that would make the output structurally invalid.
//! Coverage:
//! - top-level value: object, array, string, number, true, false, null
//! - balanced braces and brackets
//! - string escape handling (\\, \", \n, \t, \uXXXX prefix)
//! - number / true / false / null literal validity
//! What it does NOT enforce (intentional, V1):
//! - JSON Schema (use the json_schema field for tighter constraints in V2)
//! - key uniqueness
//! - UTF-8 codepoint completeness past the BPE boundary (the streaming
//! detokenizer already handles that)
//! At every byte the [`JsonState::is_complete`] predicate tells the
//! sampler whether stopping right now would produce a valid JSON value
//! (so we can let the model emit `<|im_end|>` only at top-level
//! completion).

use std::collections::VecDeque;

enum Dispatch {
    String(bool, u8),
    Number,
    Literal,
    Object,
    Array,
    Top,
}

/// Where in the JSON value the bytes-so-far end.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Frame {
    /// Open object — expecting `}` (if just opened) or `"key"` →
    /// `:` → value → `,`/`}`. `expecting_key`: about to read the next
    /// key (true after `{` or `,`); `expecting_value`: about to read
    /// the value after `key:`.
    Object {
        expecting_key: bool,
        expecting_colon: bool,
        expecting_value: bool,
        expecting_comma_or_close: bool,
    },
    /// Open array — expecting value or `]`.
    Array {
        expecting_value: bool,
        expecting_comma_or_close: bool,
    },
    /// Inside a string (key or value). `escape`: previous char was `\`.
    /// `unicode_remaining`: number of `\uXXXX` hex digits still expected.
    String {
        escape: bool,
        unicode_remaining: u8,
        is_key: bool,
    },
    /// Inside a number literal (digits, `.`, `e`, `+`, `-`).
    Number {
        seen_digit: bool,
        seen_dot: bool,
        seen_exp: bool,
        seen_exp_sign: bool,
        seen_exp_digit: bool,
    },
    /// Inside a `true`, `false`, or `null` literal — `expected` is the
    /// remaining suffix to consume.
    Literal { expected: &'static [u8] },
}

/// Streaming JSON validator. Feed it bytes via [`Self::feed`]; ask
/// [`Self::is_valid`] / [`Self::is_complete`] for the post-feed state.
#[derive(Debug, Clone)]
pub struct JsonState {
    stack: VecDeque<Frame>,
    /// `true` once the top-level value has been fully consumed and
    /// only whitespace may follow.
    finished: bool,
    /// `true` if a feed has put us into an unrecoverable state — any
    /// further byte will keep us invalid. The sampler treats this as
    /// "reject every candidate that would land here".
    invalid: bool,
}

impl JsonState {
    /// Fresh state — not yet inside any value, ready for a top-level
    /// JSON value.
    pub fn new() -> Self {
        Self {
            stack: VecDeque::new(),
            finished: false,
            invalid: false,
        }
    }

    /// `true` iff every byte fed so far has been consistent with valid
    /// JSON. (Doesn't mean the value is *complete*.)
    pub fn is_valid(&self) -> bool {
        !self.invalid
    }

    /// `true` iff stopping now would produce a syntactically-complete
    /// JSON value (top-level closed, no open frames, only whitespace
    /// after the final close).
    pub fn is_complete(&self) -> bool {
        !self.invalid && self.finished && self.stack.is_empty()
    }

    /// **#236 P0.1b** — `true` once a top-level JSON value has been
    /// opened (any `{`, `[`, string-quote, number digit, or literal
    /// keyword). Stays `true` for the rest of the parse, even after
    /// the value closes (`is_complete()` then also returns `true`).
    /// Used by the host-sampler-path JSON mask to forbid the
    /// "infinite leading whitespace" failure mode where the model
    /// emits `\n` / ` ` tokens forever instead of starting the value.
    pub fn has_started(&self) -> bool {
        !self.stack.is_empty() || self.finished
    }

    /// Feed one byte. Returns `true` if it was accepted, `false` if
    /// the state turned invalid (stays invalid forever after).
    pub fn feed(&mut self, b: u8) -> bool {
        if self.invalid {
            return false;
        }
        // Whitespace handling outside string/number/literal frames.
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
            // Inside a number literal whitespace ends it.
            if let Some(Frame::Number { seen_digit, .. }) = self.stack.back() {
                if !*seen_digit {
                    self.invalid = true;
                    return false;
                }
                self.stack.pop_back();
                self.fold_after_value();
            }
            // Otherwise whitespace is fine in any "expecting" gap.
            return true;
        }
        // Read just enough state to dispatch — releasing the borrow
        // before we call the per-state handler (which re-borrows).
        let dispatch = match self.stack.back() {
            Some(Frame::String {
                escape,
                unicode_remaining,
                ..
            }) => Dispatch::String(*escape, *unicode_remaining),
            Some(Frame::Number { .. }) => Dispatch::Number,
            Some(Frame::Literal { .. }) => Dispatch::Literal,
            Some(Frame::Object { .. }) => Dispatch::Object,
            Some(Frame::Array { .. }) => Dispatch::Array,
            None => Dispatch::Top,
        };
        match dispatch {
            Dispatch::String(escape, urem) => self.feed_string_byte(b, escape, urem),
            Dispatch::Number => self.feed_number_byte(b),
            Dispatch::Literal => self.feed_literal_byte(b),
            Dispatch::Object => self.feed_object_byte(b),
            Dispatch::Array => self.feed_array_byte(b),
            Dispatch::Top => self.feed_top_level_byte(b),
        }
    }

    /// Apply a slice of bytes; returns `true` if all accepted.
    pub fn feed_slice(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            if !self.feed(b) {
                return false;
            }
        }
        true
    }

    /// Close any trailing open number frame at EOS — numbers don't have
    /// an explicit terminator in JSON, so the streaming validator can't
    /// know `123` is complete until something non-numeric follows.
    /// Sampler calls this when the model picks an EOS token.
    pub fn finalize_at_eos(&mut self) {
        if self.invalid {
            return;
        }
        if let Some(Frame::Number {
            seen_digit,
            seen_exp,
            seen_exp_digit,
            ..
        }) = self.stack.back()
        {
            if !seen_digit || (*seen_exp && !seen_exp_digit) {
                self.invalid = true;
                return;
            }
            self.stack.pop_back();
            self.fold_after_value();
        }
    }

    // ------------------------------------------------------------------
    // Per-state byte handlers.
    // ------------------------------------------------------------------

    fn feed_top_level_byte(&mut self, b: u8) -> bool {
        if self.finished {
            // Only whitespace after the top-level close (caller filters
            // ws). Anything else is invalid extra content.
            self.invalid = true;
            return false;
        }
        self.start_value(b)
    }

    fn start_value(&mut self, b: u8) -> bool {
        match b {
            b'{' => {
                self.stack.push_back(Frame::Object {
                    expecting_key: true,
                    expecting_colon: false,
                    expecting_value: false,
                    expecting_comma_or_close: false,
                });
                true
            }
            b'[' => {
                self.stack.push_back(Frame::Array {
                    expecting_value: true,
                    expecting_comma_or_close: false,
                });
                true
            }
            b'"' => {
                self.stack.push_back(Frame::String {
                    escape: false,
                    unicode_remaining: 0,
                    is_key: false,
                });
                true
            }
            b't' => {
                self.stack.push_back(Frame::Literal { expected: b"rue" });
                true
            }
            b'f' => {
                self.stack.push_back(Frame::Literal { expected: b"alse" });
                true
            }
            b'n' => {
                self.stack.push_back(Frame::Literal { expected: b"ull" });
                true
            }
            b'-' | b'0'..=b'9' => {
                let seen_digit = b.is_ascii_digit();
                self.stack.push_back(Frame::Number {
                    seen_digit,
                    seen_dot: false,
                    seen_exp: false,
                    seen_exp_sign: false,
                    seen_exp_digit: false,
                });
                true
            }
            _ => {
                self.invalid = true;
                false
            }
        }
    }

    fn feed_string_byte(&mut self, b: u8, escape: bool, unicode_remaining: u8) -> bool {
        if let Some(Frame::String {
            escape: e,
            unicode_remaining: u,
            is_key,
        }) = self.stack.back_mut()
        {
            if unicode_remaining > 0 {
                if b.is_ascii_hexdigit() {
                    *u -= 1;
                    return true;
                }
                self.invalid = true;
                return false;
            }
            if escape {
                *e = false;
                match b {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => true,
                    b'u' => {
                        *u = 4;
                        true
                    }
                    _ => {
                        self.invalid = true;
                        false
                    }
                }
            } else if b == b'\\' {
                *e = true;
                true
            } else if b == b'"' {
                let was_key = *is_key;
                self.stack.pop_back();
                if was_key {
                    // After a key string, expect `:`.
                    if let Some(Frame::Object {
                        expecting_key,
                        expecting_colon,
                        ..
                    }) = self.stack.back_mut()
                    {
                        *expecting_key = false;
                        *expecting_colon = true;
                    }
                } else {
                    self.fold_after_value();
                }
                true
            } else if b < 0x20 {
                // Control bytes inside a string are illegal in JSON.
                self.invalid = true;
                false
            } else {
                true
            }
        } else {
            self.invalid = true;
            false
        }
    }

    fn feed_number_byte(&mut self, b: u8) -> bool {
        let mut close_number = false;
        if let Some(Frame::Number {
            seen_digit,
            seen_dot,
            seen_exp,
            seen_exp_sign,
            seen_exp_digit,
        }) = self.stack.back_mut()
        {
            if b.is_ascii_digit() {
                *seen_digit = true;
                if *seen_exp {
                    *seen_exp_digit = true;
                }
                return true;
            }
            if b == b'.' && !*seen_dot && !*seen_exp && *seen_digit {
                *seen_dot = true;
                return true;
            }
            if (b == b'e' || b == b'E') && !*seen_exp && *seen_digit {
                *seen_exp = true;
                return true;
            }
            if (b == b'+' || b == b'-') && *seen_exp && !*seen_exp_sign && !*seen_exp_digit {
                *seen_exp_sign = true;
                return true;
            }
            // Anything else closes the number; require at least one digit
            // and (if exp open) at least one exp digit.
            if !*seen_digit || (*seen_exp && !*seen_exp_digit) {
                self.invalid = true;
                return false;
            }
            close_number = true;
        }
        if close_number {
            self.stack.pop_back();
            self.fold_after_value();
            // Re-feed the byte at the parent frame.
            return self.feed(b);
        }
        self.invalid = true;
        false
    }

    fn feed_literal_byte(&mut self, b: u8) -> bool {
        let done = if let Some(Frame::Literal { expected }) = self.stack.back_mut() {
            if expected.is_empty() {
                self.invalid = true;
                return false;
            }
            if expected[0] != b {
                self.invalid = true;
                return false;
            }
            *expected = &expected[1..];
            expected.is_empty()
        } else {
            self.invalid = true;
            return false;
        };
        if done {
            self.stack.pop_back();
            self.fold_after_value();
        }
        true
    }

    fn feed_object_byte(&mut self, b: u8) -> bool {
        if let Some(Frame::Object {
            expecting_key,
            expecting_colon,
            expecting_value,
            expecting_comma_or_close,
        }) = self.stack.back_mut()
        {
            if *expecting_key {
                if b == b'"' {
                    *expecting_key = false;
                    self.stack.push_back(Frame::String {
                        escape: false,
                        unicode_remaining: 0,
                        is_key: true,
                    });
                    return true;
                }
                if b == b'}' {
                    self.stack.pop_back();
                    self.fold_after_value();
                    return true;
                }
                self.invalid = true;
                return false;
            }
            if *expecting_colon {
                if b == b':' {
                    *expecting_colon = false;
                    *expecting_value = true;
                    return true;
                }
                self.invalid = true;
                return false;
            }
            if *expecting_value {
                *expecting_value = false;
                *expecting_comma_or_close = true;
                return self.start_value(b);
            }
            if *expecting_comma_or_close {
                if b == b',' {
                    *expecting_comma_or_close = false;
                    *expecting_key = true;
                    return true;
                }
                if b == b'}' {
                    self.stack.pop_back();
                    self.fold_after_value();
                    return true;
                }
                self.invalid = true;
                return false;
            }
        }
        self.invalid = true;
        false
    }

    fn feed_array_byte(&mut self, b: u8) -> bool {
        if let Some(Frame::Array {
            expecting_value,
            expecting_comma_or_close,
        }) = self.stack.back_mut()
        {
            if *expecting_value {
                if b == b']' {
                    self.stack.pop_back();
                    self.fold_after_value();
                    return true;
                }
                *expecting_value = false;
                *expecting_comma_or_close = true;
                return self.start_value(b);
            }
            if *expecting_comma_or_close {
                if b == b',' {
                    *expecting_comma_or_close = false;
                    *expecting_value = true;
                    return true;
                }
                if b == b']' {
                    self.stack.pop_back();
                    self.fold_after_value();
                    return true;
                }
                self.invalid = true;
                return false;
            }
        }
        self.invalid = true;
        false
    }

    /// Called after a value (string/number/literal/object/array) closes.
    /// Updates the parent frame's flags so the next byte is interpreted
    /// correctly.
    fn fold_after_value(&mut self) {
        match self.stack.back_mut() {
            Some(Frame::Object {
                expecting_value,
                expecting_comma_or_close,
                ..
            }) => {
                if *expecting_value {
                    *expecting_value = false;
                    *expecting_comma_or_close = true;
                }
            }
            Some(Frame::Array {
                expecting_value,
                expecting_comma_or_close,
            }) => {
                if *expecting_value {
                    *expecting_value = false;
                    *expecting_comma_or_close = true;
                }
            }
            None => {
                self.finished = true;
            }
            _ => {}
        }
    }
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(text: &str) -> (bool, bool) {
        let mut s = JsonState::new();
        let ok = s.feed_slice(text.as_bytes());
        (ok, s.is_complete())
    }

    #[test]
    fn empty_object() {
        let (ok, done) = validate("{}");
        assert!(ok && done);
    }

    #[test]
    fn simple_object() {
        let (ok, done) = validate(r#"{"a":1,"b":"x"}"#);
        assert!(ok && done);
    }

    #[test]
    fn nested() {
        let (ok, done) = validate(r#"{"queries":["a","b"],"meta":{"n":2,"ok":true,"x":null}}"#);
        assert!(ok && done);
    }

    #[test]
    fn open_object_not_complete() {
        let (ok, done) = validate(r#"{"a":1"#);
        assert!(ok);
        assert!(!done);
    }

    #[test]
    fn rejects_extra_after_close() {
        let mut s = JsonState::new();
        assert!(s.feed_slice(b"{}"));
        assert!(s.is_complete());
        assert!(!s.feed_slice(b"x"));
    }

    #[test]
    fn rejects_mismatched_brace() {
        let mut s = JsonState::new();
        assert!(s.feed_slice(b"{"));
        assert!(!s.feed_slice(b"]"));
    }

    #[test]
    fn rejects_double_comma() {
        let mut s = JsonState::new();
        assert!(s.feed_slice(b"["));
        assert!(s.feed_slice(b"1"));
        assert!(s.feed_slice(b","));
        assert!(!s.feed_slice(b","));
    }

    #[test]
    fn unicode_escape() {
        let (ok, done) = validate(r#""hi é bye""#);
        assert!(ok && done);
    }

    #[test]
    fn negative_number() {
        // Numbers don't have an explicit terminator — we only know
        // the value is complete when something non-numeric appears
        // OR EOS is reached. Caller must `finalize_at_eos` on EOS.
        let mut s = JsonState::new();
        assert!(s.feed_slice(b"-3.14e+2"));
        s.finalize_at_eos();
        assert!(s.is_valid());
        assert!(s.is_complete());
    }

    #[test]
    fn rejects_lone_minus() {
        let mut s = JsonState::new();
        assert!(s.feed_slice(b"-"));
        // Open number with just `-` shouldn't be is_complete.
        assert!(!s.is_complete());
        // Whitespace should now invalidate (no digit consumed).
        assert!(!s.feed_slice(b" "));
    }

    #[test]
    fn literal_true() {
        let (ok, done) = validate("true");
        assert!(ok && done);
    }

    #[test]
    fn literal_null() {
        let (ok, done) = validate("null");
        assert!(ok && done);
    }
}
