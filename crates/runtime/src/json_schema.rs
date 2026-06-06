//! Schema-constrained JSON decoding for `response_format=json_schema`.
//!
//! [`SchemaState`] is a byte-fed validator with the same surface as
//! [`crate::json_grammar::JsonState`] (`feed` / `feed_slice` / `is_valid` /
//! `is_complete` / `finalize_at_eos`), so the sampler's token mask can probe
//! it identically: clone the state, feed a candidate token's bytes, reject
//! the token if it turns the state invalid. The difference is that
//! `SchemaState` also enforces the JSON **Schema** — top-level type, object
//! property names + required keys + `additionalProperties:false`, array
//! `items`, scalar types, and `enum` / `const` literal sets — not just
//! structural well-formedness.
//!
//! Supported schema keywords: `type` (object/array/string/integer/number/
//! boolean/null), `properties`, `required`, `additionalProperties`, `items`,
//! `enum`, `const`. Anything unrecognised degrades to [`Node::Any`] (free
//! JSON of any shape at that position) so an exotic schema never over-rejects.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::json_grammar::JsonState;

/// The active structured-output constraint for a decode: either plain
/// `json_object` structural validation or a `json_schema` validator. Lets
/// the sampler's token mask probe both behind one type.
#[derive(Debug, Clone)]
pub enum JsonConstraint {
    /// `response_format=json_object` — structural JSON, top value must be an
    /// object.
    Object(JsonState),
    /// `response_format=json_schema` — full schema enforcement.
    Schema(SchemaState),
}

impl JsonConstraint {
    pub fn object() -> Self {
        Self::Object(JsonState::new())
    }

    pub fn for_schema(schema: &Value) -> Self {
        Self::Schema(SchemaState::new(schema))
    }

    pub fn feed_slice(&mut self, bytes: &[u8]) -> bool {
        match self {
            Self::Object(s) => s.feed_slice(bytes),
            Self::Schema(s) => s.feed_slice(bytes),
        }
    }

    pub fn is_complete(&self) -> bool {
        match self {
            Self::Object(s) => s.is_complete(),
            Self::Schema(s) => s.is_complete(),
        }
    }

    pub fn has_started(&self) -> bool {
        match self {
            Self::Object(s) => s.has_started(),
            Self::Schema(s) => s.has_started(),
        }
    }

    pub fn finalize_at_eos(&mut self) {
        match self {
            Self::Object(s) => s.finalize_at_eos(),
            Self::Schema(s) => s.finalize_at_eos(),
        }
    }

    /// True only for `json_object`, where the top value must be an object so
    /// the mask additionally forbids non-`{` openers. Schemas gate the
    /// top-level type themselves.
    pub fn top_must_be_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }
}

/// A compiled schema node. `Any` is the permissive fallback for keywords we
/// don't model — that position accepts any well-formed JSON value.
#[derive(Debug, Clone)]
pub enum Node {
    Any,
    Bool,
    Null,
    Int,
    Num,
    Str,
    /// One of a fixed set of literal JSON values (`enum` / `const`),
    /// pre-serialised to their canonical compact bytes.
    Enum(Vec<Vec<u8>>),
    Object {
        properties: BTreeMap<String, Node>,
        required: Vec<String>,
        additional: bool,
    },
    Array {
        items: Box<Node>,
    },
}

impl Node {
    /// Compile a JSON-Schema `Value` into a [`Node`]. Unsupported constructs
    /// fall back to [`Node::Any`].
    pub fn compile(schema: &Value) -> Node {
        let obj = match schema {
            Value::Object(m) => m,
            // A bare `true`/`{}` schema (or anything non-object) allows anything.
            _ => return Node::Any,
        };

        // `const` is sugar for a single-value enum; `enum` is a literal set.
        if let Some(c) = obj.get("const") {
            return Node::Enum(vec![canonical_bytes(c)]);
        }
        if let Some(Value::Array(values)) = obj.get("enum") {
            return Node::Enum(values.iter().map(canonical_bytes).collect());
        }

        match obj.get("type").and_then(Value::as_str) {
            Some("object") => Node::compile_object(obj),
            Some("array") => {
                let items = obj
                    .get("items")
                    .map(Node::compile)
                    .unwrap_or(Node::Any);
                Node::Array {
                    items: Box::new(items),
                }
            }
            Some("string") => Node::Str,
            Some("integer") => Node::Int,
            Some("number") => Node::Num,
            Some("boolean") => Node::Bool,
            Some("null") => Node::Null,
            // No `type` but `properties` present → treat as an object.
            _ if obj.contains_key("properties") => Node::compile_object(obj),
            _ => Node::Any,
        }
    }

    fn compile_object(obj: &serde_json::Map<String, Value>) -> Node {
        let mut properties = BTreeMap::new();
        if let Some(Value::Object(props)) = obj.get("properties") {
            for (k, v) in props {
                properties.insert(k.clone(), Node::compile(v));
            }
        }
        let required = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        // OpenAI structured-output default is strict (no extra keys); honour
        // an explicit `additionalProperties:true`, else forbid extras.
        let additional = matches!(obj.get("additionalProperties"), Some(Value::Bool(true)));
        Node::Object {
            properties,
            required,
            additional,
        }
    }
}

/// Canonical compact UTF-8 bytes for a literal JSON value (used for `enum`
/// matching — `serde_json::to_vec` emits no insignificant whitespace).
fn canonical_bytes(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

/// One active parse frame: a position in the JSON annotated with the schema
/// node it must satisfy.
#[derive(Debug, Clone)]
enum Frame {
    /// Matching one of an `enum`'s literals. `alive` holds the byte offset
    /// within each still-possible candidate (None once a candidate diverges).
    Enum {
        candidates: Vec<Vec<u8>>,
        pos: usize,
    },
    Str {
        escape: bool,
        unicode_remaining: u8,
    },
    Num {
        integer: bool,
        seen_digit: bool,
        seen_dot: bool,
        seen_exp: bool,
        seen_exp_sign: bool,
        seen_exp_digit: bool,
    },
    Literal {
        expected: &'static [u8],
    },
    /// Object body. `seen` accumulates matched keys (for required-key + dup
    /// checks); `pending_key` buffers the key string currently being read.
    Object {
        node_props: BTreeMap<String, Node>,
        required: Vec<String>,
        additional: bool,
        seen: Vec<String>,
        stage: ObjStage,
        pending_key: String,
    },
    Array {
        items: Node,
        stage: ArrStage,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ObjStage {
    /// Just opened (`{`) — a key or an immediate `}` (empty object).
    ExpectKeyOrClose,
    /// Just after a `,` — a key is mandatory (no trailing-comma close).
    ExpectKey,
    InKey,
    ExpectColon,
    ExpectValue,
    ExpectCommaOrClose,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArrStage {
    ExpectValueOrClose,
    ExpectValue,
    ExpectCommaOrClose,
}

/// Schema-constrained streaming JSON validator. Drop-in for `JsonState`.
#[derive(Debug, Clone)]
pub struct SchemaState {
    root: Node,
    stack: Vec<Frame>,
    started: bool,
    finished: bool,
    invalid: bool,
}

impl SchemaState {
    /// Fresh validator for `schema` (a JSON-Schema `Value`).
    pub fn new(schema: &Value) -> Self {
        Self {
            root: Node::compile(schema),
            stack: Vec::new(),
            started: false,
            finished: false,
            invalid: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.invalid
    }

    /// True iff stopping now yields a schema-valid, structurally-complete
    /// value (top value closed, every nested frame settled).
    pub fn is_complete(&self) -> bool {
        !self.invalid && self.finished && self.stack.is_empty()
    }

    pub fn has_started(&self) -> bool {
        self.started
    }

    /// Numbers have no terminator; settle a trailing number frame at EOS.
    pub fn finalize_at_eos(&mut self) {
        if self.invalid {
            return;
        }
        if let Some(Frame::Num {
            seen_digit,
            seen_exp,
            seen_exp_digit,
            ..
        }) = self.stack.last()
        {
            if !seen_digit || (*seen_exp && !seen_exp_digit) {
                self.invalid = true;
                return;
            }
            self.stack.pop();
            self.fold_after_value();
        }
        // An enum still mid-literal that exactly equals a candidate is fine;
        // otherwise it's incomplete (caller's is_complete() catches that).
        if let Some(Frame::Enum { candidates, pos }) = self.stack.last() {
            if candidates.iter().any(|c| c.len() == *pos) {
                self.stack.pop();
                self.fold_after_value();
            }
        }
    }

    pub fn feed_slice(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            if !self.feed(b) {
                return false;
            }
        }
        true
    }

    /// Feed one byte; returns false (and latches invalid) on any schema or
    /// structural violation.
    pub fn feed(&mut self, b: u8) -> bool {
        if self.invalid {
            return false;
        }
        // Whitespace is allowed in any structural gap (not inside a string,
        // number, literal, or enum-literal).
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
            match self.stack.last() {
                Some(Frame::Str { .. } | Frame::Literal { .. } | Frame::Enum { .. }) => {}
                Some(Frame::Num { seen_digit, .. }) => {
                    if !seen_digit {
                        self.invalid = true;
                        return false;
                    }
                    self.stack.pop();
                    self.fold_after_value();
                    return true;
                }
                _ => return true,
            }
        }
        match self.stack.last() {
            None => {
                if self.finished {
                    self.invalid = true;
                    return false;
                }
                let node = self.root.clone();
                self.start_value(b, &node)
            }
            Some(Frame::Enum { .. }) => self.feed_enum(b),
            Some(Frame::Str { .. }) => self.feed_string(b),
            Some(Frame::Num { .. }) => self.feed_number(b),
            Some(Frame::Literal { .. }) => self.feed_literal(b),
            Some(Frame::Object { .. }) => self.feed_object(b),
            Some(Frame::Array { .. }) => self.feed_array(b),
        }
    }

    /// Begin a value that must conform to `node`, dispatching on the first
    /// byte. Pushes the appropriate frame.
    fn start_value(&mut self, b: u8, node: &Node) -> bool {
        self.started = true;
        match node {
            Node::Enum(cands) => {
                // Filter to candidates whose first byte matches.
                let candidates: Vec<Vec<u8>> = cands
                    .iter()
                    .filter(|c| c.first() == Some(&b))
                    .cloned()
                    .collect();
                if candidates.is_empty() {
                    self.invalid = true;
                    return false;
                }
                let done = candidates.iter().any(|c| c.len() == 1);
                self.stack.push(Frame::Enum {
                    candidates,
                    pos: 1,
                });
                if done && self.stack.last().is_some_and(single_byte_enum_done) {
                    self.stack.pop();
                    self.fold_after_value();
                }
                true
            }
            Node::Object { .. } => {
                if b != b'{' {
                    self.invalid = true;
                    return false;
                }
                self.stack.push(object_frame(node));
                true
            }
            Node::Array { items } => {
                if b != b'[' {
                    self.invalid = true;
                    return false;
                }
                self.stack.push(Frame::Array {
                    items: (**items).clone(),
                    stage: ArrStage::ExpectValueOrClose,
                });
                true
            }
            Node::Str => self.open_string(b),
            Node::Bool => match b {
                b't' => self.open_literal(b"rue"),
                b'f' => self.open_literal(b"alse"),
                _ => self.reject(),
            },
            Node::Null => {
                if b == b'n' {
                    self.open_literal(b"ull")
                } else {
                    self.reject()
                }
            }
            Node::Int | Node::Num => self.open_number(b, matches!(node, Node::Int)),
            Node::Any => self.start_any(b),
        }
    }

    /// Free-form value (schema `Any`): mirror the structural grammar with no
    /// type/key constraints.
    fn start_any(&mut self, b: u8) -> bool {
        self.started = true;
        match b {
            b'{' => {
                self.stack.push(Frame::Object {
                    node_props: BTreeMap::new(),
                    required: Vec::new(),
                    additional: true,
                    seen: Vec::new(),
                    stage: ObjStage::ExpectKeyOrClose,
                    pending_key: String::new(),
                });
                true
            }
            b'[' => {
                self.stack.push(Frame::Array {
                    items: Node::Any,
                    stage: ArrStage::ExpectValueOrClose,
                });
                true
            }
            b'"' => self.open_string(b),
            b't' => self.open_literal(b"rue"),
            b'f' => self.open_literal(b"alse"),
            b'n' => self.open_literal(b"ull"),
            b'-' | b'0'..=b'9' => self.open_number(b, false),
            _ => self.reject(),
        }
    }

    fn open_string(&mut self, b: u8) -> bool {
        if b != b'"' {
            return self.reject();
        }
        self.stack.push(Frame::Str {
            escape: false,
            unicode_remaining: 0,
        });
        true
    }

    fn open_literal(&mut self, expected: &'static [u8]) -> bool {
        self.stack.push(Frame::Literal { expected });
        true
    }

    fn open_number(&mut self, b: u8, integer: bool) -> bool {
        if b != b'-' && !b.is_ascii_digit() {
            return self.reject();
        }
        self.stack.push(Frame::Num {
            integer,
            seen_digit: b.is_ascii_digit(),
            seen_dot: false,
            seen_exp: false,
            seen_exp_sign: false,
            seen_exp_digit: false,
        });
        true
    }

    fn reject(&mut self) -> bool {
        self.invalid = true;
        false
    }

    fn feed_enum(&mut self, b: u8) -> bool {
        let Some(Frame::Enum { candidates, pos }) = self.stack.last_mut() else {
            return self.reject();
        };
        candidates.retain(|c| c.get(*pos) == Some(&b));
        if candidates.is_empty() {
            return self.reject();
        }
        *pos += 1;
        let p = *pos;
        let complete = candidates.iter().any(|c| c.len() == p);
        // If every surviving candidate is fully matched, close the value.
        if complete && candidates.iter().all(|c| c.len() == p) {
            self.stack.pop();
            self.fold_after_value();
        }
        true
    }

    fn feed_string(&mut self, b: u8) -> bool {
        let Some(Frame::Str {
            escape,
            unicode_remaining,
        }) = self.stack.last_mut()
        else {
            return self.reject();
        };
        if *unicode_remaining > 0 {
            if b.is_ascii_hexdigit() {
                *unicode_remaining -= 1;
                return true;
            }
            return self.reject();
        }
        if *escape {
            *escape = false;
            return match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => true,
                b'u' => {
                    *unicode_remaining = 4;
                    true
                }
                _ => self.reject(),
            };
        }
        if b == b'\\' {
            *escape = true;
            true
        } else if b == b'"' {
            self.stack.pop();
            self.fold_after_value();
            true
        } else if b < 0x20 {
            self.reject()
        } else {
            true
        }
    }

    fn feed_number(&mut self, b: u8) -> bool {
        let mut close = false;
        if let Some(Frame::Num {
            integer,
            seen_digit,
            seen_dot,
            seen_exp,
            seen_exp_sign,
            seen_exp_digit,
        }) = self.stack.last_mut()
        {
            if b.is_ascii_digit() {
                *seen_digit = true;
                if *seen_exp {
                    *seen_exp_digit = true;
                }
                return true;
            }
            // Integers forbid the fractional/exponent forms.
            if !*integer {
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
            }
            if !*seen_digit || (*seen_exp && !*seen_exp_digit) {
                return self.reject();
            }
            close = true;
        }
        if close {
            self.stack.pop();
            self.fold_after_value();
            return self.feed(b);
        }
        self.reject()
    }

    fn feed_literal(&mut self, b: u8) -> bool {
        let done = if let Some(Frame::Literal { expected }) = self.stack.last_mut() {
            if expected.first() != Some(&b) {
                return self.reject();
            }
            *expected = &expected[1..];
            expected.is_empty()
        } else {
            return self.reject();
        };
        if done {
            self.stack.pop();
            self.fold_after_value();
        }
        true
    }

    fn feed_object(&mut self, b: u8) -> bool {
        // Pull the fields we need; mutate via last_mut in arms.
        let stage = match self.stack.last() {
            Some(Frame::Object { stage, .. }) => stage.clone(),
            _ => return self.reject(),
        };
        match stage {
            ObjStage::ExpectKeyOrClose | ObjStage::ExpectKey => {
                if b == b'}' && stage == ObjStage::ExpectKeyOrClose {
                    return self.close_object();
                }
                if b == b'"' {
                    if let Some(Frame::Object { stage, pending_key, .. }) = self.stack.last_mut() {
                        *stage = ObjStage::InKey;
                        pending_key.clear();
                    }
                    return true;
                }
                self.reject()
            }
            ObjStage::InKey => {
                // Accumulate raw key bytes until the closing quote. (Keys are
                // plain identifiers in practice; escapes are passed through.)
                if b == b'"' {
                    let key = match self.stack.last() {
                        Some(Frame::Object { pending_key, .. }) => pending_key.clone(),
                        _ => return self.reject(),
                    };
                    // Validate key against the schema's allowed properties.
                    let (allowed, dup) = match self.stack.last() {
                        Some(Frame::Object {
                            node_props,
                            additional,
                            seen,
                            ..
                        }) => (
                            *additional || node_props.contains_key(&key),
                            seen.contains(&key),
                        ),
                        _ => (false, false),
                    };
                    if !allowed || dup {
                        return self.reject();
                    }
                    if let Some(Frame::Object { stage, seen, .. }) = self.stack.last_mut() {
                        seen.push(key);
                        *stage = ObjStage::ExpectColon;
                    }
                    return true;
                }
                if b == b'\\' || b < 0x20 {
                    // Disallow escapes/control in keys for simplicity — real
                    // property names never need them.
                    return self.reject();
                }
                // Constrain the key to remain a prefix of some allowed,
                // not-yet-seen property — otherwise the model can run away
                // emitting an unbounded key that only fails at the close quote.
                let reject = match self.stack.last() {
                    Some(Frame::Object {
                        node_props,
                        additional,
                        seen,
                        pending_key,
                        ..
                    }) => {
                        if *additional {
                            false
                        } else {
                            let mut cand = pending_key.clone();
                            cand.push(b as char);
                            !node_props
                                .keys()
                                .any(|k| !seen.contains(k) && k.starts_with(&cand))
                        }
                    }
                    _ => true,
                };
                if reject {
                    return self.reject();
                }
                if let Some(Frame::Object { pending_key, .. }) = self.stack.last_mut() {
                    pending_key.push(b as char);
                }
                true
            }
            ObjStage::ExpectColon => {
                if b == b':' {
                    if let Some(Frame::Object { stage, .. }) = self.stack.last_mut() {
                        *stage = ObjStage::ExpectValue;
                    }
                    return true;
                }
                self.reject()
            }
            ObjStage::ExpectValue => {
                let value_node = match self.stack.last() {
                    Some(Frame::Object {
                        node_props, seen, ..
                    }) => seen
                        .last()
                        .and_then(|k| node_props.get(k))
                        .cloned()
                        .unwrap_or(Node::Any),
                    _ => Node::Any,
                };
                if let Some(Frame::Object { stage, .. }) = self.stack.last_mut() {
                    *stage = ObjStage::ExpectCommaOrClose;
                }
                self.start_value(b, &value_node)
            }
            ObjStage::ExpectCommaOrClose => {
                if b == b',' {
                    // Only permit a comma if another property can still be
                    // added — else the model paints itself into a corner with
                    // no valid key, and the masked output drifts.
                    let can_add = match self.stack.last() {
                        Some(Frame::Object {
                            node_props,
                            additional,
                            seen,
                            ..
                        }) => *additional || node_props.keys().any(|k| !seen.contains(k)),
                        _ => false,
                    };
                    if !can_add {
                        return self.reject();
                    }
                    if let Some(Frame::Object { stage, .. }) = self.stack.last_mut() {
                        *stage = ObjStage::ExpectKey;
                    }
                    return true;
                }
                if b == b'}' {
                    return self.close_object();
                }
                self.reject()
            }
        }
    }

    fn close_object(&mut self) -> bool {
        // Enforce required keys.
        let ok = match self.stack.last() {
            Some(Frame::Object { required, seen, .. }) => {
                required.iter().all(|r| seen.contains(r))
            }
            _ => false,
        };
        if !ok {
            return self.reject();
        }
        self.stack.pop();
        self.fold_after_value();
        true
    }

    fn feed_array(&mut self, b: u8) -> bool {
        let stage = match self.stack.last() {
            Some(Frame::Array { stage, .. }) => stage.clone(),
            _ => return self.reject(),
        };
        match stage {
            ArrStage::ExpectValueOrClose => {
                if b == b']' {
                    self.stack.pop();
                    self.fold_after_value();
                    return true;
                }
                let items = match self.stack.last() {
                    Some(Frame::Array { items, .. }) => items.clone(),
                    _ => return self.reject(),
                };
                if let Some(Frame::Array { stage, .. }) = self.stack.last_mut() {
                    *stage = ArrStage::ExpectCommaOrClose;
                }
                self.start_value(b, &items)
            }
            ArrStage::ExpectValue => {
                let items = match self.stack.last() {
                    Some(Frame::Array { items, .. }) => items.clone(),
                    _ => return self.reject(),
                };
                if let Some(Frame::Array { stage, .. }) = self.stack.last_mut() {
                    *stage = ArrStage::ExpectCommaOrClose;
                }
                self.start_value(b, &items)
            }
            ArrStage::ExpectCommaOrClose => {
                if b == b',' {
                    if let Some(Frame::Array { stage, .. }) = self.stack.last_mut() {
                        *stage = ArrStage::ExpectValue;
                    }
                    return true;
                }
                if b == b']' {
                    self.stack.pop();
                    self.fold_after_value();
                    return true;
                }
                self.reject()
            }
        }
    }

    /// After a value closes, advance the parent frame (or mark top finished).
    fn fold_after_value(&mut self) {
        match self.stack.last_mut() {
            None => self.finished = true,
            Some(Frame::Array { stage, .. }) if *stage == ArrStage::ExpectValueOrClose => {
                *stage = ArrStage::ExpectCommaOrClose;
            }
            _ => {}
        }
    }
}

/// Build an Object frame from a compiled `Node::Object`.
fn object_frame(node: &Node) -> Frame {
    if let Node::Object {
        properties,
        required,
        additional,
    } = node
    {
        Frame::Object {
            node_props: properties.clone(),
            required: required.clone(),
            additional: *additional,
            seen: Vec::new(),
            stage: ObjStage::ExpectKeyOrClose,
            pending_key: String::new(),
        }
    } else {
        // Unreachable in practice; permissive fallback.
        Frame::Object {
            node_props: BTreeMap::new(),
            required: Vec::new(),
            additional: true,
            seen: Vec::new(),
            stage: ObjStage::ExpectKeyOrClose,
            pending_key: String::new(),
        }
    }
}

fn single_byte_enum_done(f: &Frame) -> bool {
    matches!(f, Frame::Enum { candidates, pos } if candidates.iter().all(|c| c.len() == *pos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(schema: &Value, text: &str) -> (bool, bool) {
        let mut s = SchemaState::new(schema);
        let ok = s.feed_slice(text.as_bytes());
        s.finalize_at_eos();
        (ok, s.is_complete())
    }

    #[test]
    fn object_required_ok() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        });
        let (ok, done) = run(&schema, r#"{"name":"Sam","age":30}"#);
        assert!(ok && done);
    }

    #[test]
    fn object_missing_required_incomplete() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        });
        // Closing the object without `age` must be rejected.
        let mut s = SchemaState::new(&schema);
        assert!(s.feed_slice(br#"{"name":"Sam""#));
        assert!(!s.feed_slice(b"}"));
    }

    #[test]
    fn rejects_unknown_property() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        });
        let mut s = SchemaState::new(&schema);
        assert!(s.feed_slice(br#"{"#));
        assert!(!s.feed_slice(br#""bogus""#));
    }

    #[test]
    fn allows_extra_when_additional_true() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": true,
        });
        let (ok, done) = run(&schema, r#"{"name":"x","extra":7}"#);
        assert!(ok && done);
    }

    #[test]
    fn type_mismatch_string_for_integer() {
        let schema = json!({
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        });
        let mut s = SchemaState::new(&schema);
        // `age` expects an integer; a string opener `"` must be rejected.
        assert!(s.feed_slice(br#"{"age":"#));
        assert!(!s.feed_slice(br#"""#));
    }

    #[test]
    fn integer_rejects_fraction() {
        let schema = json!({"type": "integer"});
        let mut s = SchemaState::new(&schema);
        assert!(s.feed_slice(b"12"));
        assert!(!s.feed_slice(b"."));
    }

    #[test]
    fn enum_accepts_member_rejects_other() {
        let schema = json!({"enum": ["red", "green", "blue"]});
        let (ok, done) = run(&schema, r#""green""#);
        assert!(ok && done);
        let mut s = SchemaState::new(&schema);
        // "gr" still viable (green), but "gru" diverges from every member.
        assert!(s.feed_slice(br#""gr"#));
        assert!(!s.feed_slice(b"u"));
    }

    #[test]
    fn rejects_comma_when_no_more_properties() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false,
        });
        let mut s = SchemaState::new(&schema);
        assert!(s.feed_slice(br#"{"a":1"#));
        // `a` is the only property and it's been used — a comma (more keys)
        // must be rejected so the model is forced to close.
        assert!(!s.feed_slice(b","));
    }

    #[test]
    fn rejects_runaway_key() {
        let schema = json!({
            "type": "object",
            "properties": {"items": {"type": "array"}, "label": {"type": "string"}},
        });
        let mut s = SchemaState::new(&schema);
        assert!(s.feed_slice(br#"{""#));
        // "a" is a prefix of neither "items" nor "label" → reject immediately.
        assert!(!s.feed_slice(b"a"));
    }

    #[test]
    fn array_of_integers() {
        let schema = json!({"type": "array", "items": {"type": "integer"}});
        let (ok, done) = run(&schema, "[1,2,3]");
        assert!(ok && done);
        let mut s = SchemaState::new(&schema);
        assert!(s.feed_slice(b"[1,"));
        // A string element violates items:integer.
        assert!(!s.feed_slice(br#"""#));
    }

    #[test]
    fn nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                }
            },
            "required": ["user"],
        });
        let (ok, done) = run(&schema, r#"{"user":{"id":5}}"#);
        assert!(ok && done);
    }

    #[test]
    fn top_level_type_gate() {
        let schema = json!({"type": "object", "properties": {}});
        let mut s = SchemaState::new(&schema);
        // A top-level array is wrong when the schema demands an object.
        assert!(!s.feed_slice(b"["));
    }

    #[test]
    fn boolean_and_null() {
        assert!(run(&json!({"type": "boolean"}), "true").0);
        assert!(run(&json!({"type": "null"}), "null").1);
    }
}
