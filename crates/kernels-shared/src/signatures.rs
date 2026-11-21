//! Parameter counts of every `extern "C" __global__` entry point, read from the
//! kernel sources at build time. The backend crates turn the table into a
//! launch-time check: a host launcher whose argument list drifted from the
//! kernel signature fails with an error instead of letting the driver read
//! past the argument array.

use std::collections::BTreeMap;
use std::path::Path;

/// Entry-point name → parameter count over `files`. Files that cannot be read
/// are skipped; a name defined twice keeps its first count.
pub fn param_counts<P: AsRef<Path>>(files: &[P]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for f in files {
        if let Ok(src) = std::fs::read_to_string(f.as_ref()) {
            for (name, n) in entry_points(&src) {
                out.entry(name).or_insert(n);
            }
        }
    }
    out
}

/// `pub static PARAM_COUNTS: &[(&str, u16)]`, sorted by name, for
/// `include!` in the backend kernel crate.
pub fn render_table(counts: &BTreeMap<String, usize>) -> String {
    let mut s = String::from("\npub static PARAM_COUNTS: &[(&str, u16)] = &[\n");
    for (name, n) in counts {
        s.push_str(&format!("    (\"{name}\", {n}),\n"));
    }
    s.push_str("];\n");
    s
}

fn entry_points(src: &str) -> Vec<(String, usize)> {
    let code = strip_comments_and_directives(src);
    let mut out = Vec::new();
    let mut rest = code.as_str();
    while let Some(at) = rest.find("extern \"C\"") {
        rest = &rest[at + "extern \"C\"".len()..];
        let head_end = rest.find(['{', ';']).unwrap_or(rest.len());
        let head = &rest[..head_end];
        if !head.contains("__global__") {
            continue;
        }
        if let Some(entry) = parse_head(head) {
            out.push(entry);
        }
    }
    out
}

/// `head` runs from after `extern "C"` to the body's `{`: the parameter list
/// is its last top-level parenthesised group, the name the identifier just
/// before it (`__launch_bounds__(..)` groups come earlier).
fn parse_head(head: &str) -> Option<(String, usize)> {
    let close = head.rfind(')')?;
    let bytes = head.as_bytes();
    let mut depth = 0i32;
    let mut open = None;
    for i in (0..=close).rev() {
        match bytes[i] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    open = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let open = open?;
    let before = head[..open].trim_end();
    let name_start = before
        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |i| i + 1);
    let name = &before[name_start..];
    if name.is_empty() {
        return None;
    }
    Some((name.to_string(), count_params(&head[open + 1..close])))
}

fn count_params(list: &str) -> usize {
    let trimmed = list.trim();
    if trimmed.is_empty() || trimmed == "void" {
        return 0;
    }
    let mut depth = 0i32;
    let mut commas = 0;
    for c in trimmed.chars() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => commas += 1,
            _ => {}
        }
    }
    commas + 1
}

fn strip_comments_and_directives(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    let mut at_line_start = true;
    while let Some(c) = chars.next() {
        match c {
            '/' if chars.peek() == Some(&'/') => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        out.push('\n');
                        break;
                    }
                }
                at_line_start = true;
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = ' ';
                for d in chars.by_ref() {
                    if prev == '*' && d == '/' {
                        break;
                    }
                    prev = d;
                }
                out.push(' ');
            }
            '#' if at_line_start => {
                let mut prev = ' ';
                for d in chars.by_ref() {
                    if d == '\n' && prev != '\\' {
                        out.push('\n');
                        break;
                    }
                    prev = d;
                }
            }
            '\n' => {
                out.push('\n');
                at_line_start = true;
            }
            c if c.is_whitespace() => out.push(c),
            c => {
                out.push(c);
                at_line_start = false;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_plain_entry_points() {
        let src = r#"
#include <hip/hip_runtime.h>
// extern "C" __global__ void commented(int a, int b);
extern "C" __global__ void flambeau_a(
    const float* __restrict__ x, // input
    float* y,                    /* output, a, b */
    const int n
) { y[0] = x[0]; }
extern "C" __global__ __launch_bounds__(256, 2) void flambeau_b(const Foo<int, 4>* p, int k) {}
extern "C" __global__ void flambeau_c() {}
extern "C" int host_helper(int a);
"#;
        let got = entry_points(src);
        assert_eq!(
            got,
            vec![("flambeau_a".to_string(), 3), ("flambeau_b".to_string(), 2), ("flambeau_c".to_string(), 0)]
        );
    }

    #[test]
    fn launch_bounds_after_void() {
        let src = "extern \"C\" __global__ void __launch_bounds__(64) flambeau_d(int a, int b, int c) {}";
        assert_eq!(entry_points(src), vec![("flambeau_d".to_string(), 3)]);
    }
}
