// C3.1 protocol-level test for renderMarkdown(). Pulls the three
// renderer functions out of the chat UI's <script> block and runs
// them directly.

import { readFileSync } from "fs";

const html = readFileSync("/artefact/flambeau/crates/server/assets/index.html", "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error("FAIL: no <script>"); process.exit(1); }
const js = m[1];

// Slice from the start of the first renderer fn to the end of the
// last (inlineTransforms). Bracketed by the C3.1 banner comment and
// the C2.1 banner that follows.
const start = js.indexOf("// C3.1 — minimum-viable Markdown renderer");
const end = js.indexOf("// C2.1 — sampler controls.");
if (start < 0 || end < 0 || end <= start) {
  console.error("FAIL: renderer block markers not found");
  process.exit(1);
}
const slice = js.slice(start, end);
const factory = new Function(`${slice}\n;return { escapeHtml, renderMarkdown, inlineTransforms };`);
const { renderMarkdown } = factory();

const fail = (label, got, want) => {
  console.error(`FAIL ${label}:\n  got:  ${JSON.stringify(got)}\n  want: ${JSON.stringify(want)}`);
  process.exit(1);
};
const expect = (label, got, fragments) => {
  if (typeof fragments === "string") fragments = [fragments];
  for (const f of fragments) {
    if (!got.includes(f)) fail(label, got, f);
  }
  console.log(`OK ${label}`);
};

expect("paragraph",
  renderMarkdown("hello world"),
  "<p>hello world</p>");

expect("bold + italic + inline code",
  renderMarkdown("**bold** and *italic* and `code`"),
  ["<strong>bold</strong>", "<em>italic</em>", "<code>code</code>"]);

expect("link",
  renderMarkdown("see [docs](https://example.com)"),
  '<a href="https://example.com" target="_blank" rel="noopener">docs</a>');

expect("html escape",
  renderMarkdown("<script>alert(1)</script>"),
  "&lt;script&gt;alert(1)&lt;/script&gt;");

expect("heading",
  renderMarkdown("## A heading\n\nbody"),
  ["<h2>A heading</h2>", "<p>body</p>"]);

expect("ul",
  renderMarkdown("- a\n- b\n- c"),
  ["<ul>", "<li>a</li>", "<li>b</li>", "<li>c</li>", "</ul>"]);

expect("ol",
  renderMarkdown("1. a\n2. b"),
  ["<ol>", "<li>a</li>", "<li>b</li>", "</ol>"]);

expect("blockquote",
  renderMarkdown("> quoth the raven\n> nevermore"),
  ["<blockquote>", "quoth the raven", "nevermore", "</blockquote>"]);

expect("hr",
  renderMarkdown("above\n\n---\n\nbelow"),
  ["<hr>"]);

expect("fenced code",
  renderMarkdown("```rust\nlet x = 1;\nfn main() {}\n```"),
  ['<pre><code class="lang-rust">let x = 1;\nfn main() {}</code></pre>']);

expect("fenced code (no lang)",
  renderMarkdown("```\nplain\n```"),
  ["<pre><code>plain</code></pre>"]);

expect("unterminated fence — dashed border for streaming",
  renderMarkdown("```py\nimport os\n"),
  ['<pre class="unterminated"><code class="lang-py">import os']);

expect("inline code with html chars",
  renderMarkdown("Use `Vec<T>` for that"),
  ["<code>Vec&lt;T&gt;</code>"]);

const intra = renderMarkdown("snake_case_var stays put");
if (intra.includes("<em>")) fail("italic intra-word", intra, "no <em>");
console.log("OK italic intra-word (snake_case left alone)");

expect("paragraph then list",
  renderMarkdown("intro\n\n- one\n- two"),
  ["<p>intro</p>", "<ul>"]);

console.log("\nALL OK");
