#!/usr/bin/env bash
# Regenerate `expected/*.txt` reference renders from llama.cpp's
# `test-chat-template` binary. See README.md §Regenerating.
#
# Usage: regenerate.sh <qwen3.6.gguf> <test-chat-template-binary>

set -euo pipefail

GGUF="${1:?usage: $0 <gguf> <test-chat-template>}"
TCT="${2:?usage: $0 <gguf> <test-chat-template>}"

CERT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURES="$CERT_DIR/fixtures"
EXPECTED="$CERT_DIR/expected"

mkdir -p "$EXPECTED"

# Extract the Jinja template from GGUF via flambeau's helper binary.
# The helper ships in `crates/quant` tests; easier path is a tiny grep
# of the inspect-gguf output, but that truncates long strings, so we
# compile and run the helper.
TEMPLATE_OUT="$(mktemp --suffix=.jinja)"
trap 'rm -f "$TEMPLATE_OUT"' EXIT
cargo run --quiet --release -p flambeau-cli -- \
    extract-chat-template --path "$GGUF" --out "$TEMPLATE_OUT"

if [[ ! -s "$TEMPLATE_OUT" ]]; then
    echo "extracted template is empty — did GGUF contain tokenizer.chat_template?" >&2
    exit 1
fi

# For each fixture, write the rendered output next to it.
count=0
for f in "$FIXTURES"/*.json; do
    name="$(basename "${f%.json}")"
    "$TCT" "$TEMPLATE_OUT" --json "$f" --output "$EXPECTED/$name.txt" >/dev/null
    count=$((count + 1))
done

echo "regenerated $count expected renders into $EXPECTED"
