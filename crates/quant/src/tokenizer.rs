//! Build a `tokenizers::Tokenizer` from GGUF metadata — piece 2 of 3.
//! GGUF embeds the tokenizer as a handful of metadata keys:
//! - `tokenizer.ggml.model`: BPE family name ("gpt2" for Qwen3.5/3.6/Llama)
//! - `tokenizer.ggml.pre`: pretokenizer preset ("qwen35", "llama-bpe", ...)
//! - `tokenizer.ggml.tokens`: vocab strings (unicode-escaped for byte bytes)
//! - `tokenizer.ggml.merges`: "a b" BPE merge pairs
//! - `tokenizer.ggml.bos/eos/padding_token_id`: special ids
//! llama.cpp's implementation lives in `src/llama-vocab.cpp`. This module
//! reproduces enough of its "gpt2" + "qwen35"/"qwen2" BPE-load path to give
//! byte-identical token IDs on the `parity_vs_llama_cpp.rs` prompts.
//! Covered pre-tokenizers: "default", "gpt-2", "llama-bpe", "llama3",
//! "qwen2", "qwen35". Others return an error — add them as models arrive.

use anyhow::{anyhow, Context, Result};
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::{
    byte_level::ByteLevel,
    sequence::Sequence as PreTokSequence,
    split::{Split, SplitPattern},
};
use tokenizers::{
    decoders, normalizers, AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer, TokenizerImpl,
};

use crate::gguf::GgufFile;

/// Handles returned by [`load_from_gguf`]. Wraps the `tokenizers::Tokenizer`
/// plus the special-token ids we've seen the runtime care about.
#[derive(Debug)]
pub struct GgufTokenizer {
    pub inner: Tokenizer,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub vocab_size: u32,
    /// All token ids that should terminate chat generation. Includes `eos_id`
    /// plus template-specific end-of-turn markers (e.g. Qwen's `<|im_end|>`).
    /// The server's stop-check loop uses this instead of `eos_id` alone.
    pub stop_ids: Vec<u32>,
    /// **Sampler-G** — subset of `stop_ids` that should ALWAYS stop
    /// generation, even within the early-tokens MIN_RESPONSE_TOKENS
    /// window. Currently `<think>` and `</think>` (Qwen3.6 reasoning
    /// markers) — these should never be emitted by the model when the
    /// chat template ran with `enable_thinking=false`. The early-window
    /// mask was designed to prevent immediate `<|im_end|>`, not to give
    /// the model a chance to leak reasoning artefacts. Treating these
    /// as always-stop kills the chat-truncation pattern where the
    /// model emits `</think>` after a short answer and then duplicates
    /// the response in a confused-reasoning loop.
    pub always_stop_ids: Vec<u32>,
    /// **P1.6a** — Fill-in-the-Middle special-token ids, when the
    /// vocab carries them. `None` for chat-only models. Populated from
    /// the vocab sweep at load time using each FIM family's canonical
    /// surface form (Qwen-Coder `<|fim_prefix|>`, StarCoder
    /// `<fim_prefix>`, DeepSeek `<｜fim▁begin｜>`).
    pub fim: Option<FimTokens>,
}

/// Fill-in-the-Middle (FIM) special-token ids for code-completion
/// models. Convention is: prompt is composed as
/// `prefix_tok + prefix_text + suffix_tok + suffix_text + middle_tok`
/// and the model emits the missing middle text terminated by an EOS
/// token (or, for some models, `<|file_sep|>`).
/// The optional fields are populated when the vocab carries them and
/// stay `None` otherwise. `prefix`, `suffix`, `middle` are the only
/// fields the basic `/infill` flow needs; `pad`, `repo_name`,
/// `file_sep` are required for repo-aware FIM (Qwen-Coder PSM).
#[derive(Debug, Clone, Copy)]
pub struct FimTokens {
    pub prefix: u32,
    pub suffix: u32,
    pub middle: u32,
    pub pad: Option<u32>,
    pub repo_name: Option<u32>,
    pub file_sep: Option<u32>,
}

impl GgufTokenizer {
    /// Encode `text` → token ids. No added-special-tokens; the chat template
    /// () is responsible for inserting BOS/EOS as appropriate.
    /// # Errors
    /// Returns an `anyhow` error wrapping whatever `tokenizers::Tokenizer::encode`
    /// reports — typically a malformed input or normaliser failure.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, /*add_special_tokens=*/ false)
            .map_err(|e| anyhow!("tokenizer.encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode `ids` → UTF-8 string. Skips added-special-tokens.
    /// # Errors
    /// Returns an `anyhow` error wrapping whatever `tokenizers::Tokenizer::decode`
    /// reports — typically an out-of-vocab id or invalid UTF-8 byte sequence.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, /*skip_special_tokens=*/ false)
            .map_err(|e| anyhow!("tokenizer.decode failed: {e}"))
    }
}

/// Load a tokenizer from a GGUF file's embedded metadata.
/// # Errors
/// Returns an `anyhow` error if the required GGUF metadata keys are missing
/// (`tokenizer.ggml.model`, vocab arrays, merges), if the vocab/merges can't
/// be reconstructed into a `tokenizers::Tokenizer`, or if the tokenizer
/// model name isn't one of the V1-supported families (BPE/GPT2-style).
pub fn load_from_gguf(file: &GgufFile) -> Result<GgufTokenizer> {
    let model = file
        .metadata_str("tokenizer.ggml.model")
        .context("tokenizer.ggml.model missing from GGUF")?;
    if model != "gpt2" {
        return Err(anyhow!(
            "tokenizer model `{model}` not supported (only `gpt2` BPE family so far)"
        ));
    }
    let pre = file.metadata_str("tokenizer.ggml.pre").unwrap_or("default");

    // Vocab array: index → token string. GGUF stores it as Array(String).
    let tokens_arr = file
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .context("tokenizer.ggml.tokens missing or wrong type")?;
    let mut vocab: Vocab =
        Vocab::with_capacity_and_hasher(tokens_arr.len(), std::hash::RandomState::new());
    for (id, v) in tokens_arr.iter().enumerate() {
        let s = v
            .as_str()
            .ok_or_else(|| anyhow!("non-string vocab entry at id {id}"))?;
        vocab.insert(s.to_owned(), id as u32);
    }

    // Merges: "a b" → (a, b) pairs in priority order.
    let merges_arr = file
        .metadata
        .get("tokenizer.ggml.merges")
        .and_then(|v| v.as_array())
        .context("tokenizer.ggml.merges missing or wrong type")?;
    let mut merges = Vec::with_capacity(merges_arr.len());
    for (i, v) in merges_arr.iter().enumerate() {
        let s = v
            .as_str()
            .ok_or_else(|| anyhow!("non-string merge entry at index {i}"))?;
        let mut it = s.splitn(2, ' ');
        let a = it
            .next()
            .ok_or_else(|| anyhow!("malformed merge entry `{s}`"))?;
        let b = it
            .next()
            .ok_or_else(|| anyhow!("malformed merge entry `{s}`"))?;
        merges.push((a.to_owned(), b.to_owned()));
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(false)
        .build()
        .map_err(|e| anyhow!("BPE::build: {e}"))?;

    // Pretokenizer selection — llama.cpp's tokenizer_pre cases we support.
    let pre_tok = pre_for(pre)?;

    let mut tok: TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(bpe.into());
    tok.with_pre_tokenizer(Some(pre_tok));
    tok.with_decoder(Some(DecoderWrapper::ByteLevel(
        decoders::byte_level::ByteLevel::new(
            /*add_prefix_space=*/ false,
            /*trim_offsets=*/ true,
            /*use_regex=*/ true,
        ),
    )));
    // Normalizer: gpt2 BPE is typically NFC. Qwen3 uses no normalizer
    // ("default"); leave None unless GGUF says otherwise.
    if pre == "gpt-2" {
        tok.with_normalizer(Some(NormalizerWrapper::NFC(normalizers::NFC)));
    }

    // Special tokens. Register them so encode/decode honours them as
    // single ids rather than splitting into byte-level fragments.
    // llama.cpp's `token_type == 3` or `4` flags control tokens, but GGUF
    // doesn't always reliably surface that; we use a heuristic: any vocab
    // entry wrapped in `<|...|>` (Qwen/ChatML convention) is a special token.
    let bos_id = file.metadata_u32("tokenizer.ggml.bos_token_id");
    let eos_id = file.metadata_u32("tokenizer.ggml.eos_token_id");
    let pad_id = file.metadata_u32("tokenizer.ggml.padding_token_id");
    let mut added = Vec::new();
    let mut added_ids = std::collections::HashSet::<u32>::new();
    for id in [bos_id, eos_id, pad_id].into_iter().flatten() {
        if added_ids.insert(id) {
            if let Some(v) = tokens_arr.get(id as usize).and_then(|v| v.as_str()) {
                added.push(AddedToken::from(v.to_owned(), /*special=*/ true));
            }
        }
    }
    // Sweep vocab for `<|...|>` bracket-style control tokens.
    for (id, v) in tokens_arr.iter().enumerate() {
        if let Some(s) = v.as_str() {
            if s.starts_with("<|") && s.ends_with("|>") && s.len() <= 32
                && added_ids.insert(id as u32) {
                    added.push(AddedToken::from(s.to_owned(), /*special=*/ true));
                }
        }
    }
    if !added.is_empty() {
        tok.add_special_tokens(&added);
    }

    let wrapped: Tokenizer = tok.into();

    // Build the "stop" set: eos + common end-of-turn tokens for chat models.
    // Qwen's template uses `<|im_end|>` to close turns — the model emits it
    // but its id is not eos_id, so the server must still treat it as a stop.
    let mut stop_ids: Vec<u32> = Vec::new();
    if let Some(id) = eos_id {
        stop_ids.push(id);
    }
    // Chat end-of-turn / next-turn markers. Qwen's `<|im_end|>` closes a
    // turn even when eos_id differs; llama3's `<|eot_id|>` plays the same
    // role. also stop on `<|im_start|>`: when the assistant
    // emits the next-turn-start token mid-response (observed live on
    // Coder-Next under temp=0.7 / top_k=20 — model produces
    // `<|im_start|><|im_start|>...` repeats, or hallucinates a fake
    // user-turn-start), the response has gone off the rails and we
    // should cut it. The model should never legitimately emit the
    // turn-start marker in its own response.
    // **Sampler-G (2026-04-30)** — also stop on `<think>` and `</think>`.
    // The server renders chat templates with `enable_thinking=false`,
    // which puts a CLOSED `<think>\n\n</think>\n\n` block in the prompt
    // before the assistant content. Qwen3.6 (especially 27B) sometimes
    // hallucinates a fresh `</think>` mid-response and then re-emits its
    // answer (a confused-reasoning-mode leakage where the model treats
    // its own content as "thinking"). Live-observed on `Hello` →
    // `Hello! How can I help you today?\n</think>\n\nHello! How can I
    // help...` repeating until eventual `<|im_end|>`. Treating
    // `<think>` / `</think>` as stop markers cuts the response cleanly
    // at the first leak. If a future caller needs explicit reasoning
    // mode, opt in via `enable_thinking=true` AND filter these from
    // stop_ids at the server layer.
    let mut always_stop_ids: Vec<u32> = Vec::new();
    for needle in [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<think>",
        "</think>",
    ] {
        if let Some(id) = find_vocab_id(tokens_arr, needle) {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
            // `<think>` / `</think>` (and their open variants) bypass
            // the server's MIN_RESPONSE_TOKENS early-window mask. See
            // `always_stop_ids` doc on `GgufTokenizer`.
            if (needle == "<think>" || needle == "</think>")
                && !always_stop_ids.contains(&id)
            {
                always_stop_ids.push(id);
            }
        }
    }

    let fim = detect_fim_tokens(tokens_arr);

    Ok(GgufTokenizer {
        inner: wrapped,
        bos_id,
        eos_id,
        pad_id,
        vocab_size: tokens_arr.len() as u32,
        stop_ids,
        always_stop_ids,
        fim,
    })
}

/// Probe the vocab for FIM special tokens. Returns `None` unless all
/// three required roles (prefix/suffix/middle) are present.
/// Surface-form aliases per role cover the families we target:
/// - Qwen-Coder: `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`
/// - StarCoder / Code Llama: `<fim_prefix>` / `<fim_suffix>` / `<fim_middle>`
/// - DeepSeek-Coder: `<｜fim▁begin｜>` / `<｜fim▁hole｜>` / `<｜fim▁end｜>`
/// (the pipe is U+FF5C, the separator U+2581).
fn detect_fim_tokens(tokens_arr: &[crate::gguf::Value]) -> Option<FimTokens> {
    let first = |aliases: &[&str]| -> Option<u32> {
        aliases.iter().find_map(|a| find_vocab_id(tokens_arr, a))
    };
    let prefix = first(&[
        "<|fim_prefix|>",
        "<fim_prefix>",
        "<｜fim▁begin｜>",
        "<|fim_begin|>",
    ])?;
    let suffix = first(&[
        "<|fim_suffix|>",
        "<fim_suffix>",
        "<｜fim▁hole｜>",
        "<|fim_hole|>",
    ])?;
    let middle = first(&[
        "<|fim_middle|>",
        "<fim_middle>",
        "<｜fim▁end｜>",
        "<|fim_end|>",
    ])?;
    let pad = first(&["<|fim_pad|>", "<fim_pad>"]);
    let repo_name = first(&["<|repo_name|>", "<reponame>"]);
    let file_sep = first(&[
        "<|file_sep|>",
        "<|file_separator|>",
        "<file_sep>",
        "<filename>",
    ]);
    Some(FimTokens {
        prefix,
        suffix,
        middle,
        pad,
        repo_name,
        file_sep,
    })
}

fn find_vocab_id(tokens_arr: &[crate::gguf::Value], needle: &str) -> Option<u32> {
    tokens_arr.iter().position(|v| {
        v.as_str().is_some_and(|s| s == needle)
    }).map(|i| i as u32)
}

#[cfg(test)]
mod fim_tests {
    use super::*;
    use crate::gguf::Value;

    fn vocab(words: &[&str]) -> Vec<Value> {
        words.iter().map(|w| Value::String((*w).to_string())).collect()
    }

    #[test]
    fn detects_qwen_coder_set() {
        let v = vocab(&[
            "a", "<|fim_prefix|>", "b", "<|fim_suffix|>", "<|fim_middle|>",
            "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
        ]);
        let fim = detect_fim_tokens(&v).expect("fim present");
        assert_eq!(fim.prefix, 1);
        assert_eq!(fim.suffix, 3);
        assert_eq!(fim.middle, 4);
        assert_eq!(fim.pad, Some(5));
        assert_eq!(fim.repo_name, Some(6));
        assert_eq!(fim.file_sep, Some(7));
    }

    #[test]
    fn detects_starcoder_set() {
        let v = vocab(&["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "x"]);
        let fim = detect_fim_tokens(&v).expect("fim present");
        assert_eq!((fim.prefix, fim.suffix, fim.middle), (0, 1, 2));
        assert_eq!(fim.pad, None);
    }

    #[test]
    fn detects_deepseek_set() {
        let v = vocab(&["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"]);
        let fim = detect_fim_tokens(&v).expect("fim present");
        assert_eq!((fim.prefix, fim.suffix, fim.middle), (0, 1, 2));
    }

    #[test]
    fn missing_required_role_yields_none() {
        let v = vocab(&["<|fim_prefix|>", "<|fim_middle|>"]); // no suffix
        assert!(detect_fim_tokens(&v).is_none());
    }

    #[test]
    fn chat_only_vocab_yields_none() {
        let v = vocab(&["<|im_start|>", "<|im_end|>", "<|endoftext|>"]);
        assert!(detect_fim_tokens(&v).is_none());
    }

    #[test]
    fn first_alias_wins_within_role() {
        // Both Qwen-Coder and StarCoder forms present — Qwen wins (listed
        // first in alias array). Stable selection prevents drift if a
        // GGUF carries duplicate forms.
        let v = vocab(&[
            "<fim_prefix>",
            "<fim_suffix>",
            "<fim_middle>",
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
        ]);
        let fim = detect_fim_tokens(&v).expect("fim present");
        assert_eq!(fim.prefix, 3);
        assert_eq!(fim.suffix, 4);
        assert_eq!(fim.middle, 5);
    }
}

/// Map GGUF `tokenizer.ggml.pre` → concrete pretokenizer. llama.cpp supports
/// many presets; we cover the ones in-use for our target models.
///
/// Qwen2/Qwen3 use a tokenizer-pre regex that, unlike GPT-2's, keeps optional
/// leading punctuation attached to the following letters — `"-time"` stays as
/// one chunk that BPE can merge into the single vocab token, instead of
/// splitting into `["-", "time"]`. Stock `ByteLevel(use_regex=true)` would
/// apply GPT-2's regex and emit two tokens. Fix: split with the Qwen regex
/// first, then byte-level map without re-splitting.
fn pre_for(pre: &str) -> Result<PreTokenizerWrapper> {
    match pre {
        "default" | "gpt-2" | "llama-bpe" | "llama3" => {
            Ok(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
                /*add_prefix_space=*/ false,
                /*trim_offsets=*/ true,
                /*use_regex=*/ true,
            )))
        }
        "qwen2" | "qwen35" => {
            let regex = if pre == "qwen35" {
                // llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35.
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            } else {
                // llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2.
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            };
            let split = Split::new(
                SplitPattern::Regex(regex.to_owned()),
                SplitDelimiterBehavior::Isolated,
                /*invert=*/ false,
            )
            .map_err(|e| anyhow!("qwen split pretokenizer: {e}"))?;
            let byte_level = ByteLevel::new(
                /*add_prefix_space=*/ false,
                /*trim_offsets=*/ true,
                /*use_regex=*/ false,
            );
            Ok(PreTokenizerWrapper::Sequence(PreTokSequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(byte_level),
            ])))
        }
        other => Err(anyhow!(
            "tokenizer.ggml.pre=`{other}` not supported (add a match arm in pre_for)"
        )),
    }
}
