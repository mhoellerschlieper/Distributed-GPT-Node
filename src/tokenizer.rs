// tokenizer.rs
// GGUF-Tokenizer (Unigram für "llama", Byte-Level-BPE für "gpt2")
// Autor: Marcus Schlieper, ExpChat.ai
// Datum: 2025-12-08
// Sicherheit: kein unsafe, Result-basierte Fehlerbehandlung

use std::collections::{HashMap, HashSet};

use tokenizers::{
    decoders::{byte_fallback::ByteFallback, byte_level::ByteLevel as DecByteLevel, fuse::Fuse, strip::Strip, DecoderWrapper},
    models::{bpe::BpeBuilder, unigram::Unigram, ModelWrapper},
    normalizers::{Prepend, Replace as NormReplace, NormalizerWrapper},
    pre_tokenizers::{byte_level::ByteLevel as PreByteLevel, PreTokenizerWrapper},
    processors, AddedToken, Tokenizer,
};

use crate::gguf_loader::GgufValue;

use ahash::AHashMap;

// Einheitliche API
pub struct GgufTokenizer {
    tok: Tokenizer,
    bos_id: Option<usize>,
    eos_id: Option<usize>,
    unk_id: Option<usize>,
    add_bos: bool,
    add_eos: bool,
    special_token_ids: HashSet<u32>,
}

impl GgufTokenizer {
    // Encode: fügt optional BOS/EOS hinzu (gemäß GGUF-Flags und add_special)
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<usize>, String> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| format!("encode failed: {}", e))?;
        let mut ids: Vec<usize> = enc.get_ids().iter().map(|&x| x as usize).collect();

        if add_special {
            if self.add_bos {
                if let Some(bos) = self.bos_id {
                    if ids.first().copied() != Some(bos) {
                        ids.insert(0, bos);
                    }
                }
            }
            if self.add_eos {
                if let Some(eos) = self.eos_id {
                    if ids.last().copied() != Some(eos) {
                        ids.push(eos);
                    }
                }
            }
        }
        Ok(ids)
    }

    // Decode: kann Special-Tokens überspringen
    pub fn decode(&self, ids: &[usize], skip_special: bool) -> Result<String, String> {
        let out = if skip_special {
            let filtered: Vec<u32> = ids
                .iter()
                .copied()
                .filter(|id| !self.special_token_ids.contains(&(*id as u32)))
                .map(|id| id as u32)
                .collect();
            self.tok
                .decode(filtered.as_slice(), true)
                .map_err(|e| format!("decode failed: {}", e))?
        } else {
            let v: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            self.tok
                .decode(v.as_slice(), false)
                .map_err(|e| format!("decode failed: {}", e))?
        };

        // Für Unigram üblich: U+2581 -> Space rückwandeln
        let out = out.replace('\u{2581}', " ");
        Ok(out)
    }

    pub fn bos_id(&self) -> Option<usize> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<usize> {
        self.eos_id
    }
}

// Integrations-Kleber:
// Liest aus GGUF-KV das Modell ("llama" -> Unigram, "gpt2" -> BPE) und baut den passenden Tokenizer.
pub fn gguf_tokenizer_from_kv(
    kv: &HashMap<String, GgufValue>,
) -> Result<GgufTokenizer, String> {
    // Modelltyp lesen
    let model = get_kv_str(kv, &["tokenizer.ggml.model", "tokenizer.model"])
        .ok_or_else(|| "tokenizer.ggml.model fehlt".to_string())?;

    // BOS/EOS/UNK IDs und Flags
    let bos_id = get_kv_usize(kv, &["tokenizer.ggml.bos_token_id", "tokenizer.ggml.bos_id"]);
    let eos_id = get_kv_usize(kv, &["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eos_id"]);
    let unk_id =
        get_kv_usize(kv, &["tokenizer.ggml.unknown_token_id", "tokenizer.ggml.unk_id"]);

    let add_bos = matches!(kv.get("tokenizer.ggml.add_bos_token"), Some(GgufValue::Bool(true)));
    let add_eos = matches!(kv.get("tokenizer.ggml.add_eos_token"), Some(GgufValue::Bool(true)));

    // Tokenliste
    let tokens = match kv.get("tokenizer.ggml.tokens") {
        Some(GgufValue::ArrStr(v)) => v.clone(),
        _ => return Err("tokenizer.ggml.tokens fehlt".to_string()),
    };

    // token_type (0/2/... = special, 1 = normal) optional
    let special_token_ids: HashSet<u32> = match kv.get("tokenizer.ggml.token_type") {
        Some(GgufValue::ArrI32(v)) if v.len() == tokens.len() => v
            .iter()
            .enumerate()
            .filter_map(|(i, t)| if *t != 1 { Some(i as u32) } else { None })
            .collect(),
        _ => HashSet::new(),
    };

    let tokenizer = match model.as_str() {
        "llama" | "replit" => {
            // Unigram (SentencePiece)
            build_unigram_tokenizer(kv, &tokens, unk_id)?
        }
        "gpt2" => {
            // Byte-Level-BPE
            build_bpe_tokenizer(kv, &tokens, unk_id)?
        }
        other => {
            return Err(format!(
                "Unbekannter GGUF-Tokenizer-Modelltyp: {} (erwartet 'llama' oder 'gpt2')",
                other
            ));
        }
    };

    Ok(GgufTokenizer {
        tok: tokenizer,
        bos_id,
        eos_id,
        unk_id,
        add_bos,
        add_eos,
        special_token_ids,
    })
}

// ----------------------- Unigram Aufbau -----------------------
fn build_unigram_tokenizer(
    kv: &HashMap<String, GgufValue>,
    tokens: &[String],
    unk_id: Option<usize>,
) -> Result<Tokenizer, String> {
    // Scores (f32) notwendig für Unigram
    let scores = match kv.get("tokenizer.ggml.scores") {
        Some(GgufValue::ArrF32(v)) if v.len() == tokens.len() => v.clone(),
        _ => {
            return Err("Unigram: tokenizer.ggml.scores fehlt oder Länge passt nicht".to_string())
        }
    };
    let vocab: Vec<(String, f64)> = tokens
        .iter()
        .cloned()
        .zip(scores.into_iter().map(|f| f as f64))
        .collect();

    let unk = unk_id.unwrap_or(0);
    let model =
        Unigram::from(vocab, Some(unk), true).map_err(|e| format!("Unigram build: {}", e))?;
    let mut tokenizer = Tokenizer::new(ModelWrapper::Unigram(model));

    // Normalizer: prepend '▁' und ersetze Space -> '▁'
    let normalizer = {
        let seq = tokenizers::normalizers::Sequence::new(vec![
            NormalizerWrapper::from(Prepend::new("▁".to_string())),
            NormalizerWrapper::try_from(
                NormReplace::new(" ", "▁").map_err(|e| format!("Normalizer Replace: {}", e))?,
            )
            .map_err(|e| format!("NormalizerWrapper: {}", e))?,
        ]);
        NormalizerWrapper::from(seq)
    };
    tokenizer.with_normalizer(Some(normalizer));

    // Decoder: ByteFallback + Fuse + Strip(' ', 1, 0)
    // (▁->Space ersetzen wir im decode() per String-Replacement)
    let decoder = {
        let mut seq = Vec::<DecoderWrapper>::new();
        seq.push(DecoderWrapper::from(ByteFallback::default()));
        seq.push(DecoderWrapper::from(Fuse::default()));
        seq.push(DecoderWrapper::from(Strip::new(' ', 1, 0)));
        let seq = tokenizers::decoders::sequence::Sequence::new(seq);
        DecoderWrapper::from(seq)
    };
    tokenizer.with_decoder(Some(decoder));

    // Special Tokens registrieren
    for key in &[
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.bos_id",
        "tokenizer.ggml.eos_id",
        "tokenizer.ggml.unk_id",
    ] {
        if let Some(id) = get_kv_usize(kv, &[*key]) {
            let tk = tokens
                .get(id)
                .cloned()
                .ok_or_else(|| format!("Special-Token-ID {} außerhalb des Bereichs", id))?;
            tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
        }
    }

    Ok(tokenizer)
}

// ----------------------- BPE Aufbau -----------------------
fn build_bpe_tokenizer(
    kv: &HashMap<String, GgufValue>,
    tokens: &[String],
    unk_id: Option<usize>,
) -> Result<Tokenizer, String> {
    // Merges laden (wie bei dir)
    let merges_raw = match kv.get("tokenizer.ggml.merges") {
        Some(GgufValue::ArrStr(v)) => v.clone(),
        _ => return Err("BPE: tokenizer.ggml.merges fehlt".to_string()),
    };

    // Vokabular: direkt als AHashMap aufbauen
    let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }

    // Merges als Tupel (wie bei dir)
    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_raw.len());
    for m in merges_raw {
        let mut it = m.splitn(2, ' ');
        let a = it.next().ok_or_else(|| format!("BPE merge fehlerhaft: '{}'", m))?;
        let b = it.next().ok_or_else(|| format!("BPE merge fehlerhaft: '{}'", m))?;
        merges.push((a.to_string(), b.to_string()));
    }

    // BPE aufbauen
    let mut b = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = unk_id {
        let unk_tok = tokens.get(unk).cloned().ok_or("UNK-ID ausserhalb des Bereichs")?;
        b = b.unk_token(unk_tok);
    }
    let bpe = b.build().map_err(|e| format!("BPE build: {}", e))?;
    let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(bpe));

    // NEU: Byte-Level-Flags aus GGUF + ENV
    let use_byte_level = match kv.get("tokenizer.ggml.byte_level") {
        Some(GgufValue::Bool(b)) => *b,
        _ => true,
    };
    let add_prefix_space = std::env::var("BPE_ADD_PREFIX_SPACE")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|v| v != 0)
        .or_else(|| match kv.get("tokenizer.ggml.add_prefix_space") {
            Some(GgufValue::Bool(b)) => Some(*b),
            _ => None,
        })
        .unwrap_or(false); // viele Referenzen: false

    let trim_offsets = std::env::var("BPE_TRIM_OFFSETS")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .map(|v| v != 0)
        .or(Some(true))
        .unwrap();

    if use_byte_level {
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(
            PreByteLevel::new(add_prefix_space, trim_offsets, false),
        )));
        tokenizer.with_decoder(Some(DecoderWrapper::from(DecByteLevel::new(
            add_prefix_space, trim_offsets, false,
        ))));
        tokenizer.with_post_processor(Some(processors::byte_level::ByteLevel::new(
            add_prefix_space, trim_offsets, false,
        )));
    }


    // Special Tokens registrieren
    for key in &[
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.bos_id",
        "tokenizer.ggml.eos_id",
        "tokenizer.ggml.unk_id",
    ] {
        if let Some(id) = get_kv_usize(kv, &[*key]) {
            let tk = tokens
                .get(id)
                .cloned()
                .ok_or_else(|| format!("Special-Token-ID {} ausserhalb des Bereichs", id))?;
            tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
        }
    }

    Ok(tokenizer)
}

// ----------------------- KV-Helper -----------------------
fn get_kv_str(kv: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(GgufValue::Str(s)) = kv.get(*k) {
            return Some(s.clone());
        }
    }
    None
}
fn get_kv_usize(kv: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<usize> {
    for k in keys {
        match kv.get(*k) {
            Some(GgufValue::U32(v)) => return Some(*v as usize),
            Some(GgufValue::U64(v)) => return Some(*v as usize),
            Some(GgufValue::I32(v)) => return Some((*v).max(0) as usize),
            Some(GgufValue::I64(v)) => return Some((*v).max(0) as usize),
            _ => {}
        }
    }
    None
}
