//  $env:MODEL_DEBUG="0";$env:ROPE_SCALE_APPLY="1";$env:FORCE_KV_HEADS="1";$env:RUST_DECODE_TEST="0";$env:RUST_LLAMA_CHECK="0";$env:PROMPT_TPL="turn"; cargo run --release
use std::collections::HashMap;
use crate::gguf_loader::GgufValue;

#[derive(Clone, Copy, Debug)]
pub enum RopeMode {
    AdjacentPairs, // (0,1), (2,3), ...
}

#[derive(Clone, Copy, Debug)]
pub enum ChatTpl {
    ChatMl,         // <|system|> ... <|user|> ... <|assistant|>
    ImTags,         // <|im_start|> ... <|im_end|>
    Inst,           // [INST] <<SYS>> ... [/INST]
    Simple,         // System: ... \n User: ... \n Assistant:
    Llama3,         // <|start_header_id|>, <|end_header_id|>, <|eot_id|>
    Gemma,          // <start_of_turn> ... <end_of_turn>
    TurnPipes,      // <|start_of_turn|> ... <|end_of_turn|>
    Alpaca,         // ### Instruction/Input/Response
    Vicuna,         // USER:/ASSISTANT:
}

fn kv_tokens<'a>(kv: &'a HashMap<String, GgufValue>) -> Option<&'a Vec<String>> {
    match kv.get("tokenizer.ggml.tokens") {
        Some(GgufValue::ArrStr(v)) => Some(v),
        _ => None,
    }
}

// tolerant: exakt, " pat", "▁pat"
fn token_has_any_form(tokens: &[String], pat: &str) -> bool {
    let with_space = format!(" {}", pat);
    let with_underscore = format!("\u{2581}{}", pat);
    tokens.iter().any(|t| t == pat || t == &with_space || t == &with_underscore)
}

fn has_tokens_any_form(kv: &HashMap<String, GgufValue>, need: &[&str]) -> bool {
    let Some(tokens) = kv_tokens(kv) else { return false; };
    need.iter().all(|p| token_has_any_form(tokens, p))
}

fn tokens_contain_substring(kv: &HashMap<String, GgufValue>, needle: &str) -> bool {
    let Some(tokens) = kv_tokens(kv) else { return false; };
    tokens.iter().any(|t| t.contains(needle))
}

pub struct DetectedSettings {
    pub rope_base: f32,
    pub rope_mode: RopeMode,
    pub tpl: ChatTpl,
}

pub fn detect_settings(kv: &HashMap<String, GgufValue>) -> DetectedSettings {
    // 1) Architektur lesen
    let arch = match kv.get("general.architecture") {
        Some(GgufValue::Str(s)) => s.to_lowercase(),
        _ => "llama".to_string(),
    };

    // 2) Rope-Basis (theta)
    let rope_from_kv = [
        "rope.theta",
        "rope.freq_base",
        &format!("{}.rope.freq_base", arch),
    ]
    .iter()
    .find_map(|k| match kv.get(*k) {
        Some(GgufValue::F32(v)) => Some(*v),
        Some(GgufValue::F64(v)) => Some(*v as f32),
        Some(GgufValue::U32(v)) => Some(*v as f32),
        Some(GgufValue::U64(v)) => Some(*v as f32),
        Some(GgufValue::I32(v)) => Some(*v as f32),
        Some(GgufValue::I64(v)) => Some(*v as f32),
        _ => None,
    });

    let rope_base = rope_from_kv.unwrap_or_else(|| {
        if arch.starts_with("qwen") {
            1_000_000.0
        } else {
            10_000.0
        }
    });

    let rope_mode = RopeMode::AdjacentPairs;

    // 3) Template-Erkennung (tolerant)
    let tpl = if has_tokens_any_form(kv, &["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]) {
        ChatTpl::Llama3
    } else if has_tokens_any_form(kv, &["<start_of_turn>", "<end_of_turn>"]) {
        ChatTpl::Gemma
    } else if has_tokens_any_form(kv, &["<|start_of_turn|>", "<|end_of_turn|>"]) {
        ChatTpl::TurnPipes
    } else if has_tokens_any_form(kv, &["<|im_start|>", "<|im_end|>"]) {
        ChatTpl::ImTags
    } else if has_tokens_any_form(kv, &["<|system|>", "<|user|>", "<|assistant|>"]) {
        ChatTpl::ChatMl
    } else if has_tokens_any_form(kv, &["[INST]", "[/INST]"]) {
        ChatTpl::Inst
    } else if tokens_contain_substring(kv, "### Instruction")
           && tokens_contain_substring(kv, "### Response") {
        ChatTpl::Alpaca
    } else if tokens_contain_substring(kv, "USER:")
           && tokens_contain_substring(kv, "ASSISTANT:") {
        ChatTpl::Vicuna
    } else {
        ChatTpl::Simple
    };

    DetectedSettings { rope_base, rope_mode, tpl }
}


/// Liefert ein passendes Chat-Template für das gegebene Modell.
/// Reihenfolge:
/// 1) Versuch über Token-Erkennung (detect_settings)
/// 2) Fallback per Heuristik (arch/name/description)
pub fn guess_template_from_model(kv: &HashMap<String, GgufValue>) -> ChatTpl {
    // 1) Token-basierte Erkennung zuerst (robusteste Quelle)
    let DetectedSettings { tpl, .. } = detect_settings(kv);
    if !matches!(tpl, ChatTpl::Simple) {
        return tpl;
    }

    // 2) Heuristik nach Modell-Metadaten
    let get_lower = |key: &str| -> String {
        match kv.get(key) {
            Some(GgufValue::Str(s)) => s.to_lowercase(),
            _ => String::new(),
        }
    };

    let arch = get_lower("general.architecture");
    let name = get_lower("general.name");
    let desc = get_lower("general.description");
    let info = format!("{} {} {}", arch, name, desc);

    // Bekannte Familien:
    // Qwen: <|im_start|> ... <|im_end|>
    if info.contains("qwen") {
        return ChatTpl::ImTags;
    }

    // Llama 3: oft Header-Style; wenn ChatML-Tokens vorhanden -> ChatMl, sonst Inst
    if info.contains("llama-3") || info.contains("llama 3") {
        if has_tokens_strict(kv, &["<|system|>", "<|user|>", "<|assistant|>"]) {
            return ChatTpl::ChatMl;
        } else {
            return ChatTpl::Inst;
        }
    }

    // Llama 1/2, Mistral/Mixtral, TinyLlama, Phi: meist Inst-Style ([INST] ... [/INST])
    if info.contains("llama")
        || info.contains("mistral")
        || info.contains("mixtral")
        || info.contains("tinyllama")
        || info.contains("phi")
    {
        return ChatTpl::Inst;
    }

    // Einige Modelle nutzen ChatML (z. B. Zephyr, OpenChat, Yi-Chat):
    if info.contains("zephyr")
        || info.contains("openchat")
        || info.contains("yi")
        || info.contains("chatml")
    {
        return ChatTpl::ChatMl;
    }

    // Letzter Ausweg
    ChatTpl::Simple
}

/// Strenge Tokenprüfung (exakt wie im Vokabular gespeichert).
/// Hinweis: Falls du toleranter prüfen willst (mit führendem Space oder U+2581),
/// kannst du diese Funktion später erweitern.
fn has_tokens_strict(kv: &HashMap<String, GgufValue>, need: &[&str]) -> bool {
    let Some(GgufValue::ArrStr(tokens)) = kv.get("tokenizer.ggml.tokens") else {
        return false;
    };
    need.iter().all(|pat| tokens.iter().any(|t| t == *pat))
}
