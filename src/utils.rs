use std::collections::HashMap;
use crate::gguf_loader::GgufValue;

#[derive(Clone, Copy, Debug)]
pub enum RopeMode {
    AdjacentPairs, // (0,1), (2,3), ...
}

#[derive(Clone, Copy, Debug)]
pub enum ChatTpl {
    ChatMl, // <|system|> ... <|user|> ... <|assistant|>
    ImTags, // <|im_start|> ... <|im_end|>
    Inst,   // [INST] <<SYS>> ... [/INST]
    Simple, // System: ... \n User: ... \n Assistant:
}

fn has_tokens(kv: &HashMap<String, GgufValue>, need: &[&str]) -> bool {
    let Some(GgufValue::ArrStr(tokens)) = kv.get("tokenizer.ggml.tokens") else {
        return false;
    };
    need.iter().all(|t| tokens.iter().any(|x| x == *t))
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

    // 2) Rope-Basis (theta) aus GGUF oder Fallback
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

    // 3) RoPE-Dreh-Modus: standardisiert auf adjacent pairs
    let rope_mode = RopeMode::AdjacentPairs;

    // 4) Prompt-Template über Tokens erkennen
    let tpl = if has_tokens(kv, &["<|im_start|>", "<|im_end|>"]) {
        ChatTpl::ImTags
    } else if has_tokens(kv, &["<|system|>", "<|user|>", "<|assistant|>"]) {
        ChatTpl::ChatMl
    } else if has_tokens(kv, &["[INST]", "[/INST]"]) {
        ChatTpl::Inst
    } else {
        ChatTpl::Simple
    };

    DetectedSettings { rope_base, rope_mode, tpl }
}
