// main.rs
// CPU-only GGUF v3 Inferenz: automatisches Erkennen von RoPE-Basis, RoPE-Drehung und Prompt-Template
// Autor: Marcus Schlieper, ExpChat.ai
// Kontakt: mschlieper@expchat.ai | Tel: 49 2338 8748862 | Mobil: 49 15115751864
// Firma: ExpChat.ai – Der KI Chat Client für den Mittelstand aus Breckerfeld im Sauerland.
//        RPA, KI Agents, KI Internet Research, KI Wissensmanagement.
//        Adresse: Epscheider Str21, 58339 Breckerfeld
//
// Hinweise:
// - Modelldaten und Mapping: model.rs
// - Layer/Mathe/Sampling: layer.rs, math.rs
// - GGUF-Lader: gguf_loader.rs
// - Tokenizer (Unigram + BPE aus GGUF): tokenizer.rs
// - Utils: utils.rs (enthält detect_settings, RopeMode, ChatTpl, …)
//
//  
//  $env:MODEL_DEBUG="0";$env:ROPE_SCALE_APPLY="1";$env:FORCE_KV_HEADS="1";$env:RUST_DECODE_TEST="0";$env:RUST_LLAMA_CHECK="0";$env:PROMPT_TPL="turn"; cargo run --release

mod gguf_loader;
mod layer;
mod math;
mod model;
mod tokenizer;
mod utils;

use gguf_loader::{GgufModel, GgufValue, load_gguf};
use layer::TransformerModel;
use math::{SimpleRng, sample_top_k_top_p_temperature};
use model::{build_config, init_debug_from_env, map_all_weights, mean_abs};
use tokenizer::{GgufTokenizer, gguf_tokenizer_from_kv};
use utils::{ChatTpl, DetectedSettings, detect_settings};

use std::collections::HashMap;
use std::io::{self, Write};

#[cfg(test)]
mod tests;

// =============== Kleine ENV-Helper ===============
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
        .unwrap_or(false)
}

// =============== Prompt-Helfer ===============
// Toleranter Token-Check (wie in utils.rs)
fn vocab_has_all_anyform(kv: &std::collections::HashMap<String, GgufValue>, need: &[&str]) -> bool {
    let Some(GgufValue::ArrStr(tokens)) = kv.get("tokenizer.ggml.tokens") else {
        return false;
    };
    let has_any = |pat: &str| {
        let with_space = format!(" {}", pat);
        let with_underscore = format!("\u{2581}{}", pat);
        tokens
            .iter()
            .any(|t| t == pat || t == &with_space || t == &with_underscore)
    };
    need.iter().all(|p| has_any(p))
}

// Promptbauer für alle Templates
fn build_prompt(tpl: ChatTpl, s_system: &str, s_user: &str) -> String {
    match tpl {
        ChatTpl::ChatMl => format!(
            "<|system|>\n{}\n<|user|>\n{}\n<|assistant|>\n",
            s_system, s_user
        ),
        ChatTpl::ImTags => format!(
            "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            s_system, s_user
        ),
        ChatTpl::Inst => format!(
            "[INST] <<SYS>>\n{}\n<</SYS>>\n{}\n[/INST]\n",
            s_system, s_user
        ),
        ChatTpl::Simple => format!("System: {}\nUser: {}\nAssistant:", s_system, s_user),

        // Neu:
        ChatTpl::Llama3 => format!(
            "<|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>\n\
             <|start_header_id|>user<|end_header_id|>\n{}<|eot_id|>\n\
             <|start_header_id|>assistant<|end_header_id|>\n",
            s_system, s_user
        ),
        ChatTpl::Gemma => format!(
            "<start_of_turn>system\n{}\n<end_of_turn>\n\
             <start_of_turn>user\n{}\n<end_of_turn>\n\
             <start_of_turn>model\n",
            s_system, s_user
        ),
        ChatTpl::TurnPipes => format!(
            "<|start_of_turn|>system\n{}\n<|end_of_turn|>\n\
             <|start_of_turn|>user\n{}\n<|end_of_turn|>\n\
             <|start_of_turn|>assistant\n",
            s_system, s_user
        ),
        ChatTpl::Alpaca => format!(
            "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n",
            s_system, s_user
        ),
        ChatTpl::Vicuna => format!("SYSTEM: {}\nUSER: {}\nASSISTANT:", s_system, s_user),
    }
}

// Nur für kompakte Debug-Ausgabe
fn visible_token(s_src: &str) -> String {
    let mut s_out = String::new();
    for ch in s_src.chars() {
        match ch {
            '\n' => s_out.push_str("\\n"),
            '\r' => s_out.push_str("\\r"),
            '\t' => s_out.push_str("\\t"),
            c if c.is_control() => s_out.push('?'),
            c => s_out.push(c),
        }
    }
    s_out
}

// Prüfe, ob alle benötigten Template-Tokens im Vokabular vorhanden sind
fn vocab_has_all(kv: &HashMap<String, GgufValue>, need: &[&str]) -> bool {
    let Some(GgufValue::ArrStr(tokens)) = kv.get("tokenizer.ggml.tokens") else {
        return false;
    };
    let has_any_form = |pat: &str| {
        let with_space = format!(" {}", pat);
        let with_underscore = format!("\u{2581}{}", pat);
        tokens
            .iter()
            .any(|t| t == pat || t == &with_space || t == &with_underscore)
    };
    need.iter().all(|p| has_any_form(p))
}

// =============== Optionale Tools (schnelle Checks) ===============
fn run_encode_decode_test(s_model_path: &str) -> Result<(), String> {
    println!("\n== Encode/Decode Test (Rust) ==");
    let gguf = load_gguf(s_model_path)?;
    let tok = gguf_tokenizer_from_kv(&gguf.kv)?;
    let s_text = "Hello world!";

    let ids_plain = tok.encode(s_text, false)?;
    println!("Input: {}", s_text);
    println!("Tokens (no special): {:?}", ids_plain);
    let s_back_plain = tok.decode(&ids_plain, true)?;
    println!("Decoded (skip special): {}", s_back_plain);

    let ids_special = tok.encode(s_text, true)?;
    println!("Tokens (with special): {:?}", ids_special);
    let s_back_special = tok.decode(&ids_special, true)?;
    println!("Decoded (skip special): {}", s_back_special);

    print!("Per-token decode: ");
    for &id in &ids_plain {
        let piece = tok
            .decode(&[id], true)
            .unwrap_or_default()
            .replace('\n', "\\n");
        print!("[{}:{}] ", id, piece);
    }
    println!();
    Ok(())
}

fn run_model_info_check() -> Result<(), String> {
    //let default_model = r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    //let s_model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| default_model.to_string());

    let s_model_path = r"c:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    println!("Lade Modell: {}", s_model_path);
    let gguf = load_gguf(&s_model_path)?;
    let _tok = gguf_tokenizer_from_kv(&gguf.kv)?;

    let arch = gguf
        .get_kv_str("general.architecture")
        .unwrap_or_else(|| "unknown".to_string());
    let vocab_size = gguf
        .get_kv_u32("tokenizer.vocab_size")
        .map(|v| v as usize)
        .or_else(|| {
            if let Some(GgufValue::ArrStr(v)) = gguf.kv.get("tokenizer.ggml.tokens") {
                Some(v.len())
            } else {
                None
            }
        })
        .unwrap_or(0);

    println!("{{");
    println!(
        "  \"model_path\": \"{}\",",
        s_model_path.replace('\\', "\\\\")
    );
    println!("  \"architecture\": \"{}\",", arch);
    println!("  \"vocab_size\": {}", vocab_size);
    println!("}}");

    println!("\n== Tokens (0..29) ==");
    for tid in 0..900usize {
        let piece = _tok.decode(&[tid], false).unwrap_or_default();
        let shown = escape_token_piece(&piece);
        println!("{:>6} | {}", tid, shown);
    }
    Ok(())
}

fn escape_token_piece(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c.is_control() => {
                let code = c as u32;
                if code <= 0xFF {
                    out.push_str(&format!("\\x{:02x}", code));
                } else {
                    out.push_str(&format!("\\u{{{:x}}}", code));
                }
            }
            c => out.push(c),
        }
    }
    format!("'{}'", out)
}

// =============== Hauptprogramm ===============
fn main() -> Result<(), String> {
    // Debug-Schalter aus ENV übernehmen (MODEL_DEBUG=1)
    init_debug_from_env();

    // Optionale Einzweck-Checks
    if env_bool("RUST_DECODE_TEST") {
        let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
            r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf".to_string()
        });
        return run_encode_decode_test(&path);
    }
    if env_bool("RUST_LLAMA_CHECK") {
        return run_model_info_check();
    }

    // Modellpfad (ENV überschreibbar)
    //let s_model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
    //    r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf".to_string()
    //});

    //let s_model_path = r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf".to_string(); // <= wird geladen
    //let s_model_path =r"c:\Entwicklung\rust\GPT-GGUF\model\Cinder-Phi-2-V1.F16(1).gguf".to_string(); // <= wird geladen
    let s_model_path =r"c:\Entwicklung\rust\GPT-GGUF\model\vibethinker-1.5b-q8_0.gguf".to_string(); // <= wird geladen
    ////let s_model_path =r"c:\Entwicklung\rust\GPT-GGUF\model\gemma-2-9b-it-Q4_K_M-fp16.gguf".to_string(); // <= wird NICHT geladen
    ////let s_model_path =r"c:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf".to_string(); // <= wird NICHT geladen

    println!("Lade GGUF: {}", s_model_path);
    let gguf: GgufModel = load_gguf(&s_model_path)?;

    // 1) Einstellungen automatisch erkennen (Architektur, RoPE-Basis, Template)
    let DetectedSettings {
        rope_base,
        rope_mode: _rope_mode, // aktuell nicht in main verwendet; Layer nutzt eigene Implementierung
        tpl: detected_tpl,
    } = detect_settings(&gguf.kv);

    // 2) Konfiguration aus GGUF lesen und (falls nötig) mit erkannter RoPE-Basis überschreiben
    let mut cfg = build_config(&gguf);
    // Nur überschreiben, wenn ENV nicht explizit ROPE_THETA vorgibt:
    if std::env::var("ROPE_THETA").is_err() {
        cfg.rope_base = rope_base;
    }
    println!(
        "Model config: layers={} heads={} kv_heads={} hidden={} vocab={} ctx={} rope_dim={} rope_base={}",
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.max_seq_len,
        cfg.rope_dim,
        cfg.rope_base
    );

    // 3) Tokenizer aufbauen
    let tok: GgufTokenizer =
        gguf_tokenizer_from_kv(&gguf.kv).map_err(|e| format!("Tokenizer-Fehler: {}", e))?;
    //build_unigram_tokenizer(&gguf.kv).map_err(|e| format!("Tokenizer-Fehler: {}", e))?;
    //build_bpe_tokenizer(&gguf.kv).map_err(|e| format!("Tokenizer-Fehler: {}", e))?;

    // 4) Modell instanziieren und Gewichte mappen
    let mut model = TransformerModel::new_empty(cfg.clone());
    map_all_weights(&gguf, &mut model)?;
    println!("Gewichte gemappt.");
    println!("dbg |tok_emb|mean| = {:.6}", mean_abs(&model.tok_emb));
    println!("dbg |lm_head|mean| = {:.6}", mean_abs(&model.lm_head));
    if !model.blocks.is_empty() {
        println!(
            "dbg |blk0.w_q|mean| = {:.6}",
            mean_abs(&model.blocks[0].attn.w_q.w)
        );
        println!(
            "dbg |blk0.w1|mean|  = {:.6}",
            mean_abs(&model.blocks[0].ffn.w1.w)
        );
    }

    // 5) Prompt-Template wählen: automatische Erkennung, aber ENV-PROMPT_TPL erlaubt Override
    let tpl_from_env = std::env::var("PROMPT_TPL")
        .ok()
        .map(|s| s.to_lowercase())
        .and_then(|s| match s.as_str() {
            "chatml" => Some(ChatTpl::ChatMl),
            "im" | "imtags" => Some(ChatTpl::ImTags),
            "inst" => Some(ChatTpl::Inst),
            "simple" => Some(ChatTpl::Simple),
            "llama3" => Some(ChatTpl::Llama3),
            "gemma" => Some(ChatTpl::Gemma),
            "turn" | "turnpipes" => Some(ChatTpl::TurnPipes),
            "alpaca" => Some(ChatTpl::Alpaca),
            "vicuna" => Some(ChatTpl::Vicuna),
            _ => None,
        });

    // let mut tpl = tpl_from_env.unwrap_or(detected_tpl);

    let mut tpl = detected_tpl;
    if matches!(tpl, ChatTpl::Simple) {
        tpl = utils::guess_template_from_model(&gguf.kv);
    }

    // Wenn Template-Spezialtokens fehlen, auf Simple zurückfallen (nur die, die echte Spezialtokens erwarten)
    let need = match tpl {
        ChatTpl::ImTags => vec!["<|im_start|>", "<|im_end|>"],
        ChatTpl::ChatMl => vec!["<|system|>", "<|user|>", "<|assistant|>"],
        ChatTpl::Inst => vec!["[INST]", "[/INST]"],
        ChatTpl::Llama3 => vec!["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        ChatTpl::Gemma => vec!["<start_of_turn>", "<end_of_turn>"],
        ChatTpl::TurnPipes => vec!["<|start_of_turn|>", "<|end_of_turn|>"],
        // Alpaca/Vicuna haben keine speziellen Vokabel-Tokens
        ChatTpl::Alpaca | ChatTpl::Vicuna | ChatTpl::Simple => vec![],
    };

    if !need.is_empty() && !vocab_has_all_anyform(&gguf.kv, &need) {
        println!("Warnung: Template-Tokens fehlen im Vokabular von {:?}. Fallback auf Simple.", std::env::var("PROMPT_TPL"));
        tpl = ChatTpl::Simple;
    }

    println!(
        "Hinweis: PROMPT_TPL=simple|inst|chatml|im|llama3|gemma|turn|alpaca|vicuna (ENV) wählbar."
    );

    // 6) Sampling-Parameter (ENV überschreibbar)
    let d_temperature: f32 = env_f32("TEMP", 0.8);
    let i_top_k: usize = env_usize("TOP_K", 20);
    let d_top_p: f32 = env_f32("TOP_P", 0.90);
    let i_max_new: usize = env_usize("MAX_NEW", 200);
    let mut rng = SimpleRng::new(env_u64("SEED", 0x1234_5678));

    println!("Frage das Modell. Eingabe 'exit' zum Beenden.");
    println!(
        "Nutze Sampling: temp={}, top_k={}, top_p={}",
        d_temperature, i_top_k, d_top_p
    );
    println!("Hinweis: PROMPT_TPL=simple|inst|chatml|im (ENV) wählbar.");

    // 7) Interaktive Schleife
    let mut s_input = String::new();
    loop {
        print!("> ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        s_input.clear();
        io::stdin()
            .read_line(&mut s_input)
            .map_err(|e| e.to_string())?;

        let s_line: &str = s_input.trim();
        if s_line.eq_ignore_ascii_case("exit") {
            break;
        }
        if s_line.is_empty() {
            continue;
        }

        // System-Text aus ENV oder Default
        let s_system = std::env::var("SYSTEM_PROMPT")
            .unwrap_or_else(|_| "Du bist eine hilfreiche Assistenz.".to_string());

        // Prompt bauen
        let s_prompt = build_prompt(tpl, &s_system, s_line);
        println!(
            "dbg prompt (first 200): {}",
            visible_token(&s_prompt.chars().take(200).collect::<String>())
        );

        // ENCODE: add_special = false (wir steuern BOS bei Simple selbst)
        let mut ids: Vec<usize> = tok
            .encode(&s_prompt, false)
            .map_err(|e| format!("encode fehlgeschlagen: {}", e))?;

        // Bei Simple ggf. BOS einfügen
        if matches!(tpl, ChatTpl::Simple) {
            if let Some(bos) = tok.bos_id() {
                if ids.first().copied() != Some(bos) {
                    ids.insert(0, bos);
                }
            }
        }

        // KV-Cache zurücksetzen
        model.reset_kv_cache();

        // Prompt vorwärts laufen
        let mut logits: Vec<f32> = Vec::new();
        for (pos, &tid) in ids.iter().enumerate() {
            logits = model.forward_next(tid, pos)?;
        }

        // Debug: Top-5 Logits
        let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (id, logit)) in pairs.iter().take(10).enumerate() {
            let tok_str = tok.decode(&[*id], true).unwrap_or_default();
            println!("#{i} id={id} tok={:?} logit={:.4}", tok_str, logit);
        }

        // Generierung: nur neues Suffix drucken
        let mut gen_ids: Vec<usize> = Vec::new();
        let mut s_out_prev = String::new();

        for _ in 0..i_max_new {
            let next_id =
                sample_top_k_top_p_temperature(&logits, d_temperature, i_top_k, d_top_p, &mut rng);
            ids.push(next_id);
            gen_ids.push(next_id);

            let pos = ids.len() - 1;
            logits = model.forward_next(next_id, pos)?;

            if let Ok(s_all) = tok.decode(&gen_ids, true) {
                if s_all.len() >= s_out_prev.len() {
                    let tail = s_all.get(s_out_prev.len()..).unwrap_or("");
                    if !tail.is_empty() {
                        print!("{}", tail);
                        io::stdout().flush().map_err(|e| e.to_string())?;
                    }
                    s_out_prev = s_all;
                }
            }

            if let Some(eos) = tok.eos_id() {
                if next_id == eos {
                    break;
                }
            }
        }
        println!();
    }

    Ok(())
}
