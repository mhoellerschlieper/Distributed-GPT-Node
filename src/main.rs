// main.rs
// ------------------------------------------------------------
// Chat mit Streaming, Stop-IDs, Template-Erkennung
// Backend wählbar: "candle" (schneller) oder "transformers" (eigener CPU-Forward)
// Autor: Marcus Schlieper, ExpChat.ai
// Kontakt: mschlieper@ylook.de | 49 2338 8748862 | 49 15115751864
// Firma: ExpChat.ai – Der KI Chat Client für den Mittelstand aus Breckerfeld
// Zusatz: RPA, KI Agents, KI Internet Research, KI Wissensmanagement
// Stand: 2025-12-23
// Lizenz: MIT / Apache-2.0
//
// Sicherheit:
// - kein unsafe
// - klare Fehler
// - robustes Streaming ohne zerschossene UTF-8-Ausgabe
// ------------------------------------------------------------

#![allow(warnings)]

mod model; // eigenes CPU-Transformers-Backend (safetensors)
mod models_candle; // Candle-Backend (safetensors)
mod tokenizer; // Tokenizer aus tokenizer.json

use std::io::{self, Write};

// ---------------- Env-Helper ----------------

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(d)
}
fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s) if s != "0")
}

// ---------------- Prompt-Templates ----------------

#[derive(Clone, Copy, Debug)]
enum ChatTemplate {
    ChatMLIm,
    ChatML,
    Llama3,
    Llama2,
    Mistral,
    Gemma,
    Alpaca,
    SimpleTags,
}

fn token_exists(tok: &tokenizer::GgufTokenizer, s: &str) -> bool {
    tok.encode(s, false)
        .map(|ids| ids.len() == 1)
        .unwrap_or(false)
}

fn detect_chat_template(tok: &tokenizer::GgufTokenizer) -> ChatTemplate {
    if let Ok(sel) = std::env::var("CHAT_TEMPLATE") {
        let s = sel.to_lowercase();
        return match s.as_str() {
            "qwen" | "chatml_im" | "im" => ChatTemplate::ChatMLIm,
            "chatml" => ChatTemplate::ChatML,
            "llama3" => ChatTemplate::Llama3,
            "llama2" => ChatTemplate::Llama2,
            "mistral" => ChatTemplate::Mistral,
            "gemma" => ChatTemplate::Gemma,
            "alpaca" => ChatTemplate::Alpaca,
            _ => ChatTemplate::SimpleTags,
        };
    }
    // Auto-Heuristik
    if token_exists(tok, "<|start_header_id|>")
        && token_exists(tok, "<|end_header_id|>")
        && token_exists(tok, "<|eot_id|>")
    {
        return ChatTemplate::Llama3;
    }
    if token_exists(tok, "<|im_start|>") && token_exists(tok, "<|im_end|>") {
        return ChatTemplate::ChatMLIm;
    }
    if token_exists(tok, "<|system|>")
        && token_exists(tok, "<|user|>")
        && token_exists(tok, "<|assistant|>")
    {
        return ChatTemplate::ChatML;
    }
    if token_exists(tok, "[INST]") && token_exists(tok, "[/INST]") {
        // ohne SYS-Blöcke -> Mistral
        return ChatTemplate::Mistral;
    }
    if token_exists(tok, "###") || token_exists(tok, "### Instruction:") {
        return ChatTemplate::Alpaca;
    }
    ChatTemplate::SimpleTags
}

fn build_first_turn(
    fmt: ChatTemplate,
    tok: &tokenizer::GgufTokenizer,
    system_opt: Option<&str>,
    user: &str,
) -> String {
    match fmt {
        ChatTemplate::ChatMLIm => {
            let sys = system_opt.unwrap_or("You are a helpful assistant.");
            let mut s = String::new();
            if token_exists(tok, "<|begin_of_text|>") {
                s.push_str("<|begin_of_text|>");
            }
            s.push_str(&format!(
                "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
                sys, user
            ));
            s
        }
        ChatTemplate::ChatML => {
            let sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!(
                "<|system|>\n{}\n\n<|user|>\n{}\n\n<|assistant|>\n",
                sys, user
            )
        }
        ChatTemplate::Llama3 => {
            let sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                sys, user
            )
        }
        ChatTemplate::Llama2 => {
            let sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]\n", sys, user)
        }
        ChatTemplate::Mistral => {
            format!("[INST] {}\n[/INST]\n", user)
        }
        ChatTemplate::Gemma => {
            let sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!("system\n{}\nuser\n{}\nmodel\n", sys, user)
        }
        ChatTemplate::Alpaca => {
            if let Some(sys) = system_opt {
                format!(
                    "### System:\n{}\n\n### Instruction:\n{}\n\n### Response:\n",
                    sys, user
                )
            } else {
                format!("### Instruction:\n{}\n\n### Response:\n", user)
            }
        }
        ChatTemplate::SimpleTags => {
            let mut s = String::new();
            if let Some(sys) = system_opt {
                s.push_str(&format!("<|system|>\n{}\n", sys));
            }
            s.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", user));
            s
        }
    }
}

fn build_next_turn(fmt: ChatTemplate, user: &str) -> String {
    match fmt {
        ChatTemplate::ChatMLIm => {
            format!(
                "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
                user
            )
        }
        ChatTemplate::ChatML => {
            format!("<|user|>\n{}\n\n<|assistant|>\n", user)
        }
        ChatTemplate::Llama3 => {
            format!(
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user
            )
        }
        ChatTemplate::Llama2 | ChatTemplate::Mistral => {
            format!("[INST] {}\n[/INST]\n", user)
        }
        ChatTemplate::Gemma => {
            format!("user\n{}\nmodel\n", user)
        }
        ChatTemplate::Alpaca => {
            format!("### Instruction:\n{}\n\n### Response:\n", user)
        }
        ChatTemplate::SimpleTags => {
            format!("<|user|>\n{}\n<|assistant|>\n", user)
        }
    }
}

fn default_stops_for_template(fmt: ChatTemplate) -> Vec<String> {
    match fmt {
        ChatTemplate::ChatMLIm => vec!["<|im_end|>".to_string()],
        ChatTemplate::ChatML => vec!["<|end|>".to_string()],
        ChatTemplate::Llama3 => vec!["<|eot_id|>".to_string()],
        ChatTemplate::Llama2 => vec!["[/INST]".to_string(), "</s>".to_string()],
        ChatTemplate::Mistral => vec!["[/INST]".to_string()],
        ChatTemplate::Gemma => vec!["model".to_string()],
        ChatTemplate::Alpaca => vec!["### Instruction:".to_string(), "### Input:".to_string()],
        ChatTemplate::SimpleTags => vec!["<|user|>".to_string()],
    }
}

fn env_or_default_stops(tok: &tokenizer::GgufTokenizer, fmt: ChatTemplate) -> Vec<String> {
    if let Ok(raw) = std::env::var("STOP") {
        let it = if raw.contains("||") {
            raw.split("||")
        } else {
            raw.split(",")
        };
        it.map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        let mut v = default_stops_for_template(fmt);
        // nur Strings behalten, die als 1-Token existieren – falls nicht, behalten wir sie trotzdem (konservativ)
        // v.retain(|s| token_exists(tok, s)); // optional
        v
    }
}

// ---------------- Stop-IDs ----------------

fn compile_stop_id_sequences(tok: &tokenizer::GgufTokenizer, stop_str: &[String]) -> Vec<Vec<u32>> {
    let mut v_seqs: Vec<Vec<u32>> = Vec::new();
    for s in stop_str {
        if let Ok(ids) = tok.encode(s, false) {
            if !ids.is_empty() {
                v_seqs.push(ids);
            }
        }
    }
    v_seqs
}

fn max_stop_len(v_stop_ids: &[Vec<u32>]) -> usize {
    v_stop_ids.iter().map(|v| v.len()).max().unwrap_or(0)
}

// ---------------- Sampler-Fallback ----------------

fn pick_top1(v_logits: &[f32], d_temp: f32) -> usize {
    let inv_t = if d_temp > 0.0 { 1.0 / d_temp } else { 1.0 };
    let mut i_best = 0usize;
    let mut d_best = f32::MIN;
    for (i, &v) in v_logits.iter().enumerate() {
        let vt = v * inv_t;
        if vt > d_best {
            d_best = vt;
            i_best = i;
        }
    }
    i_best
}

// ---------------- Gemeinsames Backend-Trait ----------------

trait LmBackend {
    fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String>;
    fn vocab_size(&self) -> usize;
}

// Custom: TransformerModel (safetensors)
impl LmBackend for model::TransformerModel {
    fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        model::TransformerModel::forward_tokens(self, ids)
    }
    fn vocab_size(&self) -> usize {
        model::TransformerModel::vocab_size(self)
    }
}

// Candle (safetensors)
impl LmBackend for models_candle::CandleLlamaModel {
    fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        models_candle::CandleLlamaModel::forward_tokens(self, ids)
    }
    fn vocab_size(&self) -> usize {
        models_candle::CandleLlamaModel::vocab_size(self)
    }
}

// Backend wählen
fn build_backend() -> Result<Box<dyn LmBackend>, String> {
    let s_weights =
        std::env::var("LLAMA_WEIGHTS").map_err(|_| "Env LLAMA_WEIGHTS fehlt".to_string())?;
    let s_config =
        std::env::var("LLAMA_CONFIG").map_err(|_| "Env LLAMA_CONFIG fehlt".to_string())?;
    let which = std::env::var("BACKEND")
        .unwrap_or_else(|_| "candle".to_string())
        .to_lowercase();
    let dt = match std::env::var("LLAMA_DTYPE")
        .unwrap_or_else(|_| "f32".to_string())
        .as_str()
    {
        "f16" => candle::DType::F16,
        "bf16" => candle::DType::BF16,
        _ => candle::DType::F32,
    };
    
    match which.as_str() {
        "transformers" => {
            let m = model::TransformerModel::from_safetensors(&s_weights, &s_config, dt)?;
            Ok(Box::new(m))
        }
        // candle (Default)
        _ => {
            let m = models_candle::CandleLlamaModel::from_safetensors(&s_weights, &s_config, dt)?;
            Ok(Box::new(m))
        }
    }
}

// ---------------- Streaming-Helfer: stabiler UTF-8-Ausdruck ----------------

fn common_prefix_bytes(a: &str, b: &str) -> usize {
    // vergleicht nach chars -> erzeugt gültige Byte-Grenzen für UTF-8
    let mut n = 0usize;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca == cb {
            n += ca.len_utf8();
        } else {
            break;
        }
    }
    n
}

// ---------------- Chat-Loop ----------------

fn chat_loop(
    tok: tokenizer::GgufTokenizer,
    mut ctx_ids: Vec<u32>,
    mut mdl: Box<dyn LmBackend>,
    d_temp: f32,
    i_max_new: usize,
    i_topk: usize,
    d_topp: f32,
    d_minp: f32,
    d_rep_pen: f32,
    i_rep_win: usize,
    v_stop_ids: Vec<Vec<u32>>,
    fmt: ChatTemplate,
) -> Result<(), String> {
    println!("Chat gestartet. Tippe exit zum Beenden.");

    let mut rng_state: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64 ^ 0xC0FFEE_BAAD_F00D_u64)
        .unwrap_or(0x1234_5678_9ABC_DEF0_u64);

    let _i_max_stop = max_stop_len(&v_stop_ids);
    let mut s_input = String::new();

    loop {
        print!("> ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        s_input.clear();
        io::stdin()
            .read_line(&mut s_input)
            .map_err(|e| e.to_string())?;
        let s_line = s_input.trim();
        if s_line.eq_ignore_ascii_case("exit") {
            println!("Tschuess");
            break;
        }
        if s_line.is_empty() {
            continue;
        }

        // Benutzer-Zugabe in den Kontext einbetten
        if ctx_ids.is_empty() {
            if let Some(bos) = tok.bos_id() {
                ctx_ids.push(bos);
            }
            let s_sys = std::env::var("SYSTEM_PROMPT")
                .unwrap_or_else(|_| "You are a helpful assistant.".to_string());
            let s_first = build_first_turn(fmt, &tok, Some(&s_sys), s_line);
            let ids = tok.encode(&s_first, false)?;
            ctx_ids.extend_from_slice(&ids);

            if debug_on() {
                println!(
                    "[DBG] first turn: template={:?}, sys_len={}, user_len={}",
                    fmt,
                    s_sys.len(),
                    s_line.len()
                );
                println!("[DBG] ctx_len_after_first = {}", ctx_ids.len());
            }
        } else {
            let s_next = build_next_turn(fmt, s_line);
            let ids = tok.encode(&s_next, false)?;
            ctx_ids.extend_from_slice(&ids);
            if debug_on() {
                println!("[DBG] appended next turn, ctx_len = {}", ctx_ids.len());
            }
        }

        let mut pending: Vec<u32> = Vec::new();
        let mut printed_any = false;
        let mut s_printed = String::new(); // stabiler Streaming-Puffer

        if debug_on() {
            println!("[DBG] generation start: ctx_len = {}", ctx_ids.len());
        }

        for step in 0..i_max_new {
            // Repetition window
            let start = ctx_ids.len().saturating_sub(i_rep_win);
            let v_recent = &ctx_ids[start..];

            // Logits
            let v_logits = mdl.forward_tokens(&ctx_ids)?;
            let next_idx = if (d_topp > 0.0 && d_topp <= 1.0) || d_minp > 0.0 || d_rep_pen > 1.0 {
                model::sample_topk_topp_minp_with_repeat(
                    &v_logits,
                    d_temp,
                    i_topk,
                    d_topp,
                    d_minp,
                    v_recent,
                    d_rep_pen,
                    &mut rng_state,
                ) as u32
            } else if i_topk > 0 {
                model::sample_topk(&v_logits, d_temp, i_topk, &mut rng_state) as u32
            } else {
                pick_top1(&v_logits, d_temp) as u32
            };

            ctx_ids.push(next_idx);
            pending.push(next_idx);

            // Stop-IDs prüfen (vollständige Sequenz am Ende)
            let mut hit_stop = false;
            let mut stop_n = 0usize;
            'stopcheck: for seq in &v_stop_ids {
                let n = seq.len();
                if n == 0 || ctx_ids.len() < n {
                    continue;
                }
                let tail = &ctx_ids[ctx_ids.len() - n..];
                if tail == seq.as_slice() {
                    stop_n = n;
                    hit_stop = true;
                    break 'stopcheck;
                }
            }

            if hit_stop {
                // Nur sicheren Teil (ohne Stop-Sequenz) ausgeben:
                if pending.len() > stop_n {
                    let safe = &pending[..pending.len() - stop_n];
                    if !safe.is_empty() {
                        let rest = tok.decode(safe, true).unwrap_or_default();
                        let i_cp = common_prefix_bytes(&s_printed, &rest);
                        let new_bytes = &rest.as_bytes()[i_cp..];
                        if !new_bytes.is_empty() {
                            if let Ok(new_str) = std::str::from_utf8(new_bytes) {
                                print!("{}", new_str);
                                io::stdout().flush().map_err(|e| e.to_string())?;
                                printed_any = true;
                            }
                        }
                    }
                }
                pending.clear();
                s_printed.clear();
                println!();
                println!();
                if debug_on() {
                    println!(
                        "[DBG] stop hit at step {}, ctx_len = {}",
                        step,
                        ctx_ids.len()
                    );
                }
                break;
            }

            // Streaming stabil: gesamte Pending decodieren, nur neuen Suffix drucken
            let s_all = tok.decode(&pending, true).unwrap_or_default();
            let i_cp = common_prefix_bytes(&s_printed, &s_all);
            let new_bytes = &s_all.as_bytes()[i_cp..];
            if !new_bytes.is_empty() {
                if let Ok(new_str) = std::str::from_utf8(new_bytes) {
                    print!("{}", new_str);
                    io::stdout().flush().map_err(|e| e.to_string())?;
                    printed_any = true;
                }
            }
            s_printed = s_all;

            // EOS?
            if let Some(eos) = tok.eos_id() {
                if next_idx == eos {
                    // Rest final ausgeben (falls noch nicht gezeigt)
                    if !pending.is_empty() {
                        let rest = tok.decode(&pending, true).unwrap_or_default();
                        let i_cp = common_prefix_bytes(&s_printed, &rest);
                        let new_bytes = &rest.as_bytes()[i_cp..];
                        if !new_bytes.is_empty() {
                            if let Ok(new_str) = std::str::from_utf8(new_bytes) {
                                print!("{}", new_str);
                                io::stdout().flush().map_err(|e| e.to_string())?;
                            }
                        }
                        pending.clear();
                        s_printed.clear();
                    }
                    println!();
                    println!();
                    if debug_on() {
                        println!(
                            "[DBG] eos reached at step {}, ctx_len = {}",
                            step,
                            ctx_ids.len()
                        );
                    }
                    break;
                }
            }
        }

        // Falls kein Stop/EOS ausgelöst, eventuell verbliebene Ausgabe zeigen
        if !pending.is_empty() {
            let rest = tok.decode(&pending, true).unwrap_or_default();
            let i_cp = common_prefix_bytes(&s_printed, &rest);
            let new_bytes = &rest.as_bytes()[i_cp..];
            if !new_bytes.is_empty() {
                if let Ok(new_str) = std::str::from_utf8(new_bytes) {
                    print!("{}", new_str);
                    io::stdout().flush().map_err(|e| e.to_string())?;
                }
            }
            pending.clear();
            s_printed.clear();
        }
        if !printed_any {
            println!("[kein Token erzeugt]");
        }
        println!();
    }

    Ok(())
}

// ---------------- main ----------------

fn main() -> Result<(), String> {
    // Erforderlich: Weights, Config, Tokenizer
    let s_tok_json = std::env::var("TOKENIZER_JSON")
        .map_err(|_| "Bitte TOKENIZER_JSON auf tokenizer.json setzen".to_string())?;
    let o_tok = tokenizer::load_tokenizer_from_json_force(&s_tok_json)
        .map_err(|e| format!("Tokenizer-Load Fehler: {}", e))?;

    // Sampling
    let d_temperature = env_f32("TEMP", 0.8);
    let i_max_new = env_usize("MAX_NEW", 64);
    let i_topk = env_usize("TOPK", 40);
    let d_topp = env_f32("TOPP", 0.9);
    let d_minp = env_f32("MINP", 0.0);
    let d_rep_pen = env_f32("REP_PENALTY", 1.1);
    let i_rep_win = env_usize("REP_WINDOW", 256);

    // Backend
    let mut o_backend = build_backend()?;
    println!("Vokabular (Backend): {}", o_backend.vocab_size());

    // Template + Stops
    let e_fmt = detect_chat_template(&o_tok);
    let v_stop_str = env_or_default_stops(&o_tok, e_fmt);
    let v_stop_ids = compile_stop_id_sequences(&o_tok, &v_stop_str);

    if debug_on() {
        println!("[CHK] BOS={:?} | EOS={:?}", o_tok.bos_id(), o_tok.eos_id());
        let stop_raw = std::env::var("STOP").unwrap_or_else(|_| "".to_string());
        println!("[CHK] STOP strings = {}", stop_raw);
        println!("[CHK] DETECTED_TEMPLATE = {:?}", e_fmt);
    }

    println!(
        "Temperature = {}, Max New = {}, Top-k = {}, Top-p = {}, Min-p = {}, RepPenalty = {}, RepWindow = {}, Stop = {:?}, Template = {:?}",
        d_temperature, i_max_new, i_topk, d_topp, d_minp, d_rep_pen, i_rep_win, v_stop_str, e_fmt
    );

    // Chat
    chat_loop(
        o_tok,
        Vec::new(),
        o_backend,
        d_temperature,
        i_max_new,
        i_topk,
        d_topp,
        d_minp,
        d_rep_pen,
        i_rep_win,
        v_stop_ids,
        e_fmt,
    )
}
