// main.rs
// ------------------------------------------------------------
// Chat mit Streaming, Stop-IDs, Template-Erkennung
// Erweiterung: Chat Befehle plus Menu Ausgabe
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-23 Marcus Schlieper: Basis Version
// - 2025-12-26 Marcus Schlieper: Menu und 10 Chat Befehle in main und chat loop

//  $env:LLAMA_DTYPE="f16"; $env:OMP_NUM_THREADS="8";$env:RAYON_NUM_THREADS="8";$env:DEBUG_MODEL="0"; $env:LLAMA_WEIGHTS="d:\Models\Llama-3.2-3B-I\model.safetensors.index.json"; $env:LLAMA_CONFIG="D:\models\Llama-3.2-3B-I\config.json"; $env:TOKENIZER_JSON="D:\models\Llama-3.2-3B-I\tokenizer.json"; $env:BACKEND="candle"; $env:CHAT_TEMPLATE="Llama3"; $env:STOP="<|eot_id|>";  cargo run --bin llm_node --release
// ------------------------------------------------------------
// main.rs
// ------------------------------------------------------------
// Chat mit Streaming, Stop-IDs, Template-Erkennung
// Erweiterung: Menu, save und load, und 20 weitere Tools
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-23 Marcus Schlieper: Basis Version
// - 2025-12-26 Marcus Schlieper: Menu und Chat Befehle erweitert, save und load, weitere Tools

// Aufruf:
// $env:LLAMA_DTYPE="f16"; $env:OMP_NUM_THREADS="8";$env:RAYON_NUM_THREADS="8";$env:DEBUG_MODEL="0"; $env:LLAMA_WEIGHTS="d:\Models\Llama-3.2-3B-I\model.safetensors.index.json"; $env:LLAMA_CONFIG="D:\models\Llama-3.2-3B-I\config.json"; $env:TOKENIZER_JSON="D:\models\Llama-3.2-3B-I\tokenizer.json"; $env:BACKEND="candle"; $env:CHAT_TEMPLATE="Llama3"; $env:STOP="<|eot_id|>";  cargo run --bin llm_node --release
// ------------------------------------------------------------

#![allow(warnings)]

mod local_llama;
mod model;

mod model_inspect;
mod models_candle;
mod tokenizer;

use std::io::{self, Write};
use std::path::Path;

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal;
use std::time::Duration;

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
            let mut s_out = String::new();
            if token_exists(tok, "<|begin_of_text|>") {
                s_out.push_str("<|begin_of_text|>");
            }
            s_out.push_str(&format!(
                "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
                sys, user
            ));
            s_out
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
        ChatTemplate::Mistral => format!("[INST] {}\n[/INST]\n", user),
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
            let mut s_out = String::new();
            if let Some(sys) = system_opt {
                s_out.push_str(&format!("<|system|>\n{}\n", sys));
            }
            s_out.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", user));
            s_out
        }
    }
}

fn build_next_turn(fmt: ChatTemplate, user: &str) -> String {
    match fmt {
        ChatTemplate::ChatMLIm => format!(
            "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            user
        ),
        ChatTemplate::ChatML => format!("<|user|>\n{}\n\n<|assistant|>\n", user),
        ChatTemplate::Llama3 => format!(
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user
        ),
        ChatTemplate::Llama2 | ChatTemplate::Mistral => format!("[INST] {}\n[/INST]\n", user),
        ChatTemplate::Gemma => format!("user\n{}\nmodel\n", user),
        ChatTemplate::Alpaca => format!("### Instruction:\n{}\n\n### Response:\n", user),
        ChatTemplate::SimpleTags => format!("<|user|>\n{}\n<|assistant|>\n", user),
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
        default_stops_for_template(fmt)
    }
}

// ---------------- Stop-IDs ----------------

fn compile_stop_id_sequences(tok: &tokenizer::GgufTokenizer, stop_str: &[String]) -> Vec<Vec<u32>> {
    let mut v_seqs: Vec<Vec<u32>> = Vec::new();
    for s_val in stop_str {
        if let Ok(ids) = tok.encode(s_val, false) {
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

// ---------------- Backend-Trait ----------------

trait LmBackend {
    fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String>;
    fn vocab_size(&self) -> usize;
}

impl LmBackend for models_candle::CandleLlamaModel {
    fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        models_candle::CandleLlamaModel::forward_tokens(self, ids)
    }
    fn vocab_size(&self) -> usize {
        models_candle::CandleLlamaModel::vocab_size(self)
    }
}

fn build_backend() -> Result<Box<dyn LmBackend>, String> {
    let s_weights =
        std::env::var("LLAMA_WEIGHTS").map_err(|_| "Env LLAMA_WEIGHTS fehlt".to_string())?;
    let s_config =
        std::env::var("LLAMA_CONFIG").map_err(|_| "Env LLAMA_CONFIG fehlt".to_string())?;

    let dt = match std::env::var("LLAMA_DTYPE")
        .unwrap_or_else(|_| "f32".to_string())
        .as_str()
    {
        "f16" => candle::DType::F16,
        "bf16" => candle::DType::BF16,
        _ => candle::DType::F32,
    };

    println!("RUNNING: Candle");
    let m = models_candle::CandleLlamaModel::from_safetensors(&s_weights, &s_config, dt)?;
    Ok(Box::new(m))
}

// ---------------- Streaming Helper ----------------

fn common_prefix_bytes(a: &str, b: &str) -> usize {
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

// ---------------- Chat Tools ----------------

#[derive(Clone)]
struct ChatState {
    s_system_prompt: String,
    d_temp: f32,
    i_max_new: usize,
    i_topk: usize,
    d_topp: f32,
    d_minp: f32,
    d_rep_pen: f32,
    i_rep_win: usize,
    b_echo_prompt: bool,
    b_debug_show_tokens: bool,
    i_last_gen_tokens: usize,
}

fn default_chat_state() -> ChatState {
    let s_system_prompt = std::env::var("SYSTEM_PROMPT")
        .unwrap_or_else(|_| "You are a helpful assistant.".to_string());
    ChatState {
        s_system_prompt,
        d_temp: env_f32("TEMP", 0.8),
        i_max_new: env_usize("MAX_NEW", 64),
        i_topk: env_usize("TOPK", 40),
        d_topp: env_f32("TOPP", 0.9),
        d_minp: env_f32("MINP", 0.0),
        d_rep_pen: env_f32("REP_PENALTY", 1.1),
        i_rep_win: env_usize("REP_WINDOW", 256),
        b_echo_prompt: false,
        b_debug_show_tokens: false,
        i_last_gen_tokens: 0,
    }
}
fn is_abort_space_pressed() -> Result<bool, String> {
    if event::poll(Duration::from_millis(0)).map_err(|e| e.to_string())? {
        if let Event::Key(k) = event::read().map_err(|e| e.to_string())? {
            if k.code == KeyCode::Char(' ') {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
fn print_menu() {
    println!("------------------------------------------------------------");
    println!("Chat Menu");
    println!("exit                         beendet das programm");
    println!("help                         zeigt dieses menu");
    println!("Space                        bricht die laufende ausgabe ab");
    println!("------------------------------------------------------------");
    println!("save <file>                  speichert kontext token ids");
    println!("load <file>                  laedt kontext token ids");
    println!("sys <text>                   setzt system prompt");
    println!("show sys                     zeigt system prompt");
    println!("temp <f32>                   setzt temperatur");
    println!("topk <usize>                 setzt top k");
    println!("topp <f32>                   setzt top p");
    println!("minp <f32>                   setzt min p");
    println!("maxnew <usize>               setzt max neue tokens");
    println!("rep <pen> <win>              setzt repetition penalty und window");
    println!("stats                        zeigt kontext und sampler");
    println!("vocab                        zeigt vocab size");
    println!("ctx                          zeigt kontext laenge");
    println!("ctx pop <n>                  entfernt die letzten n tokens");
    println!("ctx keep <n>                 behaelt nur die letzten n tokens");
    println!("ctx head <n>                 zeigt die ersten n tokens als text");
    println!("ctx tail <n>                 zeigt die letzten n tokens als text");
    println!("tok <text>                   encodiert text und zeigt token ids");
    println!("detok <ids>                  decodiert ids, format: 1,2,3");
    println!("stop show                    zeigt stop strings aus env oder default");
    println!("template show                zeigt erkanntes template");
    println!("echo on|off                  zeigt prompt vor generation");
    println!("show params                  zeigt sampler parameter");
    println!("reset params                 setzt sampler auf env defaults");
    println!("debug tokens on|off          zeigt token ids pro generation");
    println!("bench <n> <text>             generiert n mal fuer test");
    println!("------------------------------------------------------------");
}

fn try_parse_f32(s_val: &str) -> Option<f32> {
    s_val.trim().parse::<f32>().ok()
}

fn try_parse_usize(s_val: &str) -> Option<usize> {
    s_val.trim().parse::<usize>().ok()
}

fn parse_cmd_arg_rest(s_line: &str) -> (String, String) {
    let mut it = s_line.splitn(2, ' ');
    let s_cmd = it.next().unwrap_or("").trim().to_string();
    let s_rest = it.next().unwrap_or("").trim().to_string();
    (s_cmd, s_rest)
}

fn parse_two_args(s_line: &str) -> Option<(String, String)> {
    let mut it = s_line.split_whitespace();
    let s_a = it.next()?.to_string();
    let s_b = it.next()?.to_string();
    Some((s_a, s_b))
}

fn parse_three_args(s_line: &str) -> Option<(String, String, String)> {
    let mut it = s_line.split_whitespace();
    let s_a = it.next()?.to_string();
    let s_b = it.next()?.to_string();
    let s_c = it.next()?.to_string();
    Some((s_a, s_b, s_c))
}

fn parse_ids_csv(s_ids: &str) -> Result<Vec<u32>, String> {
    let mut v_ids: Vec<u32> = Vec::new();
    for s_part in s_ids.split(',') {
        let s_t = s_part.trim();
        if s_t.is_empty() {
            continue;
        }
        let i_val = s_t
            .parse::<u32>()
            .map_err(|_| "detok: ungueltige id liste".to_string())?;
        v_ids.push(i_val);
    }
    if v_ids.is_empty() {
        return Err("detok: keine ids".to_string());
    }
    Ok(v_ids)
}

fn is_safe_file_name(s_file: &str) -> bool {
    if s_file.is_empty() {
        return false;
    }
    if s_file.contains("..") {
        return false;
    }
    if s_file.contains(':') {
        return false;
    }
    if s_file.contains('\\') {
        return false;
    }
    if s_file.contains('/') {
        return false;
    }
    true
}

fn save_ctx_to_file(s_file: &str, v_ctx_ids: &[u32]) -> Result<(), String> {
    if !is_safe_file_name(s_file) {
        return Err("save: ungueltiger dateiname".to_string());
    }
    if v_ctx_ids.is_empty() {
        return Err("save: kontext ist leer".to_string());
    }

    let mut s_out = String::new();
    for (i, id) in v_ctx_ids.iter().enumerate() {
        if i > 0 {
            s_out.push(',');
        }
        s_out.push_str(&id.to_string());
    }

    std::fs::write(s_file, s_out).map_err(|e| format!("save: schreiben fehlgeschlagen: {}", e))?;
    Ok(())
}

fn load_ctx_from_file(s_file: &str) -> Result<Vec<u32>, String> {
    if !is_safe_file_name(s_file) {
        return Err("load: ungueltiger dateiname".to_string());
    }
    if !Path::new(s_file).exists() {
        return Err("load: datei fehlt".to_string());
    }

    let s_raw = std::fs::read_to_string(s_file)
        .map_err(|e| format!("load: lesen fehlgeschlagen: {}", e))?;
    let v_ids = parse_ids_csv(&s_raw)?;
    Ok(v_ids)
}

fn print_params(o_state: &ChatState) {
    println!(
        "temp={} maxnew={} topk={} topp={} minp={} rep_penalty={} rep_window={}",
        o_state.d_temp,
        o_state.i_max_new,
        o_state.i_topk,
        o_state.d_topp,
        o_state.d_minp,
        o_state.d_rep_pen,
        o_state.i_rep_win
    );
}

fn reset_params_from_env(o_state: &mut ChatState) {
    o_state.d_temp = env_f32("TEMP", 0.8);
    o_state.i_max_new = env_usize("MAX_NEW", 64);
    o_state.i_topk = env_usize("TOPK", 40);
    o_state.d_topp = env_f32("TOPP", 0.9);
    o_state.d_minp = env_f32("MINP", 0.0);
    o_state.d_rep_pen = env_f32("REP_PENALTY", 1.1);
    o_state.i_rep_win = env_usize("REP_WINDOW", 256);
}

// ---------------- Chat-Loop ----------------

// ------------------------------------------------------------
// Chat Loop
// Aenderung:
// - raw mode aktiv fuer sofortige taste
// - generation kann mit space abgebrochen werden
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-26 Marcus Schlieper: space abort
// ------------------------------------------------------------
fn chat_loop(
    tok: tokenizer::GgufTokenizer,
    mut ctx_ids: Vec<u32>,
    mut mdl: Box<dyn LmBackend>,
    fmt: ChatTemplate,
) -> Result<(), String> {
    terminal::enable_raw_mode().map_err(|e| e.to_string())?;

    let r_out = (|| -> Result<(), String> {
        print_menu();
        println!("Chat gestartet. Tippe help fuer menu.");

        let mut rng_state: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 ^ 0xC0FFEE_BAAD_F00D_u64)
            .unwrap_or(0x1234_5678_9ABC_DEF0_u64);

        let v_stop_str = env_or_default_stops(&tok, fmt);
        let v_stop_ids = compile_stop_id_sequences(&tok, &v_stop_str);

        let mut s_input = String::new();

        loop {
            // raw mode: du kannst nicht normal read_line nutzen, weil enter nicht wie gewohnt geht.
            // Loesung: wir bleiben fuer eingabe bei normal mode? Oder du nutzt eine eigene line editor.
            // Minimal invasiv: raw mode nur waehrend generation aktivieren.

            // daher: hier raw mode kurz aus, eingabe lesen, dann wieder an
            terminal::disable_raw_mode().map_err(|e| e.to_string())?;
            print!("> ");
            io::stdout().flush().map_err(|e| e.to_string())?;
            s_input.clear();
            io::stdin()
                .read_line(&mut s_input)
                .map_err(|e| e.to_string())?;
            terminal::enable_raw_mode().map_err(|e| e.to_string())?;

            let s_line = s_input.trim();
            if s_line.is_empty() {
                continue;
            }

            let s_line_lc = s_line.to_lowercase();
            if s_line_lc == "exit" {
                println!("Tschuess");
                break;
            }
            if s_line_lc == "help" {
                print_menu();
                continue;
            }

            // prompt bauen wie gehabt
            if ctx_ids.is_empty() {
                if let Some(bos) = tok.bos_id() {
                    ctx_ids.push(bos);
                }
                let s_sys = std::env::var("SYSTEM_PROMPT")
                    .unwrap_or_else(|_| "You are a helpful assistant.".to_string());
                let s_first = build_first_turn(fmt, &tok, Some(&s_sys), s_line);
                let ids = tok.encode(&s_first, true)?;
                ctx_ids.extend_from_slice(&ids);
            } else {
                let s_next = build_next_turn(fmt, s_line);
                let ids = tok.encode(&s_next, false)?;
                ctx_ids.extend_from_slice(&ids);
            }

            // generation
            let mut pending: Vec<u32> = Vec::new();
            let mut s_printed = String::new();
            let mut printed_any = false;

            for _step in 0..env_usize("MAX_NEW", 64) {
                // Abbruch mit Space
                if is_abort_space_pressed()? {
                    println!();
                    println!("[abbruch durch space]");
                    println!();
                    break;
                }

                let v_logits = mdl.forward_tokens(&ctx_ids)?;
                let next_idx = pick_top1(&v_logits, env_f32("TEMP", 0.8)) as u32;

                ctx_ids.push(next_idx);
                pending.push(next_idx);

                // streaming stabil
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

                // stop ids check wie gehabt (hier kurz gelassen)
                for seq in &v_stop_ids {
                    let n = seq.len();
                    if n == 0 || ctx_ids.len() < n {
                        continue;
                    }
                    let tail = &ctx_ids[ctx_ids.len() - n..];
                    if tail == seq.as_slice() {
                        println!();
                        println!();
                        pending.clear();
                        s_printed.clear();
                        break;
                    }
                }
            }

            if !printed_any {
                println!("[kein token erzeugt]");
            }
        }

        Ok(())
    })();

    terminal::disable_raw_mode().map_err(|e| e.to_string())?;
    r_out
}

// ---------------- main ----------------

fn main() -> Result<(), String> {
    let s_tok_json = std::env::var("TOKENIZER_JSON")
        .map_err(|_| "Bitte TOKENIZER_JSON auf tokenizer.json setzen".to_string())?;
    let o_tok = tokenizer::load_tokenizer_from_json_force(&s_tok_json)
        .map_err(|e| format!("Tokenizer Load Fehler: {}", e))?;

    let mut o_backend = build_backend()?;
    println!("Vokabular (Backend): {}", o_backend.vocab_size());

    let e_fmt = detect_chat_template(&o_tok);
    if debug_on() {
        println!("[CHK] BOS={:?} | EOS={:?}", o_tok.bos_id(), o_tok.eos_id());
        println!("[CHK] DETECTED_TEMPLATE = {:?}", e_fmt);
    }

    chat_loop(o_tok, Vec::new(), o_backend, e_fmt)
}
