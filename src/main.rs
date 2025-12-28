// main.rs
// ------------------------------------------------------------
// Chat mit Streaming, Stop-IDs, Template-Erkennung
// Erweiterung: Menu, save und load, Tools, P2P auto connect und P2P server handler
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-23 Marcus Schlieper: Basis Version
// - 2025-12-26 Marcus Schlieper: Menu und Chat Befehle erweitert, save und load, weitere Tools
// - 2025-12-28 Marcus Schlieper: P2P peers aus blocks_map, auto connect, server handler fuer block run
//
// Hinweis
// - Sonderzeichen in Code und Beschreibung sind vermieden
// - sichere Dateinamen checks sind enthalten
// - server antwortet immer mit RunBlockResponse inkl s_error
// p2p-Aufruf:
// $env:P2P_PORT="4002"; $env:LLAMA_DTYPE="f16"; $env:OMP_NUM_THREADS="8";$env:RAYON_NUM_THREADS="8";$env:DEBUG_MODEL="0"; $env:LLAMA_WEIGHTS="d:\Models\Llama-3.2-3B-I\model.safetensors.index.json"; $env:LLAMA_CONFIG="D:\models\Llama-3.2-3B-I\config.json"; $env:TOKENIZER_JSON="D:\models\Llama-3.2-3B-I\tokenizer.json"; $env:BACKEND="p2p"; $env:CHAT_TEMPLATE="Llama3"; $env:STOP="<|eot_id|>";  cargo run --bin llm_node --release

// ------------------------------------------------------------

#![allow(warnings)]

mod local_llama;
mod model;

mod model_inspect;
mod models_candle;
mod tokenizer;

mod models_p2p;
mod p2p_blocks_map;
mod p2p_client_libp2p;
mod p2p_codec;
mod p2p_identity;
mod p2p_llama_forward;
mod p2p_node;
mod p2p_runtime_handle;
mod p2p_tensor_conv;
mod p2p_wire;

use async_trait::async_trait;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal;
use libp2p::PeerId;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::p2p_tensor_conv::{tensor_to_wire, wire_to_tensor};
use crate::p2p_wire::RunBlockResponse;

fn env_u16(k: &str, d: u16) -> u16 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(d)
}

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

fn env_string(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s_val) if s_val != "0")
}

// ---------------- Prompt Templates ----------------

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

fn token_exists(tok: &tokenizer::GgufTokenizer, s_val: &str) -> bool {
    tok.encode(s_val, false)
        .map(|ids| ids.len() == 1)
        .unwrap_or(false)
}

fn detect_chat_template(tok: &tokenizer::GgufTokenizer) -> ChatTemplate {
    if let Ok(sel) = std::env::var("CHAT_TEMPLATE") {
        let s_lc = sel.to_lowercase();
        return match s_lc.as_str() {
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
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            let mut s_out = String::new();
            if token_exists(tok, "<|begin_of_text|>") {
                s_out.push_str("<|begin_of_text|>");
            }
            s_out.push_str(&format!(
                "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
                s_sys, user
            ));
            s_out
        }
        ChatTemplate::ChatML => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!(
                "<|system|>\n{}\n\n<|user|>\n{}\n\n<|assistant|>\n",
                s_sys, user
            )
        }
        ChatTemplate::Llama3 => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                s_sys, user
            )
        }
        ChatTemplate::Llama2 => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]\n", s_sys, user)
        }
        ChatTemplate::Mistral => format!("[INST] {}\n[/INST]\n", user),
        ChatTemplate::Gemma => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            format!("system\n{}\nuser\n{}\nmodel\n", s_sys, user)
        }
        ChatTemplate::Alpaca => {
            if let Some(s_sys) = system_opt {
                format!(
                    "### System:\n{}\n\n### Instruction:\n{}\n\n### Response:\n",
                    s_sys, user
                )
            } else {
                format!("### Instruction:\n{}\n\n### Response:\n", user)
            }
        }
        ChatTemplate::SimpleTags => {
            let mut s_out = String::new();
            if let Some(s_sys) = system_opt {
                s_out.push_str(&format!("<|system|>\n{}\n", s_sys));
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

// ---------------- Backend Trait ----------------

#[async_trait]
trait LmBackend: Send {
    async fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String>;
    fn vocab_size(&self) -> usize;
}

#[async_trait]
impl LmBackend for models_candle::CandleLlamaModel {
    async fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        models_candle::CandleLlamaModel::forward_tokens(self, ids)
    }

    fn vocab_size(&self) -> usize {
        models_candle::CandleLlamaModel::vocab_size(self)
    }
}

#[async_trait]
impl LmBackend for crate::models_p2p::P2pLlamaModel {
    async fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.forward_tokens_async(ids).await
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

fn build_backend(o_peer_id: PeerId) -> Result<Box<dyn LmBackend>, String> {
    let s_weights =
        std::env::var("LLAMA_WEIGHTS").map_err(|_| "Env LLAMA_WEIGHTS fehlt".to_string())?;
    let s_config =
        std::env::var("LLAMA_CONFIG").map_err(|_| "Env LLAMA_CONFIG fehlt".to_string())?;
    let s_backend = std::env::var("BACKEND").unwrap_or_else(|_| "local".to_string());

    let dt = match std::env::var("LLAMA_DTYPE")
        .unwrap_or_else(|_| "f32".to_string())
        .as_str()
    {
        "f16" => candle::DType::F16,
        "bf16" => candle::DType::BF16,
        _ => candle::DType::F32,
    };

    println!("RUNNING: {}", s_backend);

    let o_backend: Box<dyn LmBackend> = match s_backend.as_str() {
        "local" | "candle" => {
            let o_mdl = models_candle::CandleLlamaModel::from_safetensors(&s_weights, &s_config, dt)?;
            Box::new(o_mdl)
        }
        "p2p" => {
            let s_blocks_map_file =
                std::env::var("BLOCKS_MAP").unwrap_or_else(|_| "blocks_map.json".to_string());
            let s_model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "llama".to_string());

            let o_mdl = crate::models_p2p::P2pLlamaModel::from_safetensors(
                &s_weights,
                &s_config,
                dt,
                &s_blocks_map_file,
                &s_model_name,
                o_peer_id,
            )?;
            Box::new(o_mdl)
        }
        _ => {
            let o_mdl = models_candle::CandleLlamaModel::from_safetensors(&s_weights, &s_config, dt)?;
            Box::new(o_mdl)
        }
    };

    Ok(o_backend)
}

// ---------------- Chat State ----------------

#[derive(Clone)]
struct ChatState {
    s_system_prompt: String,
    d_temp: f32,
    i_max_new: usize,
    b_echo_prompt: bool,
    b_debug_show_tokens: bool,
}

fn default_chat_state() -> ChatState {
    let s_system_prompt = std::env::var("SYSTEM_PROMPT")
        .unwrap_or_else(|_| "You are a helpful assistant.".to_string());
    ChatState {
        s_system_prompt,
        d_temp: env_f32("TEMP", 0.8),
        i_max_new: env_usize("MAX_NEW", 64),
        b_echo_prompt: false,
        b_debug_show_tokens: false,
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
    println!("peers                        zeigt verbundene peers");
    println!("connect <peer_id> <addr>     verbindet manuell zu peer");
    println!("Space                        bricht die laufende ausgabe ab");
    println!("------------------------------------------------------------");
}

fn is_safe_file_name(s_file: &str) -> bool {
    if s_file.trim().is_empty() {
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
    for (i_idx, id) in v_ctx_ids.iter().enumerate() {
        if i_idx > 0 {
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

    let mut v_ids: Vec<u32> = Vec::new();
    for s_part in s_raw.split(',') {
        let s_t = s_part.trim();
        if s_t.is_empty() {
            continue;
        }
        let i_val = s_t
            .parse::<u32>()
            .map_err(|_| "load: ungueltige id liste".to_string())?;
        v_ids.push(i_val);
    }
    if v_ids.is_empty() {
        return Err("load: keine ids".to_string());
    }

    Ok(v_ids)
}

// ---------------- P2P Auto Connect ----------------

async fn auto_connect_route_peers(
    o_rt: &crate::p2p_node::P2pRuntime,
    o_blocks_map: &crate::p2p_blocks_map::BlocksMap,
) -> Result<(), String> {
    let s_model_name = env_string("MODEL_NAME", "llama");
    let set_needed = o_blocks_map.needed_peers_for_model(&s_model_name);

    for s_peer_id in set_needed {
        if s_peer_id == o_blocks_map.s_self_peer_id {
            continue;
        }

        let s_addr = match o_blocks_map.get_addr_for_peer(&s_peer_id) {
            Some(v) => v,
            None => {
                return Err("blocks_map: addr fehlt fuer peer".to_string());
            }
        };

        let o_msg = crate::p2p_node::SwarmControlMsg::ConnectPeer { s_peer_id, s_addr };
        o_rt.o_ctl_tx
            .send(o_msg)
            .await
            .map_err(|_| "auto connect: control channel closed".to_string())?;
    }

    Ok(())
}

// ---------------- P2P Server Handler State ----------------

struct P2pServerState {
    o_model: Arc<crate::local_llama::LocalLlama>,
    o_cache: Arc<Mutex<crate::local_llama::Cache>>,
}
type P2pServerStateRef = Arc<P2pServerState>;

fn build_server_state_from_env() -> Result<P2pServerStateRef, String> {
    let s_weights = std::env::var("LLAMA_WEIGHTS").map_err(|_| "LLAMA_WEIGHTS fehlt".to_string())?;
    let s_config = std::env::var("LLAMA_CONFIG").map_err(|_| "LLAMA_CONFIG fehlt".to_string())?;

    let e_dtype = match std::env::var("LLAMA_DTYPE")
        .unwrap_or_else(|_| "f32".to_string())
        .as_str()
    {
        "f16" => candle::DType::F16,
        "bf16" => candle::DType::BF16,
        _ => candle::DType::F32,
    };

    let o_dev = candle::Device::Cpu;

    let v_cfg_bytes =
        std::fs::read(&s_config).map_err(|e| format!("config json lesen fehlgeschlagen: {}", e))?;
    let o_cfg_raw: crate::local_llama::LlamaConfig =
        serde_json::from_slice(&v_cfg_bytes).map_err(|e| format!("config json parse: {}", e))?;
    let o_config = o_cfg_raw.into_config(false);

    let v_weight_files = crate::model_inspect::build_weight_files(&s_weights)?;
    let o_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&v_weight_files, e_dtype, &o_dev)
            .map_err(|e| format!("safetensors mmap fehlgeschlagen: {}", e))?
    };

    let o_model = crate::local_llama::LocalLlama::load(o_vb, &o_config)
        .map_err(|e| format!("llama load: {}", e))?;
    let o_cache = crate::local_llama::Cache::new(true, e_dtype, &o_config, &o_dev)
        .map_err(|e| format!("llama cache: {}", e))?;

    Ok(Arc::new(P2pServerState {
        o_model: Arc::new(o_model),
        o_cache: Arc::new(Mutex::new(o_cache)),
    }))
}

fn make_error_response(s_err: &str) -> Result<Vec<u8>, String> {
    let o_dummy = candle::Tensor::zeros((1, 1, 1), candle::DType::F32, &candle::Device::Cpu)
        .map_err(|e| format!("server: dummy tensor: {}", e))?;

    let o_resp = RunBlockResponse {
        o_y: tensor_to_wire(&o_dummy).map_err(|e| format!("server: tensor_to_wire: {}", e))?,
        s_error: s_err.to_string(),
    };

    bincode::serialize(&o_resp).map_err(|_| "server: bincode response encode fehlgeschlagen".to_string())
}

fn build_p2p_server_handler(o_state: P2pServerStateRef) -> crate::p2p_node::ServerHandler {
    let o_state_outer = o_state.clone();

    Arc::new(move |o_peer, v_req| {
        let o_state_inner = o_state_outer.clone();

        Box::pin(async move {
            let o_req: crate::p2p_wire::RunBlockRequest = match bincode::deserialize(&v_req) {
                Ok(v) => v,
                Err(_) => return make_error_response("server: request decode fehlgeschlagen"),
            };

            let o_x = match wire_to_tensor(&o_req.o_x) {
                Ok(v) => v,
                Err(e) => return make_error_response(&format!("server: wire_to_tensor: {}", e)),
            };

            let o_y = {
                let mut o_cache_guard = o_state_inner.o_cache.lock().await;

                let o_llama = o_state_inner.o_model.inner_ref();
                match o_llama.forward_one_block(&o_x, o_req.i_pos, o_req.i_block_no, &mut *o_cache_guard) {
                    Ok(v) => v,
                    Err(e) => return make_error_response(&format!("server: forward_one_block: {}", e)),
                }
            };

            let o_resp = RunBlockResponse {
                o_y: match tensor_to_wire(&o_y) {
                    Ok(v) => v,
                    Err(e) => return make_error_response(&format!("server: tensor_to_wire: {}", e)),
                },
                s_error: String::new(),
            };

            bincode::serialize(&o_resp).map_err(|_| "server: response encode fehlgeschlagen".to_string())
        })
    })
}

// ---------------- Chat Loop ----------------

struct RawModeGuard;

impl RawModeGuard {
    fn enable() -> Result<Self, String> {
        terminal::enable_raw_mode().map_err(|e| e.to_string())?;
        Ok(Self)
    }
}

fn build_session_id() -> String {
    let i_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("sess_{}", i_now)
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

async fn chat_loop(
    tok: tokenizer::GgufTokenizer,
    mut ctx_ids: Vec<u32>,
    mut mdl: Box<dyn LmBackend>,
    fmt: ChatTemplate,
    o_p2p_rt_opt: Option<crate::p2p_node::P2pRuntime>,
) -> Result<(), String> {
    let _o_raw_guard = RawModeGuard::enable()?;

    print_menu();
    println!("Chat gestartet. Tippe help fuer menu.");

    let v_stop_str = env_or_default_stops(&tok, fmt);
    let v_stop_ids = compile_stop_id_sequences(&tok, &v_stop_str);

    let mut s_input = String::new();

    loop {
        terminal::disable_raw_mode().map_err(|e| e.to_string())?;
        print!("> ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        s_input.clear();
        io::stdin().read_line(&mut s_input).map_err(|e| e.to_string())?;
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

        if s_line_lc == "peers" {
            if let Some(o_rt) = &o_p2p_rt_opt {
                let v_peers = crate::p2p_node::get_connected_peers(o_rt).await;
                println!("------------------------------------------------------------");
                println!("connected peers count: {}", v_peers.len());
                for o_peer in v_peers {
                    println!("peer: {}", o_peer);
                }
                println!("------------------------------------------------------------");
            } else {
                println!("p2p ist nicht aktiv");
            }
            continue;
        }

        if s_line_lc.starts_with("connect ") {
            if let Some(o_rt) = &o_p2p_rt_opt {
                let mut it = s_line.split_whitespace();
                let _ = it.next();
                let s_peer_id = it.next().unwrap_or("").trim().to_string();
                let s_addr = it.next().unwrap_or("").trim().to_string();

                if s_peer_id.is_empty() || s_addr.is_empty() {
                    println!("connect usage: connect peer_id addr");
                    continue;
                }

                let o_msg = crate::p2p_node::SwarmControlMsg::ConnectPeer { s_peer_id, s_addr };
                if o_rt.o_ctl_tx.send(o_msg).await.is_err() {
                    println!("connect: swarm control channel closed");
                }
                continue;
            } else {
                println!("p2p ist nicht aktiv");
                continue;
            }
        }

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

        let mut pending: Vec<u32> = Vec::new();
        let mut s_printed = String::new();
        let mut printed_any = false;

        for _step in 0..env_usize("MAX_NEW", 64) {
            if is_abort_space_pressed()? {
                println!();
                println!("[abbruch durch space]");
                println!();
                break;
            }

            let v_logits = mdl.forward_tokens(&ctx_ids).await?;
            let next_idx = pick_top1(&v_logits, env_f32("TEMP", 0.8)) as u32;

            ctx_ids.push(next_idx);
            pending.push(next_idx);

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
}

// ---------------- main ----------------

#[tokio::main]
async fn main() -> Result<(), String> {
    let i_p2p_port = env_u16("P2P_PORT", 4001);

    let s_blocks_map_file = std::env::var("BLOCKS_MAP").unwrap_or_else(|_| "blocks_map.json".to_string());
    let s_peer_key_file = std::env::var("PEER_KEY_FILE").unwrap_or_else(|_| "peer_key.bin".to_string());

    let (o_keypair, o_peer_id, _o_map) = crate::p2p_identity::init_peer_identity(&s_blocks_map_file, &s_peer_key_file)?;
    println!("peer id: {}", o_peer_id);

    let (o_p2p_rt, o_swarm, o_rx, o_ctl_rx) = crate::p2p_node::build_runtime(i_p2p_port, o_keypair)?;
    let o_connected_peers = o_p2p_rt.o_connected_peers.clone();

    let o_server_state = build_server_state_from_env()?;
    let o_server_handler: crate::p2p_node::ServerHandler = build_p2p_server_handler(o_server_state);

    crate::p2p_runtime_handle::set_p2p_runtime(o_p2p_rt.clone())?;

    crate::p2p_node::spawn_swarm_task(
        o_swarm,
        o_rx,
        o_ctl_rx,
        o_server_handler,
        o_connected_peers,
    );

    println!("p2p peer id: {}", o_p2p_rt.o_self_peer_id);
    println!("p2p listen tcp port: {}", i_p2p_port);

    let o_blocks_map = crate::p2p_blocks_map::BlocksMap::from_file(&s_blocks_map_file)?;
    if std::env::var("BACKEND").unwrap_or_else(|_| "local".to_string()) == "p2p" {
        auto_connect_route_peers(&o_p2p_rt, &o_blocks_map).await?;
    }

    let s_tok_json = std::env::var("TOKENIZER_JSON")
        .map_err(|_| "Bitte TOKENIZER_JSON auf tokenizer.json setzen".to_string())?;
    let o_tok = tokenizer::load_tokenizer_from_json_force(&s_tok_json)
        .map_err(|e| format!("Tokenizer Load Fehler: {}", e))?;

    let mut o_backend = build_backend(o_peer_id)?;
    println!("Vokabular (Backend): {}", o_backend.vocab_size());

    let e_fmt = detect_chat_template(&o_tok);
    let o_p2p_rt_opt = Some(o_p2p_rt.clone());

    chat_loop(o_tok, Vec::new(), o_backend, e_fmt, o_p2p_rt_opt).await
}
