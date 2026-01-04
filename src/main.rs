// main.rs
// ------------------------------------------------------------
// Description
// This module implements an interactive CLI chat client with token streaming,
// stop-sequence detection, prompt-template auto detection, context save and load,
// and an optional P2P backend based on libp2p that supports block forwarding and
// a server-side handler for block execution.
//
// Core features
// - Prompt template auto detection (ChatML, Llama, Mistral, Gemma, Alpaca, Simple)
// - Stop tokens as strings or token id sequences
// - Greedy decoding (top1) with temperature scaling
// - Interactive menu, commands (help, exit, peers, connect, clear)
// - Save and load of token context to a file
// - P2P auto connect based on blocks map
// - P2P server handler with peer-specific KV cache (per peer hash)
//
// Security and robustness
// - File name validation for save and load to prevent path traversal
// - Error handling for IO, tokenizer, P2P requests, model forward
// - Safe defaults, no unvalidated paths, best-effort broadcast on clear
//
// Author
// Marcus Schlieper, ExpChat.ai
// Contact
// mschlieper@ylook.de
// Phone 49 2338 8748862
// Mobile 49 15115751864
// Company ExpChat.ai, Epscheider Str21 58339 Breckerfeld
// Additional
// Der KI Chat Client fuer den Mittelstand aus Breckerfeld im Sauerland. RPA, KI Agents,
// KI Internet Research, KI Wissensmanagement, Wir bringen KI in den Mittelstand.
//
// History
// - 2025-12-23 Marcus Schlieper: base version
// - 2025-12-26 Marcus Schlieper: extended menu and chat commands, save and load, tools
// - 2025-12-28 Marcus Schlieper: P2P peers from blocks_map, auto connect, server handler for block run
// - 2026-01-03 Marcus Schlieper: clear creates new server cache via build_server_state_from_env
// - 2026-01-03 Marcus Schlieper: cache map per peer hash, reset per peer
// ------------------------------------------------------------

#![allow(warnings)]

// ------------------------------------------------------------
// Section: Module imports
// Description
// These modules encapsulate tokenizer, local model logic (Candle, LocalLlama),
// and P2P functionality (wire format, codec, node runtime, blocks map, identity).
// ------------------------------------------------------------
mod local_llama;
mod model_inspect;
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
mod device_select;

use async_trait::async_trait;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal;
use libp2p::PeerId;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::p2p_tensor_conv::{tensor_to_wire, wire_to_tensor};
use crate::p2p_wire::RunBlockResponse;
use crate::device_select::get_default_device;

use candle::Device;

// ------------------------------------------------------------
// Section: Environment helpers
// Description
// Converts environment variables into concrete types and uses safe defaults if
// variables are missing or invalid.
// ------------------------------------------------------------

/// Reads an environment variable as u16 and falls back to a safe default.
/// - Input: key name k, default value d
/// - Output: parsed u16 if present and valid, otherwise d
/// - Robustness: parsing errors do not propagate, they cause default usage
fn env_u16(k: &str, d: u16) -> u16 {
    // 1) Read env var
    // 2) Parse into u16
    // 3) Default on missing or invalid
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(d)
}

/// Reads an environment variable as f32 and falls back to a safe default.
/// - Input: key name k, default value d
/// - Output: parsed f32 if present and valid, otherwise d
/// - Robustness: parsing errors do not propagate, they cause default usage
fn env_f32(k: &str, d: f32) -> f32 {
    // 1) Read env var
    // 2) Parse into f32
    // 3) Default on missing or invalid
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(d)
}

/// Reads an environment variable as usize and falls back to a safe default.
/// - Input: key name k, default value d
/// - Output: parsed usize if present and valid, otherwise d
/// - Robustness: parsing errors do not propagate, they cause default usage
fn env_usize(k: &str, d: usize) -> usize {
    // 1) Read env var
    // 2) Parse into usize
    // 3) Default on missing or invalid
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(d)
}

/// Reads an environment variable as String and falls back to a safe default.
/// - Input: key name k, default string d
/// - Output: env value if present, otherwise d
/// - Robustness: missing env var uses default
fn env_string(k: &str, d: &str) -> String {
    // 1) Read env var
    // 2) Default on missing
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Enables debug logic if DEBUG_MODEL is set and not equal to "0".
/// - Output: true if debug is enabled, else false
/// - Rationale: supports quick feature toggling without recompilation
fn debug_on() -> bool {
    // The `matches!` guards against missing env var and treats any non-"0" as enabled.
    matches!(std::env::var("DEBUG_MODEL"), Ok(s_val) if s_val != "0")
}

// ------------------------------------------------------------
// Section: Prompt templates
// Description
// Contains template detection using tokenizer tokens and builders for the first
// turn (system and user) and follow-up turns. Stop sequences are template
// dependent and may be overridden via ENV STOP.
// ------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum ChatTemplate {
    /// ChatML with im_start/im_end tokens.
    ChatMLIm,
    /// ChatML with system/user/assistant tokens.
    ChatML,
    /// Llama 3 header and EOT tokens.
    Llama3,
    /// Llama 2 INST syntax.
    Llama2,
    /// Mistral INST syntax.
    Mistral,
    /// Gemma role syntax.
    Gemma,
    /// Alpaca instruction/response syntax.
    Alpaca,
    /// Simple fallback tags.
    SimpleTags,
}

/// Checks whether the given string maps to exactly one token in the tokenizer.
/// - Input: tokenizer reference, token candidate string s_val
/// - Output: true if encoding produces exactly one token id, else false
/// - Purpose: robust template detection without model specific config
fn token_exists(tok: &tokenizer::GgufTokenizer, s_val: &str) -> bool {
    // Encode with `add_special_tokens=false` to test raw representability.
    tok.encode(s_val, false)
        .map(|ids| ids.len() == 1)
        .unwrap_or(false)
}

/// Detects the prompt template in priority order:
/// 1) explicit ENV CHAT_TEMPLATE
/// 2) token based heuristics for known special tokens
/// - Input: tokenizer reference
/// - Output: selected ChatTemplate variant
/// - Robustness: falls back to SimpleTags
fn detect_chat_template(tok: &tokenizer::GgufTokenizer) -> ChatTemplate {
    // Section 1: explicit override via environment
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

    // Section 2: tokenizer heuristics
    // Llama3: header markers and end-of-turn token
    if token_exists(tok, "<|start_header_id|>")
        && token_exists(tok, "<|end_header_id|>")
        && token_exists(tok, "<|eot_id|>")
    {
        return ChatTemplate::Llama3;
    }

    // ChatML with <|im_start|> and <|im_end|>
    if token_exists(tok, "<|im_start|>") && token_exists(tok, "<|im_end|>") {
        return ChatTemplate::ChatMLIm;
    }

    // ChatML with explicit role tokens
    if token_exists(tok, "<|system|>")
        && token_exists(tok, "<|user|>")
        && token_exists(tok, "<|assistant|>")
    {
        return ChatTemplate::ChatML;
    }

    // Mistral style: INST tags
    if token_exists(tok, "[INST]") && token_exists(tok, "[/INST]") {
        return ChatTemplate::Mistral;
    }

    // Alpaca style: "###" markers
    if token_exists(tok, "###") || token_exists(tok, "### Instruction:") {
        return ChatTemplate::Alpaca;
    }

    // Section 3: fallback
    ChatTemplate::SimpleTags
}

/// Builds the initial prompt for the first user input.
/// - Input:
///   - fmt: chosen chat template
///   - tok: tokenizer (used for optional BOS and special token checks)
///   - system_opt: optional system prompt
///   - user: user message string
/// - Output: fully formatted prompt string for first turn
/// - Notes: some templates require explicit assistant marker at the end
fn build_first_turn(
    fmt: ChatTemplate,
    tok: &tokenizer::GgufTokenizer,
    system_opt: Option<&str>,
    user: &str,
) -> String {
    // Section: per-template formatting
    match fmt {
        ChatTemplate::ChatMLIm => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            let mut s_out = String::new();

            // Optional BOS marker, only used if tokenizer supports it as single token.
            if token_exists(tok, "<|begin_of_text|>") {
                s_out.push_str("<|begin_of_text|>");
            }

            // Important: ends with assistant marker to prime decoding in assistant role.
            s_out.push_str(&format!(
                "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
                s_sys, user
            ));
            s_out
        }
        ChatTemplate::ChatML => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            // Important: separate blocks with blank lines to align with common ChatML variants.
            format!(
                "<|system|>\n{}\n\n<|user|>\n{}\n\n<|assistant|>\n",
                s_sys, user
            )
        }
        ChatTemplate::Llama3 => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            // Important: uses EOT token between turns and assistant header at end.
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                s_sys, user
            )
        }
        ChatTemplate::Llama2 => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            // Important: Llama2 wraps system prompt inside <<SYS>> markers.
            format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]\n", s_sys, user)
        }
        ChatTemplate::Mistral => {
            // Mistral typically does not include system prompt in basic INST.
            format!("[INST] {}\n[/INST]\n", user)
        }
        ChatTemplate::Gemma => {
            let s_sys = system_opt.unwrap_or("You are a helpful assistant.");
            // Important: Gemma uses "model" as assistant role marker.
            format!("system\n{}\nuser\n{}\nmodel\n", s_sys, user)
        }
        ChatTemplate::Alpaca => {
            // Alpaca supports an optional system section.
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
            // Minimal fallback, aligns with many finetune tag-based formats.
            let mut s_out = String::new();
            if let Some(s_sys) = system_opt {
                s_out.push_str(&format!("<|system|>\n{}\n", s_sys));
            }
            s_out.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", user));
            s_out
        }
    }
}

/// Builds a prompt segment for subsequent turns (user part plus assistant marker).
/// - Input: template fmt and user text
/// - Output: formatted string that appends the new turn
/// - Note: keeps earlier context in ctx_ids, so only the incremental text is needed
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

/// Returns default stop strings per template.
/// - Input: template fmt
/// - Output: list of stop markers as strings
/// - Note: stop markers are used as high-level stop criteria and later compiled into token id sequences
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

/// Uses ENV STOP (comma or || separated) or falls back to template defaults.
/// - Input: tokenizer (currently unused for parsing, but kept for future checks), template fmt
/// - Output: list of stop markers as strings
/// - Security: trims input and removes empty entries
fn env_or_default_stops(_tok: &tokenizer::GgufTokenizer, fmt: ChatTemplate) -> Vec<String> {
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

/// Compiles stop strings into token id sequences.
/// - Input: tokenizer reference, stop strings slice
/// - Output: list of token id sequences, one per stop string
/// - Robustness: silently ignores stop strings that do not encode or encode to empty
fn compile_stop_id_sequences(tok: &tokenizer::GgufTokenizer, stop_str: &[String]) -> Vec<Vec<u32>> {
    let mut v_seqs: Vec<Vec<u32>> = Vec::new();

    // Section: encode each stop marker into token ids
    for s_val in stop_str {
        if let Ok(ids) = tok.encode(s_val, false) {
            if !ids.is_empty() {
                v_seqs.push(ids);
            }
        }
    }

    v_seqs
}

/// Computes the byte length of the common UTF-8 prefix of two strings.
/// - Input: a and b strings
/// - Output: number of bytes that are identical at the start
/// - Purpose: streaming output prints only the new delta bytes after re-decoding
fn common_prefix_bytes(a: &str, b: &str) -> usize {
    let mut n = 0usize;

    // Iterate as char pairs and accumulate UTF-8 byte length
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca == cb {
            n += ca.len_utf8();
        } else {
            break;
        }
    }

    n
}

/// Greedy decoding: picks the index of the maximum logit.
/// - Input: logits vector, temperature d_temp
/// - Output: argmax index
/// - Note: applies temperature as a scaling factor (inv_t) to logits
fn pick_top1(v_logits: &[f32], d_temp: f32) -> usize {
    // Important: safe default when temp <= 0.0
    let inv_t = if d_temp > 0.0 { 1.0 / d_temp } else { 1.0 };

    let mut i_best = 0usize;
    let mut d_best = f32::MIN;

    // Section: scan all logits to find maximum
    for (i, &v) in v_logits.iter().enumerate() {
        let vt = v * inv_t;
        if vt > d_best {
            d_best = vt;
            i_best = i;
        }
    }

    i_best
}

// ------------------------------------------------------------
// Section: Backend trait
// Description
// Abstraction across different inference backends (local Candle or P2P).
// Goal: chat loop remains backend-agnostic and uses a unified forward_tokens API.
// ------------------------------------------------------------

#[async_trait]
trait LmBackend: Send {
    /// Runs a forward pass over the full token sequence and returns logits.
    /// - Input: token id slice ids
    /// - Output: logits vector (size equals vocab size), or error string
    /// - Contract: backend must be thread-safe with respect to internal state (Send)
    async fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String>;

    /// Returns the vocabulary size, used for diagnostics and plausibility checks.
    fn vocab_size(&self) -> usize;
}

#[async_trait]
impl LmBackend for crate::models_p2p::P2pLlamaModel {
    /// Adapter for P2P backend.
    /// - Delegates to forward_tokens_async
    async fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.forward_tokens_async(ids).await
    }

    /// Adapter for P2P backend vocabulary size.
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

/// Builds the selected backend according to ENV BACKEND.
/// - Input: local peer id (used by P2P backend for addressing and identity)
/// - Output: boxed backend implementing LmBackend
/// - Required ENV:
///   - LLAMA_WEIGHTS, LLAMA_CONFIG
/// - Optional ENV:
///   - LLAMA_DTYPE (f32, f16, bf16)
/// - P2P additional ENV:
///   - BLOCKS_MAP, MODEL_NAME
/// - Robustness: unknown BACKEND values fall back to local backend
// ------------------------------------------------------------
// Section: Backend trait and backend build (refactored)
// ------------------------------------------------------------
// New: shared config struct for unified construction
struct ModelEnvConfig {
    s_weights: String,
    s_config: String,
    e_dtype: candle::DType,
}

fn read_model_env_config() -> Result<ModelEnvConfig, String> {
    let s_weights =
        std::env::var("LLAMA_WEIGHTS").map_err(|_| "Env LLAMA_WEIGHTS fehlt".to_string())?;
    let s_config =
        std::env::var("LLAMA_CONFIG").map_err(|_| "Env LLAMA_CONFIG fehlt".to_string())?;

    let e_dtype = match std::env::var("LLAMA_DTYPE")
        .unwrap_or_else(|_| "f32".to_string())
        .to_lowercase()
        .as_str()
    {
        "f16" => candle::DType::F16,
        "bf16" => candle::DType::BF16,
        _ => candle::DType::F32,
    };

    Ok(ModelEnvConfig {
        s_weights,
        s_config,
        e_dtype,
    })
}

fn load_llama_config_from_json(s_config: &str) -> Result<crate::local_llama::Config, String> {
    let v_cfg_bytes =
        std::fs::read(s_config).map_err(|e| format!("config json lesen fehlgeschlagen: {}", e))?;
    let o_cfg_raw: crate::local_llama::LlamaConfig =
        serde_json::from_slice(&v_cfg_bytes).map_err(|e| format!("config json parse: {}", e))?;
    Ok(o_cfg_raw.into_config(false))
}

fn load_local_llama_from_env_config(
    o_env: &ModelEnvConfig,
    o_blocks_map: &crate::p2p_blocks_map::BlocksMap,
) -> Result<crate::local_llama::LocalLlama, String> {
    let o_dev = get_default_device();

    let o_config = load_llama_config_from_json(&o_env.s_config)?;

    let v_weight_files = crate::model_inspect::build_weight_files(&o_env.s_weights)?;
    let o_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&v_weight_files, o_env.e_dtype, &o_dev)
            .map_err(|e| format!("safetensors mmap fehlgeschlagen: {}", e))?
    };

    crate::local_llama::LocalLlama::load(o_vb, &o_config, o_blocks_map)
        .map_err(|e| format!("llama load: {}", e))
}

// ------------------------------------------------------------
// Description
// Adjust build_backend to pass the already loaded Arc<LocalLlama> into P2pLlamaModel,
// avoiding double weight loading.
//
// Author
// Marcus Schlieper, ExpChat.ai
//
// History
// - 2026-01-04 Marcus Schlieper: pass injected shared LocalLlama into P2pLlamaModel
// ------------------------------------------------------------

fn build_backend(
    o_peer_id: libp2p::PeerId,
    o_blocks_map: &crate::p2p_blocks_map::BlocksMap,
) -> Result<(Box<dyn LmBackend>, P2pServerStateRef), String> {
    let o_env = read_model_env_config()?;

    // Load LocalLlama once.
    let o_local_llama = load_local_llama_from_env_config(&o_env, o_blocks_map)?;
    let o_local_llama_arc: Arc<crate::local_llama::LocalLlama> = Arc::new(o_local_llama);

    // Build server state from the same LocalLlama instance.
    let o_server_state: P2pServerStateRef = Arc::new(P2pServerState {
        o_model: o_local_llama_arc.clone(),
        o_cache_by_peer_hash: Arc::new(Mutex::new(HashMap::new())),
    });

    // Build backend using injected model (no weight load inside).
    let s_blocks_map_file =
        std::env::var("BLOCKS_MAP").unwrap_or_else(|_| "blocks_map.json".to_string());
    let s_model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "llama".to_string());

    let o_mdl = crate::models_p2p::P2pLlamaModel::new_with_local_model(
        o_local_llama_arc.clone(),
        &s_blocks_map_file,
        &s_model_name,
        o_peer_id,
    )?;

    let o_backend: Box<dyn LmBackend> = Box::new(o_mdl);

    Ok((o_backend, o_server_state))
}

// ------------------------------------------------------------
// Section: Chat state
// Description
// Configuration values for the chat session (partly via ENV).
// Note: in this excerpt ChatState is initialized but not used everywhere,
// because some values are read directly via env_* calls.
// ------------------------------------------------------------

#[derive(Clone)]
struct ChatState {
    s_system_prompt: String,
    d_temp: f32,
    i_max_new: usize,
    b_echo_prompt: bool,
    b_debug_show_tokens: bool,
}

/// Builds default chat configuration.
/// - Output: ChatState with defaults and ENV overrides
/// - Notes: SYSTEM_PROMPT defaults to a safe general assistant prompt
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

/// Non-blocking check whether Space has been pressed to stop streaming.
/// - Output: true if Space pressed, else false
/// - Error: returns String on terminal event read errors
fn is_abort_space_pressed() -> Result<bool, String> {
    // Section: poll for key events with zero timeout to avoid blocking generation loop
    if event::poll(Duration::from_millis(0)).map_err(|e| e.to_string())? {
        if let Event::Key(k) = event::read().map_err(|e| e.to_string())? {
            // Important: only Space triggers abort
            if k.code == KeyCode::Char(' ') {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Prints interactive menu and supported commands.
/// - Output: writes help text to stdout
fn print_menu() {
    println!("------------------------------------------------------------");
    println!("Chat Menu");
    println!("exit                         beendet das programm");
    println!("help                         zeigt dieses menu");
    println!("peers                        zeigt verbundene peers");
    println!("connect <peer_id> <addr>     verbindet manuell zu peer");
    println!("clear                        setzt chat kontext zurueck und resettet server cache fuer self peer");
    println!("Space                        bricht die laufende ausgabe ab");
    println!("------------------------------------------------------------");
}

// ------------------------------------------------------------
// Section: Save and load context
// Description
// Persists ctx_ids (token ids) as CSV in a local file.
// Security logic: file name is strictly validated to prevent path traversal.
// ------------------------------------------------------------

/// Validates file names to be safe for use in the current working directory.
/// - Input: s_file file name
/// - Output: true if safe, else false
/// - Security policy:
///   - must not be empty
///   - must not contain ".."
///   - must not contain ":" (windows drive prefix)
///   - must not contain path separators "\" or "/"
fn is_safe_file_name(s_file: &str) -> bool {
    // Section: basic checks
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

/// Saves token ids as CSV into a local file.
/// - Input: safe file name, ctx id slice
/// - Output: Ok on success, Err on validation or IO errors
/// - Security: rejects unsafe file names and empty contexts
fn save_ctx_to_file(s_file: &str, v_ctx_ids: &[u32]) -> Result<(), String> {
    // Section: validation
    if !is_safe_file_name(s_file) {
        return Err("save: ungueltiger dateiname".to_string());
    }
    if v_ctx_ids.is_empty() {
        return Err("save: kontext ist leer".to_string());
    }

    // Section: build CSV payload
    let mut s_out = String::new();
    for (i_idx, id) in v_ctx_ids.iter().enumerate() {
        if i_idx > 0 {
            s_out.push(',');
        }
        s_out.push_str(&id.to_string());
    }

    // Important: writes into current directory only due to strict filename policy
    std::fs::write(s_file, s_out).map_err(|e| format!("save: schreiben fehlgeschlagen: {}", e))?;
    Ok(())
}

/// Loads token ids from a CSV file.
/// - Input: safe file name
/// - Output: Vec<u32> token ids
/// - Errors:
///   - unsafe name
///   - file not found
///   - parse errors
/// - Robustness: ignores empty CSV segments after trimming
fn load_ctx_from_file(s_file: &str) -> Result<Vec<u32>, String> {
    // Section: validation
    if !is_safe_file_name(s_file) {
        return Err("load: ungueltiger dateiname".to_string());
    }
    if !Path::new(s_file).exists() {
        return Err("load: datei fehlt".to_string());
    }

    // Section: file read
    let s_raw = std::fs::read_to_string(s_file)
        .map_err(|e| format!("load: lesen fehlgeschlagen: {}", e))?;

    // Section: parse CSV into ids
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

    // Important: empty context is treated as error to prevent silent no-op loads
    if v_ids.is_empty() {
        return Err("load: keine ids".to_string());
    }

    Ok(v_ids)
}

// ------------------------------------------------------------
// Section: P2P auto connect
// Description
// Automatically connects to all peers required by the model, based on BlocksMap.
// Goal: establish routing for block execution across multiple peers.
// ------------------------------------------------------------

/// Connects to all peers required for the selected model, as derived from BlocksMap.
/// - Input:
///   - o_rt: P2P runtime reference (control channel sender)
///   - o_blocks_map: blocks map with peer addresses and model routing
/// - Output: Ok on success, Err on missing addresses or channel errors
/// - Notes:
///   - skips self peer id
///   - uses SwarmControlMsg::ConnectPeer for each needed peer
async fn auto_connect_route_peers(
    o_rt: &crate::p2p_node::P2pRuntime,
    o_blocks_map: &crate::p2p_blocks_map::BlocksMap,
) -> Result<(), String> {
    // Section: determine model name and required peers
    let s_model_name = env_string("MODEL_NAME", "llama");
    let set_needed = o_blocks_map.needed_peers_for_model(&s_model_name);

    // Section: connect to each needed peer
    for s_peer_id in set_needed {
        // Skip connecting to self
        if s_peer_id == o_blocks_map.s_self_peer_id {
            continue;
        }

        // Security/robustness: require an explicit address for each peer
        let s_addr = match o_blocks_map.get_addr_for_peer(&s_peer_id) {
            Some(v) => v,
            None => {
                return Err("blocks_map: addr fehlt fuer peer".to_string());
            }
        };

        // Important: send connect message to swarm task
        let o_msg = crate::p2p_node::SwarmControlMsg::ConnectPeer { s_peer_id, s_addr };
        o_rt.o_ctl_tx
            .send(o_msg)
            .await
            .map_err(|_| "auto connect: control channel closed".to_string())?;
    }

    Ok(())
}

// ------------------------------------------------------------
// Section: P2P server handler state
// Description
// Holds the locally loaded model plus a KV cache per remote peer.
// Cache keying uses a peer hash so each client peer gets an independent context.
// ------------------------------------------------------------

struct P2pServerState {
    /// Local model that executes blocks on the server side.
    o_model: Arc<crate::local_llama::LocalLlama>,
    /// Map peer_hash -> Cache, guarded by mutex for async access.
    o_cache_by_peer_hash: Arc<Mutex<HashMap<u64, crate::local_llama::Cache>>>,
}
type P2pServerStateRef = Arc<P2pServerState>;

/// Builds a stable non-cryptographic hash from a PeerId.
/// - Input: peer id reference
/// - Output: u64 hash value
/// - Purpose: HashMap keying for per-peer caches
/// - Note: not used for security, only for efficiency and stable indexing
fn build_peer_hash_from_peer_id(o_peer: &PeerId) -> u64 {
    let mut o_hasher = std::collections::hash_map::DefaultHasher::new();
    o_peer.to_string().hash(&mut o_hasher);
    o_hasher.finish()
}

// New central function: reset only cache for peer, keep model loaded
// History:
// - 2026-01-03 Marcus Schlieper: initial, peer hashed cache map, reset only for peer

/// Resets only the KV cache for a specific peer while keeping the model loaded.
/// - Input:
///   - o_state: shared server state
///   - o_peer_id: requesting peer id
/// - Output: Ok on success, Err on config read, parse, or cache init errors
/// - Concurrency: uses mutex to avoid races while replacing cache entry
async fn reset_server_cache_only(
    o_state: &P2pServerStateRef,
    o_peer_id: &PeerId,
) -> Result<(), String> {
    // Section: read config and dtype
    let s_config = std::env::var("LLAMA_CONFIG").map_err(|_| "LLAMA_CONFIG fehlt".to_string())?;
    let e_dtype = match std::env::var("LLAMA_DTYPE")
        .unwrap_or_else(|_| "f32".to_string())
        .as_str()
    {
        "f16" => candle::DType::F16,
        "bf16" => candle::DType::BF16,
        _ => candle::DType::F32,
    };

    // Section: device selection for cache
    let o_dev = get_default_device();

    // Section: parse model config to build cache correctly
    let v_cfg_bytes =
        std::fs::read(&s_config).map_err(|e| format!("config json lesen fehlgeschlagen: {}", e))?;
    let o_cfg_raw: crate::local_llama::LlamaConfig =
        serde_json::from_slice(&v_cfg_bytes).map_err(|e| format!("config json parse: {}", e))?;
    let o_config = o_cfg_raw.into_config(false);

    // Section: allocate new cache instance
    let o_new_cache = crate::local_llama::Cache::new(true, e_dtype, &o_config, &o_dev)
        .map_err(|e| format!("llama cache: {}", e))?;

    // Section: map key and swap cache in map
    let i_peer_hash = build_peer_hash_from_peer_id(o_peer_id);

    // Important: mutex guard ensures atomic replace with respect to other tasks
    let mut o_map_guard = o_state.o_cache_by_peer_hash.lock().await;
    o_map_guard.insert(i_peer_hash, o_new_cache);

    Ok(())
}

/// Creates a valid RunBlockResponse even in error cases.
/// - Input: error string
/// - Output: serialized bincode response bytes, or Err if serialization fails
/// - Robustness: uses a dummy tensor to satisfy response schema
fn make_error_response(s_err: &str) -> Result<Vec<u8>, String> {
    // Section: build dummy tensor
    let o_dummy = candle::Tensor::zeros((1, 1, 1), candle::DType::F32, &get_default_device())
        .map_err(|e| format!("server: dummy tensor: {}", e))?;

    // Section: build response payload
    let o_resp = RunBlockResponse {
        o_y: tensor_to_wire(&o_dummy).map_err(|e| format!("server: tensor_to_wire: {}", e))?,
        s_error: s_err.to_string(),
    };

    // Important: always serialize into bincode for wire transport
    bincode::serialize(&o_resp)
        .map_err(|_| "server: bincode response encode fehlgeschlagen".to_string())
}

// Server handler uses a state holder so clear can replace state at runtime
type P2pServerStateHolder = Arc<Mutex<P2pServerStateRef>>;

/// Builds the P2P server handler closure.
/// - Input: state holder (indirection allows dynamic state swapping)
/// - Output: ServerHandler closure compatible with p2p_node
/// - Behavior:
///   - decodes RunBlockRequest from bincode
///   - converts wire tensor to candle tensor
///   - executes requested blocks sequentially using peer-specific KV cache
///   - encodes RunBlockResponse
/// - Robustness: returns error response bytes on failures instead of crashing
fn build_p2p_server_handler(
    o_state_holder: P2pServerStateHolder,
) -> crate::p2p_node::ServerHandler {
    Arc::new(move |o_peer, v_req| {
        let o_state_holder_inner = o_state_holder.clone();

        Box::pin(async move {
            // Section: decode request
            let o_req: crate::p2p_wire::RunBlockRequest = match bincode::deserialize(&v_req) {
                Ok(v) => v,
                Err(_) => return make_error_response("server: request decode fehlgeschlagen"),
            };

            // Section: convert input tensor from wire to candle tensor
            let o_x = match wire_to_tensor(&o_req.o_x) {
                Ok(v) => v,
                Err(e) => return make_error_response(&format!("server: wire_to_tensor: {}", e)),
            };

            // Section: read current state reference (supports dynamic replacement)
            let o_state_ref: P2pServerStateRef = {
                let o_guard = o_state_holder_inner.lock().await;
                o_guard.clone()
            };

            // Section: compute peer hash for per-peer cache selection
            let i_peer_hash = build_peer_hash_from_peer_id(&o_peer);

            let mut calc_x = o_x;

            // Section: execute each requested block
            for i_block_no in o_req.v_block_nos {
                calc_x = {
                    // Section: obtain or create peer cache
                    let mut o_map_guard = o_state_ref.o_cache_by_peer_hash.lock().await;

                    if !o_map_guard.contains_key(&i_peer_hash) {
                        // First contact: initialize cache for this peer
                        let s_config = match std::env::var("LLAMA_CONFIG") {
                            Ok(v) => v,
                            Err(_) => return make_error_response("server: LLAMA_CONFIG fehlt"),
                        };

                        let e_dtype = match std::env::var("LLAMA_DTYPE")
                            .unwrap_or_else(|_| "f32".to_string())
                            .as_str()
                        {
                            "f16" => candle::DType::F16,
                            "bf16" => candle::DType::BF16,
                            _ => candle::DType::F32,
                        };

                        let o_dev = get_default_device();

                        // Parse config to allocate proper cache structure
                        let v_cfg_bytes = match std::fs::read(&s_config) {
                            Ok(v) => v,
                            Err(e) => {
                                return make_error_response(&format!(
                                    "server: config json lesen fehlgeschlagen: {}",
                                    e
                                ))
                            }
                        };

                        let o_cfg_raw: crate::local_llama::LlamaConfig =
                            match serde_json::from_slice(&v_cfg_bytes) {
                                Ok(v) => v,
                                Err(e) => {
                                    return make_error_response(&format!(
                                        "server: config json parse: {}",
                                        e
                                    ))
                                }
                            };

                        let o_config = o_cfg_raw.into_config(false);

                        // Allocate cache
                        let o_new_cache = match crate::local_llama::Cache::new(
                            true, e_dtype, &o_config, &o_dev,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                return make_error_response(&format!("server: llama cache: {}", e))
                            }
                        };

                        // Important: insert cache into map
                        o_map_guard.insert(i_peer_hash, o_new_cache);
                    };

                    // Section: get cache and forward block
                    let o_cache_mut = match o_map_guard.get_mut(&i_peer_hash) {
                        Some(v) => v,
                        None => return make_error_response("server: cache not found after init"),
                    };

                    // Important: inner_ref provides access to underlying llama model implementation
                    let o_llama = o_state_ref.o_model.inner_ref();
                    match o_llama.forward_one_block(&calc_x, o_req.i_pos, i_block_no, o_cache_mut) {
                        Ok(v) => v,
                        Err(e) => {
                            return make_error_response(&format!(
                                "server: forward_one_block: {}",
                                e
                            ))
                        }
                    }
                };
            }

            // Section: encode response
            let o_resp = RunBlockResponse {
                o_y: match tensor_to_wire(&calc_x) {
                    Ok(v) => v,
                    Err(e) => {
                        return make_error_response(&format!("server: tensor_to_wire: {}", e))
                    }
                },
                s_error: String::new(),
            };

            // Important: serialize response for transport
            bincode::serialize(&o_resp)
                .map_err(|_| "server: response encode fehlgeschlagen".to_string())
        })
    })
}

/// Builds handler that processes remote clear-cache requests.
/// - Input: state holder
/// - Output: ClearCacheHandler closure
/// - Behavior:
///   - retrieves current state reference
///   - resets only the cache for requesting peer
/// - Observability: prints far peer id for tracing
fn build_clear_cache_handler(
    o_state_holder: P2pServerStateHolder,
) -> crate::p2p_node::ClearCacheHandler {
    Arc::new(move |o_far_peer_id: PeerId| {
        let o_state_holder_inner = o_state_holder.clone();
        Box::pin(async move {
            // Section: read current state
            let o_state_ref: P2pServerStateRef = {
                let o_guard = o_state_holder_inner.lock().await;
                o_guard.clone()
            };

            // Important: operator log for debugging and audits
            println!(
                "build_clear_cache_handler for far peer-ID: {}",
                o_far_peer_id
            );

            // Section: reset per-peer cache only
            reset_server_cache_only(&o_state_ref, &o_far_peer_id).await
        })
    })
}

// ------------------------------------------------------------
// Section: Chat loop infrastructure
// Description
// RawModeGuard enables terminal raw mode for non-blocking key polling.
// Session id is a simple timestamp for possible future extensions.
// ------------------------------------------------------------

struct RawModeGuard;

impl RawModeGuard {
    /// Enables terminal raw mode and returns a guard that disables it on drop.
    /// - Output: RawModeGuard on success
    /// - Error: returns String on terminal errors
    /// - Robustness: uses RAII to ensure raw mode is restored
    fn enable() -> Result<Self, String> {
        terminal::enable_raw_mode().map_err(|e| e.to_string())?;
        Ok(Self)
    }
}

/// Builds a session id based on UNIX time (nanos).
/// - Output: string like "sess_123..."
/// - Note: not cryptographically unique, only operationally convenient
fn build_session_id() -> String {
    let i_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("sess_{}", i_now)
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        // Important: always attempt to disable raw mode when guard leaves scope
        let _ = terminal::disable_raw_mode();
    }
}

/// Sets an environment variable only if it is missing.
/// - Input: key and value
/// - Output: none
/// - Purpose: allows blocks_map.json to provide defaults without overriding operator settings
fn set_env_if_missing(s_key: &str, s_val: &str) {
    if std::env::var(s_key).is_err() {
        std::env::set_var(s_key, s_val);
    }
}

/// Applies blocks map settings into environment variables without overriding existing values.
/// - Input: blocks map reference
/// - Output: Ok on success, Err on missing model settings usage issues
/// - Behavior:
///   - sets global parameters
///   - sets per-model settings for MODEL_NAME, if present
fn apply_blocks_map_to_env(o_map: &crate::p2p_blocks_map::BlocksMap) -> Result<(), String> {
    // Section: global parameters
    for (s_key, s_val) in &o_map.map_parameters {
        set_env_if_missing(s_key, s_val);
    }

    // Section: model specific parameters
    let s_model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "llama".to_string());
    if let Some(o_m) = o_map.get_model_settings(&s_model_name) {
        if let Some(s_v) = o_m.CHAT_TEMPLATE {
            set_env_if_missing("CHAT_TEMPLATE", &s_v);
        }
        if let Some(s_v) = o_m.STOP {
            set_env_if_missing("STOP", &s_v);
        }

        if let Some(s_v) = o_m.TOKENIZER_JSON {
            set_env_if_missing("TOKENIZER_JSON", &s_v);
        }
        if let Some(s_v) = o_m.LLAMA_WEIGHTS {
            set_env_if_missing("LLAMA_WEIGHTS", &s_v);
        }
        if let Some(s_v) = o_m.LLAMA_CONFIG {
            set_env_if_missing("LLAMA_CONFIG", &s_v);
        }
        if let Some(s_v) = o_m.LLAMA_DTYPE {
            set_env_if_missing("LLAMA_DTYPE", &s_v);
        }
    }

    Ok(())
}

/// Resets cache entries for all remote peers known in blocks_map.
/// - Input: state holder, blocks map reference
/// - Output: Ok on completion, Err only if acquiring state reference fails
/// - Observability: prints per-peer progress and a summary
/// - History:
///   - 2026-01-03 Marcus Schlieper: initial, robust iteration and parsing
pub async fn reset_server_cache_for_all_remote_peers_from_blocks_map(
    o_state_holder: P2pServerStateHolder,
    o_blocks_map: &crate::p2p_blocks_map::BlocksMap,
) -> Result<(), String> {
    // Section: read current state reference
    let o_state_ref: P2pServerStateRef = {
        let o_guard = o_state_holder.lock().await;
        o_guard.clone()
    };

    let s_self_peer_id = o_blocks_map.s_self_peer_id.clone();

    let mut i_ok: usize = 0;
    let mut i_skip: usize = 0;
    let mut i_err: usize = 0;

    // Section: iterate known peers from blocks_map
    for (o_peer_id, _addr) in &o_blocks_map.map_peer_addr {
        let s_peer_id = o_peer_id.trim().to_string();

        println!("Clear Peer: {}", s_peer_id);

        if s_peer_id.is_empty() {
            i_err += 1;
            continue;
        }

        // Skip self to avoid unintended local context loss
        if s_peer_id == s_self_peer_id {
            i_skip += 1;
            continue;
        }

        // Parse peer id into libp2p PeerId
        let o_far_peer_id: libp2p::PeerId = match s_peer_id.parse::<libp2p::PeerId>() {
            Ok(v) => v,
            Err(_) => {
                i_err += 1;
                continue;
            }
        };

        // Reset cache for this peer
        if let Err(_e) = reset_server_cache_only(&o_state_ref, &o_far_peer_id).await {
            i_err += 1;
            continue;
        }

        i_ok += 1;
    }

    // Important: summary for operator diagnostics
    println!(
        "reset_server_cache_for_all_remote_peers_from_blocks_map: ok={} skip_self={} err={}",
        i_ok, i_skip, i_err
    );

    Ok(())
}

/// Clears local chat context and resets server-side caches.
/// - Input:
///   - ctx_ids: mutable local token context
///   - o_p2p_rt_opt: optional P2P runtime (None means local-only mode)
///   - o_state_holder: server state holder for cache operations
/// - Output: none (prints status)
/// - History:
///   - 2026-01-03 Marcus Schlieper: initial, ctx clear plus local reset plus remote broadcast
pub async fn clear_all_caches(
    ctx_ids: &mut Vec<u32>,
    o_p2p_rt_opt: Option<crate::p2p_node::P2pRuntime>,
    o_state_holder: P2pServerStateHolder,
) {
    // Section 1: clear local context
    ctx_ids.clear();

    // Section 2: if P2P inactive, stop here
    let o_rt = match &o_p2p_rt_opt {
        Some(v) => v,
        None => {
            println!("clear: context reset aktiv (p2p nicht aktiv)");
            return;
        }
    };

    // Section 3: read current server state
    let o_state_ref: P2pServerStateRef = {
        let o_guard = o_state_holder.lock().await;
        o_guard.clone()
    };

    // Section 4: reset self cache
    let o_self_peer_id = o_rt.o_self_peer_id;
    if let Err(e) = reset_server_cache_only(&o_state_ref, &o_self_peer_id).await {
        println!("clear: cache reset fehlgeschlagen: {}", e);
        return;
    }

    // Section 5: notify remote peers (best effort)
    if let Err(e) = crate::p2p_node::send_clear_cache_to_all(o_rt).await {
        println!("clear: broadcast fehlgeschlagen: {}", e);
    }

    println!("clear: context reset und cache reset aktiv");
}

/// Interactive chat loop:
/// - reads user input
/// - processes commands
/// - builds prompts (initial or follow-up)
/// - generates tokens until stop sequence or MAX_NEW or Space abort
/// - Input:
///   - tok: tokenizer instance
///   - ctx_ids: initial context ids (can be empty)
///   - mdl: backend implementing LmBackend
///   - fmt: chat template
///   - o_p2p_rt_opt: optional runtime (for peers/connect)
///   - o_state_holder: server state holder (for clear)
/// - Output: Ok on normal termination, Err on terminal errors or tokenizer errors
async fn chat_loop(
    tok: tokenizer::GgufTokenizer,
    mut ctx_ids: Vec<u32>,
    mut mdl: Box<dyn LmBackend>,
    fmt: ChatTemplate,
    o_p2p_rt_opt: Option<crate::p2p_node::P2pRuntime>,
    o_state_holder: P2pServerStateHolder,
) -> Result<(), String> {
    // Section: enable raw mode for non-blocking key polling during generation
    let _o_raw_guard = RawModeGuard::enable()?;

    // Section: initial UI
    print_menu();
    println!("Chat gestartet. Tippe help fuer menu.");

    // Section: stop sequences setup
    let v_stop_str = env_or_default_stops(&tok, fmt);
    let v_stop_ids = compile_stop_id_sequences(&tok, &v_stop_str);

    let mut s_input = String::new();

    loop {
        // Section: temporarily disable raw mode for line input
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

        // Section: command handling
        if s_line_lc == "exit" {
            println!("Tschuess");
            break;
        }
        if s_line_lc == "help" {
            print_menu();
            continue;
        }
        if s_line_lc == "clear" {
            clear_all_caches(&mut ctx_ids, o_p2p_rt_opt.clone(), o_state_holder.clone()).await;
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

                // Important: send connect request to swarm control channel
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

        // Section: prompt construction (first turn vs follow-up turn)
        if ctx_ids.is_empty() {
            // Important: push BOS if tokenizer defines it
            if let Some(bos) = tok.bos_id() {
                ctx_ids.push(bos);
            }

            // System prompt is read at runtime to allow ENV override without recompilation
            let s_sys = std::env::var("SYSTEM_PROMPT")
                .unwrap_or_else(|_| "You are a helpful assistant.".to_string());

            let s_first = build_first_turn(fmt, &tok, Some(&s_sys), s_line);

            // Important: encode with add_special_tokens=true for first turn
            let ids = tok.encode(&s_first, true)?;
            ctx_ids.extend_from_slice(&ids);
        } else {
            let s_next = build_next_turn(fmt, s_line);

            // Important: encode follow-up without special tokens in most formats
            let ids = tok.encode(&s_next, false)?;
            ctx_ids.extend_from_slice(&ids);
        }

        // Section: token generation loop
        let mut pending: Vec<u32> = Vec::new();
        let mut s_printed = String::new();
        let mut printed_any = false;

        // Metrics: measure per assistant response generation
        // - i_gen_tokens: number of tokens generated in this response
        // - o_t0: start time
        let mut i_gen_tokens: usize = 0;
        let o_t0 = Instant::now();

        for _step in 0..env_usize("MAX_NEW", 64) {
            // Section: user abort via Space key
            if is_abort_space_pressed()? {
                println!();
                println!("[abbruch durch space]");
                println!();
                break;
            }

            // Section: forward and logits
            let v_logits = match mdl.forward_tokens(&ctx_ids).await {
                Ok(v) => v,
                Err(s_err) => {
                    // Important: treat backend errors as recoverable for interactive UX
                    println!();
                    println!("[p2p fehler, ignored] {}", s_err);
                    println!();
                    break;
                }
            };

            // Section: select next token (greedy decoding)
            let next_idx = pick_top1(&v_logits, env_f32("TEMP", 0.8)) as u32;

            // Important: append token to context and pending buffer
            ctx_ids.push(next_idx);
            pending.push(next_idx);

            // Metrics: count generated tokens (one per loop iteration)
            i_gen_tokens += 1;

            // Section: streaming output by delta decoding
            let s_all = tok.decode(&pending, true).unwrap_or_default();
            let i_cp = common_prefix_bytes(&s_printed, &s_all);
            let new_bytes = &s_all.as_bytes()[i_cp..];

            if !new_bytes.is_empty() {
                if let Ok(new_str) = std::str::from_utf8(new_bytes) {
                    // Important: print without newline to simulate streaming
                    print!("{}", new_str);
                    io::stdout().flush().map_err(|e| e.to_string())?;
                    printed_any = true;
                }
            }
            s_printed = s_all;

            // Section: stop sequences in token space
            for seq in &v_stop_ids {
                let n = seq.len();
                if n == 0 || ctx_ids.len() < n {
                    continue;
                }
                let tail = &ctx_ids[ctx_ids.len() - n..];

                // Important: token-space stop match avoids issues with partial UTF-8 output
                if tail == seq.as_slice() {
                    println!();
                    println!();

                    // Reset pending decode state for next output phase
                    pending.clear();
                    s_printed.clear();
                    break;
                }
            }
        }

        // Metrics: finalize and print after generation finished
        let d_elapsed_s: f64 = o_t0.elapsed().as_secs_f64();
        let d_tps: f64 = if d_elapsed_s > 0.0 {
            (i_gen_tokens as f64) / d_elapsed_s
        } else {
            0.0
        };

        println!("");
        // Print metrics in a compact, operator friendly block
        println!("------------------------------------------------------------");
        println!("metrics token_count: {}", i_gen_tokens);
        println!("metrics tokens_per_sec: {:.3}", d_tps);
        println!("metrics total_duration_sec: {:.3}", d_elapsed_s);
        println!("------------------------------------------------------------");

        if !printed_any {
            println!("[kein token erzeugt]");
        }
    }

    Ok(())
}

// ------------------------------------------------------------
// Section: main
// Description
// Entry point:
// - loads blocks_map and applies configuration into ENV
// - initializes peer identity and P2P runtime
// - builds server state holder and handlers
// - optional auto connect when BACKEND=p2p
// - loads tokenizer
// - builds backend (local or p2p)
// - starts chat loop
// ------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), String> {
    let s_blocks_map_file =
        std::env::var("BLOCKS_MAP").unwrap_or_else(|_| "blocks_map.json".to_string());
    let s_peer_key_file =
        std::env::var("PEER_KEY_FILE").unwrap_or_else(|_| "peer_key.bin".to_string());

    println!("1");
    let o_blocks_map = crate::p2p_blocks_map::BlocksMap::from_file(&s_blocks_map_file)?;
    apply_blocks_map_to_env(&o_blocks_map)?;
    println!("2");

    let i_p2p_port = env_u16("P2P_PORT", 4001);

    let (o_keypair, o_peer_id, _o_map) =
        crate::p2p_identity::init_peer_identity(&s_blocks_map_file, &s_peer_key_file)?;
    println!("peer id: {}", o_peer_id);
    println!("3");

    let (o_p2p_rt, o_swarm, o_rx, o_ctl_rx) =
        crate::p2p_node::build_runtime(i_p2p_port, o_keypair)?;
    let o_connected_peers = o_p2p_rt.o_connected_peers.clone();
    println!("4");

    // New: build both backend and server state in one call
    let (mut o_backend, o_server_state) = build_backend(o_peer_id, &o_blocks_map)?;
    let o_state_holder: P2pServerStateHolder = Arc::new(Mutex::new(o_server_state));
    println!("5");

    let o_server_handler: crate::p2p_node::ServerHandler =
        build_p2p_server_handler(o_state_holder.clone());
    println!("6");

    crate::p2p_runtime_handle::set_p2p_runtime(o_p2p_rt.clone())?;

    let o_clear_cache_handler = build_clear_cache_handler(o_state_holder.clone());
    println!("7");

    crate::p2p_node::spawn_swarm_task(
        o_swarm,
        o_rx,
        o_ctl_rx,
        o_server_handler,
        o_connected_peers,
        o_clear_cache_handler,
    );
    println!("p2p peer id: {}", o_p2p_rt.o_self_peer_id);
    println!("p2p listen tcp port: {}", i_p2p_port);
    println!("8");

    // NOTE: keep blocks_map reload if needed, otherwise reuse o_blocks_map
    let o_blocks_map2 = crate::p2p_blocks_map::BlocksMap::from_file(&s_blocks_map_file)?;
    if std::env::var("BACKEND").unwrap_or_else(|_| "local".to_string()) == "p2p" {
        auto_connect_route_peers(&o_p2p_rt, &o_blocks_map2).await?;
    }
    println!("9");

    let s_tok_json = std::env::var("TOKENIZER_JSON")
        .map_err(|_| "Bitte TOKENIZER_JSON auf tokenizer.json setzen".to_string())?;
    let o_tok = tokenizer::load_tokenizer_from_json_force(&s_tok_json)
        .map_err(|e| format!("Tokenizer Load Fehler: {}", e))?;
    println!("10");

    println!("Vokabular (Backend): {}", o_backend.vocab_size());

    let e_fmt = detect_chat_template(&o_tok);
    println!("11");

    let o_p2p_rt_opt = Some(o_p2p_rt.clone());

    let ctx_ids: Vec<u32> = Vec::new();
    println!("12");

    chat_loop(
        o_tok,
        ctx_ids,
        o_backend,
        e_fmt,
        o_p2p_rt_opt,
        o_state_holder,
    )
    .await
}
