// main.rs
// CPU-only GGUF v3 Inferenz: KV-Cache, Sampling (temp, top-k, top-p), RoPE, SwiGLU
// Autor: Marcus Schlieper, ExpChat.ai
// Kontakt: mschlieper@ylook.de | Tel: +49 2338 8748862 | Mobil: +49 151 15751864
// Firma: ExpChat.ai – Der KI Chat Client für den Mittelstand aus Breckerfeld im Sauerland.
//        RPA, KI Agents, KI Internet Research, KI Wissensmanagement.
//        Wir bringen KI in den Mittelstand. Adresse: Epscheider Str21, 58339 Breckerfeld
//
// Hinweise:
// - Modellrepräsentation und Gewichts-Mapping: model.rs
// - Layer/Mathe/Sampling: layer.rs, math.rs
// - GGUF-Lader: gguf_loader.rs
// - Tokenizer (Unigram + BPE aus GGUF): tokenizer.rs
//
// Sicherheit: kein unsafe, durchgehende Fehlerbehandlung per Result
// Starten (Beispiele):
//   PowerShell: $env:MODEL_DEBUG="0"; $env:PROMPT_TPL="simple"; cargo run --release
//   Tests:      cargo test
//
// Optional-Checks (per ENV):
//   - RUST_DECODE_TEST=1   -> kleiner Encode/Decode-Test, beendet dann.
//   - RUST_LLAMA_CHECK=1   -> kompakter Modell-Check wie im Python-Beispiel, beendet dann.
//   - MODEL_PATH=<pfad.gguf>
//   - ROPE_THETA=<float> (kann in build_config berücksichtigt werden, je nach Implementierung)

mod gguf_loader;
mod layer;
mod math;
mod model;
mod tokenizer;

use gguf_loader::{GgufModel, GgufValue, load_gguf};
use layer::TransformerModel;
use math::{SimpleRng, sample_top_k_top_p_temperature};
use model::{
    build_config, count_layers_in_tensors, init_debug_from_env, map_all_weights, mean_abs,
};
use tokenizer::{GgufTokenizer, gguf_tokenizer_from_kv};

use std::collections::HashMap;
use std::io::{self, Write};

#[cfg(test)]
mod tests;

// Einfaches Chat-Template
#[derive(Clone, Copy, Debug)]
enum ChatTpl {
    ChatMl, // <|system|> ... <|user|> ... <|assistant|>
    ImTags, // <|im_start|> ... <|im_end|>
    Inst,   // [INST] <<SYS>> ... </SYS>> ... [/INST]
    Simple, // System: ... \n User: ... \n Assistant:
}

// Template-Auswahl: per ENV PROMPT_TPL = simple | inst | chatml | im
fn tpl_from_env_or_default() -> ChatTpl {
    match std::env::var("PROMPT_TPL")
        .unwrap_or_else(|_| "simple".to_string())
        .to_lowercase()
        .as_str()
    {
        "chatml" => ChatTpl::ChatMl,
        "im" => ChatTpl::ImTags,
        "inst" => ChatTpl::Inst,
        _ => ChatTpl::Simple,
    }
}

// Prompt bauen
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
    }
}

// Steuerzeichen sichtbar machen (nur für Debug-Ausgaben)
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

// kleine ENV-Helper
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

// Prüft, ob alle benötigten Template-Token im Vokabular vorhanden sind
fn vocab_has_all(kv: &HashMap<String, GgufValue>, need: &[&str]) -> bool {
    let Some(GgufValue::ArrStr(tokens)) = kv.get("tokenizer.ggml.tokens") else {
        return false;
    };
    need.iter().all(|t| tokens.iter().any(|x| x == *t))
}

// Encode/Decode Test (Rust): "Hello world!"
// Nutzung: $env:RUST_DECODE_TEST="1"; cargo run --release
fn run_encode_decode_test(s_model_path: &str) -> Result<(), String> {
    println!("\n== Encode/Decode Test (Rust) ==");
    let o_gguf = load_gguf(s_model_path)?;
    let tok = gguf_tokenizer_from_kv(&o_gguf.kv)?;
    let s_text = "Hello world!";

    // 1) Ohne Special-Tokens
    let v_ids_plain = tok.encode(s_text, false)?;
    println!("Input: {}", s_text);
    println!("Tokens (no special): {:?}", v_ids_plain);
    let s_back_plain = tok
        .decode(&v_ids_plain, true)
        .map_err(|e| format!("decode failed: {}", e))?;
    println!("Decoded (skip special): {}", s_back_plain);

    // 2) Mit Special-Tokens (falls gesetzt)
    let v_ids_special = tok.encode(s_text, true)?;
    println!("Tokens (with special): {:?}", v_ids_special);
    let s_back_special = tok
        .decode(&v_ids_special, true)
        .map_err(|e| format!("decode failed: {}", e))?;
    println!("Decoded (skip special): {}", s_back_special);

    // 3) Per-Token-Decode (hilft bei ByteLevel-BPE)
    print!("Per-token decode: ");
    for &id in &v_ids_plain {
        let piece = tok
            .decode(&[id], true)
            .unwrap_or_else(|_| "".to_string())
            .replace('\n', "\\n");
        print!("[{}:{}] ", id, piece);
    }
    println!();

    Ok(())
}

// Kompakter Info-/Token-Check ähnlich Python-Demo
// Nutzung: $env:RUST_LLAMA_CHECK="1"; cargo run --release
fn run_rust_llama_check() -> Result<(), String> {
    use tokenizer::gguf_tokenizer_from_kv;

    let default_model = r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    let s_model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| default_model.to_string());
    let n_ctx_runtime = env_usize("N_CTX", 2048);

    println!("Lade Modell: {}", s_model_path);
    let gguf = load_gguf(&s_model_path)?;
    let tok = gguf_tokenizer_from_kv(&gguf.kv)?;

    // Infos aus KV (robust)
    let architecture = get_kv_str_multi(&gguf.kv, &["general.architecture"])
        .unwrap_or_else(|| "unknown".to_string());
    let tokenizer_model = get_kv_str_multi(&gguf.kv, &["tokenizer.ggml.model", "tokenizer.model"]);
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
    let n_layers = get_kv_usize_multi(
        &gguf.kv,
        &[
            "block_count",
            &format!("{}.block_count", architecture),
            "llama.block_count",
        ],
    );
    let hidden_size = get_kv_usize_multi(
        &gguf.kv,
        &[
            "embedding_length",
            "hidden_size",
            &format!("{}.embedding_length", architecture),
            "llama.embedding_length",
        ],
    );
    let n_heads = get_kv_usize_multi(
        &gguf.kv,
        &[
            "attention.head_count",
            &format!("{}.attention.head_count", architecture),
            "llama.attention.head_count",
        ],
    );
    let n_kv_heads = get_kv_usize_multi(
        &gguf.kv,
        &[
            "attention.head_count_kv",
            &format!("{}.attention.head_count_kv", architecture),
            "llama.attention.head_count_kv",
        ],
    )
    .or(n_heads);
    let max_pos = get_kv_usize_multi(
        &gguf.kv,
        &[
            "context_length",
            &format!("{}.context_length", architecture),
            "llama.context_length",
        ],
    );
    let rope_theta = get_kv_f32_multi(
        &gguf.kv,
        &[
            "rope.theta",
            "rope.freq_base",
            &format!("{}.rope.freq_base", architecture),
        ],
    )
    .unwrap_or(10000.0);
    let rope_dim = get_kv_usize_multi(
        &gguf.kv,
        &[
            "rope.dimension_count",
            &format!("{}.rope.dimension_count", architecture),
            "llama.rope.dimension_count",
        ],
    );
    let bos_id = get_kv_usize_multi(
        &gguf.kv,
        &["tokenizer.ggml.bos_token_id", "tokenizer.ggml.bos_id"],
    );
    let eos_id = get_kv_usize_multi(
        &gguf.kv,
        &["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eos_id"],
    );
    let unk_id = get_kv_usize_multi(
        &gguf.kv,
        &["tokenizer.ggml.unknown_token_id", "tokenizer.ggml.unk_id"],
    );

    // JSON-ähnliche Ausgabe (ohne externe Crates)
    println!("{{");
    println!(
        "  \"model_path\": \"{}\",",
        s_model_path.replace('\\', "\\\\")
    );
    println!("  \"architecture\": \"{}\",", architecture);
    println!(
        "  \"tokenizer.model\": {},",
        tokenizer_model
            .as_ref()
            .map(|s| format!("\"{}\"", s))
            .unwrap_or("null".to_string())
    );
    println!("  \"vocab_size\": {},", vocab_size);
    println!(
        "  \"n_layers\": {},",
        n_layers
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!(
        "  \"n_heads\": {},",
        n_heads
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!(
        "  \"n_kv_heads\": {},",
        n_kv_heads
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!(
        "  \"hidden_size\": {},",
        hidden_size
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!(
        "  \"max_position_embeddings\": {},",
        max_pos
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!("  \"rope_theta\": {},", rope_theta);
    println!(
        "  \"rope_dim\": {},",
        rope_dim
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!(
        "  \"bos_id\": {},",
        bos_id.map(|v| v.to_string()).unwrap_or("null".to_string())
    );
    println!(
        "  \"eos_id\": {},",
        eos_id.map(|v| v.to_string()).unwrap_or("null".to_string())
    );
    println!(
        "  \"unk_id\": {},",
        unk_id
            .map(|v| format!("\"{}\"", v))
            .unwrap_or("null".to_string())
    );
    println!("  \"n_tokens_in_file\": null,");
    println!("  \"n_ctx_runtime\": {}", n_ctx_runtime);
    println!("}}");

    // Tokens 0..29
    println!("\n== Tokens (0..29) ==");
    for tid in 0..30usize {
        let piece = tok.decode(&[tid], true).unwrap_or_default();
        let shown = escape_token_piece(&piece);
        println!("{:>6} | {}", tid, shown);
    }

    // Encode/Decode-Test
    println!("\n== Encode/Decode Test ==");
    let text_in = "Hello world!";
    let ids = tok.encode(text_in, false)?;
    println!("Tokens: {:?}", ids);
    let text_out = tok.decode(&ids, true).unwrap_or_default();
    println!("Text: {}", text_out);

    Ok(())
}

fn get_kv_str_multi(kv: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(GgufValue::Str(s)) = kv.get(*k) {
            return Some(s.clone());
        }
    }
    None
}
fn get_kv_usize_multi(kv: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<usize> {
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
fn get_kv_f32_multi(kv: &HashMap<String, GgufValue>, keys: &[&str]) -> Option<f32> {
    for k in keys {
        match kv.get(*k) {
            Some(GgufValue::F32(v)) => return Some(*v),
            Some(GgufValue::F64(v)) => return Some(*v as f32),
            Some(GgufValue::U32(v)) => return Some(*v as f32),
            Some(GgufValue::U64(v)) => return Some(*v as f32),
            Some(GgufValue::I32(v)) => return Some(*v as f32),
            Some(GgufValue::I64(v)) => return Some(*v as f32),
            _ => {}
        }
    }
    None
}


// In main.rs, nach map_all_weights(...) einfügen:
fn hash64(v: &[f32]) -> u64 {
    // einfacher Fingerabdruck: Summe der quantisierten Werte
    let mut h: u64 = 0;
    for &x in v.iter().take(2048) { // klein halten
        let q = (x * 1_000_000.0).round() as i64;
        h = h.wrapping_add(q as u64);
        h = h.rotate_left(5);
    }
    h
}

fn print_layer_report(model: &layer::TransformerModel) {
    println!("\n== Rust Layer Report ==");
    for (i, blk) in model.blocks.iter().enumerate() {
        let q_m = model::mean_abs(&blk.attn.w_q.w);
        let k_m = model::mean_abs(&blk.attn.w_k.w);
        let v_m = model::mean_abs(&blk.attn.w_v.w);
        let o_m = model::mean_abs(&blk.attn.w_o.w);
        let w1_m = model::mean_abs(&blk.ffn.w1.w);
        let w3_m = model::mean_abs(&blk.ffn.w3.w);
        let w2_m = model::mean_abs(&blk.ffn.w2.w);

        println!(
            "L{} | Q:{:.6} K:{:.6} V:{:.6} O:{:.6} | W1:{:.6} G(W3):{:.6} W2:{:.6}",
            i, q_m, k_m, v_m, o_m, w1_m, w3_m, w2_m
        );

        println!(
            "    hash Q:{:016x} K:{:016x} V:{:016x} O:{:016x} | W1:{:016x} W3:{:016x} W2:{:016x}",
            hash64(&blk.attn.w_q.w),
            hash64(&blk.attn.w_k.w),
            hash64(&blk.attn.w_v.w),
            hash64(&blk.attn.w_o.w),
            hash64(&blk.ffn.w1.w),
            hash64(&blk.ffn.w3.w),
            hash64(&blk.ffn.w2.w)
        );

        // optional: Formen prüfen
        println!(
            "    shapes Q:{:?}->{:?} K:{:?}->{:?} V:{:?}->{:?} O:{:?}->{:?}",
            blk.attn.w_q.in_dim, blk.attn.w_q.out_dim,
            blk.attn.w_k.in_dim, blk.attn.w_k.out_dim,
            blk.attn.w_v.in_dim, blk.attn.w_v.out_dim,
            blk.attn.w_o.in_dim, blk.attn.w_o.out_dim
        );
    }
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

fn main() -> Result<(), String> {
    // Debug-Schalter aus Umgebungsvariable übernehmen
    init_debug_from_env();

    // Optionale Einzweck-Checks
    if env_bool("RUST_DECODE_TEST") {
        let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
            r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf".to_string()
        });
        return run_encode_decode_test(&path);
    }
    if env_bool("RUST_LLAMA_CHECK") {
        return run_rust_llama_check();
    }

    // Modell-Datei (ENV überschreibbar)
    let s_model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        // Beispielpfad (bitte anpassen)
        //r"C:\Entwicklung\rust\GPT-GGUF\model\vibethinker-1.5b-q8_0.gguf".to_string()
        r"C:\Entwicklung\rust\GPT-GGUF\model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf".to_string()
    });

    println!("Lade GGUF: {}", s_model_path);
    let o_gguf: GgufModel = load_gguf(&s_model_path)?;

    // Debug: Tensor-Typen-Verteilung grob ausgeben
    {
        use std::collections::BTreeMap;
        let mut seen: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for (name, t) in &o_gguf.tensors {
            seen.entry(t.type_code).or_default().push(name.clone());
        }
        println!("Tensor-Typen im Modell:");
        for (ty, names) in seen.iter() {
            let mut sample = names.clone();
            sample.sort();
            if sample.len() > 5 {
                sample.truncate(5);
            }
            println!(
                "  type_code={}  count={}  z.B.: {:?}",
                ty,
                names.len(),
                sample
            );
        }
    }

    // Konfiguration aus GGUF lesen und Modell anlegen
    let o_cfg = build_config(&o_gguf);
    // Optional: RoPE-Basis sichtbar loggen (theta)
    println!("Rope Base (theta)= {}", o_cfg.rope_base);
    println!(
        "Model config: layers={}, heads={}, hidden={}, vocab={}, ctx={}",
        o_cfg.n_layers, o_cfg.n_heads, o_cfg.hidden_size, o_cfg.vocab_size, o_cfg.max_seq_len
    );

    println!("arch = {:?}", o_gguf.get_kv_str("general.architecture"));
    println!(
        "tokenizer.model = {:?}",
        o_gguf.get_kv_str("tokenizer.ggml.model")
    );
    println!(
        "rope.scaling.type={:?} factor={:?}",
        o_cfg.rope_scaling_type, o_cfg.rope_scaling_factor
    );

    let mut o_model = TransformerModel::new_empty(o_cfg.clone());
    map_all_weights(&o_gguf, &mut o_model)?;
    println!("Gewichte gemappt.");

    // ===================================================================================
    //print_layer_report(&o_model);
    
    // ===================================================================================

    // Layer-Infos: erwartet (cfg), in GGUF gefunden, im Modell angelegt
    let i_layers_cfg = o_model.cfg.n_layers;
    let i_layers_model = o_model.blocks.len();
    let i_layers_gguf = count_layers_in_tensors(&o_gguf);
    println!(
        "Layer-Info: erwartet(cfg)={}, gguf_vorhanden={}, modell_angelegt={}",
        i_layers_cfg, i_layers_gguf, i_layers_model
    );

    // kleine Plausibilitätswerte
    println!("dbg |tok_emb|mean| = {:.6}", mean_abs(&o_model.tok_emb));
    println!("dbg |lm_head|mean| = {:.6}", mean_abs(&o_model.lm_head));
    println!(
        "dbg rope_dim = {}, head_dim = {}",
        o_model.cfg.rope_dim,
        o_model.cfg.hidden_size / o_model.cfg.n_heads
    );
    println!(
        "dbg heads = {}, kv_heads = {}",
        o_model.cfg.n_heads, o_model.cfg.n_kv_heads
    );
    if !o_model.blocks.is_empty() {
        println!(
            "dbg |blk0.w_q|mean| = {:.6}",
            mean_abs(&o_model.blocks[0].attn.w_q.w)
        );
        println!(
            "dbg |blk0.w1|mean|  = {:.6}",
            mean_abs(&o_model.blocks[0].ffn.w1.w)
        );
    }

    // Tokenizer aus GGUF-Metadaten (llama -> Unigram | gpt2 -> BPE)
    let tok: GgufTokenizer = gguf_tokenizer_from_kv(&o_gguf.kv)
        .map_err(|e| format!("Tokenizer aus GGUF konnte nicht gebaut werden: {}", e))?;

    // Template wählen (vor dem Prompt prüfen, ob nötige Tokens da sind)
    let mut tpl = tpl_from_env_or_default();
    let need = match tpl {
        ChatTpl::ImTags => vec!["<|im_start|>", "<|im_end|>"],
        ChatTpl::ChatMl => vec!["<|system|>", "<|user|>", "<|assistant|>"],
        _ => vec![],
    };
    if !need.is_empty() && !vocab_has_all(&o_gguf.kv, &need) {
        println!("Warnung: Template-Tokens fehlen im Vokabular. Fallback auf Simple.");
        tpl = ChatTpl::Simple;
    }

    // Sampling-Parameter (ENV überschreibbar)
    let d_temperature: f32 = env_f32("TEMP", 0.2);
    let i_top_k: usize = env_usize("TOP_K", 20);
    let d_top_p: f32 = env_f32("TOP_P", 0.90);
    let i_max_new: usize = env_usize("MAX_NEW", 200);
    let mut rng = SimpleRng::new(env_u64("SEED", 0x1234_5678));

    println!("Frag das Modell. Eingabe 'exit' zum Beenden.");
    println!(
        "Nutze Sampling: temp={}, top_k={}, top_p={}",
        d_temperature, i_top_k, d_top_p
    );
    println!("Hinweis: PROMPT_TPL=simple|inst|chatml|im (ENV) wählbar.");

    // Prompt-Schleife
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

        // Prompt bauen
        let s_prompt = " 2 + 2?";
        
        //build_prompt(tpl, "You are a helpful assistant.", s_line);
        println!(
            "dbg prompt (first 200): {}",
            visible_token(&s_prompt.chars().take(200).collect::<String>())
        );

       
        // Encode (hier: add_special=false; BOS manuell ergänzen, falls vorhanden und Simple)
        let mut v_ids: Vec<usize> = tok
            .encode(&s_prompt, false)
            .map_err(|e| format!("encode fehlgeschlagen: {}", e))?;

        /*if matches!(tpl, ChatTpl::Simple) {
            if let Some(bos) = tok.bos_id() {
                if v_ids.first().copied() != Some(bos) {
                    v_ids.insert(0, bos);
                }
            }
        }*/

        // KV-Cache reset
        o_model.reset_kv_cache();

        // Prompt vorwärts laufen
        let mut v_logits: Vec<f32> = Vec::new();
        for (i_pos, &tid) in v_ids.iter().enumerate() {
            v_logits = o_model.forward_next(tid, i_pos)?;
        }

        // Top-5 Logits (Debug)
        let mut v_pairs: Vec<(usize, f32)> = v_logits.iter().copied().enumerate().collect();
        v_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (id, logit)) in v_pairs.iter().take(5).enumerate() {
            let tok_str = tok.decode(&[*id], true).unwrap_or_default();
            println!("#{i} id={id} tok={:?} logit={:.4}", tok_str, logit);
        }

        // Generiere i_max_new Tokens (gepolstertes Decoding: nur Delta drucken)
        let mut v_gen_ids: Vec<usize> = Vec::new();
        let mut s_out_prev = String::new();

        for _ in 0..i_max_new {
            let i_next = sample_top_k_top_p_temperature(
                &v_logits,
                d_temperature,
                i_top_k,
                d_top_p,
                &mut rng,
            );
            v_ids.push(i_next);
            v_gen_ids.push(i_next);

            let i_pos = v_ids.len() - 1;
            v_logits = o_model.forward_next(i_next, i_pos)?;

            // Block-Decode und nur neues Suffix drucken (stabil für BPE)
            if let Ok(s_all) = tok.decode(&v_gen_ids, true) {
                if s_all.len() >= s_out_prev.len() {
                    let tail = s_all.get(s_out_prev.len()..).unwrap_or("");
                    if !tail.is_empty() {
                        print!("{}", tail);
                        io::stdout().flush().map_err(|e| e.to_string())?;
                    }
                    s_out_prev = s_all;
                }
            }

            // stop, falls EOS
            if let Some(eos) = tok.eos_id() {
                if i_next == eos {
                    break;
                }
            }
        }
        println!();
    }

    Ok(())
}
