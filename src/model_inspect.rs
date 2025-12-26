// src/model_inspect.rs
// ------------------------------------------------------------
// Modell Inspektion fuer Candle Safetensors Shards plus config.json
//
// Ziel
// - config.json laden und wichtige Felder drucken
// - safetensors (einzeln oder shards via index json) lesen
// - tensor liste, shapes, parameter summe berechnen
// - top 10 groesste tensoren ausgeben
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-26 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfigJson {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndexJson {
    weight_map: HashMap<String, String>,
}

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s_val) if s_val != "0")
}

fn is_valid_shard_filename(s_name: &str) -> bool {
    if s_name.is_empty() {
        return false;
    }
    if s_name.contains('/') || s_name.contains('\\') {
        return false;
    }
    if !s_name.ends_with(".safetensors") {
        return false;
    }
    true
}

fn load_shard_paths_from_index_json(s_index_json_path: &str) -> Result<Vec<String>, String> {
    let p_index = Path::new(s_index_json_path);
    if !p_index.exists() {
        return Err("index json datei fehlt".to_string());
    }

    let p_dir = p_index
        .parent()
        .ok_or_else(|| "index json hat kein parent verzeichnis".to_string())?;

    let s_raw = fs::read_to_string(p_index).map_err(|e| format!("index json lesen: {}", e))?;
    let o_idx: SafetensorsIndexJson =
        serde_json::from_str(&s_raw).map_err(|e| format!("index json parse: {}", e))?;

    let mut set_names: BTreeSet<String> = BTreeSet::new();
    for (_k, v_name) in o_idx.weight_map.iter() {
        if !is_valid_shard_filename(v_name) {
            return Err("ungueltiger shard dateiname in index json".to_string());
        }
        set_names.insert(v_name.to_string());
    }

    if set_names.is_empty() {
        return Err("keine shard dateien im index json".to_string());
    }

    let mut v_paths: Vec<String> = Vec::new();
    for s_name in set_names {
        let mut p_file = PathBuf::from(p_dir);
        p_file.push(&s_name);
        if !p_file.exists() {
            return Err(format!("shard datei fehlt: {}", p_file.display()));
        }
        v_paths.push(p_file.to_string_lossy().to_string());
    }

    Ok(v_paths)
}

pub fn build_weight_files(s_weights_path: &str) -> Result<Vec<String>, String> {
    if s_weights_path.to_lowercase().ends_with(".index.json") {
        return load_shard_paths_from_index_json(s_weights_path);
    }
    Ok(vec![s_weights_path.to_string()])
}

fn parse_config_json(s_config_path: &str) -> Result<LlamaConfigJson, String> {
    let v_bytes = fs::read(s_config_path).map_err(|e| format!("config json lesen: {}", e))?;
    let o_cfg: LlamaConfigJson =
        serde_json::from_slice(&v_bytes).map_err(|e| format!("config json parse: {}", e))?;
    Ok(o_cfg)
}

fn u128_to_string(i_val: u128) -> String {
    i_val.to_string()
}

fn human_bytes(i_bytes: u128) -> String {
    let d = i_bytes as f64;
    let kb = 1024.0;
    let mb = kb * 1024.0;
    let gb = mb * 1024.0;

    if d >= gb {
        format!("{:.2} gb", d / gb)
    } else if d >= mb {
        format!("{:.2} mb", d / mb)
    } else if d >= kb {
        format!("{:.2} kb", d / kb)
    } else {
        format!("{} b", i_bytes)
    }
}

fn dtype_size_bytes(s_dtype: &str) -> Option<u128> {
    match s_dtype.to_lowercase().as_str() {
        "f32" => Some(4),
        "f16" => Some(2),
        "bf16" => Some(2),
        _ => None,
    }
}

fn product_u128(v_dims: &[usize]) -> u128 {
    let mut i_prod: u128 = 1;
    for &d in v_dims {
        i_prod = i_prod.saturating_mul(d as u128);
    }
    i_prod
}

fn format_shape(v_dims: &[usize]) -> String {
    if v_dims.is_empty() {
        return "[]".to_string();
    }
    let mut s_out = String::from("[");
    for (i, d) in v_dims.iter().enumerate() {
        if i > 0 {
            s_out.push_str(", ");
        }
        s_out.push_str(&d.to_string());
    }
    s_out.push(']');
    s_out
}

#[derive(Debug, Clone)]
struct TensorInfo {
    s_name: String,
    v_shape: Vec<usize>,
    i_params: u128,
}

fn read_safetensors_metadata(s_file: &str) -> Result<Vec<TensorInfo>, String> {
    let v_bytes = fs::read(s_file).map_err(|e| format!("safetensors lesen: {}: {}", s_file, e))?;
    let o_st = safetensors::SafeTensors::deserialize(&v_bytes)
        .map_err(|e| format!("safetensors parse: {}: {}", s_file, e))?;

    let mut v_info: Vec<TensorInfo> = Vec::new();

    for s_name in o_st.names() {
        let o_view = o_st
            .tensor(s_name)
            .map_err(|e| format!("tensor view: {}: {}", s_name, e))?;

        let v_shape_u64 = o_view.shape();
        let mut v_shape: Vec<usize> = Vec::with_capacity(v_shape_u64.len());
        for &d in v_shape_u64 {
            v_shape.push(d as usize);
        }

        let i_params = product_u128(&v_shape);

        v_info.push(TensorInfo {
            s_name: s_name.to_string(),
            v_shape,
            i_params,
        });
    }

    Ok(v_info)
}

pub fn print_model_report_candle(s_weights_path: &str, s_config_path: &str) -> Result<(), String> {
    let o_cfg = parse_config_json(s_config_path)?;

    let i_kv = o_cfg.num_key_value_heads.unwrap_or(o_cfg.num_attention_heads);
    let d_rope = o_cfg.rope_theta.unwrap_or(10000.0);
    let b_tie = o_cfg.tie_word_embeddings.unwrap_or(false);

    println!("------------------------------------------------------------");
    println!("MODEL REPORT (candle)");
    println!("config file: {}", s_config_path);
    println!("weights path: {}", s_weights_path);
    println!("------------------------------------------------------------");
    println!("config summary");
    println!("hidden_size: {}", o_cfg.hidden_size);
    println!("intermediate_size: {}", o_cfg.intermediate_size);
    println!("num_hidden_layers: {}", o_cfg.num_hidden_layers);
    println!("num_attention_heads: {}", o_cfg.num_attention_heads);
    println!("num_key_value_heads: {}", i_kv);
    println!("vocab_size: {}", o_cfg.vocab_size);
    println!("max_position_embeddings: {}", o_cfg.max_position_embeddings);
    println!("rms_norm_eps: {}", o_cfg.rms_norm_eps);
    println!("rope_theta: {}", d_rope);
    println!("tie_word_embeddings: {}", b_tie);

    if let Some(o_rs) = &o_cfg.rope_scaling {
        let s_rt = o_rs.rope_type.clone().unwrap_or_else(|| "unknown".to_string());
        println!("rope_scaling: yes");
        println!("rope_scaling_type: {}", s_rt);
        println!("rope_scaling_factor: {}", o_rs.factor);
        println!("rope_scaling_low_freq_factor: {}", o_rs.low_freq_factor);
        println!("rope_scaling_high_freq_factor: {}", o_rs.high_freq_factor);
        println!(
            "rope_scaling_original_max_position_embeddings: {}",
            o_rs.original_max_position_embeddings
        );
    } else {
        println!("rope_scaling: no");
    }

    let v_files = build_weight_files(s_weights_path)?;
    println!("------------------------------------------------------------");
    println!("weight files");
    println!("count: {}", v_files.len());
    for (i, s_file) in v_files.iter().enumerate() {
        println!("file_{}: {}", i, s_file);
    }

    let s_dtype = std::env::var("LLAMA_DTYPE").unwrap_or_else(|_| "f32".to_string());
    let i_dtype_bytes = dtype_size_bytes(&s_dtype).unwrap_or(0);

    let mut map_unique: BTreeMap<String, TensorInfo> = BTreeMap::new();
    let mut i_total_params: u128 = 0;
    let mut i_total_tensors: u128 = 0;

    for s_file in &v_files {
        let v_ti = read_safetensors_metadata(s_file)?;
        for o_t in v_ti {
            i_total_tensors = i_total_tensors.saturating_add(1);

            if map_unique.contains_key(&o_t.s_name) {
                continue;
            }

            i_total_params = i_total_params.saturating_add(o_t.i_params);
            map_unique.insert(o_t.s_name.clone(), o_t);
        }
    }

    let i_unique_tensors = map_unique.len() as u128;
    let i_bytes_est = if i_dtype_bytes > 0 {
        i_total_params.saturating_mul(i_dtype_bytes)
    } else {
        0
    };

    println!("------------------------------------------------------------");
    println!("safetensors summary");
    println!("tensors_total_seen: {}", u128_to_string(i_total_tensors));
    println!("tensors_unique_by_name: {}", u128_to_string(i_unique_tensors));
    println!("params_total_unique: {}", u128_to_string(i_total_params));
    if i_bytes_est > 0 {
        println!("estimated_weight_bytes_from_dtype_env: {}", human_bytes(i_bytes_est));
        println!("dtype_env: {}", s_dtype);
    } else {
        println!("estimated_weight_bytes_from_dtype_env: unknown");
        println!("dtype_env: {}", s_dtype);
    }

    let mut v_all: Vec<TensorInfo> = map_unique.values().cloned().collect();
    v_all.sort_by(|a, b| b.i_params.cmp(&a.i_params));

    println!("------------------------------------------------------------");
    println!("top 10 tensors by params");
    let i_topn = 10usize.min(v_all.len());
    for i in 0..i_topn {
        let o_t = &v_all[i];
        println!(
            "rank: {} name: {} shape: {} params: {}",
            i + 1,
            o_t.s_name,
            format_shape(&o_t.v_shape),
            u128_to_string(o_t.i_params)
        );
    }

    if debug_on() {
        println!("------------------------------------------------------------");
        println!("all tensor names (debug)");
        for (i, o_t) in v_all.iter().enumerate() {
            println!(
                "idx: {} name: {} shape: {} params: {}",
                i,
                o_t.s_name,
                format_shape(&o_t.v_shape),
                u128_to_string(o_t.i_params)
            );
        }
    }

    println!("------------------------------------------------------------");
    Ok(())
}
