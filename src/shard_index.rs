// shard_index.rs
// ------------------------------------------------------------
// Laedt Safetensors Shards anhand einer index json Datei
// Ziel:
// - model.safetensors.index.json lesen
// - shard dateien sammeln
// - als sortierte liste von pfaden zurueckgeben
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-26 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SafetensorsIndexJson {
    weight_map: std::collections::HashMap<String, String>,
}

fn is_valid_shard_filename(s_name: &str) -> bool {
    // sehr konservativ: nur relative dateinamen ohne pfad trenner
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

pub fn load_shard_paths_from_index_json(s_index_json_path: &str) -> Result<Vec<String>, String> {
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
        let mut p = PathBuf::from(p_dir);
        p.push(&s_name);

        if !p.exists() {
            return Err(format!("shard datei fehlt: {}", p.display()));
        }

        let s_path = p.to_string_lossy().to_string();

        println!("{}",s_path);
        v_paths.push(s_path);
    }

    Ok(v_paths)
}
