// src/p2p_blocks_map.rs
// ------------------------------------------------------------
// Blocks Map plus peer list plus parameters and models
//
// Ziel
// - blocks_map json enthaelt peers, routes, parameters und models
// - parameters und model settings koennen env variablen ersetzen
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: peers und routes basis
// - 2025-12-28 Marcus Schlieper: parameters und models hinzugefuegt
// ------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerEntry {
    pub peer_id: String,
    pub addr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRoute {
    pub model_name: String,
    pub block_no: usize,
    pub peer_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    pub model_name: String,

    pub CHAT_TEMPLATE: Option<String>,
    pub STOP: Option<String>,

    pub TOKENIZER_JSON: Option<String>,
    pub LLAMA_WEIGHTS: Option<String>,
    pub LLAMA_CONFIG: Option<String>,
    pub LLAMA_DTYPE: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlocksMapFile {
    pub self_peer_id: String,

    pub peers: Vec<PeerEntry>,
    pub routes: Vec<BlockRoute>,

    pub parameters: Option<HashMap<String, serde_json::Value>>,
    pub models: Option<Vec<ModelSettings>>,
}

#[derive(Debug, Clone)]
pub struct BlocksMap {
    pub s_self_peer_id: String,
    pub map_peer_addr: HashMap<String, String>,
    pub map_routes: HashMap<(String, usize), String>,

    pub map_parameters: HashMap<String, String>,
    pub v_models: Vec<ModelSettings>,
}

impl BlocksMap {
    pub fn from_file(s_file: &str) -> Result<Self, String> {
        let s_raw = std::fs::read_to_string(s_file)
            .map_err(|e| format!("blocks_map lesen fehlgeschlagen: {}", e))?;

        let o_file: BlocksMapFile =
            serde_json::from_str(&s_raw).map_err(|e| format!("blocks_map json parse: {}", e))?;

        let mut map_peer_addr: HashMap<String, String> = HashMap::new();
        for o_peer in &o_file.peers {
            if o_peer.peer_id.trim().is_empty() {
                return Err("blocks_map peers: peer_id ist leer".to_string());
            }
            if o_peer.addr.trim().is_empty() {
                return Err("blocks_map peers: addr ist leer".to_string());
            }
            map_peer_addr.insert(o_peer.peer_id.clone(), o_peer.addr.clone());
        }

        let mut map_routes: HashMap<(String, usize), String> = HashMap::new();
        for o_r in &o_file.routes {
            map_routes.insert((o_r.model_name.clone(), o_r.block_no), o_r.peer_id.clone());
        }

        let mut map_parameters: HashMap<String, String> = HashMap::new();
        if let Some(o_params) = &o_file.parameters {
            for (s_key, o_val) in o_params {
                let s_val = match o_val {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => {
                        if *b { "1".to_string() } else { "0".to_string() }
                    }
                    _ => {
                        return Err("blocks_map parameters: ungueltiger werttyp".to_string());
                    }
                };
                map_parameters.insert(s_key.clone(), s_val);
            }
        }

        let v_models = o_file.models.unwrap_or_default();

        Ok(Self {
            s_self_peer_id: o_file.self_peer_id,
            map_peer_addr,
            map_routes,
            map_parameters,
            v_models,
        })
    }

    pub fn get_peer_for_block(&self, s_model_name: &str, i_block_no: usize) -> Option<String> {
        self.map_routes
            .get(&(s_model_name.to_string(), i_block_no))
            .cloned()
    }

    pub fn get_addr_for_peer(&self, s_peer_id: &str) -> Option<String> {
        self.map_peer_addr.get(s_peer_id).cloned()
    }

    pub fn needed_peers_for_model(&self, s_model_name: &str) -> HashSet<String> {
        let mut set_out: HashSet<String> = HashSet::new();
        for ((s_m, _i_block), s_peer_id) in &self.map_routes {
            if s_m == s_model_name {
                set_out.insert(s_peer_id.clone());
            }
        }
        set_out
    }

    pub fn get_model_settings(&self, s_model_name: &str) -> Option<ModelSettings> {
        for o_m in &self.v_models {
            if o_m.model_name == s_model_name {
                return Some(o_m.clone());
            }
        }
        None
    }
}
