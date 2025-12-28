// src/p2p_blocks_map.rs
// ------------------------------------------------------------
// Blocks Map plus Peer Adressen
//
// Ziel
// - blocks_map.json enthaelt peers (peer_id plus addr) und routes
// - benoetigte peers fuer ein model werden berechnet
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: peers und auto connect basis
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
pub struct BlocksMapFile {
    pub self_peer_id: String,
    pub peers: Vec<PeerEntry>,
    pub routes: Vec<BlockRoute>,
}

#[derive(Debug, Clone)]
pub struct BlocksMap {
    pub s_self_peer_id: String,
    pub map_peer_addr: HashMap<String, String>,
    pub map_routes: HashMap<(String, usize), String>,
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

        Ok(Self {
            s_self_peer_id: o_file.self_peer_id,
            map_peer_addr,
            map_routes,
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
}
