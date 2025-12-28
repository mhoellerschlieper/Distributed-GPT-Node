// src/p2p_identity.rs
// ------------------------------------------------------------
// Peer Identity: stabile PeerId ueber peer_key.bin und blocks_map.json
//
// Ziel
// - peer_key.bin laden
// - wenn peer_key.bin fehlt oder kaputt: neu erzeugen und speichern
// - blocks_map.json erstellen, falls fehlt
// - self_peer_id in blocks_map.json nur setzen, wenn "auto"
// - wenn self_peer_id nicht passt: fehler
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-27 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use libp2p::{identity, PeerId};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRoute {
    pub model_name: String,
    pub block_no: usize,
    pub peer_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlocksMapFile {
    pub self_peer_id: String,
    pub routes: Vec<BlockRoute>,
}

fn is_auto(s_val: &str) -> bool {
    let s_lc = s_val.trim().to_lowercase();
    s_lc.is_empty() || s_lc == "auto"
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

fn ensure_blocks_map_file_exists(s_blocks_map_file: &str) -> Result<(), String> {
    if !is_safe_file_name(s_blocks_map_file) {
        return Err("blocks_map file name ungueltig".to_string());
    }
    if Path::new(s_blocks_map_file).exists() {
        return Ok(());
    }

    let o_default = BlocksMapFile {
        self_peer_id: "auto".to_string(),
        routes: Vec::new(),
    };

    let s_out = serde_json::to_string_pretty(&o_default)
        .map_err(|e| format!("blocks_map serialize fehlgeschlagen: {}", e))?;

    std::fs::write(s_blocks_map_file, s_out)
        .map_err(|e| format!("blocks_map erstellen fehlgeschlagen: {}", e))?;

    Ok(())
}

fn load_blocks_map_file(s_blocks_map_file: &str) -> Result<BlocksMapFile, String> {
    let s_raw = std::fs::read_to_string(s_blocks_map_file)
        .map_err(|e| format!("blocks_map lesen fehlgeschlagen: {}", e))?;
    let o_map: BlocksMapFile =
        serde_json::from_str(&s_raw).map_err(|e| format!("blocks_map json parse: {}", e))?;
    Ok(o_map)
}

fn save_blocks_map_file(s_blocks_map_file: &str, o_map: &BlocksMapFile) -> Result<(), String> {
    let s_out = serde_json::to_string_pretty(o_map)
        .map_err(|e| format!("blocks_map serialize: {}", e))?;
    std::fs::write(s_blocks_map_file, s_out)
        .map_err(|e| format!("blocks_map schreiben fehlgeschlagen: {}", e))?;
    Ok(())
}

fn write_keypair_file(s_key_file: &str, o_key: &identity::Keypair) -> Result<(), String> {
    if !is_safe_file_name(s_key_file) {
        return Err("peer key file name ungueltig".to_string());
    }

    let v_out = o_key
        .to_protobuf_encoding()
        .map_err(|_| "peer key encode fehlgeschlagen".to_string())?;

    std::fs::write(s_key_file, v_out)
        .map_err(|e| format!("peer key schreiben fehlgeschlagen: {}", e))?;

    Ok(())
}

fn backup_bad_key_file(s_key_file: &str) -> Result<(), String> {
    let s_backup = format!("{}.bad", s_key_file);

    if Path::new(&s_backup).exists() {
        std::fs::remove_file(&s_backup)
            .map_err(|e| format!("peer key backup loeschen fehlgeschlagen: {}", e))?;
    }

    std::fs::rename(s_key_file, &s_backup)
        .map_err(|e| format!("peer key backup rename fehlgeschlagen: {}", e))?;

    Ok(())
}

pub fn load_or_create_keypair_file(s_key_file: &str) -> Result<identity::Keypair, String> {
    if Path::new(s_key_file).exists() {
        let v_bytes =
            std::fs::read(s_key_file).map_err(|e| format!("peer key lesen fehlgeschlagen: {}", e))?;

        match identity::Keypair::from_protobuf_encoding(&v_bytes) {
            Ok(o_key) => Ok(o_key),
            Err(_) => {
                backup_bad_key_file(s_key_file)?;
                let o_new = identity::Keypair::generate_ed25519();
                write_keypair_file(s_key_file, &o_new)?;
                Ok(o_new)
            }
        }
    } else {
        let o_key = identity::Keypair::generate_ed25519();
        write_keypair_file(s_key_file, &o_key)?;
        Ok(o_key)
    }
}

pub fn init_peer_identity(
    s_blocks_map_file: &str,
    s_key_file: &str,
) -> Result<(identity::Keypair, PeerId, BlocksMapFile), String> {
    ensure_blocks_map_file_exists(s_blocks_map_file)?;

    let o_key = load_or_create_keypair_file(s_key_file)?;
    let o_peer_id = PeerId::from(o_key.public());
    let s_peer_id = o_peer_id.to_string();

    let mut o_map = load_blocks_map_file(s_blocks_map_file)?;

    if is_auto(&o_map.self_peer_id) {
        o_map.self_peer_id = s_peer_id.clone();

        for o_r in o_map.routes.iter_mut() {
            if is_auto(&o_r.peer_id) {
                o_r.peer_id = s_peer_id.clone();
            }
        }

        save_blocks_map_file(s_blocks_map_file, &o_map)?;
    } else if o_map.self_peer_id.trim() != s_peer_id {
        return Err("blocks_map self_peer_id passt nicht zum peer key file".to_string());
    }

    Ok((o_key, o_peer_id, o_map))
}
