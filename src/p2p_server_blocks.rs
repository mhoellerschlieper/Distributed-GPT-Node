// src/p2p_server_blocks.rs
// ------------------------------------------------------------
// P2P Server Blocks: RunBlockRequest und RunBlocksRequest
//
// Ziel
// - cache pro session_id
// - mehrere blocks in einem request ausfuehren
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: session cache map und multi block handler
// ------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use candle::Tensor;
use libp2p::PeerId;

use crate::local_llama::{Cache, LocalLlama};
use crate::p2p_tensor_conv::{tensor_to_wire, wire_to_tensor};
use crate::p2p_wire::{RunBlockRequest, RunBlockResponse, RunBlocksRequest, RunBlocksResponse};

pub struct P2pServerState {
    pub o_model: Arc<LocalLlama>,
    pub map_session_cache: Arc<Mutex<HashMap<String, Cache>>>,
}

pub type P2pServerStateRef = Arc<P2pServerState>;

fn make_error_response_blocks(s_err: &str) -> Result<Vec<u8>, String> {
    let o_dummy = Tensor::zeros((1, 1, 1), candle::DType::F32, &candle::Device::Cpu)
        .map_err(|e| format!("server: dummy tensor: {}", e))?;

    let o_resp = RunBlocksResponse {
        o_y: tensor_to_wire(&o_dummy).map_err(|e| format!("server: tensor_to_wire: {}", e))?,
        s_error: s_err.to_string(),
    };

    bincode::serialize(&o_resp).map_err(|_| "server: bincode response encode fehlgeschlagen".to_string())
}

fn make_error_response_block(s_err: &str) -> Result<Vec<u8>, String> {
    let o_dummy = Tensor::zeros((1, 1, 1), candle::DType::F32, &candle::Device::Cpu)
        .map_err(|e| format!("server: dummy tensor: {}", e))?;

    let o_resp = RunBlockResponse {
        o_y: tensor_to_wire(&o_dummy).map_err(|e| format!("server: tensor_to_wire: {}", e))?,
        s_error: s_err.to_string(),
    };

    bincode::serialize(&o_resp).map_err(|_| "server: bincode response encode fehlgeschlagen".to_string())
}

async fn get_or_create_cache_for_session(
    o_state: &P2pServerState,
    s_session_id: &str,
) -> Result<Cache, String> {
    if s_session_id.trim().is_empty() {
        return Err("server: session_id ist leer".to_string());
    }

    let mut map_guard = o_state.map_session_cache.lock().await;
    if let Some(o_cache) = map_guard.get(s_session_id) {
        return Ok(o_cache.clone());
    }

    // Hinweis: clone geht nur wenn Cache Clone ist. Wenn nicht, musst du Cache anders verwalten.
    // Falls Cache nicht Clone ist: aendere map_session_cache auf HashMap<String, Cache> mit take/insert pattern.
    return Err("server: cache clone nicht verfuegbar, bitte cache verwaltung anpassen".to_string());
}

pub fn build_p2p_server_handler(o_state: P2pServerStateRef) -> crate::p2p_node::ServerHandler {
    let o_state_outer = o_state.clone();

    Arc::new(move |_o_peer: PeerId, v_req: Vec<u8>| {
        let o_state_inner = o_state_outer.clone();

        Box::pin(async move {
            // Versuch 1: RunBlocksRequest
            if let Ok(o_req) = bincode::deserialize::<RunBlocksRequest>(&v_req) {
                let o_x = match wire_to_tensor(&o_req.o_x) {
                    Ok(v) => v,
                    Err(e) => return make_error_response_blocks(&format!("server: wire_to_tensor: {}", e)),
                };

                // Cache pro Session: hier brauchst du eine echte Cache Instanz pro session.
                // Minimal: du legst sie in der map an, wenn nicht vorhanden.
                let mut map_guard = o_state_inner.map_session_cache.lock().await;
                let o_cache = map_guard.get_mut(&o_req.s_session_id);
                if o_cache.is_none() {
                    return make_error_response_blocks("server: session cache fehlt, bitte init implementieren");
                }
                let o_cache_ref = o_cache.unwrap();

                let mut o_y = o_x;
                for i_block_no in &o_req.v_block_nos {
                    let o_llama = o_state_inner.o_model.inner_ref();
                    match o_llama.forward_one_block(&o_y, o_req.i_pos, *i_block_no, o_cache_ref) {
                        Ok(v) => o_y = v,
                        Err(e) => {
                            return make_error_response_blocks(&format!("server: forward_one_block: {}", e));
                        }
                    }
                }

                let o_resp = RunBlocksResponse {
                    o_y: match tensor_to_wire(&o_y) {
                        Ok(v) => v,
                        Err(e) => return make_error_response_blocks(&format!("server: tensor_to_wire: {}", e)),
                    },
                    s_error: String::new(),
                };

                return bincode::serialize(&o_resp)
                    .map_err(|_| "server: bincode response encode fehlgeschlagen".to_string());
            }

            // Versuch 2: RunBlockRequest (fallback)
            if let Ok(o_req) = bincode::deserialize::<RunBlockRequest>(&v_req) {
                let o_x = match wire_to_tensor(&o_req.o_x) {
                    Ok(v) => v,
                    Err(e) => return make_error_response_block(&format!("server: wire_to_tensor: {}", e)),
                };

                let mut map_guard = o_state_inner.map_session_cache.lock().await;
                let o_cache = map_guard.get_mut(&o_req.s_session_id);
                if o_cache.is_none() {
                    return make_error_response_block("server: session cache fehlt, bitte init implementieren");
                }
                let o_cache_ref = o_cache.unwrap();

                let o_llama = o_state_inner.o_model.inner_ref();
                let o_y = match o_llama.forward_one_block(&o_x, o_req.i_pos, o_req.i_block_no, o_cache_ref) {
                    Ok(v) => v,
                    Err(e) => return make_error_response_block(&format!("server: forward_one_block: {}", e)),
                };

                let o_resp = RunBlockResponse {
                    o_y: match tensor_to_wire(&o_y) {
                        Ok(v) => v,
                        Err(e) => return make_error_response_block(&format!("server: tensor_to_wire: {}", e)),
                    },
                    s_error: String::new(),
                };

                return bincode::serialize(&o_resp)
                    .map_err(|_| "server: bincode response encode fehlgeschlagen".to_string());
            }

            make_error_response_block("server: request decode fehlgeschlagen")
        })
    })
}
