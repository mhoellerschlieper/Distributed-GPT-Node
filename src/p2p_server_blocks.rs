// src/p2p_server_blocks.rs
// ------------------------------------------------------------
// P2P Server: RunBlockRequest verarbeiten
//
// Ziel
// - request decode
// - forward_one_block lokal ausfuehren
// - response encode
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use std::sync::Arc;
use tokio::sync::Mutex;

use candle::Tensor;
use libp2p::PeerId;

use crate::local_llama::{Cache, LocalLlama};
use crate::p2p_tensor_conv::{tensor_to_wire, wire_to_tensor};
use crate::p2p_wire::{RunBlockRequest, RunBlockResponse};

pub struct P2pBlockServerState {
    pub o_model: LocalLlama,
    pub o_cache: Cache,
}

pub type P2pBlockServerStateRef = Arc<Mutex<P2pBlockServerState>>;

pub fn build_server_handler(o_state: P2pBlockServerStateRef) -> crate::p2p_node::ServerHandler {
    Arc::new(move |o_peer: PeerId, v_req: Vec<u8>| {
        let o_state = o_state.clone();
        Box::pin(async move {
            let o_req: RunBlockRequest = bincode::deserialize(&v_req)
                .map_err(|_| "server: bincode request decode fehlgeschlagen".to_string())?;

            // debug fuer dich: jetzt siehst du, dass wirklich was ankommt
            println!(
                "server got request peer={} model={} block_no={} pos={} bytes={}",
                o_peer,
                o_req.s_model_name,
                o_req.i_block_no,
                o_req.i_pos,
                v_req.len()
            );

            let o_x: Tensor = wire_to_tensor(&o_req.o_x)
                .map_err(|e| format!("server: wire_to_tensor fehlgeschlagen: {}", e))?;

            // block run
            let o_y = {
                let mut o_guard = o_state.lock().await;

                o_guard
                    .o_model
                    .forward_one_block_via_self(
                        &o_x,
                        o_req.i_pos,
                        o_req.i_block_no,
                        &mut o_guard.o_cache.clone(),
                    )
                    .map_err(|e| format!("server: forward_one_block fehlgeschlagen: {}", e))?
            };

            let o_resp = RunBlockResponse {
                o_y: tensor_to_wire(&o_y).map_err(|e| format!("server: tensor_to_wire: {}", e))?,
                s_error: String::new(),
            };

            let v_out = bincode::serialize(&o_resp)
                .map_err(|_| "server: bincode response encode fehlgeschlagen".to_string())?;

            Ok(v_out)
        })
    })
}
