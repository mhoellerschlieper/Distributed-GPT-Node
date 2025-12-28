// src/p2p_runtime_handle.rs
// ------------------------------------------------------------
// Globaler Zugriff auf P2P Runtime (minimal)
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-27 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use tokio::sync::OnceCell;

use crate::p2p_node::P2pRuntime;
use libp2p::PeerId;

static G_P2P_RT: OnceCell<P2pRuntime> = OnceCell::const_new();

pub fn set_p2p_runtime(o_rt: P2pRuntime) -> Result<(), String> {
    G_P2P_RT
        .set(o_rt)
        .map_err(|_| "p2p runtime schon gesetzt".to_string())
}

pub async fn with_p2p_runtime(o_peer_id: PeerId, v_req: Vec<u8>) -> Result<Vec<u8>, String> {
    let o_rt = G_P2P_RT
        .get()
        .ok_or_else(|| "p2p runtime nicht gesetzt".to_string())?;
    crate::p2p_node::send_request_bytes(o_rt, o_peer_id, v_req).await
}
