// src/p2p_wire.rs
// ------------------------------------------------------------
// P2P Wire Types fuer Block Run
//
// Ziel
// - session_id fuer cache pro session
// - multi block request fuer weniger peer wechsel
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: session_id und multi block request hinzugefuegt
// ------------------------------------------------------------

use serde::{Deserialize, Serialize};



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClearRequest {
    pub s_session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClearResponse {
    pub b_ok: bool,
    pub s_error: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBlockRequest {
    pub s_model_name: String,
    pub i_block_no: usize,
    pub i_pos: usize,
    pub s_session_id: String,
    pub o_x: WireTensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBlockResponse {
    pub o_y: WireTensor,
    pub s_error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBlocksRequest {
    pub s_model_name: String,
    pub v_block_nos: Vec<usize>,
    pub i_pos: usize,
    pub s_session_id: String,
    pub o_x: WireTensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBlocksResponse {
    pub o_y: WireTensor,
    pub s_error: String,
}

// ------------------------------------------------------------
// WireTensor und WireDType sind in dieser datei definiert
// ------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WireDType {
    F32,
    F16,
    BF16,
    I64,
    U32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensor {
    pub v_shape: Vec<usize>,
    pub e_dtype: WireDType,
    pub v_data: Vec<u8>,
}
