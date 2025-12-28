// src/p2p_wire.rs
// ------------------------------------------------------------
// Wire Format fuer Block Run ueber libp2p
//
// Ziel
// - Request und Response als bincode structs
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-26 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WireDType {
    F32,
    F16,
    BF16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensor {
    pub v_shape: Vec<usize>,
    pub e_dtype: WireDType,
    pub v_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBlockRequest {
    pub s_model_name: String,
    pub i_block_no: usize,
    pub i_pos: usize,
    pub o_x: WireTensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBlockResponse {
    pub o_y: WireTensor,
    pub s_error: String,
}
