// src/web_api.rs
// ------------------------------------------------------------
// Web API Modelle
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: web api models
// ------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeersGetResponse {
    pub v_peers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeersStatsGetResponse {
    pub map_stats: HashMap<String, crate::p2p_node::PeerStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatHistoryGetResponse {
    pub map_history: HashMap<String, Vec<crate::web_state::ChatTurn>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSendRequest {
    pub s_session_id: String,
    pub s_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSendResponse {
    pub s_session_id: String,
    pub s_answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClearRequestWeb {
    pub s_session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OkResponse {
    pub b_ok: bool,
    pub s_error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemGetResponse {
    pub s_system_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSetRequest {
    pub s_system_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamsGetResponse {
    pub d_temp: f32,
    pub i_max_new: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamsSetRequest {
    pub d_temp: Option<f32>,
    pub i_max_new: Option<usize>,
}
