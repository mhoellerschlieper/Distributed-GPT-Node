// src/web_state.rs
// ------------------------------------------------------------
// Web State
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: session state und history
// ------------------------------------------------------------
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub s_user: String,
    pub s_assistant: String,
}

#[derive(Clone)]
pub struct WebAppState {
    pub o_chat_history: Arc<Mutex<HashMap<String, Vec<ChatTurn>>>>,
    pub o_p2p_rt: crate::p2p_node::P2pRuntime,
    pub o_chat_state: Arc<Mutex<crate::ChatState>>,
}

impl WebAppState {
    pub fn new(o_p2p_rt: crate::p2p_node::P2pRuntime, o_chat_state: crate::ChatState) -> Self {
        Self {
            o_chat_history: Arc::new(Mutex::new(HashMap::new())),
            o_p2p_rt,
            o_chat_state: Arc::new(Mutex::new(o_chat_state)),
        }
    }
}
