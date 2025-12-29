// src/web_server.rs
// ------------------------------------------------------------
// Axum Web Server
//
// Aenderung
// - Clear im UI loescht nur den Tab Kontext: tab_id
// - Peer Neustart loescht alles: global_reset_id steigt
// - WebSocket send Nachricht traegt tab_id
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: api endpoints basis
// - 2025-12-28 Marcus Schlieper: api_chat_send voll implementiert
// - 2025-12-29 Marcus Schlieper: websocket chunk streaming 100ms
// - 2025-12-29 Marcus Schlieper: tab reset und peer restart reset
// ------------------------------------------------------------

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::p2p_wire::SessionClearRequest;
use crate::web_api::*;
use crate::web_state::*;
use tower_http::cors::{Any, CorsLayer};

// ------------------------------------------------------------
// Global Reset Zaehler
// - steigt bei jedem Peer Neustart (Prozess Start)
// - ein Tab merkt sich den letzten Wert
// - wenn Wert anders ist, wird der Kontext fuer diesen Tab verworfen
// ------------------------------------------------------------
static ATOMIC_GLOBAL_RESET_ID: AtomicU64 = AtomicU64::new(1);

fn get_global_reset_id() -> u64 {
    ATOMIC_GLOBAL_RESET_ID.load(Ordering::Relaxed)
}

// ------------------------------------------------------------
// Session Speicher fuer Token Kontext
// - key: session_id + tab_id
// ------------------------------------------------------------
#[derive(Default)]
pub struct SessionCtxStore {
    pub map_ctx_ids: HashMap<String, Vec<u32>>,
    pub map_last_reset_id: HashMap<String, u64>,
}

fn build_ctx_key(s_session_id: &str, s_tab_id: &str) -> String {
    format!("{}_{}", s_session_id.trim(), s_tab_id.trim())
}

// ------------------------------------------------------------
// App State
// ------------------------------------------------------------
#[derive(Clone)]
pub struct WebAppStateFull {
    pub o_chat_history: Arc<Mutex<HashMap<String, Vec<ChatTurn>>>>,
    pub o_chat_state: Arc<Mutex<crate::ChatState>>,

    pub o_p2p_rt: crate::p2p_node::P2pRuntime,

    pub o_tok: Arc<crate::tokenizer::GgufTokenizer>,
    pub e_fmt: crate::ChatTemplate,

    pub o_backend: Arc<Mutex<Box<dyn crate::LmBackend>>>,
    pub o_sessions: Arc<Mutex<SessionCtxStore>>,
}

// ------------------------------------------------------------
// WebSocket Protokoll
// Client -> Server
// - s_type: send
// - s_session_id: stabil pro browser
// - s_tab_id: pro tab eindeutig
// - i_reset_id: der tab speichert was er zuletzt sah
// ------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WsClientMsg {
    s_type: String,
    s_session_id: String,
    s_tab_id: String,
    s_text: String,
    i_reset_id: u64,
}

// Server -> Client
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WsServerMsg {
    s_type: String,
    s_session_id: String,
    s_tab_id: String,
    s_text: String,
    i_reset_id: u64,
}

fn ws_msg_start(s_session_id: &str, s_tab_id: &str, i_reset_id: u64) -> WsServerMsg {
    WsServerMsg {
        s_type: "start".to_string(),
        s_session_id: s_session_id.to_string(),
        s_tab_id: s_tab_id.to_string(),
        s_text: String::new(),
        i_reset_id,
    }
}

fn ws_msg_chunk(s_session_id: &str, s_tab_id: &str, s_text: &str, i_reset_id: u64) -> WsServerMsg {
    WsServerMsg {
        s_type: "chunk".to_string(),
        s_session_id: s_session_id.to_string(),
        s_tab_id: s_tab_id.to_string(),
        s_text: s_text.to_string(),
        i_reset_id,
    }
}

fn ws_msg_done(s_session_id: &str, s_tab_id: &str, i_reset_id: u64) -> WsServerMsg {
    WsServerMsg {
        s_type: "done".to_string(),
        s_session_id: s_session_id.to_string(),
        s_tab_id: s_tab_id.to_string(),
        s_text: String::new(),
        i_reset_id,
    }
}

fn ws_msg_error(s_session_id: &str, s_tab_id: &str, s_err: &str, i_reset_id: u64) -> WsServerMsg {
    WsServerMsg {
        s_type: "error".to_string(),
        s_session_id: s_session_id.to_string(),
        s_tab_id: s_tab_id.to_string(),
        s_text: s_err.to_string(),
        i_reset_id,
    }
}

fn ws_to_text(o_msg: &WsServerMsg) -> String {
    serde_json::to_string(o_msg).unwrap_or_else(|_| {
        "{\"s_type\":\"error\",\"s_session_id\":\"\",\"s_tab_id\":\"\",\"s_text\":\"encode failed\",\"i_reset_id\":0}".to_string()
    })
}

// ------------------------------------------------------------
// Router
// ------------------------------------------------------------
pub fn build_router(o_state: WebAppStateFull) -> Router {
    let o_cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/", get(ui_index))
        .route("/api/chat/ws", get(api_chat_ws))
        .route("/api/session/clear", post(api_session_clear))
        .layer(o_cors)
        .with_state(o_state)
}

async fn ui_index() -> impl IntoResponse {
    Html(include_str!("./static/chat.html"))
}

// ------------------------------------------------------------
// Helper
// ------------------------------------------------------------
fn ends_with_seq(v_ctx_ids: &[u32], v_seq: &[u32]) -> bool {
    if v_seq.is_empty() || v_ctx_ids.len() < v_seq.len() {
        return false;
    }
    let i_n = v_seq.len();
    &v_ctx_ids[v_ctx_ids.len() - i_n..] == v_seq
}

fn common_prefix_bytes(a: &str, b: &str) -> usize {
    let mut n = 0usize;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca == cb {
            n += ca.len_utf8();
        } else {
            break;
        }
    }
    n
}

async fn ws_send_json(o_socket: &mut WebSocket, o_msg: WsServerMsg) -> Result<(), String> {
    let s_txt = ws_to_text(&o_msg);
    o_socket
        .send(Message::Text(s_txt))
        .await
        .map_err(|_| "ws send failed".to_string())
}

// ------------------------------------------------------------
// WebSocket Endpoint
// - chunk streaming 100ms
// - tab reset: Clear loescht nur den key session_id + tab_id
// - peer restart reset: global_reset_id aendert sich, dann wird tab ctx ignoriert
// ------------------------------------------------------------
async fn api_chat_ws(
    ws: WebSocketUpgrade,
    State(o_state): State<WebAppStateFull>,
) -> impl IntoResponse {
    ws.on_upgrade(move |o_socket| ws_chat_session(o_socket, o_state))
}

async fn ws_chat_session(mut o_socket: WebSocket, o_state: WebAppStateFull) {
    let o_in = match o_socket.recv().await {
        Some(Ok(v)) => v,
        _ => return,
    };

    let s_payload = match o_in {
        Message::Text(s_val) => s_val,
        _ => {
            let _ = ws_send_json(
                &mut o_socket,
                ws_msg_error("", "", "ws text message required", get_global_reset_id()),
            )
            .await;
            return;
        }
    };

    let o_req: WsClientMsg = match serde_json::from_str(&s_payload) {
        Ok(v) => v,
        Err(_) => {
            let _ = ws_send_json(
                &mut o_socket,
                ws_msg_error("", "", "ws json parse failed", get_global_reset_id()),
            )
            .await;
            return;
        }
    };

    if o_req.s_type != "send" {
        let _ = ws_send_json(
            &mut o_socket,
            ws_msg_error(
                o_req.s_session_id.trim(),
                o_req.s_tab_id.trim(),
                "ws unknown type",
                get_global_reset_id(),
            ),
        )
        .await;
        return;
    }

    if o_req.s_session_id.trim().is_empty() || o_req.s_tab_id.trim().is_empty() {
        let _ = ws_send_json(
            &mut o_socket,
            ws_msg_error("", "", "ws session id or tab id empty", get_global_reset_id()),
        )
        .await;
        return;
    }

    if o_req.s_text.trim().is_empty() {
        let _ = ws_send_json(
            &mut o_socket,
            ws_msg_error(
                o_req.s_session_id.trim(),
                o_req.s_tab_id.trim(),
                "ws text empty",
                get_global_reset_id(),
            ),
        )
        .await;
        return;
    }

    let s_session_id = o_req.s_session_id.trim().to_string();
    let s_tab_id = o_req.s_tab_id.trim().to_string();
    let s_user_text = o_req.s_text.trim().to_string();
    let i_global_reset_id = get_global_reset_id();

    let _ = ws_send_json(
        &mut o_socket,
        ws_msg_start(&s_session_id, &s_tab_id, i_global_reset_id),
    )
    .await;

    // ctx key
    let s_ctx_key = build_ctx_key(&s_session_id, &s_tab_id);

    // ctx laden mit reset logik
    let mut v_ctx_ids: Vec<u32> = {
        let mut o_sess = o_state.o_sessions.lock().await;

        let i_last = o_sess
            .map_last_reset_id
            .get(&s_ctx_key)
            .cloned()
            .unwrap_or(0);

        if i_last != i_global_reset_id {
            o_sess.map_ctx_ids.remove(&s_ctx_key);
            o_sess.map_last_reset_id.insert(s_ctx_key.clone(), i_global_reset_id);
            Vec::new()
        } else {
            o_sess
                .map_ctx_ids
                .get(&s_ctx_key)
                .cloned()
                .unwrap_or_else(Vec::new)
        }
    };

    // prompt
    let s_system_prompt = {
        let o_cs = o_state.o_chat_state.lock().await;
        o_cs.s_system_prompt.clone()
    };

    let b_is_first = v_ctx_ids.is_empty();
    if b_is_first {
        if let Some(i_bos) = o_state.o_tok.bos_id() {
            v_ctx_ids.push(i_bos);
        }
        let s_first = crate::build_first_turn(
            o_state.e_fmt,
            &o_state.o_tok,
            Some(&s_system_prompt),
            &s_user_text,
        );
        let v_ids = match o_state.o_tok.encode(&s_first, true) {
            Ok(v) => v,
            Err(e) => {
                let _ = ws_send_json(
                    &mut o_socket,
                    ws_msg_error(
                        &s_session_id,
                        &s_tab_id,
                        &format!("tokenize failed {}", e),
                        i_global_reset_id,
                    ),
                )
                .await;
                return;
            }
        };
        v_ctx_ids.extend_from_slice(&v_ids);
    } else {
        let s_next = crate::build_next_turn(o_state.e_fmt, &s_user_text);
        let v_ids = match o_state.o_tok.encode(&s_next, false) {
            Ok(v) => v,
            Err(e) => {
                let _ = ws_send_json(
                    &mut o_socket,
                    ws_msg_error(
                        &s_session_id,
                        &s_tab_id,
                        &format!("tokenize failed {}", e),
                        i_global_reset_id,
                    ),
                )
                .await;
                return;
            }
        };
        v_ctx_ids.extend_from_slice(&v_ids);
    }

    // stops
    let v_stop_str = crate::env_or_default_stops(&o_state.o_tok, o_state.e_fmt);
    let v_stop_ids = crate::compile_stop_id_sequences(&o_state.o_tok, &v_stop_str);

    // params
    let (d_temp, i_max_new) = {
        let o_cs = o_state.o_chat_state.lock().await;
        (o_cs.d_temp, o_cs.i_max_new)
    };

    // generation
    let d_chunk_tick = Duration::from_millis(100);
    let mut v_pending: Vec<u32> = Vec::new();
    let mut s_printed = String::new();
    let mut s_answer_full = String::new();
    let mut s_chunk_buf = String::new();
    let mut o_last_flush = Instant::now();

    for _step in 0..i_max_new {
        // peer reset check waehrend lauf
        if get_global_reset_id() != i_global_reset_id {
            let _ = ws_send_json(
                &mut o_socket,
                ws_msg_error(&s_session_id, &s_tab_id, "peer reset detected", get_global_reset_id()),
            )
            .await;
            return;
        }

        let v_logits = {
            let mut o_backend = o_state.o_backend.lock().await;
            match o_backend.forward_tokens(&v_ctx_ids).await {
                Ok(v) => v,
                Err(e) => {
                    let _ = ws_send_json(
                        &mut o_socket,
                        ws_msg_error(
                            &s_session_id,
                            &s_tab_id,
                            &format!("forward failed {}", e),
                            i_global_reset_id,
                        ),
                    )
                    .await;
                    return;
                }
            }
        };

        let i_next = crate::pick_top1(&v_logits, d_temp) as u32;
        v_ctx_ids.push(i_next);
        v_pending.push(i_next);

        let s_all = o_state.o_tok.decode(&v_pending, true).unwrap_or_default();
        let i_cp = common_prefix_bytes(&s_printed, &s_all);
        let v_new_bytes = &s_all.as_bytes()[i_cp..];
        if !v_new_bytes.is_empty() {
            if let Ok(s_new) = std::str::from_utf8(v_new_bytes) {
                s_answer_full.push_str(s_new);
                s_chunk_buf.push_str(s_new);
            }
        }
        s_printed = s_all;

        if o_last_flush.elapsed() >= d_chunk_tick && !s_chunk_buf.is_empty() {
            let s_send = s_chunk_buf.clone();
            s_chunk_buf.clear();
            o_last_flush = Instant::now();
            if ws_send_json(
                &mut o_socket,
                ws_msg_chunk(&s_session_id, &s_tab_id, &s_send, i_global_reset_id),
            )
            .await
            .is_err()
            {
                return;
            }
        }

        let mut b_stop = false;
        for v_seq in &v_stop_ids {
            if ends_with_seq(&v_ctx_ids, v_seq) {
                b_stop = true;
                break;
            }
        }
        if b_stop {
            break;
        }
    }

    if !s_chunk_buf.is_empty() {
        let _ = ws_send_json(
            &mut o_socket,
            ws_msg_chunk(&s_session_id, &s_tab_id, &s_chunk_buf, i_global_reset_id),
        )
        .await;
        s_chunk_buf.clear();
    }

    // speichern
    {
        let mut o_sess = o_state.o_sessions.lock().await;
        o_sess.map_ctx_ids.insert(s_ctx_key.clone(), v_ctx_ids);
        o_sess.map_last_reset_id.insert(s_ctx_key.clone(), i_global_reset_id);
    }

    // history optional: hier nur pro session_id, nicht pro tab
    {
        let mut o_hist = o_state.o_chat_history.lock().await;
        let v_hist = o_hist.entry(s_session_id.clone()).or_insert_with(Vec::new);
        v_hist.push(ChatTurn {
            s_user: s_user_text.clone(),
            s_assistant: s_answer_full.clone(),
        });
    }

    let _ = ws_send_json(
        &mut o_socket,
        ws_msg_done(&s_session_id, &s_tab_id, i_global_reset_id),
    )
    .await;
}

// ------------------------------------------------------------
// Clear nur fuer diesen Tab
// - loescht ctx key session_id + tab_id
// - loescht nicht andere tabs
// ------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClearRequestWeb {
    pub s_session_id: String,
    pub s_tab_id: String,
}

async fn api_session_clear(
    State(o_state): State<WebAppStateFull>,
    Json(o_req): Json<SessionClearRequestWeb>,
) -> impl IntoResponse {
    if o_req.s_session_id.trim().is_empty() || o_req.s_tab_id.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(OkResponse {
                b_ok: false,
                s_error: "session id or tab id empty".to_string(),
            }),
        );
    }

    let s_session_id = o_req.s_session_id.trim().to_string();
    let s_tab_id = o_req.s_tab_id.trim().to_string();
    let s_ctx_key = build_ctx_key(&s_session_id, &s_tab_id);

    {
        let mut o_sess = o_state.o_sessions.lock().await;
        o_sess.map_ctx_ids.remove(&s_ctx_key);
        o_sess.map_last_reset_id.insert(s_ctx_key.clone(), get_global_reset_id());
    }

    // optional: p2p clear fuer session id ist global, daher hier nicht senden
    // wenn du auch tab genauen cache im p2p willst, brauchst du tab_id im p2p protokoll

    (
        StatusCode::OK,
        Json(OkResponse {
            b_ok: true,
            s_error: String::new(),
        }),
    )
}
