// src/p2p_llama_forward.rs
// ------------------------------------------------------------
// P2P Forward fuer Llama
//
// Ziel
// - pro block: lokal oder remote
// - keine zugriffe auf private felder (wte ln_f lm_head)
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-27 Marcus Schlieper: initiale version
// - 2025-12-27 Marcus Schlieper: fix private felder, nutze public helper
// ------------------------------------------------------------

use candle::{Result, Tensor};
use libp2p::PeerId;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::local_llama::{Cache, Llama};
use crate::p2p_blocks_map::BlocksMap;
use crate::p2p_client_libp2p::send_blocks_run_and_wait;


fn session_id_from_env() -> String {
    std::env::var("SESSION_ID").unwrap_or_else(|_| "default_session".to_string())
}

fn push_segment(v_segments: &mut Vec<(PeerId, Vec<usize>)>, o_peer: PeerId, i_block_no: usize) {
    if let Some((o_last_peer, v_last_blocks)) = v_segments.last_mut() {
        if *o_last_peer == o_peer {
            v_last_blocks.push(i_block_no);
            return;
        }
    }
    v_segments.push((o_peer, vec![i_block_no]));
}

pub async fn forward_input_embed_p2p(
    o_llama: &Llama,
    o_blocks_map: &BlocksMap,
    o_my_peer_id: PeerId,
    s_model_name: &str,
    o_input: &Tensor,
    i_pos: usize,
    o_cache: &mut Cache,
) -> Result<Tensor> {
    let (_bs, i_seq_len, _h) = o_input.dims3()?;
    let mut o_x = o_input.clone();
    let s_session_id =
        std::env::var("SESSION_ID").unwrap_or_else(|_| "default_session".to_string());

    let mut i_block_no: usize = 0;
    while i_block_no < o_llama.blocks_len() {
        let s_peer_id = o_blocks_map
            .get_peer_for_block(s_model_name, i_block_no)
            .ok_or_else(|| candle::Error::Msg("blocks_map peer id fehlt".to_string()))?;

        let o_peer_id: PeerId = s_peer_id
            .parse()
            .map_err(|_| candle::Error::Msg("blocks_map peer id ungueltig".to_string()))?;

        // ======================================
        // wenn localer Block, dann....
        if o_peer_id == o_my_peer_id {
            // local block
            o_x = o_llama.forward_one_block(&o_x, i_pos, i_block_no, o_cache)?;
            i_block_no += 1;
            continue;
        }

        // ======================================
        // remote segment: sammle alle aufeinanderfolgenden blocks fuer denselben peer
        let o_remote_peer_id = o_peer_id;
        let mut v_blocks: Vec<usize> = Vec::new();

        while i_block_no < o_llama.blocks_len() {
            let s_curr_peer_id = o_blocks_map
                .get_peer_for_block(s_model_name, i_block_no)
                .ok_or_else(|| candle::Error::Msg("blocks_map peer id fehlt".to_string()))?;

            let o_curr_peer_id: PeerId = s_curr_peer_id
                .parse()
                .map_err(|_| candle::Error::Msg("blocks_map peer id ungueltig".to_string()))?;

            // stop, sobald peer wechselt oder lokal wird
            if o_curr_peer_id != o_remote_peer_id || o_curr_peer_id == o_my_peer_id {
                break;
            }

            v_blocks.push(i_block_no);
            i_block_no += 1;
        }

        if v_blocks.is_empty() {
            return Err(candle::Error::Msg("p2p forward: remote blocks ist leer".to_string()));
        }

        // sende ein request fuer dieses zusammenhaengende block segment
        o_x = crate::p2p_client_libp2p::send_blocks_run_and_wait(
            o_remote_peer_id,
            s_model_name,
            &v_blocks,
            i_pos,
            &s_session_id,
            &o_x,
        )
        .await?;
    }

    o_llama.forward_final_from_hidden(&o_x, i_seq_len)
}


pub async fn forward_p2p(
    o_llama: &Llama,
    o_blocks_map: &BlocksMap,
    o_my_peer_id: PeerId,
    s_model_name: &str,
    o_ids: &Tensor,
    i_pos: usize,
    o_cache: &mut Cache,
) -> Result<Tensor> {
    let (_bs, i_seq_len) = o_ids.dims2()?;

    // stabile session id
    let s_session_id = session_id_from_env();

    // Embedding Tokens
    let mut o_x = o_llama.embed_tokens(o_ids)?;

    let mut i_block_no: usize = 0;
    while i_block_no < o_llama.blocks_len() {
        let s_peer_id = o_blocks_map
            .get_peer_for_block(s_model_name, i_block_no)
            .ok_or_else(|| candle::Error::Msg("blocks_map peer id fehlt".to_string()))?;

        let o_peer_id: PeerId = s_peer_id
            .parse()
            .map_err(|_| candle::Error::Msg("blocks_map peer id ungueltig".to_string()))?;

        if o_peer_id == o_my_peer_id {
            // local block
            o_x = o_llama.forward_one_block(&o_x, i_pos, i_block_no, o_cache)?;
            i_block_no += 1;
            continue;
        }

        // remote segment: sammle alle aufeinanderfolgenden blocks fuer denselben peer
        let o_remote_peer_id = o_peer_id;
        let mut v_blocks: Vec<usize> = Vec::new();

        while i_block_no < o_llama.blocks_len() {
            let s_curr_peer_id = o_blocks_map
                .get_peer_for_block(s_model_name, i_block_no)
                .ok_or_else(|| candle::Error::Msg("blocks_map peer id fehlt".to_string()))?;

            let o_curr_peer_id: PeerId = s_curr_peer_id
                .parse()
                .map_err(|_| candle::Error::Msg("blocks_map peer id ungueltig".to_string()))?;

            // stop, sobald peer wechselt oder lokal wird
            if o_curr_peer_id != o_remote_peer_id || o_curr_peer_id == o_my_peer_id {
                break;
            }

            v_blocks.push(i_block_no);
            i_block_no += 1;
        }

        if v_blocks.is_empty() {
            return Err(candle::Error::Msg("p2p forward: remote blocks ist leer".to_string()));
        }

        // sende ein request fuer dieses zusammenhaengende block segment
        o_x = crate::p2p_client_libp2p::send_blocks_run_and_wait(
            o_remote_peer_id,
            s_model_name,
            &v_blocks,
            i_pos,
            &s_session_id,
            &o_x,
        )
        .await?;
    }

    // Hidden -> Logits
    o_llama.forward_final_from_hidden(&o_x, i_seq_len)
}
