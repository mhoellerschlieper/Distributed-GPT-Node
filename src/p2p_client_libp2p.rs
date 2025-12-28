// src/p2p_client_libp2p.rs
// ------------------------------------------------------------
// Client helper: send block run to peer and wait
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-27 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use candle::{Result, Tensor};
use libp2p::PeerId;

use crate::p2p_runtime_handle::with_p2p_runtime;
use crate::p2p_tensor_conv::{tensor_to_wire, wire_to_tensor};
use crate::p2p_wire::{RunBlockRequest, RunBlockResponse};

pub async fn send_block_run_and_wait(
    o_peer_id: PeerId,
    s_model_name: &str,
    i_block_no: usize,
    i_pos: usize,
    o_x: &Tensor,
) -> Result<Tensor> {
    let o_req = RunBlockRequest {
        s_model_name: s_model_name.to_string(),
        i_block_no,
        i_pos,
        o_x: tensor_to_wire(o_x).map_err(|e| candle::Error::Msg(e))?,
    };

    let v_req: Vec<u8> = bincode::serialize(&o_req)
        .map_err(|_| candle::Error::Msg("bincode request encode fehlgeschlagen".to_string()))?;

    let v_resp: Vec<u8> = with_p2p_runtime(o_peer_id, v_req)
        .await
        .map_err(|e| candle::Error::Msg(e))?;

    let o_resp: RunBlockResponse = bincode::deserialize(&v_resp)
        .map_err(|_| candle::Error::Msg("bincode response decode fehlgeschlagen".to_string()))?;

    if !o_resp.s_error.trim().is_empty() {
        return Err(candle::Error::Msg(format!(
            "remote error: {}",
            o_resp.s_error
        )));
    }

    let o_y = wire_to_tensor(&o_resp.o_y).map_err(|e| candle::Error::Msg(e))?;
    Ok(o_y)
}
