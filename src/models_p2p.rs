// src/models_p2p.rs
// ------------------------------------------------------------
// Description
// Refactor P2pLlamaModel to reuse a single already loaded LocalLlama instance
// provided via Arc, avoiding double loading of model weights.
//
// Key changes
// - Remove internal weight loading in new_with_local_model.
// - Store injected Arc<LocalLlama> as the model reference used for inference.
// - Keep client-side KV cache owned by P2pLlamaModel, because it is session
//   specific and should not be shared across peers.
//
// Security and robustness
// - Validate inputs.
// - Use explicit error messages.
// - Do not load weights inside P2pLlamaModel in the injected-model path.
//
// Author
// Marcus Schlieper, ExpChat.ai
//
// History
// - 2026-01-04 Marcus Schlieper: refactor to avoid double model load, use injected Arc<LocalLlama>
// ------------------------------------------------------------

use candle::{DType, Device, Tensor};
use libp2p::PeerId;
use std::sync::Arc;

use crate::local_llama::{Cache, Config, LlamaConfig, LocalLlama};
use crate::p2p_blocks_map::BlocksMap;
use crate::p2p_llama_forward;
use crate::device_select::get_default_device;

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s_val) if s_val != "0")
}

pub struct P2pLlamaModel {
    // Device is used for input tensor construction and cache allocation.
    o_dev: Device,

    // Shared model instance (single weights load in process).
    o_local_model: Arc<LocalLlama>,

    // Client-side KV cache (not shared).
    o_cache: Cache,

    // Config is kept for cache resets and for vocab_size.
    o_config: Config,
    e_dtype: DType,

    i_prev_len: usize,
    i_vocab_size: usize,

    s_model_name: String,
    o_blocks_map: BlocksMap,
    o_my_peer_id: PeerId,
}

impl P2pLlamaModel {
    // Build a P2P model backend that reuses an already loaded LocalLlama.
    // The blocks map is loaded to support routing decisions, but weights are not loaded here.
    pub fn new_with_local_model(
        o_local_model: Arc<LocalLlama>,
        s_blocks_map_file: &str,
        s_model_name: &str,
        o_peer_id: PeerId,
    ) -> Result<Self, String> {
        // Validation
        if s_blocks_map_file.trim().is_empty() {
            return Err("new_with_local_model: blocks_map_file empty".to_string());
        }
        if s_model_name.trim().is_empty() {
            return Err("new_with_local_model: model_name empty".to_string());
        }

        // Read config json to get Config and vocab_size.
        // This avoids requiring LocalLlama to expose its Config.
        let s_config_json =
            std::env::var("LLAMA_CONFIG").map_err(|_| "Env LLAMA_CONFIG fehlt".to_string())?;

        let e_dtype = match std::env::var("LLAMA_DTYPE")
            .unwrap_or_else(|_| "f32".to_string())
            .to_lowercase()
            .as_str()
        {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        };

        let o_dev = get_default_device();

        let v_cfg_bytes = std::fs::read(&s_config_json)
            .map_err(|e| format!("config json lesen fehlgeschlagen: {}", e))?;

        let o_cfg_raw: LlamaConfig =
            serde_json::from_slice(&v_cfg_bytes).map_err(|e| format!("config json parse: {}", e))?;

        let o_config = o_cfg_raw.into_config(false);
        let i_vocab_size = o_config.vocab_size;

        // Load blocks map (routing, peer addrs, model settings).
        let o_blocks_map = BlocksMap::from_file(s_blocks_map_file)?;

        // Allocate client-side cache.
        let o_cache =
            Cache::new(true, e_dtype, &o_config, &o_dev).map_err(|e| format!("llama cache: {}", e))?;

        if debug_on() {
            println!(
                "[P2P] injected model_name={} vocab_size={} dtype={:?} my_peer_id={}",
                s_model_name, i_vocab_size, e_dtype, o_peer_id
            );
        }

        Ok(Self {
            o_dev,
            o_local_model,
            o_cache,
            o_config,
            e_dtype,
            i_prev_len: 0,
            i_vocab_size,
            s_model_name: s_model_name.to_string(),
            o_blocks_map,
            o_my_peer_id: o_peer_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.i_vocab_size
    }

    pub fn reset_kv_cache(&mut self) -> Result<(), String> {
        // History:
        // - 2026-01-04 Marcus Schlieper: keep reset local to client cache only
        self.o_cache = Cache::new(true, self.e_dtype, &self.o_config, &self.o_dev)
            .map_err(|e| format!("cache reset: {}", e))?;
        self.i_prev_len = 0;
        Ok(())
    }

    pub async fn forward_tokens_async(&mut self, v_ids: &[u32]) -> Result<Vec<f32>, String> {
        if v_ids.is_empty() {
            return Err("forward_tokens: leere eingabe".to_string());
        }

        if v_ids.len() < self.i_prev_len {
            self.reset_kv_cache()?;
        }

        // Use the injected shared model for forward.
        let o_llama = self.o_local_model.inner_ref();

        if self.i_prev_len == 0 {
            let o_inp = Tensor::new(v_ids, &self.o_dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let o_logits_res = p2p_llama_forward::forward_p2p(
                o_llama,
                &self.o_blocks_map,
                self.o_my_peer_id,
                &self.s_model_name,
                &o_inp,
                0,
                &mut self.o_cache,
            )
            .await;

            let o_logits = match o_logits_res {
                Ok(v) => v,
                Err(e) => {
                    let _ = self.reset_kv_cache();
                    return Err(format!("p2p forward fehler: {}", e));
                }
            };

            let v_rows: Vec<Vec<f32>> = o_logits.to_vec2().map_err(|e| format!("to_vec2: {}", e))?;
            self.i_prev_len = v_ids.len();
            return Ok(v_rows.into_iter().next().unwrap_or_default());
        }

        let mut v_out: Vec<f32> = Vec::new();

        for &i_tid in &v_ids[self.i_prev_len..] {
            let o_inp = Tensor::new(&[i_tid], &self.o_dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let i_pos = self.i_prev_len;

            let o_logits_res = p2p_llama_forward::forward_p2p(
                o_llama,
                &self.o_blocks_map,
                self.o_my_peer_id,
                &self.s_model_name,
                &o_inp,
                i_pos,
                &mut self.o_cache,
            )
            .await;

            let o_logits = match o_logits_res {
                Ok(v) => v,
                Err(e) => {
                    let _ = self.reset_kv_cache();
                    return Err(format!("p2p forward fehler: {}", e));
                }
            };

            let v_rows: Vec<Vec<f32>> = o_logits.to_vec2().map_err(|e| format!("to_vec2: {}", e))?;
            v_out = v_rows.into_iter().next().unwrap_or_default();
            self.i_prev_len += 1;
        }

        if v_out.is_empty() {
            return Err("forward_tokens: keine logits".to_string());
        }

        Ok(v_out)
    }
}
