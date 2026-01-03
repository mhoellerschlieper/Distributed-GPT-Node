// src/models_p2p.rs
// ------------------------------------------------------------
// P2P Backend: Llama Safetensors via Candle plus p2p forward
//
// Ziel
// - kompatibel zur Struktur von models_candle.rs
// - nutzt p2p_llama_forward fuer block routing
//
// Hinweis
// - forward_tokens ist hier async (forward_tokens_async)
// - du brauchst in local_llama.rs: LocalLlama::inner_ref()
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-28 Marcus Schlieper: initiale version
// - 2025-12-28 Marcus Schlieper: borrow fix (self cache mut + llama ref)
// ------------------------------------------------------------

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use libp2p::PeerId;

use crate::local_llama::{Cache, Config, LlamaConfig, LocalLlama};
use crate::model_inspect;
use crate::model_inspect::print_model_report_candle;
use crate::p2p_blocks_map::BlocksMap;
use crate::p2p_llama_forward;

use async_trait::async_trait;

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s_val) if s_val != "0")
}

pub struct P2pLlamaModel {
    o_dev: Device,
    o_model: LocalLlama,
    o_cache: Cache,
    o_config: Config,
    e_dtype: DType,

    i_prev_len: usize,
    i_vocab_size: usize,

    s_model_name: String,
    o_blocks_map: BlocksMap,
    o_my_peer_id: PeerId,
}

impl P2pLlamaModel {
    pub fn from_safetensors(
        s_weights_path: &str,
        s_config_json: &str,
        e_dtype: DType,
        s_blocks_map_file: &str,
        s_model_name: &str,
        o_my_peer_id: PeerId,
    ) -> Result<Self, String> {
        let o_dev = Device::Cpu;

        //print_model_report_candle(s_weights_path, s_config_json)?;

        let v_cfg_bytes = std::fs::read(s_config_json)
            .map_err(|e| format!("config json lesen fehlgeschlagen: {}", e))?;

        let o_cfg_raw: LlamaConfig = serde_json::from_slice(&v_cfg_bytes)
            .map_err(|e| format!("config json parse: {}", e))?;

        let o_config = o_cfg_raw.into_config(false);
        let i_vocab_size = o_config.vocab_size;

        let v_weight_files = model_inspect::build_weight_files(s_weights_path)?;

        let o_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&v_weight_files, e_dtype, &o_dev)
                .map_err(|e| format!("safetensors mmap fehlgeschlagen: {}", e))?
        };

        let o_model =
            LocalLlama::load(o_vb, &o_config).map_err(|e| format!("llama load: {}", e))?;
        let o_cache = Cache::new(true, e_dtype, &o_config, &o_dev)
            .map_err(|e| format!("llama cache: {}", e))?;

        let o_blocks_map = BlocksMap::from_file(s_blocks_map_file)?;

        if debug_on() {
            println!(
                "[P2P] model_name={} vocab_size={} dtype={:?} my_peer_id={}",
                s_model_name, i_vocab_size, e_dtype, o_my_peer_id
            );
        }

        Ok(Self {
            o_dev,
            o_model,
            o_cache,
            o_config,
            e_dtype,
            i_prev_len: 0,
            i_vocab_size,
            s_model_name: s_model_name.to_string(),
            o_blocks_map,
            o_my_peer_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.i_vocab_size
    }

    pub fn reset_kv_cache(&mut self) -> Result<(), String> {
        self.o_cache = Cache::new(true, self.e_dtype, &self.o_config, &self.o_dev)
            .map_err(|e| format!("cache reset: {}", e))?;
        self.i_prev_len = 0;
        Ok(())
    }

    // ------------------------------------------------------------
    // async forward tokens
    // - exakt wie models_candle.rs aufgebaut (erst full prompt, dann token by token)
    // - nutzt p2p_llama_forward::forward_p2p
    // ------------------------------------------------------------
    pub async fn forward_tokens_async(&mut self, v_ids: &[u32]) -> Result<Vec<f32>, String> {
        if v_ids.is_empty() {
            return Err("forward_tokens: leere eingabe".to_string());
        }

        // falls ctx gekuerzt wurde, cache reset
        if v_ids.len() < self.i_prev_len {
            self.reset_kv_cache()?;
        }

        // 1) erster call: kompletter prompt
        if self.i_prev_len == 0 {
            let o_inp = Tensor::new(v_ids, &self.o_dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            // borrow fix: llama ref in lokale variable
            let o_llama = self.o_model.inner_ref();

            let o_logits_res  = p2p_llama_forward::forward_p2p(
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

            let v_rows: Vec<Vec<f32>> =
                o_logits.to_vec2().map_err(|e| format!("to_vec2: {}", e))?;

            self.i_prev_len = v_ids.len();
            return Ok(v_rows.into_iter().next().unwrap_or_default());
        }

        // 2) danach: nur neue tokens inkrementell
        let mut v_out: Vec<f32> = Vec::new();

        for &i_tid in &v_ids[self.i_prev_len..] {
            let o_inp = Tensor::new(&[i_tid], &self.o_dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let i_pos = self.i_prev_len;

            let o_llama = self.o_model.inner_ref();

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

            let v_rows: Vec<Vec<f32>> =
                o_logits.to_vec2().map_err(|e| format!("to_vec2: {}", e))?;

            v_out = v_rows.into_iter().next().unwrap_or_default();
            self.i_prev_len += 1;
        }

        if v_out.is_empty() {
            return Err("forward_tokens: keine logits".to_string());
        }

        Ok(v_out)
    }
}
