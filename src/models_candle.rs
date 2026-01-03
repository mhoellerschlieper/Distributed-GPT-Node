// src/models_candle.rs
// ------------------------------------------------------------
// Candle Backend: Llama Safetensors
//
// Aenderung
// - vor dem laden werden modell infos aus config plus gewichten gedruckt
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie
// - 2025-12-26 Marcus Schlieper: modell report einbau
// ------------------------------------------------------------

use crate::model_inspect::print_model_report_candle;

use crate::local_llama::{Cache, Config, LlamaConfig, LocalLlama};
use async_trait::async_trait;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::p2p_blocks_map::BlocksMap;

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s_val) if s_val != "0")
}

pub struct CandleLlamaModel {
    dev: Device,
    model: LocalLlama,
    cache: Cache,
    vocab_size: usize,
    prev_len: usize,
    dtype: DType,
    config: Config,
}

impl CandleLlamaModel {
    pub fn from_safetensors(
        weights_path: &str,
        config_json: &str,
        dtype: DType,
    ) -> Result<Self, String> {
        let dev = Device::Cpu;

        // print_model_report_candle(weights_path, config_json)?;

        let cfg_bytes = std::fs::read(config_json)
            .map_err(|e| format!("config json lesen fehlgeschlagen: {}", e))?;

        let cfg_raw: LlamaConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| format!("config json parse: {}", e))?;
        let config = cfg_raw.into_config(false);

        let vocab_size = config.vocab_size as usize;

        let v_weight_files = crate::model_inspect::build_weight_files(weights_path)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&v_weight_files, dtype, &dev)
                .map_err(|e| format!("safetensors mmap fehlgeschlagen: {}", e))?
        };

        let o_blocks_map = BlocksMap::from_file("")?;

        let model = LocalLlama::load(vb, &config, &o_blocks_map).map_err(|e| format!("llama load: {}", e))?;
        let cache =
            Cache::new(true, dtype, &config, &dev).map_err(|e| format!("llama cache: {}", e))?;

        if debug_on() {
            println!("[CANDLE] vocab_size: {} dtype: {:?}", vocab_size, dtype);
        }

        Ok(Self {
            dev,
            model,
            cache,
            vocab_size,
            prev_len: 0,
            dtype,
            config,
        })
    }

    pub fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("forward_tokens: leere eingabe".to_string());
        }

        if ids.len() < self.prev_len {
            self.cache = Cache::new(true, self.dtype, &self.config, &self.dev)
                .map_err(|e| format!("cache reset: {}", e))?;
            self.prev_len = 0;
        }

        if self.prev_len == 0 {
            let inp = Tensor::new(ids, &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let logits = self
                .model
                .forward(&inp, 0, &mut self.cache)
                .map_err(|e| format!("llama forward: {}", e))?;

            let rows: Vec<Vec<f32>> = logits
                .to_vec2()
                .map_err(|e| format!("to_vec2: {}", e))?;
            let out = rows.into_iter().next().unwrap_or_default();

            self.prev_len = ids.len();
            return Ok(out);
        }

        let mut out: Vec<f32> = Vec::new();
        for &tid in &ids[self.prev_len..] {
            let inp = Tensor::new(&[tid], &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let logits = self
                .model
                .forward(&inp, self.prev_len, &mut self.cache)
                .map_err(|e| format!("llama forward: {}", e))?;

            let rows: Vec<Vec<f32>> = logits
                .to_vec2()
                .map_err(|e| format!("to_vec2: {}", e))?;
            out = rows.into_iter().next().unwrap_or_default();

            self.prev_len += 1;
        }

        if out.is_empty() {
            return Err("forward_tokens: keine neuen tokens seit prev_len".to_string());
        }

        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
