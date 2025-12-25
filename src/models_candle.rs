// src/models_candle.rs
// Candle-Backend: Llama (Safetensors) – kleines API, lokal.
// Kompatibel zu deinem Trait (forward_tokens, vocab_size).

use crate::local_llama::{Cache, Config, LlamaConfig, LocalLlama};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama;

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s) if s != "0")
}

pub struct CandleLlamaModel {
    dev: Device,
    model: LocalLlama,
    cache: Cache, // <-- dein eigener Cache
    vocab_size: usize,
    prev_len: usize,
    dtype: DType,
    config: Config, // <-- ebenfalls dein eigener Config-Typ
}

impl CandleLlamaModel {
    pub fn from_safetensors(
        weights_path: &str,
        config_json: &str,
        dtype: DType,
    ) -> Result<Self, String> {
        let dev = Device::Cpu;

        // Config laden
        let cfg_bytes = std::fs::read(config_json)
            .map_err(|e| format!("config.json lesen fehlgeschlagen: {e}"))?;

        let cfg_raw: LlamaConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| format!("config.json parse: {e}"))?;
        let config = cfg_raw.into_config(false); // false = kein Flash-Attn

        // Cache erzeugen
        let cache =
            Cache::new(true, dtype, &config, &dev).map_err(|e| format!("Llama Cache: {e}"))?;

        let vocab_size = config.vocab_size as usize;

        // Candle lädt Safetensors per mmap (API ist als unsafe markiert).
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_string()], dtype, &dev)
                .map_err(|e| format!("safetensors mmap fehlgeschlagen: {}", e))?
        };

        let model = LocalLlama::load(vb, &config).map_err(|e| format!("Llama load: {}", e))?;
        let cache = Cache::new(true, dtype, &config, &dev)
            .map_err(|e| format!("Llama Cache: {}", e))?;

        if debug_on() {
            println!("[CANDLE-LLAMA] vocab={}, dtype={:?}", vocab_size, dtype);
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
            return Err("forward_tokens: leere Eingabe".to_string());
        }

        // Neuer Dialog? -> Cache neu
        if ids.len() < self.prev_len {
            self.cache = Cache::new(true, self.dtype, &self.config, &self.dev)
                .map_err(|e| format!("Cache Reset: {}", e))?;
            self.prev_len = 0;
        }

        // Erster Pass: gesamten Prompt auf einmal (pos = 0)
        if self.prev_len == 0 {
            let inp = Tensor::new(ids, &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let logits = self
                .model
                .forward(&inp, 0, &mut self.cache)
                .map_err(|e| format!("Llama forward: {}", e))?;

            let rank = logits.rank();
            let out: Vec<f32> = match rank {
                3 => {
                    let (_b, t, _v) = logits.dims3().map_err(|e| format!("dims3: {}", e))?;
                    let last = logits
                        .narrow(1, t.saturating_sub(1), 1)
                        .map_err(|e| e.to_string())?
                        .squeeze(1)
                        .map_err(|e| e.to_string())?;
                    let rows: Vec<Vec<f32>> = last
                        .to_vec2()
                        .map_err(|e| format!("to_vec2 (rank3): {}", e))?;
                    rows.into_iter().next().unwrap_or_default()
                }
                2 => {
                    let rows: Vec<Vec<f32>> = logits
                        .to_vec2()
                        .map_err(|e| format!("to_vec2 (rank2): {}", e))?;
                    rows.into_iter().next().unwrap_or_default()
                }
                other => return Err(format!("unexpected rank: {}", other)),
            };

            self.prev_len = ids.len();
            return Ok(out);
        }

        // Spätere Pässe: neue Tokens einzeln einspeisen (Prefill)
        // Beispiel: vorher 99, jetzt 171 -> wir füttern 72 Tokens nacheinander
        let mut out: Vec<f32> = Vec::new();
        for &tid in &ids[self.prev_len..] {
            let inp = Tensor::new(&[tid], &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let logits = self
                .model
                .forward(&inp, self.prev_len, &mut self.cache)
                .map_err(|e| format!("Llama forward: {}", e))?;

            let rank = logits.rank();
            let row: Vec<f32> = match rank {
                3 => {
                    let (_b, t, _v) = logits.dims3().map_err(|e| format!("dims3: {}", e))?;
                    let last = logits
                        .narrow(1, t.saturating_sub(1), 1)
                        .map_err(|e| e.to_string())?
                        .squeeze(1)
                        .map_err(|e| e.to_string())?;
                    let rows: Vec<Vec<f32>> = last
                        .to_vec2()
                        .map_err(|e| format!("to_vec2 (rank3): {}", e))?;
                    rows.into_iter().next().unwrap_or_default()
                }
                2 => {
                    let rows: Vec<Vec<f32>> = logits
                        .to_vec2()
                        .map_err(|e| format!("to_vec2 (rank2): {}", e))?;
                    rows.into_iter().next().unwrap_or_default()
                }
                other => return Err(format!("unexpected rank: {}", other)),
            };

            out = row; // Logits des jeweils letzten neuen Tokens
            self.prev_len += 1; // Cache fortschreiben
        }

        // Keine neuen Tokens? Dann das letzte Token “replayen”, um Logits zu liefern.
        if out.is_empty() {
            let last_tid = *ids.last().unwrap();
            // pos = prev_len - 1, weil dieses Token schon im Kontext steht
            let pos = self.prev_len.saturating_sub(1);
            let inp = Tensor::new(&[last_tid], &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;

            let logits = self
                .model
                .forward(&inp, pos, &mut self.cache)
                .map_err(|e| format!("Llama forward (replay): {}", e))?;

            let rank = logits.rank();
            let rows: Vec<Vec<f32>> = match rank {
                3 => {
                    let (_b, t, _v) = logits.dims3().map_err(|e| format!("dims3: {}", e))?;
                    let last = logits
                        .narrow(1, t.saturating_sub(1), 1)
                        .map_err(|e| e.to_string())?
                        .squeeze(1)
                        .map_err(|e| e.to_string())?;
                    last.to_vec2()
                        .map_err(|e| format!("to_vec2 (rank3 replay): {}", e))?
                }
                2 => logits
                    .to_vec2()
                    .map_err(|e| format!("to_vec2 (rank2 replay): {}", e))?,
                other => return Err(format!("unexpected rank (replay): {}", other)),
            };
            return Ok(rows.into_iter().next().unwrap_or_default());
        }

        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
