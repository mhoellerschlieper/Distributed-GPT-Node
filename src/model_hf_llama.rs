// src/model_hf_llama.rs
// Minimaler HF-LLaMA Backend ohne with_tracing-Abhängigkeit
// - Verwendet candle und candle_nn direkt
// - RoPE (paarweise), GQA, KV-Cache, kausale Maske
// - Lädt Safetensors via VarBuilder (mmap)
// - Liefert Logits (letztes Token) als Vec<f32>

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use candle::{DType, Device, IndexOp, Result as CResult, Tensor, D};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s) if s != "0")
}

// ---------------- HF Config ----------------

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

fn default_rope_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<LlamaEosToks>,
    #[serde(default)]
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub use_flash_attn: bool,
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            use_flash_attn,
        }
    }
}

// ---------------- RoPE ----------------

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

fn build_rope_tables(cfg: &Config, dtype: DType, dev: &Device) -> CResult<(Tensor, Tensor)> {
    let base: Vec<f32> = match &cfg.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => calculate_default_inv_freq(cfg),
        Some(rs) => {
            use std::f32::consts::PI;
            let low_wavelen = rs.original_max_position_embeddings as f32 / rs.low_freq_factor;
            let high_wavelen = rs.original_max_position_embeddings as f32 / rs.high_freq_factor;
            calculate_default_inv_freq(cfg)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2.0 * PI / freq;
                    if wavelen < high_wavelen {
                        freq
                    } else if wavelen > low_wavelen {
                        freq / rs.factor
                    } else {
                        let smooth = (rs.original_max_position_embeddings as f32 / wavelen
                            - rs.low_freq_factor)
                            / (rs.high_freq_factor - rs.low_freq_factor);
                        (1.0 - smooth) * freq / rs.factor + smooth * freq
                    }
                })
                .collect::<Vec<_>>()
        }
    };

    let theta = Tensor::new(base, dev)?;
    let idx = Tensor::arange(0, cfg.max_position_embeddings as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((cfg.max_position_embeddings, 1))?;

    let cos = idx
        .matmul(&theta.reshape((1, theta.elem_count()))?)?
        .cos()?
        .to_dtype(dtype)?;
    let sin = idx
        .matmul(&theta.reshape((1, theta.elem_count()))?)?
        .sin()?
        .to_dtype(dtype)?;
    Ok((cos, sin))
}

// ---------------- Cache ----------------

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>, // seq_len -> [seq, seq]
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>, // per layer: (k, v)
    cos: Tensor,                        // [max_pos, head_dim/2]
    sin: Tensor,                        // [max_pos, head_dim/2]
    device: Device,
    max_pos: usize,
}

impl Cache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
    ) -> CResult<Self> {
        let (cos, sin) = build_rope_tables(config, dtype, device)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
            device: device.clone(),
            max_pos: config.max_position_embeddings,
        })
    }

    fn mask(&mut self, t: usize) -> CResult<Tensor> {
        if let Some(m) = self.masks.get(&t) {
            return Ok(m.clone());
        }
        let mut v = vec![0u8; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                v[i * t + j] = 1;
            }
        }
        let m = Tensor::from_slice(&v, (t, t), &self.device)?;
        self.masks.insert(t, m.clone());
        Ok(m)
    }

    fn rope_slice(&self, index_pos: usize, seq_len: usize) -> CResult<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        Ok((cos, sin))
    }

    pub fn reset(&mut self, dtype: DType, config: &Config) -> CResult<()> {
        self.kvs.fill(None);
        let (cos, sin) = build_rope_tables(config, dtype, &self.device)?;
        self.cos = cos;
        self.sin = sin;
        Ok(())
    }
}

// ---------------- Attention, MLP, Block ----------------

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool, // nicht genutzt auf CPU
    max_position_embeddings: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> CResult<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        let q_proj = candle_nn::linear_no_bias(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(size_q, size_in, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }

    fn apply_rope(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> CResult<Tensor> {
        // x: [b, heads, seq, head_dim]
        let (_b, _h, seq, _d) = x.dims4()?;
        let (cos, sin) = cache.rope_slice(index_pos, seq)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn repeat_kv(&self, x: &Tensor) -> CResult<Tensor> {
        // x: [b, kv_heads, seq, head_dim] -> repeat to match attn heads
        let repeat = self.num_attention_heads / self.num_key_value_heads;
        if repeat == 1 {
            return Ok(x.clone());
        }
        let mut xs: Vec<Tensor> = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            xs.push(x.clone());
        }
        Tensor::cat(&xs, 1)
    }

    fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> CResult<Tensor> {
        let shape = mask.shape();
        let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
        mask.where_cond(&on_true, on_false)
    }

    fn forward(
        &self,
        x: &Tensor,       // [b, seq, hidden]
        index_pos: usize, // Startposition im Kontext
        block_idx: usize,
        cache: &mut Cache,
    ) -> CResult<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;

        // Proj
        let q = self.q_proj.forward(x)?; // [b, seq, size_q]
        let k = self.k_proj.forward(x)?; // [b, seq, size_kv]
        let v = self.v_proj.forward(x)?; // [b, seq, size_kv]

        // Reshape/transpose
        let q = q
            .reshape((b, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)? // [b, heads, seq, head_dim]
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)? // [b, kv_heads, seq, head_dim]
            .contiguous()?;
        let v = v
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?; // [b, kv_heads, seq, head_dim]

        // RoPE
        let q = self.apply_rope(&q, index_pos, cache)?;
        let mut k = self.apply_rope(&k, index_pos, cache)?;
        let mut v = v;

        // KV-Cache
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;

                // Beschneiden, falls zu lang
                let k_sl = k.dims()[2];
                if k_sl > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_sl - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
                let v_sl = v.dims()[2];
                if v_sl > self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_sl - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        // GQA expand
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Attention
        let in_dtype = q.dtype();
        let qf = q.to_dtype(DType::F32)?;
        let kf = k.to_dtype(DType::F32)?;
        let vf = v.to_dtype(DType::F32)?;

        let kt = kf.transpose(2, 3)?; // [b, heads, head_dim, seq_all]

        let mut att = qf.matmul(&kt)?;
        let scale_t = Tensor::new(1.0f32 / (self.head_dim as f32).sqrt(), att.device())?;
        att = att.broadcast_mul(&scale_t)?;

        // Kausale Maske nur wenn seq_len > 1
        if seq_len > 1 {
            let mask = cache.mask(seq_len)?; // [seq, seq], upper-tri
            let mask = mask.broadcast_as(att.shape())?;
            att = Self::masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        }

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&vf.contiguous()?)?; // [b, heads, seq, head_dim]
        let y = y.transpose(1, 2)?.reshape(&[b, seq_len, hidden])?; // [b, seq, hidden]
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: candle_nn::Linear,
    c_fc2: candle_nn::Linear,
    c_proj: candle_nn::Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> CResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let c_fc1 = candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let c_fc2 = candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?;
        let c_proj = candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let act = candle_nn::ops::silu(&self.c_fc1.forward(x)?)?;
        let up = self.c_fc2.forward(x)?;
        let mix = (act * up)?;
        self.c_proj.forward(&mix)
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: candle_nn::RmsNorm,
    attn: CausalSelfAttention,
    rms_2: candle_nn::RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Config) -> CResult<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> CResult<Tensor> {
        let residual = x;
        let x1 = self.rms_1.forward(x)?;
        let x2 = self.attn.forward(&x1, index_pos, block_idx, cache)?;
        let x = (x2 + residual)?;
        let residual = &x;
        let x3 = self.rms_2.forward(&x)?;
        let x4 = self.mlp.forward(&x3)?;
        (x4 + residual)
    }
}

// ---------------- Llama Modell ----------------

#[derive(Debug, Clone)]
pub struct Llama {
    wte: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::RmsNorm,
    // lm_head als Linear (oder Embedding^T bei Tying)
    lm_head: Option<candle_nn::Linear>,
    // weight tying fallback (wenn tie_word_embeddings=true)
    lm_head_hv: Option<Tensor>, // [hidden, vocab]
}

impl Llama {
    pub fn load(vb: VarBuilder, cfg: &Config) -> CResult<Self> {
        let wte =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        // lm_head
        let (lm_head, lm_head_hv) = if cfg.tie_word_embeddings {
            // Weight tying: lm_head = embed^T
            let emb = wte.embeddings(); // [vocab, hidden]
            let w = emb.transpose(0, 1)?; // [hidden, vocab]
            (None, Some(w))
        } else {
            let lin = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
            (Some(lin), None)
        };

        let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(Block::load(vb.pp(format!("model.layers.{i}")), cfg)?);
        }
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            lm_head_hv,
        })
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> CResult<Tensor> {
        // x: [b, seq]
        let (_b, seq_len) = x.dims2()?;
        let mut h = self.wte.forward(x)?; // [b, seq, hidden]

        for (blk_idx, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, index_pos, blk_idx, cache)?;
        }

        let h = self.ln_f.forward(&h)?;
        let h_last = h.i((.., seq_len - 1, ..))?.contiguous()?; // [b, hidden]

        // logits [b, vocab]
        let logits = if let Some(ref lin) = self.lm_head {
            lin.forward(&h_last)?
        } else if let Some(ref hv) = self.lm_head_hv {
            h_last.matmul(hv)?
        } else {
            // sollte nicht passieren
            return Err(candle::Error::Msg("lm_head fehlt".into()));
        };

        logits.to_dtype(DType::F32)
    }

    pub fn embed(&self, x: &Tensor) -> CResult<Tensor> {
        self.wte.forward(x)
    }
}

// ---------------- Öffentliches Backend ----------------

pub struct HfLlamaModel {
    dev: Device,
    model: Llama,
    cache: Cache,
    vocab_size: usize,
    prev_len: usize,
    dtype: DType,
    config: Config,
}

impl HfLlamaModel {
    pub fn from_safetensors(
        weights_path: &str,
        config_json: &str,
        dtype: DType,
    ) -> Result<Self, String> {
        let dev = Device::Cpu;

        // Config laden
        let cfg_bytes =
            std::fs::read(config_json).map_err(|e| format!("config.json read: {}", e))?;
        let cfg_hf: LlamaConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| format!("config.json parse: {}", e))?;
        let config = cfg_hf.into_config(false); // CPU: kein Flash-Attn

        // Safetensors mappen
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_string()], dtype, &dev)
                .map_err(|e| format!("safetensors mmap: {}", e))?
        };

        // Model
        // HfLlamaModel::from_safetensors(...)
        let model = Llama::load(vb, &config).map_err(|e| format!("Llama load: {}", e))?;

        let cache =
            Cache::new(true, dtype, &config, &dev).map_err(|e| format!("Cache new: {}", e))?;

        if debug_on() {
            println!(
                "[HF-LLAMA] vocab={}, hidden={}, layers={}, heads={}, kv_heads={}, dtype={:?}",
                config.vocab_size,
                config.hidden_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads,
                dtype
            );
        }

        Ok(Self {
            dev,
            model,
            cache,
            vocab_size: config.vocab_size,
            prev_len: 0,
            dtype,
            config,
        })
    }

    pub fn forward_tokens(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("forward_tokens: empty input".to_string());
        }

        // Reset-Kriterium: neuer Dialog / kürzerer Kontext
        if ids.len() < self.prev_len {
            self.cache
                .reset(self.dtype, &self.config)
                .map_err(|e| format!("cache reset: {}", e))?;
            self.prev_len = 0;
        }

        // Erster Pass: gesamten Prompt
        if self.prev_len == 0 {
            let inp = Tensor::new(ids, &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;
            let logits = self
                .model
                .forward(&inp, 0, &mut self.cache)
                .map_err(|e| format!("forward: {}", e))?;
            let rows: Vec<Vec<f32>> = logits.to_vec2().map_err(|e| e.to_string())?;
            self.prev_len = ids.len();
            return Ok(rows.into_iter().next().unwrap_or_default());
        }

        // Spätere Schritte: neue Tokens einzeln
        let mut out: Vec<f32> = Vec::new();
        for &tid in &ids[self.prev_len..] {
            let inp = Tensor::new(&[tid], &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;
            let logits = self
                .model
                .forward(&inp, self.prev_len, &mut self.cache)
                .map_err(|e| format!("forward: {}", e))?;
            let rows: Vec<Vec<f32>> = logits.to_vec2().map_err(|e| e.to_string())?;
            out = rows.into_iter().next().unwrap_or_default();
            self.prev_len += 1;
        }

        // Falls nichts Neues, letztes Token "replayen"
        if out.is_empty() {
            let last_tid = *ids.last().unwrap();
            let pos = self.prev_len.saturating_sub(1);
            let inp = Tensor::new(&[last_tid], &self.dev)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| e.to_string())?;
            let logits = self
                .model
                .forward(&inp, pos, &mut self.cache)
                .map_err(|e| format!("forward: {}", e))?;
            let rows: Vec<Vec<f32>> = logits.to_vec2().map_err(|e| e.to_string())?;
            return Ok(rows.into_iter().next().unwrap_or_default());
        }

        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
