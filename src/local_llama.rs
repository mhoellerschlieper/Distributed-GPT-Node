// local_llama.rs
// -----------------------------------------------------------
// Full llama inference
// -----------------------------------------------------------
// Autor  : Marcus Schlieper, ExpChat.ai
// Datum  : 25-12-2025
// Lizenz : MIT / Apache-2.0
//
// Hinweis
// Dieses File ist aus candle-transformers abgeleitet,
// damit du es lokal ohne private Items nutzen kannst.
//
// Aenderung
// - Compile Fixes fuer fehlende Generics
// - Public API fuer Config, Cache, Llama, LocalLlama
// - Helper fuer p2p block offload (blocks_len, forward_one_block, embed_tokens)
// - Helper fuer final stage (norm plus lm_head) nach remote blocks
//
// Historie
// - 2025-12-25 Marcus Schlieper: basis version
// - 2025-12-27 Marcus Schlieper: compile fixes und p2p helper
// -----------------------------------------------------------

use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use candle_transformers::utils;
use std::collections::HashMap;

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug)]
pub struct LocalLlama {
    // internal llama network
    inner: Llama,
    // vocab size
    vocab_size: usize,
}

impl LocalLlama {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let inner = Llama::load(vb, cfg)?;
        Ok(Self {
            inner,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn forward(&self, ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        self.inner.forward(ids, pos, cache)
    }

    pub fn forward_one_block(
        &self,
        o_x: &Tensor,
        i_pos: usize,
        i_block_no: usize,
        o_cache: &mut Cache,
    ) -> Result<Tensor> {
        self.inner
            .forward_one_block(o_x, i_pos, i_block_no, o_cache)
    }

    pub fn blocks_len(&self) -> usize {
        self.inner.blocks_len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.inner.embed_tokens(ids)
    }

    pub fn forward_final_from_hidden(&self, o_hidden: &Tensor, i_seq_len: usize) -> Result<Tensor> {
        self.inner.forward_final_from_hidden(o_hidden, i_seq_len)
    }
}

// -----------------------------------------------------------
// Rope config
// -----------------------------------------------------------

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

// -----------------------------------------------------------
// Config json
// -----------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            use_flash_attn,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

fn default_rope() -> f32 {
    10_000.0
}

// -----------------------------------------------------------
// Internal config
// -----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

// -----------------------------------------------------------
// KV cache
// -----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let v_theta = calculate_default_inv_freq(cfg);
        let o_theta = Tensor::new(v_theta, dev)?;

        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&o_theta.reshape((1, o_theta.elem_count()))?)?;

        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; cfg.num_hidden_layers],
            device: dev.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(m) = self.masks.get(&t) {
            return Ok(m.clone());
        }

        let data: Vec<u8> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();

        let m = Tensor::from_slice(&data, (t, t), &self.device)?;
        self.masks.insert(t, m.clone());
        Ok(m)
    }
}

// -----------------------------------------------------------
// Attention
// -----------------------------------------------------------

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _e = self.span_rot.enter();
        let (_b, _h, seq_len, _d) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _e = self.span.enter();
        let (bs, seq_len, hidden) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((bs, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = k
            .reshape((bs, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut v = v
            .reshape((bs, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((ck, cv)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[ck, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cv, &v], 2)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let idt = q.dtype();
        let qf = q.to_dtype(DType::F32)?;
        let kf = k.to_dtype(DType::F32)?;
        let vf = v.to_dtype(DType::F32)?;

        let att = (qf.matmul(&kf.t()?)? / (self.head_dim as f64).sqrt())?;

        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&vf.contiguous()?)?.to_dtype(idt)?;

        let y = y.transpose(1, 2)?.reshape(&[bs, seq_len, hidden])?;
        self.o_proj.forward(&y)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");

        let size_in = cfg.hidden_size;
        let size_q = cfg.hidden_size;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        Ok(Self {
            q_proj: linear(size_in, size_q, vb.pp("q_proj"))?,
            k_proj: linear(size_in, size_kv, vb.pp("k_proj"))?,
            v_proj: linear(size_in, size_kv, vb.pp("v_proj"))?,
            o_proj: linear(size_q, size_in, vb.pp("o_proj"))?,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
        })
    }
}

fn masked_fill(a: &Tensor, m: &Tensor, val: f32) -> Result<Tensor> {
    let fill = Tensor::new(val, a.device())?.broadcast_as(m.shape().dims())?;
    m.where_cond(&fill, a)
}

// -----------------------------------------------------------
// MLP
// -----------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    out: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _e = self.span.enter();
        let x = (candle_nn::ops::silu(&self.fc1.forward(x)?)? * self.fc2.forward(x)?)?;
        self.out.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            fc2: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            out: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            span,
        })
    }
}

// -----------------------------------------------------------
// Transformer block
// -----------------------------------------------------------

#[derive(Debug, Clone)]
struct Block {
    norm1: RmsNorm,
    attn: CausalSelfAttention,
    norm2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(&self, x: &Tensor, pos: usize, idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let _e = self.span.enter();
        let res = x;
        let x = self.norm1.forward(x)?;
        let x = (self.attn.forward(&x, pos, idx, cache)? + res)?;
        let res = &x;
        let x = (self.mlp.forward(&self.norm2.forward(&x)?)? + res)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        Ok(Self {
            norm1: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: CausalSelfAttention::load(vb.pp("self_attn"), cfg)?,
            norm2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            span,
        })
    }
}

// -----------------------------------------------------------
// Llama top model
// -----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Llama {
    // keep private, expose via methods
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }

    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(ids)
    }

    pub fn blocks_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn forward_one_block(
        &self,
        o_x: &Tensor,
        i_pos: usize,
        i_block_no: usize,
        o_cache: &mut Cache,
    ) -> Result<Tensor> {
        let o_b = self
            .blocks
            .get(i_block_no)
            .ok_or_else(|| candle::Error::Msg("block not found".to_string()))?;
        o_b.forward(o_x, i_pos, i_block_no, o_cache)
    }

    pub fn forward_input_embed(
        &self,
        input: &Tensor,
        pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_bs, seq_len, _h) = input.dims3()?;
        let mut x = input.clone();
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x, pos, i, cache)?;
        }
        self.forward_final_from_hidden(&x, seq_len)
    }

    pub fn forward(&self, ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_bs, seq_len) = ids.dims2()?;
        let mut x = self.wte.forward(ids)?;
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x, pos, i, cache)?;
        }
        self.forward_final_from_hidden(&x, seq_len)
    }

    pub fn forward_final_from_hidden(&self, o_hidden: &Tensor, i_seq_len: usize) -> Result<Tensor> {
        let x = self.ln_f.forward(o_hidden)?;
        let x = x.i((.., i_seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

impl LocalLlama {
    pub fn inner_ref(&self) -> &Llama {
        &self.inner
    }
}

impl LocalLlama {
    pub fn forward_one_block_via_self(
        &self,
        o_x: &candle::Tensor,
        i_pos: usize,
        i_block_no: usize,
        o_cache: &mut crate::local_llama::Cache,
    ) -> candle::Result<candle::Tensor> {
        let o_llama = self.inner_ref();
        o_llama.forward_one_block(o_x, i_pos, i_block_no, o_cache)
    }
}