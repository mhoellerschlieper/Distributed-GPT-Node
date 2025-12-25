// models.rs
// ------------------------------------------------------------
// Transformer Modell (CPU) ohne GGUF
// - Lädt HuggingFace Llama Gewichte aus safetensors + config.json
// - Forward: Embedding -> N x [RMSNorm + MHA(GQA, RoPE) + Residual
//            + RMSNorm + SwiGLU-MLP + Residual]
// - Liefert Logits des letzten Tokens (Vec<f32>)
//
// Autor: Marcus Schlieper, ExpChat.ai
// Kontakt: 49 2338 8748862 | 49 15115751864 | mschlieper@ylook.de
// Firma: ExpChat.ai – Der KI Chat Client für den Mittelstand aus Breckerfeld
// Zusatz: RPA, KI Agents, KI Internet Research, KI Wissensmanagement
// Adresse: Epscheider Str21, 58339 Breckerfeld
//
// Stand: 2025-12-23
// Lizenz: MIT / Apache-2.0
//
// Sicherheit:
// - Kein unsafe
// - Klare Result-Fehler
// - Defensive Checks (Dtypen, Dimensionen, Masken)
//
// Hinweise:
// - BACKEND="transformers" nutzt dieses Modell
// - Datentyp per LLAMA_DTYPE: "f32" (Standard) oder "f16"
// - Für stabile Ausgabe: passendes Prompt-Template (z. B. Llama3) und Stop-Token
// ------------------------------------------------------------

use candle::{DType, Device, Tensor};
use candle_nn::ops as nn_ops;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::Deserialize;

// ------------------------------------------------------------
// Debug per ENV: DEBUG_MODEL != "0"
// ------------------------------------------------------------
fn debug_on() -> bool {
    matches!(std::env::var("DEBUG_MODEL"), Ok(s) if s != "0")
}

// ------------------------------------------------------------
// Hilfen: Dims, Ausrichtung, beste Teiler
// ------------------------------------------------------------
fn tensor_dims_2d(t: &Tensor) -> Result<(usize, usize), String> {
    t.dims2().map_err(|e| e.to_string())
}

// Für x[T, in_dim].matmul(W) soll W => [in_dim, out_dim] sein.
// HF Linear-Gewichte sind oft [out, in]; ggf. transponieren.
fn orient_for_right_matmul(w: &Tensor, in_dim: usize) -> Result<Tensor, String> {
    let (a, b) = tensor_dims_2d(w)?;
    if a == in_dim {
        Ok(w.clone())
    } else if b == in_dim {
        w.t().map_err(|e| e.to_string())
    } else {
        Err(format!(
            "Gewicht passt nicht zu in_dim {} ({} x {})",
            in_dim, a, b
        ))
    }
}

// Fallback-Teiler (für head_dim)
fn best_divisor(total: usize, prefer: usize) -> usize {
    if prefer > 0 && total % prefer == 0 {
        return prefer;
    }
    let mut d = prefer.max(1);
    while d > 1 {
        if total % d == 0 {
            return d;
        }
        d -= 1;
    }
    for d2 in (1..=total).rev() {
        if total % d2 == 0 {
            return d2;
        }
    }
    1
}

// ------------------------------------------------------------
// DType-Utilities
// ------------------------------------------------------------
fn to_dtype_if_needed(t: Tensor, dtype: DType) -> Result<Tensor, String> {
    if t.dtype() == dtype {
        Ok(t)
    } else {
        t.to_dtype(dtype).map_err(|e| e.to_string())
    }
}

// ------------------------------------------------------------
// RoPE: Tabellen und Anwendung (voll / partiell)
// ------------------------------------------------------------
fn build_rope_cos_sin(
    dev: &Device,
    seq_len: usize,
    rope_dim: usize,
    rope_base: f32,
    dtype: DType,
) -> Result<(Tensor, Tensor), String> {
    let half = rope_dim / 2;
    if rope_dim == 0 || half == 0 {
        return Err("rope_dim zu klein für RoPE".to_string());
    }
    let dim_idx_v: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let dim_idx = Tensor::from_vec(dim_idx_v, half, dev).map_err(|e| e.to_string())?;

    let pos_v: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let pos = Tensor::from_vec(pos_v, seq_len, dev).map_err(|e| e.to_string())?;
    let pos = pos.unsqueeze(1).map_err(|e| e.to_string())?; // [T,1]

    let ln_base = rope_base.ln();
    let scale = -2.0f32 / (rope_dim as f32);
    let inv_log = dim_idx
        .broadcast_mul(&Tensor::new(scale * ln_base, dev).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let inv_freq = inv_log.exp().map_err(|e| e.to_string())?; // [half]
    let inv_freq = inv_freq.unsqueeze(0).map_err(|e| e.to_string())?; // [1,half]

    let angles = pos.broadcast_mul(&inv_freq).map_err(|e| e.to_string())?; // [T,half]
    let cos = angles.cos().map_err(|e| e.to_string())?;
    let sin = angles.sin().map_err(|e| e.to_string())?;

    let cos = to_dtype_if_needed(cos, dtype)?;
    let sin = to_dtype_if_needed(sin, dtype)?;
    Ok((cos, sin))
}

fn apply_rope_full(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, String> {
    let (t, d) = x.dims2().map_err(|e| e.to_string())?;
    if d % 2 != 0 {
        return Err("head_dim muss gerade sein".to_string());
    }
    let half = d / 2;

    // [T, d] -> [T, half, 2] (Paare bilden: (0,1), (2,3), ...)
    let x_pair = x.reshape((t, half, 2)).map_err(|e| e.to_string())?;
    let x_even = x_pair.narrow(2, 0, 1).map_err(|e| e.to_string())?.squeeze(2).map_err(|e| e.to_string())?;
    let x_odd  = x_pair.narrow(2, 1, 1).map_err(|e| e.to_string())?.squeeze(2).map_err(|e| e.to_string())?;

    // cos, sin: [T, half]
    let y_even = x_even.broadcast_mul(cos).map_err(|e| e.to_string())?
        .broadcast_sub(&x_odd.broadcast_mul(sin).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let y_odd  = x_even.broadcast_mul(sin).map_err(|e| e.to_string())?
        .broadcast_add(&x_odd.broadcast_mul(cos).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;

    // zurück zu [T, d]
    let y_even = y_even.unsqueeze(2).map_err(|e| e.to_string())?;
    let y_odd  = y_odd.unsqueeze(2).map_err(|e| e.to_string())?;
    let y_pair = Tensor::cat(&[y_even, y_odd], 2).map_err(|e| e.to_string())?;
    y_pair.reshape((t, d)).map_err(|e| e.to_string())
}

fn apply_rope_partial(x: &Tensor, cos: &Tensor, sin: &Tensor, rope_dim: usize) -> Result<Tensor, String> {
    let (t, d) = x.dims2().map_err(|e| e.to_string())?;
    if rope_dim == 0 || rope_dim > d || rope_dim % 2 != 0 {
        return Err(format!("ungueltiger rope_dim: {}, gesamt_dim: {}", rope_dim, d));
    }
    // vorderer Teil rotieren, Rest passt durch
    let x_rot = x.narrow(1, 0, rope_dim).map_err(|e| e.to_string())?;
    let x_pas = x.narrow(1, rope_dim, d - rope_dim).map_err(|e| e.to_string())?;

    let half = rope_dim / 2;
    let x_pair = x_rot.reshape((t, half, 2)).map_err(|e| e.to_string())?;
    let x_even = x_pair.narrow(2, 0, 1).map_err(|e| e.to_string())?.squeeze(2).map_err(|e| e.to_string())?;
    let x_odd  = x_pair.narrow(2, 1, 1).map_err(|e| e.to_string())?.squeeze(2).map_err(|e| e.to_string())?;

    // cos/sin haben Form [T, half]
    let y_even = x_even.broadcast_mul(cos).map_err(|e| e.to_string())?
        .broadcast_sub(&x_odd.broadcast_mul(sin).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let y_odd  = x_even.broadcast_mul(sin).map_err(|e| e.to_string())?
        .broadcast_add(&x_odd.broadcast_mul(cos).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;

    let y_even = y_even.unsqueeze(2).map_err(|e| e.to_string())?;
    let y_odd  = y_odd.unsqueeze(2).map_err(|e| e.to_string())?;
    let y_pair = Tensor::cat(&[y_even, y_odd], 2).map_err(|e| e.to_string())?;
    let y_rot  = y_pair.reshape((t, rope_dim)).map_err(|e| e.to_string())?;

    Tensor::cat(&[y_rot, x_pas], 1).map_err(|e| e.to_string())
}


// ------------------------------------------------------------
// RMSNorm und kausale Maske
// ------------------------------------------------------------
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, String> {
    let (_t, h) = x.dims2().map_err(|e| e.to_string())?;
    let dtype = x.dtype();

    let x2 = x.broadcast_mul(x).map_err(|e| e.to_string())?;
    let sum = x2.sum_keepdim(1).map_err(|e| e.to_string())?;
    let h_t = Tensor::new(h as f32, x.device())
        .map_err(|e| e.to_string())?
        .to_dtype(dtype)
        .map_err(|e| e.to_string())?;
    let eps_t = Tensor::new(eps, x.device())
        .map_err(|e| e.to_string())?
        .to_dtype(dtype)
        .map_err(|e| e.to_string())?;

    let denom = sum
        .broadcast_div(&h_t)
        .map_err(|e| e.to_string())?
        .broadcast_add(&eps_t)
        .map_err(|e| e.to_string())?
        .sqrt()
        .map_err(|e| e.to_string())?;

    let y = x.broadcast_div(&denom).map_err(|e| e.to_string())?;
    let w2 = w.unsqueeze(0).map_err(|e| e.to_string())?;
    y.broadcast_mul(&w2).map_err(|e| e.to_string())
}

pub fn causal_mask(dev: &Device, t: usize, dtype: DType) -> Result<Tensor, String> {
    let neg_inf_f32 = f32::NEG_INFINITY;
    let mut v = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            v[i * t + j] = neg_inf_f32;
        }
    }
    let m = Tensor::from_vec(v, (t, t), dev).map_err(|e| e.to_string())?;
    to_dtype_if_needed(m, dtype)
}


// ------------------------------------------------------------
// Konvertierung F16/BF16 -> F32, Safetensors-Reader
// ------------------------------------------------------------
fn f16_to_f32_bits(h: u16) -> f32 {
    let s = ((h >> 15) & 0x0001) as u32;
    let e = ((h >> 10) & 0x001f) as u32;
    let f = (h & 0x03ff) as u32;
    let bits: u32 = if e == 0 {
        if f == 0 {
            s << 31
        } else {
            // subnormal
            let mut e32: i32 = 127 - 15 + 1;
            let mut f32m = f;
            while (f32m & 0x0400) == 0 {
                f32m <<= 1;
                e32 -= 1;
            }
            let f32m = f32m & 0x03ff;
            (s << 31) | ((e32 as u32) << 23) | (f32m << 13)
        }
    } else if e == 31 {
        if f == 0 {
            (s << 31) | 0x7f800000
        } else {
            (s << 31) | 0x7f800000 | (f << 13)
        }
    } else {
        let e32 = e + (127 - 15);
        (s << 31) | (e32 << 23) | (f << 13)
    };
    f32::from_le_bytes(bits.to_le_bytes())
}

fn bf16_to_f32_bits(h: u16) -> f32 {
    let bits: u32 = (h as u32) << 16;
    f32::from_le_bytes(bits.to_le_bytes())
}

fn read_view_to_f32_vec(tv: &TensorView) -> Result<Vec<f32>, String> {
    let data = tv.data();
    Ok(match tv.dtype() {
        Dtype::F32 => {
            let mut out = vec![0f32; data.len() / 4];
            for (i, ch) in data.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
            }
            out
        }
        Dtype::F16 => {
            let mut out = vec![0f32; data.len() / 2];
            for (i, ch) in data.chunks_exact(2).enumerate() {
                let h = u16::from_le_bytes([ch[0], ch[1]]);
                out[i] = f16_to_f32_bits(h);
            }
            out
        }
        Dtype::BF16 => {
            let mut out = vec![0f32; data.len() / 2];
            for (i, ch) in data.chunks_exact(2).enumerate() {
                let h = u16::from_le_bytes([ch[0], ch[1]]);
                out[i] = bf16_to_f32_bits(h);
            }
            out
        }
        other => return Err(format!("Nicht unterstützter Dtype: {:?}", other)),
    })
}

fn load_2d(st: &SafeTensors, name: &str, dev: &Device) -> Result<Tensor, String> {
    let tv = st.tensor(name).map_err(|e| format!("{}: {}", name, e))?;
    let shape = tv.shape();
    if shape.len() != 2 {
        return Err(format!("{} ist nicht 2D (shape={:?})", name, shape));
    }
    let v = read_view_to_f32_vec(&tv)?;
    let (a, b) = (shape[0], shape[1]);
    Tensor::from_vec(v, (a, b), dev).map_err(|e| e.to_string())
}

fn load_1d(st: &SafeTensors, name: &str, dev: &Device) -> Result<Tensor, String> {
    let tv = st.tensor(name).map_err(|e| format!("{}: {}", name, e))?;
    let shape = tv.shape();
    if shape.len() != 1 {
        return Err(format!("{} ist nicht 1D (shape={:?})", name, shape));
    }
    let v = read_view_to_f32_vec(&tv)?;
    let n = shape[0];
    Tensor::from_vec(v, n, dev).map_err(|e| e.to_string())
}

fn try_load_lm_head(
    st: &SafeTensors,
    dev: &Device,
    hidden: usize,
    vocab: usize,
) -> Result<Tensor, String> {
    let cand = [
        "lm_head.weight",
        "model.lm_head.weight",
        "output.weight",
        "model.output.weight",
    ];
    for name in cand {
        if let Ok(w) = load_2d(st, name, dev) {
            let (a, b) = w.dims2().map_err(|e| e.to_string())?;
            if a == hidden && b == vocab {
                return Ok(w);
            } else if a == vocab && b == hidden {
                return w.t().map_err(|e| e.to_string());
            } else {
                let wt = w.t().map_err(|e| e.to_string())?;
                let (aa, bb) = wt.dims2().map_err(|e| e.to_string())?;
                if aa == hidden && bb == vocab {
                    return Ok(wt);
                }
            }
        }
    }
    Err("lm_head nicht gefunden".to_string())
}

// ------------------------------------------------------------
// HF Llama Config (Auszug)
// ------------------------------------------------------------
#[derive(Debug, Deserialize)]
struct HfLlamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

// ------------------------------------------------------------
// Layer-Gewichte und Modell
// ------------------------------------------------------------
struct LayerWeights {
    // Attention Projektionen
    w_q: Tensor, // [hidden, q_out]
    w_k: Tensor, // [hidden, k_out]
    w_v: Tensor, // [hidden, v_out]
    w_o: Tensor, // Rohform; im Forward ggf. transponieren

    // Normen
    attn_norm_w: Tensor, // [hidden]
    ffn_norm_w: Tensor,  // [hidden]

    // MLP (SwiGLU)
    w_gate: Tensor, // [hidden, inter]
    w_up: Tensor,   // [hidden, inter]
    w_down: Tensor, // [inter, hidden]
}

pub struct TransformerModel {
    dev: Device,

    // Größen
    i_vocab_size: usize,
    i_hidden_size: usize,
    i_n_layers: usize,
    i_n_heads_hint: usize,
    i_n_kv_heads_hint: usize,

    // RoPE
    d_rope_base: f32,
    i_rope_dim_hint: usize,

    // Norm
    d_rms_eps: f32,

    // Aktiver DType des Modells
    model_dtype: DType,

    // Embeddings / Head
    emb_vh: Tensor,     // [vocab, hidden] in model_dtype
    lm_head_hv: Tensor, // [hidden, vocab] in model_dtype
    output_norm_w: Option<Tensor>,

    // Layer
    layers: Vec<LayerWeights>,
}

// ------------------------------------------------------------
// Konstruktor
// ------------------------------------------------------------
impl TransformerModel {
    pub fn from_safetensors(
        weights_path: &str,
        config_json: &str,
        dtype: DType,
    ) -> Result<Self, String> {
        let dev = Device::Cpu;

        // Config laden
        let cfg_bytes = std::fs::read(config_json)
            .map_err(|e| format!("config.json lesen fehlgeschlagen: {}", e))?;
        let cfg: HfLlamaConfig =
            serde_json::from_slice(&cfg_bytes).map_err(|e| format!("config.json parse: {}", e))?;

        let i_hidden = cfg.hidden_size;
        let i_vocab = cfg.vocab_size;
        let i_layers = cfg.num_hidden_layers;
        let i_heads_hint = cfg.num_attention_heads.max(1);
        let i_kv_heads = cfg.num_key_value_heads.unwrap_or(i_heads_hint).max(1);
        let d_rope_base = cfg.rope_theta.unwrap_or(10000.0);
        let i_rope_dim_hint = i_hidden / i_heads_hint;
        let d_rms_eps = cfg.rms_norm_eps.unwrap_or(1e-5);

        // Weights laden
        let w_bytes = std::fs::read(weights_path)
            .map_err(|e| format!("safetensors lesen fehlgeschlagen: {}", e))?;
        let st = SafeTensors::deserialize(&w_bytes)
            .map_err(|e| format!("safetensors parse fehlgeschlagen: {}", e))?;

        if debug_on() {
            let mut shown = 0usize;
            for n in st.names() {
                if n.contains("head") || n.contains("output") {
                    if shown < 16 {
                        println!("[DBG] key: {}", n);
                    }
                    shown += 1;
                }
            }
            if shown > 16 {
                println!("[DBG] ... {} weitere", shown - 16);
            }
        }

        // Embedding f32 -> dtype
        let emb_vh_f32 = load_2d(&st, "model.embed_tokens.weight", &dev)?;
        let emb_vh = to_dtype_if_needed(emb_vh_f32.clone(), dtype)?;

        // lm_head: try -> sonst Weight-Tying
        let tying = cfg.tie_word_embeddings.unwrap_or(false);
        let lm_head_hv_f32 = if tying {
            if debug_on() {
                println!("[INFO] tie_word_embeddings=true -> Weight Tying");
            }
            emb_vh_f32.t().map_err(|e| e.to_string())?
        } else {
            match try_load_lm_head(&st, &dev, i_hidden, i_vocab) {
                Ok(t) => t,
                Err(_) => {
                    if debug_on() {
                        println!("[WARN] lm_head fehlt -> Weight Tying via embed_tokens");
                    }
                    emb_vh_f32.t().map_err(|e| e.to_string())?
                }
            }
        };
        let lm_head_hv = to_dtype_if_needed(lm_head_hv_f32, dtype)?;

        if debug_on() {
            let (ev, eh) = emb_vh.dims2().map_err(|e| e.to_string())?;
            let (lh, lv) = lm_head_hv.dims2().map_err(|e| e.to_string())?;
            let emb_ok = ev == i_vocab && eh == i_hidden;
            let head_ok = lh == i_hidden && lv == i_vocab;
            println!(
                "[CHK-EMB] emb_vh dims (vocab,hidden)=({},{}) expected=({},{}) -> {}",
                ev,
                eh,
                i_vocab,
                i_hidden,
                if emb_ok { "OK" } else { "MISMATCH" }
            );
            println!(
                "[CHK-HEAD] lm_head_hv dims (hidden,vocab)=({},{}) expected=({},{}) -> {}",
                lh,
                lv,
                i_hidden,
                i_vocab,
                if head_ok { "OK" } else { "MISMATCH" }
            );
        }

        // optionale finale Norm
        let output_norm_w = match st.tensor("model.norm.weight") {
            Ok(_) => {
                let w = load_1d(&st, "model.norm.weight", &dev)?;
                Some(to_dtype_if_needed(w, dtype)?)
            }
            Err(_) => None,
        };

        // Layer
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(i_layers);
        for l in 0..i_layers {
            let pre = format!("model.layers.{}", l);

            let w_q = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.self_attn.q_proj.weight", pre), &dev)?,
                    i_hidden,
                )?,
                dtype,
            )?;
            let w_k = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.self_attn.k_proj.weight", pre), &dev)?,
                    i_hidden,
                )?,
                dtype,
            )?;
            let w_v = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.self_attn.v_proj.weight", pre), &dev)?,
                    i_hidden,
                )?,
                dtype,
            )?;
            // o_proj Rohform (Transpose im Forward bei Bedarf)
            let w_o = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.self_attn.o_proj.weight", pre), &dev)?,
                    i_hidden, // Eingangs-Dim = hidden
                )?,
                dtype,
            )?;

            let attn_norm_w = to_dtype_if_needed(
                load_1d(&st, &format!("{}.input_layernorm.weight", pre), &dev)?,
                dtype,
            )?;
            let ffn_norm_w = to_dtype_if_needed(
                load_1d(
                    &st,
                    &format!("{}.post_attention_layernorm.weight", pre),
                    &dev,
                )?,
                dtype,
            )?;

            let w_gate = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.mlp.gate_proj.weight", pre), &dev)?,
                    i_hidden,
                )?,
                dtype,
            )?;
            let w_up = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.mlp.up_proj.weight", pre), &dev)?,
                    i_hidden,
                )?,
                dtype,
            )?;
            let (_, inter_u) = w_up.dims2().map_err(|e| e.to_string())?;
            let w_down = to_dtype_if_needed(
                orient_for_right_matmul(
                    &load_2d(&st, &format!("{}.mlp.down_proj.weight", pre), &dev)?,
                    inter_u,
                )?,
                dtype,
            )?;

            if debug_on() && l == 0 {
                let (q_in, q_out) = tensor_dims_2d(&w_q)?;
                let (k_in, k_out) = tensor_dims_2d(&w_k)?;
                let (v_in, v_out) = tensor_dims_2d(&w_v)?;
                let (wo_a, wo_b) = tensor_dims_2d(&w_o)?;
                println!(
                    "[CHK-L0] Q=({},{}), K=({},{}), V=({},{}), O=({},{})",
                    q_in, q_out, k_in, k_out, v_in, v_out, wo_a, wo_b
                );
            }

            layers.push(LayerWeights {
                w_q,
                w_k,
                w_v,
                w_o,
                attn_norm_w,
                ffn_norm_w,
                w_gate,
                w_up,
                w_down,
            });
        }

        Ok(Self {
            dev,
            i_vocab_size: i_vocab,
            i_hidden_size: i_hidden,
            i_n_layers: i_layers,
            i_n_heads_hint: i_heads_hint,
            i_n_kv_heads_hint: i_kv_heads,
            d_rope_base,
            i_rope_dim_hint,
            d_rms_eps,
            model_dtype: dtype,
            emb_vh,
            lm_head_hv,
            output_norm_w,
            layers,
        })
    }

    // --------------------------------------------------------
    // Forward: Logits (letztes Token) als Vec<f32>
    // --------------------------------------------------------
    pub fn forward_tokens(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("forward_tokens: leere Eingabe".to_string());
        }

        // Embedding stapeln: [T, hidden] (bereits model_dtype)
        let mut rows: Vec<Tensor> = Vec::with_capacity(ids.len());
        for &tid in ids {
            let i = tid as usize;
            if i >= self.i_vocab_size {
                return Err(format!(
                    "Token-ID {} >= vocab_size {}",
                    i, self.i_vocab_size
                ));
            }
            let row = self
                .emb_vh
                .narrow(0, i, 1)
                .map_err(|e| e.to_string())?
                .squeeze(0)
                .map_err(|e| e.to_string())?;
            rows.push(row);
        }
        let mut x = Tensor::stack(&rows, 0).map_err(|e| e.to_string())?; // [T, hidden]
        let t = ids.len();

        // Maske (selber DType wie x)
        let mask = causal_mask(&self.dev, t, x.dtype())?;

        // Layer
        for (_l_idx, lw) in self.layers.iter().enumerate() {
            // Pre-Attention RMSNorm
            let x_attn_in = rms_norm(&x, &lw.attn_norm_w, self.d_rms_eps)?;

            // Proj
            let q = x_attn_in.matmul(&lw.w_q).map_err(|e| e.to_string())?; // [T, q_w]
            let k = x_attn_in.matmul(&lw.w_k).map_err(|e| e.to_string())?; // [T, k_w]
            let v = x_attn_in.matmul(&lw.w_v).map_err(|e| e.to_string())?; // [T, v_w]
            let (_tq, q_w) = tensor_dims_2d(&q)?;
            let (_tk, k_w) = tensor_dims_2d(&k)?;
            let (_tv, v_w) = tensor_dims_2d(&v)?;

            // Head-Aufteilung
            let base_head_dim = if self.i_n_heads_hint > 0 {
                self.i_hidden_size / self.i_n_heads_hint
            } else {
                0
            };
            let head_dim = if base_head_dim > 0 && q_w % base_head_dim == 0 {
                base_head_dim
            } else {
                best_divisor(q_w, base_head_dim.max(1))
            };
            if head_dim == 0 || q_w % head_dim != 0 || k_w % head_dim != 0 || v_w % head_dim != 0 {
                return Err(format!(
                    "Head-Aufteilung passt nicht: q={}, k={}, v={}, head_dim={}",
                    q_w, k_w, v_w, head_dim
                ));
            }
            let n_heads_eff = q_w / head_dim;
            let n_kv_heads_eff = (k_w / head_dim).max(1);
            if (v_w / head_dim).max(1) != n_kv_heads_eff {
                return Err("k/v-Heads ungleich".to_string());
            }

            // in [heads, T, head_dim]
            let q = q
                .reshape((t, n_heads_eff, head_dim))
                .map_err(|e| e.to_string())?
                .transpose(0, 1)
                .map_err(|e| e.to_string())?;
            let k = k
                .reshape((t, n_kv_heads_eff, head_dim))
                .map_err(|e| e.to_string())?
                .transpose(0, 1)
                .map_err(|e| e.to_string())?;
            let v = v
                .reshape((t, n_kv_heads_eff, head_dim))
                .map_err(|e| e.to_string())?
                .transpose(0, 1)
                .map_err(|e| e.to_string())?;

            // RoPE
            let rope_dim_eff = (self.i_rope_dim_hint.min(head_dim).max(2)) & !1;
            let (cos, sin) =
                build_rope_cos_sin(&self.dev, t, rope_dim_eff, self.d_rope_base, x.dtype())?;

            // GQA: teile Q auf KV-Heads (mod)
            let mut ctx_heads: Vec<Tensor> = Vec::with_capacity(n_heads_eff);
            for h in 0..n_heads_eff {
                let hk = h % n_kv_heads_eff;

                let qh = q
                    .narrow(0, h, 1)
                    .map_err(|e| e.to_string())?
                    .squeeze(0)
                    .map_err(|e| e.to_string())?;
                let kh = k
                    .narrow(0, hk, 1)
                    .map_err(|e| e.to_string())?
                    .squeeze(0)
                    .map_err(|e| e.to_string())?;
                let vh = v
                    .narrow(0, hk, 1)
                    .map_err(|e| e.to_string())?
                    .squeeze(0)
                    .map_err(|e| e.to_string())?;

                let qh = apply_rope_partial(&qh, &cos, &sin, rope_dim_eff)?;
                let kh = apply_rope_partial(&kh, &cos, &sin, rope_dim_eff)?;

                // scores [T,T]
                let kt = kh.transpose(0, 1).map_err(|e| e.to_string())?;
                let mut scores = qh.matmul(&kt).map_err(|e| e.to_string())?;
                let scale = (head_dim as f32).sqrt();
                let scale_t = Tensor::new(scale, &self.dev)
                    .map_err(|e| e.to_string())?
                    .to_dtype(x.dtype())
                    .map_err(|e| e.to_string())?;
                scores = scores.broadcast_div(&scale_t).map_err(|e| e.to_string())?;
                scores = scores.broadcast_add(&mask).map_err(|e| e.to_string())?;
                let attn = nn_ops::softmax(&scores, 1).map_err(|e| e.to_string())?;
                let ctx = attn.matmul(&vh).map_err(|e| e.to_string())?;
                ctx_heads.push(ctx);
            }

            // concat heads -> [T, q_w]
            let ctx = Tensor::cat(&ctx_heads, 1).map_err(|e| e.to_string())?;

            // o-Proj ausrichten
            let (wo_a, wo_b) = tensor_dims_2d(&lw.w_o)?;
            let w_o = if wo_a == q_w {
                lw.w_o.clone()
            } else if wo_b == q_w {
                lw.w_o.t().map_err(|e| e.to_string())?
            } else {
                return Err(format!(
                    "o_proj Form ungueltig: erwartet {}, ist {} x {}",
                    q_w, wo_a, wo_b
                ));
            };

            // Attention-Output + Residual
            let attn_out = ctx.matmul(&lw.w_o).map_err(|e| e.to_string())?;

            let x_res1 = x.broadcast_add(&attn_out).map_err(|e| e.to_string())?;

            // Pre-FFN RMSNorm
            let x_mlp_in = rms_norm(&x_res1, &lw.ffn_norm_w, self.d_rms_eps)?;

            // MLP: SwiGLU
            let gate = x_mlp_in.matmul(&lw.w_gate).map_err(|e| e.to_string())?;
            let up = x_mlp_in.matmul(&lw.w_up).map_err(|e| e.to_string())?;
            let act = nn_ops::silu(&gate).map_err(|e| e.to_string())?;
            let ff = act.broadcast_mul(&up).map_err(|e| e.to_string())?;
            let down = ff.matmul(&lw.w_down).map_err(|e| e.to_string())?;

            x = x_res1.broadcast_add(&down).map_err(|e| e.to_string())?;
        }

        // Finale Norm optional
        let x_out = if let Some(w) = &self.output_norm_w {
            rms_norm(&x, w, self.d_rms_eps)?
        } else {
            x
        };

        // Logits letztes Token
        let last = x_out
            .narrow(0, t - 1, 1)
            .map_err(|e| e.to_string())?
            .squeeze(0)
            .map_err(|e| e.to_string())?;
        let last = last.unsqueeze(0).map_err(|e| e.to_string())?;
        let logits = last.matmul(&self.lm_head_hv).map_err(|e| e.to_string())?;
        let logits = logits.squeeze(0).map_err(|e| e.to_string())?;

        // In f32 konvertieren (unabhängig vom Rechen-DType)
        let logits_f32 = logits.to_dtype(DType::F32).map_err(|e| e.to_string())?;
        let v_logits: Vec<f32> = logits_f32.to_vec1::<f32>().map_err(|e| e.to_string())?;

        if debug_on() {
            let ok = v_logits.len() == self.i_vocab_size;
            println!(
                "[CHK-LOGITS] len={} vs vocab={} -> {}",
                v_logits.len(),
                self.i_vocab_size,
                if ok { "OK" } else { "MISMATCH" }
            );
            let (mn, mx) = v_logits
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
                    (a.min(x), b.max(x))
                });
            println!("[LOGITS] min/max = {:.5}/{:.5}", mn, mx);
        }

        Ok(v_logits)
    }

    pub fn vocab_size(&self) -> usize {
        self.i_vocab_size
    }
}

// ------------------------------------------------------------
// Sampler & Repetition Penalty (von main genutzt)
// ------------------------------------------------------------
pub fn prng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9e3779b97f4a7c15u64;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

pub fn sample_topk(logits: &[f32], temp: f32, k: usize, rng_state: &mut u64) -> usize {
    let t = if temp > 0.0 { temp } else { 1.0 };
    let mut pairs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / t))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_len = pairs.len().min(k.max(1));
    let top = &pairs[..top_len];
    let max_v = top
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = top.iter().map(|(_, v)| (*v - max_v).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    let r = {
        let x = prng_next(rng_state);
        (x as f64) / (u64::MAX as f64)
    };
    let mut acc = 0f64;
    for (idx, p) in top.iter().map(|(i, _)| *i).zip(probs.iter()) {
        acc += *p as f64;
        if r <= acc {
            return idx;
        }
    }
    top[0].0
}

pub fn apply_repetition_penalty(v_logits: &mut [f32], v_recent: &[u32], penalty: f32) {
    let p = penalty.max(1.0);
    if p == 1.0 || v_recent.is_empty() {
        return;
    }
    use std::collections::HashSet;
    let mut seen: HashSet<u32> = HashSet::new();
    for &id in v_recent {
        seen.insert(id);
    }
    for (i, logit) in v_logits.iter_mut().enumerate() {
        if seen.contains(&(i as u32)) {
            if *logit > 0.0 {
                *logit /= p;
            } else {
                *logit *= p;
            }
        }
    }
}

pub fn sample_topk_topp_minp_with_repeat(
    logits_in: &[f32],
    temp: f32,
    topk: usize,
    topp: f32,
    minp: f32,
    recent: &[u32],
    rep_penalty: f32,
    rng_state: &mut u64,
) -> usize {
    let mut scores: Vec<f32> = logits_in.to_vec();
    apply_repetition_penalty(&mut scores, recent, rep_penalty);

    let t = if temp > 0.0 { temp } else { 1.0 };
    for s in &mut scores {
        *s /= t;
    }
    let m = scores
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    for s in &mut scores {
        *s -= m;
    }
    let mut pairs: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.exp()))
        .collect();
    let sum: f32 = pairs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits_in.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        return best;
    }
    for (_, p) in &mut pairs {
        *p /= sum;
    }
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // top-k
    let slice_k: &[(usize, f32)] = if topk > 0 {
        let keep = pairs.len().min(topk);
        &pairs[..keep]
    } else {
        &pairs[..]
    };

    // top-p
    let use_p = if topp <= 0.0 || !topp.is_finite() {
        1.0
    } else {
        topp.min(1.0)
    };
    let mut keep = slice_k.len();
    if use_p < 1.0 {
        let mut acc = 0.0f32;
        keep = 0usize;
        for &(_, pr) in slice_k.iter() {
            acc += pr;
            keep += 1;
            if acc >= use_p {
                break;
            }
        }
        if keep == 0 {
            keep = 1;
        }
    }
    let mut work: Vec<(usize, f32)> = slice_k[..keep].to_vec();

    // min-p
    if minp > 0.0 && minp.is_finite() {
        let pmax = work.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
        let th = pmax * minp;
        work.retain(|(_, p)| *p >= th);
        if work.is_empty() {
            work.push(slice_k[0]);
        }
    }

    // Ziehen
    let ren: f32 = work.iter().map(|(_, p)| *p).sum();
    let probs: Vec<f32> = if ren > 0.0 && ren.is_finite() {
        work.iter().map(|(_, p)| *p / ren).collect()
    } else {
        let mut v = vec![0.0f32; work.len()];
        if !v.is_empty() {
            v[0] = 1.0;
        }
        v
    };
    let r = {
        let x = prng_next(rng_state);
        (x as f64) / (u64::MAX as f64)
    };
    let mut acc = 0.0f64;
    for ((idx, _), p) in work.iter().zip(probs.iter()) {
        acc += *p as f64;
        if r <= acc {
            return *idx;
        }
    }
    work[0].0
}
