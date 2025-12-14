// model.rs
// Zweck: Alles, was das Modell repraesentiert
//  - Konfiguration aus GGUF laden (build_config, robust fuer mehrere Architekturen)
//  - Gewichte aus GGUF in das interne Modell abbilden (map_all_weights)
//  - Hilfsfunktionen fuer das sichere Umordnen von ggml-Matrizen
//  - Schaltbare Debug-Ausgaben
// Autor: Marcus Schlieper, ExpChat.ai
// Datum: 2025-12-08
// Sicherheit: kein unsafe, durchgehende Fehlerbehandlung

use crate::gguf_loader::{GgufModel, GgufTensor, GgufValue};
use crate::layer::{Linear, ModelConfig, TransformerModel};
use std::sync::atomic::{AtomicBool, Ordering};

pub static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn init_debug_from_env() {
    let on = std::env::var("MODEL_DEBUG")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    set_debug(on);
}

macro_rules! mdl_dbg {
    ($($arg:tt)*) => {
        if $crate::model::DEBUG_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
            println!($($arg)*);
        }
    }
}

pub fn mean_abs(v_vals: &[f32]) -> f32 {
    if v_vals.is_empty() {
        return 0.0;
    }
    let mut d_sum = 0.0;
    for &x in v_vals {
        d_sum += x.abs();
    }
    d_sum / (v_vals.len() as f32)
}

// ============ Hilfsfunktionen: robustes Key-Lesen ============

fn kv_u32_any(kv: &std::collections::HashMap<String, GgufValue>, keys: &[String]) -> Option<u32> {
    for k in keys {
        match kv.get(k) {
            Some(GgufValue::U32(v)) => return Some(*v),
            Some(GgufValue::U64(v)) => return Some(*v as u32),
            Some(GgufValue::I32(v)) => return Some((*v).max(0) as u32),
            Some(GgufValue::I64(v)) => return Some((*v).max(0) as u32),
            _ => {}
        }
    }
    None
}

fn kv_f32_any(kv: &std::collections::HashMap<String, GgufValue>, keys: &[String]) -> Option<f32> {
    for k in keys {
        match kv.get(k) {
            Some(GgufValue::F32(v)) => return Some(*v),
            Some(GgufValue::F64(v)) => return Some(*v as f32),
            Some(GgufValue::U32(v)) => return Some(*v as f32),
            Some(GgufValue::U64(v)) => return Some(*v as f32),
            Some(GgufValue::I32(v)) => return Some(*v as f32),
            Some(GgufValue::I64(v)) => return Some(*v as f32),
            _ => {}
        }
    }
    None
}

fn arch_keys(arch: &str, base: &str) -> Vec<String> {
    // Liefert Prioritätenliste: arch.base, base, llama.base (fallback)
    vec![
        format!("{}.{}", arch, base),
        base.to_string(),
        format!("llama.{}", base),
    ]
}

fn many_arch_aliases(arch: &str, bases: &[&str]) -> Vec<String> {
    // baut fuer jede basis die drei Varianten wie oben
    let mut out = Vec::new();
    for b in bases {
        out.push(format!("{}.{}", arch, b));
        out.push((*b).to_string());
        out.push(format!("llama.{}", b));
    }
    out
}

fn find_tensor<'a>(
    m: &'a std::collections::HashMap<String, GgufTensor>,
    names: &[&str],
) -> Option<&'a GgufTensor> {
    for n in names {
        if let Some(t) = m.get(*n) {
            return Some(t);
        }
    }
    None
}

// ============ GGML -> Row-Major Mapping mit strengen Checks ============

fn map_ggml_2d_to_rowmajor_labeled(
    label: &str,
    v_dst: &mut [f32],
    i_rows: usize,
    i_cols: usize,
    v_src: &[f32],
    i_ne0: usize,
    i_ne1: usize,
) -> Result<(), String> {
    let i_dst_need = i_rows
        .checked_mul(i_cols)
        .ok_or_else(|| "dst size overflow".to_string())?;
    let i_src_have = i_ne0
        .checked_mul(i_ne1)
        .ok_or_else(|| "src size overflow".to_string())?;
    if v_dst.len() != i_dst_need {
        return Err(format!(
            "dst len mismatch: {} != {}",
            v_dst.len(),
            i_dst_need
        ));
    }
    if v_src.len() != i_src_have {
        return Err(format!(
            "src len mismatch: {} != {}",
            v_src.len(),
            i_src_have
        ));
    }

    // Branch A: gleiche Orientierung (ggml rows = ne1, cols = ne0)
    if i_ne1 == i_rows && i_ne0 == i_cols {
        mdl_dbg!(
            "map {}: A (no transpose) rows={} cols={} ne0={} ne1={}",
            label,
            i_rows,
            i_cols,
            i_ne0,
            i_ne1
        );
        for r in 0..i_rows {
            for c in 0..i_cols {
                let src_idx = c + r * i_ne0;
                let dst_idx = r * i_cols + c;
                v_dst[dst_idx] = v_src[src_idx];
            }
        }
        return Ok(());
    }

    // Branch B: transponiert gespeichert
    if i_ne1 == i_cols && i_ne0 == i_rows {
        mdl_dbg!(
            "map {}: B (transpose) rows={} cols={} ne0={} ne1={}",
            label,
            i_rows,
            i_cols,
            i_ne0,
            i_ne1
        );
        for r in 0..i_rows {
            for c in 0..i_cols {
                let src_r = c;
                let src_c = r;
                let src_idx = src_c + src_r * i_ne0;
                let dst_idx = r * i_cols + c;
                v_dst[dst_idx] = v_src[src_idx];
            }
        }
        return Ok(());
    }

    Err(format!(
        "unerwartete ggml shape: (ne0={}, ne1={}) passt nicht zu (rows={}, cols={})",
        i_ne0, i_ne1, i_rows, i_cols
    ))
}

fn ensure_linear_shape_or_err(
    in_dim: usize,
    out_dim: usize,
    ne0: usize,
    ne1: usize,
    name: &str,
) -> Result<(), String> {
    let ok = (ne1 == in_dim && ne0 == out_dim) || (ne1 == out_dim && ne0 == in_dim);
    if ok {
        Ok(())
    } else {
        Err(format!(
            "linear '{}' unexpected shape src=({}x{}), dst=({}x{})",
            name, ne1, ne0, in_dim, out_dim
        ))
    }
}

fn set_linear_weight_from_tensor(o_lin: &mut Linear, o_t: &GgufTensor) -> Result<(), String> {
    if o_t.shape.len() != 2 {
        return Err("tensor shape not 2D".to_string());
    }
    let v_data = o_t.to_f32_vec()?;
    let i_ne0 = o_t.shape[0];
    let i_ne1 = o_t.shape[1];

    ensure_linear_shape_or_err(o_lin.in_dim, o_lin.out_dim, i_ne0, i_ne1, &o_t.name)?;

    mdl_dbg!(
        "map linear {}: src=({}x{}), dst=({}x{}), type={}",
        o_t.name,
        i_ne1,
        i_ne0,
        o_lin.in_dim,
        o_lin.out_dim,
        o_t.type_code
    );

    map_ggml_2d_to_rowmajor_labeled(
        &o_t.name,
        &mut o_lin.w,
        o_lin.in_dim,
        o_lin.out_dim,
        &v_data,
        i_ne0,
        i_ne1,
    )
}

fn set_vector_from_tensor(v_dst: &mut [f32], o_t: &GgufTensor) -> Result<(), String> {
    let v_data = o_t.to_f32_vec()?;
    if v_data.len() != v_dst.len() {
        return Err(format!(
            "vector len mismatch: {} != {}",
            v_data.len(),
            v_dst.len()
        ));
    }
    mdl_dbg!(
        "map vector {}: len={} type={}",
        o_t.name,
        v_data.len(),
        o_t.type_code
    );
    v_dst.copy_from_slice(&v_data);
    Ok(())
}

fn set_bias_from_tensor(v_dst: &mut [f32], o_t: &GgufTensor) -> Result<(), String> {
    let v_data = o_t.to_f32_vec()?;
    if v_data.len() != v_dst.len() {
        return Err(format!(
            "bias len mismatch: {} != {}",
            v_data.len(),
            v_dst.len()
        ));
    }
    mdl_dbg!(
        "map bias {}: len={} type={}",
        o_t.name,
        v_data.len(),
        o_t.type_code
    );
    v_dst.copy_from_slice(&v_data);
    Ok(())
}

fn tie_lm_head_from_tok_emb(o_model: &mut TransformerModel) {
    let h = o_model.cfg.hidden_size;
    let v = o_model.cfg.vocab_size;
    mdl_dbg!(
        "lm_head: tie from tok_emb (transpose) with hidden={}, vocab={}",
        h,
        v
    );
    for i_h in 0..h {
        for i_v in 0..v {
            o_model.lm_head[i_h * v + i_v] = o_model.tok_emb[i_v * h + i_h];
        }
    }
}

// ============ Build Config: erweiterte Keys und Aliase ============

pub fn build_config(o_gguf: &GgufModel) -> ModelConfig {
    let s_arch = o_gguf
        .get_kv_str("general.architecture")
        .unwrap_or_else(|| "llama".to_string())
        .to_lowercase();

    // Vokab
    let mut i_vocab_size = o_gguf
        .get_kv_u32("tokenizer.vocab_size")
        .map(|v| v as usize)
        .unwrap_or(0);
    if i_vocab_size == 0 {
        if let Some(GgufValue::ArrStr(v)) = o_gguf.kv.get("tokenizer.ggml.tokens") {
            i_vocab_size = v.len();
        }
    }
    if i_vocab_size == 0 {
        i_vocab_size = 32000;
    }

    // Embedding-Fallback (Form)
    let emb_t = find_tensor(
        &o_gguf.tensors,
        &[
            "tok_embeddings.weight",
            "token_embd.weight",
            "token_embeddings.weight",
            "embed_tokens.weight",
        ],
    );
    let (hidden_guess, vocab_guess) = if let Some(t) = emb_t {
        if t.shape.len() == 2 {
            (t.shape[0], t.shape[1])
        } else {
            (128usize, i_vocab_size)
        }
    } else {
        (128usize, i_vocab_size)
    };

    // Aliase pro Feld
    let hidden_keys = {
        let v = many_arch_aliases(&s_arch, &["embedding_length", "hidden_size"]);
        v
    };
    let layers_keys = many_arch_aliases(&s_arch, &["block_count"]);
    let heads_keys = {
        let mut v = many_arch_aliases(&s_arch, &["attention.head_count", "num_attention_heads"]);
        v.push("num_attention_heads".to_string());
        v
    };
    let kv_heads_keys = {
        let mut v = many_arch_aliases(&s_arch, &["attention.head_count_kv", "num_key_value_heads"]);
        v.push("num_key_value_heads".to_string());
        v
    };
    let ffn_keys = {
        let mut v = many_arch_aliases(&s_arch, &["feed_forward_length", "intermediate_size"]);
        v.push("intermediate_size".to_string());
        v
    };
    let ctx_keys = {
        let mut v = many_arch_aliases(&s_arch, &["context_length", "max_position_embeddings"]);
        v.push("max_position_embeddings".to_string());
        v
    };
    let rope_dim_keys = {
        let mut v = many_arch_aliases(&s_arch, &["rope.dimension_count"]);
        // weitere Aliase (NeoX)
        v.push("attention.rotary_ndims".to_string());
        v.push("gptneox.rope.dimension_count".to_string());
        v
    };
    let rope_base_keys = {
        // Bevorzugt theta, dann freq_base
        let mut v = many_arch_aliases(&s_arch, &["rope.theta", "rope.freq_base"]);
        v.push("rope.freq_base".to_string());
        v
    };
    let rms_eps_keys = {
        let v = vec![
            "rms_norm_eps".to_string(),
            format!("{}.rms_norm_eps", s_arch),
            "attention.layer_norm_rms_epsilon".to_string(),
            format!("{}.attention.layer_norm_rms_epsilon", s_arch),
            "llama.rms_norm_eps".to_string(),
        ];
        v
    };

    let i_hidden_size = kv_u32_any(&o_gguf.kv, &hidden_keys)
        .map(|v| v as usize)
        .unwrap_or(hidden_guess);

    let i_n_layers = kv_u32_any(&o_gguf.kv, &layers_keys)
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            let mut i = 0usize;
            loop {
                let p = format!("blk.{}", i);
                let has = o_gguf.tensors.keys().any(|k| k.starts_with(&p));
                if !has {
                    break;
                }
                i += 1;
            }
            i.max(1)
        });

    let i_n_heads = kv_u32_any(&o_gguf.kv, &heads_keys)
        .map(|v| v as usize)
        .unwrap_or((i_hidden_size / 64).max(1));

    let mut i_n_kv_heads = kv_u32_any(&o_gguf.kv, &kv_heads_keys)
        .map(|v| v as usize)
        .unwrap_or(i_n_heads);

    // Intermediate (FFN)
    let i_intermediate_size = kv_u32_any(&o_gguf.kv, &ffn_keys)
        .map(|v| v as usize)
        .or_else(|| {
            find_tensor(&o_gguf.tensors, &["blk.0.ffn_up.weight"]).and_then(|t| {
                if t.shape.len() == 2 {
                    Some(t.shape[1])
                } else {
                    None
                }
            })
        })
        .unwrap_or(i_hidden_size * 4);

    let i_max_seq_len = kv_u32_any(&o_gguf.kv, &ctx_keys)
        .map(|v| v as usize)
        .unwrap_or(2048);

    // Rope Dim, Default: head_dim
    let i_head_dim = (i_hidden_size / i_n_heads.max(1)).max(1);
    let i_rope_dim = kv_u32_any(&o_gguf.kv, &rope_dim_keys)
        .map(|v| v as usize)
        .unwrap_or(i_head_dim);

    // Rope Base (theta)
    let d_rope_base = if let Ok(s) = std::env::var("ROPE_THETA") {
        s.parse::<f32>().unwrap_or_else(|_| {
            kv_f32_any(&o_gguf.kv, &rope_base_keys).unwrap_or_else(|| {
                if s_arch.starts_with("qwen") {
                    1_000_000.0
                } else {
                    10000.0
                }
            })
        })
    } else {
        kv_f32_any(&o_gguf.kv, &rope_base_keys).unwrap_or_else(|| {
            if s_arch.starts_with("qwen") {
                1_000_000.0
            } else {
                10000.0
            }
        })
    };

    println!("Rope Base (theta)= {}", d_rope_base);
    
    // Rope Scaling
    let rope_scaling_type = o_gguf
        .get_kv_str("rope.scaling.type")
        .or_else(|| o_gguf.get_kv_str(&format!("{}.rope.scaling.type", s_arch)));
    let rope_scaling_factor = kv_f32_any(
        &o_gguf.kv,
        &vec![
            format!("{}.rope.scaling.factor", s_arch),
            "rope.scaling.factor".to_string(),
        ],
    );

    // RMSNorm eps
    let d_rms_eps = kv_f32_any(&o_gguf.kv, &rms_eps_keys).unwrap_or(1e-5);

    mdl_dbg!("rms_norm_eps from GGUF = {}", d_rms_eps);
    mdl_dbg!("rope_base (theta) from GGUF = {}", d_rope_base);

    if let Some(ref t) = rope_scaling_type {
        mdl_dbg!("rope.scaling.type = {}", t);
    }
    if let Some(f) = rope_scaling_factor {
        mdl_dbg!("rope.scaling.factor = {}", f);
    }

    // GQA Plausibilitaet aus Gewichten pruefen und Heads ggf. korrigieren (nur Debug)
    if let Some(tk) = o_gguf
        .tensors
        .get("blk.0.attn_k.weight")
        .or_else(|| o_gguf.tensors.get("layers.0.self_attn.k_proj.weight"))
        .or_else(|| o_gguf.tensors.get("layers.0.attention.wk.weight"))
    {
        let (ne0, ne1) = (tk.shape[0], tk.shape[1]);
        let out_k = if ne1 == i_hidden_size {
            ne0
        } else if ne0 == i_hidden_size {
            ne1
        } else {
            0
        };
        if out_k > 0 {
            let kv_guess = (out_k / i_head_dim).max(1);
            mdl_dbg!(
                "dbg kv_heads: from weights = {}, from kv = {}",
                kv_guess,
                i_n_kv_heads
            );
            if std::env::var("FORCE_KV_HEADS")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                mdl_dbg!("FORCE_KV_HEADS=1 -> override kv_heads to {}", kv_guess);
                i_n_kv_heads = kv_guess;
            }
        }
    }

    // Heads % KV-Heads muss aufgehen
    if i_n_kv_heads == 0 {
        i_n_kv_heads = 1;
    }
    assert!(
        i_n_heads % i_n_kv_heads == 0,
        "n_heads must be divisible by n_kv_heads"
    );

    let cfg = ModelConfig {
        vocab_size: i_vocab_size.max(vocab_guess),
        hidden_size: i_hidden_size,
        intermediate_size: i_intermediate_size,
        n_layers: i_n_layers,
        n_heads: i_n_heads,
        n_kv_heads: i_n_kv_heads,
        max_seq_len: i_max_seq_len,
        rope_dim: i_rope_dim,
        rope_base: d_rope_base,

        rms_eps: d_rms_eps,
        rope_scaling_type,
        rope_scaling_factor,
    };

    mdl_dbg!(
        "build_config: arch='{}' vocab={} hidden={} layers={} heads={} kv_heads={} ffn={} ctx={} rope_dim={} rope_base={}",
        s_arch,
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.intermediate_size,
        cfg.max_seq_len,
        cfg.rope_dim,
        cfg.rope_base
    );

    // Hinweis: d_rms_eps bitte in layer.rs in RMSNorm.new(cfg_rms_eps) uebernehmen.
    cfg
}

// die Layer zählen
pub fn count_layers_in_tensors(o_gguf: &crate::gguf_loader::GgufModel) -> usize {
    // Zähle fortlaufend i = 0,1,2,... solange wir irgendeinen Tensor
    // mit Präfix "blk.i." oder "layers.i." finden.
    let mut i = 0usize;
    loop {
        let p_blk = format!("blk.{}.", i);
        let p_old = format!("layers.{}.", i);
        let has = o_gguf
            .tensors
            .keys()
            .any(|k| k.starts_with(&p_blk) || k.starts_with(&p_old));
        if !has {
            break;
        }
        i += 1;
    }
    i
}

// ============ Mapping: Namensvarianten erweitern, Checks verschaerfen ============

pub fn map_all_weights(o_gguf: &GgufModel, o_model: &mut TransformerModel) -> Result<(), String> {
    mdl_dbg!("map_all_weights: start");

    // Embedding
    if let Some(t) = find_tensor(
        &o_gguf.tensors,
        &[
            "tok_embeddings.weight",
            "token_embd.weight",
            "token_embeddings.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
    ) {
        let v_data = t.to_f32_vec()?;
        let i_ne0 = t.shape[0];
        let i_ne1 = t.shape[1];
        mdl_dbg!(
            "map embedding {}: src=({}x{}), dst=({}x{}), type={}",
            t.name,
            i_ne1,
            i_ne0,
            o_model.cfg.vocab_size,
            o_model.cfg.hidden_size,
            t.type_code
        );
        map_ggml_2d_to_rowmajor_labeled(
            &t.name,
            &mut o_model.tok_emb,
            o_model.cfg.vocab_size,
            o_model.cfg.hidden_size,
            &v_data,
            i_ne0,
            i_ne1,
        )?;
    } else {
        mdl_dbg!("map embedding: MISSING");
    }

    // LM Head
    let mut lm_head_loaded = false;
    if let Some(t) = find_tensor(
        &o_gguf.tensors,
        &["lm_head.weight", "output.weight", "model.lm_head.weight"],
    ) {
        let v_data = t.to_f32_vec()?;
        let i_ne0 = t.shape[0];
        let i_ne1 = t.shape[1];
        mdl_dbg!(
            "map lm_head {}: src=({}x{}), dst=({}x{}), type={}",
            t.name,
            i_ne1,
            i_ne0,
            o_model.cfg.hidden_size,
            o_model.cfg.vocab_size,
            t.type_code
        );
        map_ggml_2d_to_rowmajor_labeled(
            &t.name,
            &mut o_model.lm_head,
            o_model.cfg.hidden_size,
            o_model.cfg.vocab_size,
            &v_data,
            i_ne0,
            i_ne1,
        )?;
        lm_head_loaded = true;
    } else {
        mdl_dbg!("map lm_head: no explicit head found -> tie from tok_emb");
        tie_lm_head_from_tok_emb(o_model);
        lm_head_loaded = true;
    }
    if !lm_head_loaded {
        return Err("lm_head could not be initialized".to_string());
    }

    // Final Norm
    if let Some(t) = o_gguf
        .tensors
        .get("output_norm.weight")
        .or_else(|| o_gguf.tensors.get("norm.weight"))
        .or_else(|| o_gguf.tensors.get("model.norm.weight"))
        .or_else(|| o_gguf.tensors.get("final_layernorm.weight"))
    {
        set_vector_from_tensor(&mut o_model.final_norm.weight, t)?;
    } else {
        mdl_dbg!("final_norm.weight: MISSING");
    }

    // Blocks
    let head_dim = o_model.cfg.hidden_size / o_model.cfg.n_heads.max(1);
    let kv_expected = o_model.cfg.n_kv_heads * head_dim;

    for i in 0..o_model.cfg.n_layers {
        let s_old = format!("layers.{}", i);
        let s_new = format!("blk.{}", i);

        // Normen: ln1
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attention_norm.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.attn_norm.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.input_layernorm.weight", s_old))
            })
        {
            set_vector_from_tensor(&mut o_model.blocks[i].ln1.weight, t)?;
        } else {
            mdl_dbg!("blk {} ln1.weight: MISSING", i);
        }

        // Normen: ln2
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.ffn_norm.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.ffn_norm.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.post_attention_layernorm.weight", s_old))
            })
        {
            set_vector_from_tensor(&mut o_model.blocks[i].ln2.weight, t)?;
        } else {
            mdl_dbg!("blk {} ln2.weight: MISSING", i);
        }

        // Attention Q
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attention.wq.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.attn_q.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.q_proj.weight", s_old))
            })
        {
            set_linear_weight_from_tensor(&mut o_model.blocks[i].attn.w_q, t)?;
        } else if o_gguf
            .tensors
            .contains_key(&format!("{}.self_attn.query_key_value.weight", s_old))
        {
            return Err(format!(
                "blk {} has fused QKV (query_key_value.weight): not supported in this loader",
                i
            ));
        } else {
            mdl_dbg!("blk {} attn.w_q.weight: MISSING", i);
        }

        // Attention K
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attention.wk.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.attn_k.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.k_proj.weight", s_old))
            })
        {
            // strenger Check fuer K-Outputdim: n_kv_heads * head_dim
            let ne_out = if t.shape[1] == o_model.blocks[i].attn.w_k.in_dim {
                t.shape[0]
            } else {
                t.shape[1]
            };
            if ne_out != kv_expected {
                mdl_dbg!(
                    "warn blk {} attn.k out dim {} != expected kv={} (n_kv_heads * head_dim)",
                    i,
                    ne_out,
                    kv_expected
                );
            }
            set_linear_weight_from_tensor(&mut o_model.blocks[i].attn.w_k, t)?;
        } else {
            mdl_dbg!("blk {} attn.w_k.weight: MISSING", i);
        }

        // Attention V
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attention.wv.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.attn_v.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.v_proj.weight", s_old))
            })
        {
            let ne_out = if t.shape[1] == o_model.blocks[i].attn.w_v.in_dim {
                t.shape[0]
            } else {
                t.shape[1]
            };
            if ne_out != kv_expected {
                mdl_dbg!(
                    "warn blk {} attn.v out dim {} != expected kv={} (n_kv_heads * head_dim)",
                    i,
                    ne_out,
                    kv_expected
                );
            }
            set_linear_weight_from_tensor(&mut o_model.blocks[i].attn.w_v, t)?;
        } else {
            mdl_dbg!("blk {} attn.w_v.weight: MISSING", i);
        }

        // Attention O
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attention.wo.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.attn_output.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.o_proj.weight", s_old))
            })
        {
            set_linear_weight_from_tensor(&mut o_model.blocks[i].attn.w_o, t)?;
        } else {
            mdl_dbg!("blk {} attn.w_o.weight: MISSING", i);
        }

        // Attention Bias (optional)
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attn_q.bias", s_new))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.q_proj.bias", s_old))
            })
        {
            set_bias_from_tensor(&mut o_model.blocks[i].attn.w_q.b, t)?;
        }
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attn_k.bias", s_new))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.k_proj.bias", s_old))
            })
        {
            set_bias_from_tensor(&mut o_model.blocks[i].attn.w_k.b, t)?;
        }
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attn_v.bias", s_new))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.v_proj.bias", s_old))
            })
        {
            set_bias_from_tensor(&mut o_model.blocks[i].attn.w_v.b, t)?;
        }
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.attn_output.bias", s_new))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.self_attn.o_proj.bias", s_old))
            })
        {
            set_bias_from_tensor(&mut o_model.blocks[i].attn.w_o.b, t)?;
        }

        // MLP: up (w1), gate (w3), down (w2)
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.feed_forward.w1.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.ffn_up.weight", s_new)))
            .or_else(|| o_gguf.tensors.get(&format!("{}.mlp.up_proj.weight", s_old)))
        {
            set_linear_weight_from_tensor(&mut o_model.blocks[i].ffn.w1, t)?;
        } else {
            mdl_dbg!("blk {} ffn.w1 (up) weight: MISSING", i);
        }

        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.feed_forward.w3.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.ffn_gate.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.mlp.gate_proj.weight", s_old))
            })
        {
            set_linear_weight_from_tensor(&mut o_model.blocks[i].ffn.w3, t)?;
        } else {
            mdl_dbg!("blk {} ffn.w3 (gate) weight: MISSING", i);
        }

        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.feed_forward.w2.weight", s_old))
            .or_else(|| o_gguf.tensors.get(&format!("{}.ffn_down.weight", s_new)))
            .or_else(|| {
                o_gguf
                    .tensors
                    .get(&format!("{}.mlp.down_proj.weight", s_old))
            })
        {
            set_linear_weight_from_tensor(&mut o_model.blocks[i].ffn.w2, t)?;
        } else {
            mdl_dbg!("blk {} ffn.w2 (down) weight: MISSING", i);
        }

        // MLP Bias (optional)
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.ffn_up.bias", s_new))
            .or_else(|| o_gguf.tensors.get(&format!("{}.mlp.up_proj.bias", s_old)))
        {
            set_bias_from_tensor(&mut o_model.blocks[i].ffn.w1.b, t)?;
        }
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.ffn_gate.bias", s_new))
            .or_else(|| o_gguf.tensors.get(&format!("{}.mlp.gate_proj.bias", s_old)))
        {
            set_bias_from_tensor(&mut o_model.blocks[i].ffn.w3.b, t)?;
        }
        if let Some(t) = o_gguf
            .tensors
            .get(&format!("{}.ffn_down.bias", s_new))
            .or_else(|| o_gguf.tensors.get(&format!("{}.mlp.down_proj.bias", s_old)))
        {
            set_bias_from_tensor(&mut o_model.blocks[i].ffn.w2.b, t)?;
        }
    }

    mdl_dbg!("map_all_weights: done");

    validate_layers_nonzero(o_model)?; // wirft Err mit Layer-Index bei Problem

    Ok(())
}

// ======================================================================================================
// model.rs
// Helper: prüft, ob ein Vektor mindestens einen Wert != 0 (mit Toleranz) enthält
fn has_nonzero_eps(v: &[f32], eps: f32) -> bool {
    v.iter().any(|&x| x.abs() > eps)
}

// Prüfe einen einzelnen Transformer-Block auf mindestens ein nicht-null Gewicht
fn block_has_nonzero_weights(blk: &crate::layer::TransformerBlock) -> bool {
    let eps = 1e-12;

    // LayerNorm-Gewichte
    if has_nonzero_eps(&blk.ln1.weight, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.ln2.weight, eps) {
        return true;
    }

    // Attention: Gewichte und Bias
    if has_nonzero_eps(&blk.attn.w_q.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.attn.w_q.b, eps) {
        return true;
    }

    if has_nonzero_eps(&blk.attn.w_k.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.attn.w_k.b, eps) {
        return true;
    }

    if has_nonzero_eps(&blk.attn.w_v.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.attn.w_v.b, eps) {
        return true;
    }

    if has_nonzero_eps(&blk.attn.w_o.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.attn.w_o.b, eps) {
        return true;
    }

    // FeedForward: up (w1), gate (w3), down (w2) + Bias
    if has_nonzero_eps(&blk.ffn.w1.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.ffn.w1.b, eps) {
        return true;
    }

    if has_nonzero_eps(&blk.ffn.w3.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.ffn.w3.b, eps) {
        return true;
    }

    if has_nonzero_eps(&blk.ffn.w2.w, eps) {
        return true;
    }
    if has_nonzero_eps(&blk.ffn.w2.b, eps) {
        return true;
    }

    false
}

// Öffentliche Validierung: alle Layer müssen mindestens ein nicht-null Gewicht haben
pub fn validate_layers_nonzero(model: &TransformerModel) -> Result<(), String> {
    for (i, blk) in model.blocks.iter().enumerate() {
        if !block_has_nonzero_weights(blk) {
            return Err(format!(
                "Layer {} hat nur Null-Gewichte (Verdacht auf Mapping-Fehler)",
                i
            ));
        }
    }
    Ok(())
}
