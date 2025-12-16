// layer.rs
// Transformer-Layer mit KV-Cache (GQA) und SwiGLU-MLP
// CPU-Pfad, kein unsafe, mit schaltbaren Debug-Ausgaben über MODEL_DEBUG="1"
// Autor: Marcus Schlieper, ExpChat.ai
// Datum: 2025-12-07
//
// Hinweise:
// - Linear-Gewichte werden als Row-Major [in_dim, out_dim] erwartet.
// - Attention nutzt RoPE mit frei wählbarer Basis (rope_base) und partial-dim (rope_dim).
// - SwiGLU: y = W2( SiLU(W_gate x) * (W_up x) ).
// - Debug-Ausgaben aktivieren: Umgebungsvariable MODEL_DEBUG=1 (oder "true") setzen.
//
// Historie:
// - 2025-12-07: Erste vollständige, robuste Version mit GQA, RoPE, SwiGLU und Debug-Schalter.

use crate::math::{
    add_bias, apply_rope_partial_base, hadamard_inplace, matmul_blocked, rms_norm, silu_inplace,
    softmax_inplace,
};
use std::sync::OnceLock;

// Debug-Schalter (lazy aus ENV gelesen)
fn dbg_on() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| {
        let v = std::env::var("MODEL_DEBUG").unwrap_or_default();
        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
    })
}

macro_rules! lyr_dbg {
    ($($arg:tt)*) => {
        if crate::layer::dbg_on() {
            println!($($arg)*);
        }
    };
}

// kleine Hilfsfunktion für kompakte Debug-Werte
fn mean_abs(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &x in v {
        s += x.abs();
    }
    s / (v.len() as f32)
}

// kleine Hilfsfunktion: wende optionales Scaling an
fn apply_rope_scaling_if_enabled(base: f32, kind: Option<&str>, factor: Option<f32>) -> f32 {
    let do_apply = std::env::var("ROPE_SCALE_APPLY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !do_apply {
        return base;
    }
    let f = factor.unwrap_or(1.0);
    let k = kind.unwrap_or("");
    // einfache, konservative Variante: base * factor
    // (richtige NTK/Yarn-Formeln sind komplexer und können später ergänzt werden)
    match k {
        "linear" | "llama3" | "yarn" | "ntk" => base * f,
        _ => base,
    }
}

// Konfiguration des Modells
#[derive(Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    pub rope_dim: usize,
    pub rope_base: f32,

    pub rms_eps: f32,
    pub rope_scaling_type: Option<String>, // z. B. "linear", "yarn", "ntk", "llama3"
    pub rope_scaling_factor: Option<f32>,  // z. B. 1.0 .. 8.0
}

// Lineare Projektion: out = x * W + b
#[derive(Clone)]
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f32>, // [in_dim, out_dim] Row-Major
    pub b: Vec<f32>, // [out_dim]
}

impl Linear {
    pub fn new(i_in: usize, i_out: usize) -> Self {
        Self {
            in_dim: i_in,
            out_dim: i_out,
            w: vec![0.0; i_in * i_out],
            b: vec![0.0; i_out],
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // x: [1, in_dim]
        let mut out = vec![0f32; self.out_dim];
        matmul_blocked(x, 1, self.in_dim, &self.w, self.out_dim, &mut out);
        add_bias(&mut out, &self.b);
        out
    }
}

// RMSNorm (pre-LN)
#[derive(Clone)]
pub struct RMSNorm {
    pub dim: usize,
    pub eps: f32,
    pub weight: Vec<f32>,
}

impl RMSNorm {
    pub fn new(i_dim: usize, d_eps: f32) -> Self {
        Self {
            dim: i_dim,
            eps: d_eps,
            weight: vec![1.0; i_dim],
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        let _scale = rms_norm(&mut y, self.eps);
        for i in 0..self.dim {
            y[i] *= self.weight[i];
        }
        y
    }
}

// Attention mit Grouped-Query Attention (GQA)
#[derive(Clone)]
pub struct Attention {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_dim: usize,
    pub rope_base: f32,

    pub w_q: Linear, // hidden -> hidden
    pub w_k: Linear, // hidden -> (n_kv_heads * head_dim)
    pub w_v: Linear, // hidden -> (n_kv_heads * head_dim)
    pub w_o: Linear, // hidden -> hidden

    pub k_cache: Vec<f32>, // [n_kv_heads, max_seq_len, head_dim]
    pub v_cache: Vec<f32>, // [n_kv_heads, max_seq_len, head_dim]
}

impl Attention {
    pub fn new(
        i_hidden: usize,
        i_heads: usize,
        i_kv_heads: usize,
        i_max_seq: usize,
        i_rope_dim: usize,
        i_rope_base: f32,
    ) -> Self {
        assert!(i_heads > 0, "n_heads must be > 0");
        assert!(
            i_hidden % i_heads == 0,
            "hidden_size must be divisible by n_heads"
        );

        let i_group = i_heads / i_kv_heads;
        let i_head_dim = i_hidden / i_heads;
        let i_kv_out = i_kv_heads * i_head_dim;

        /*if dbg_on() {
            println!(
                "Attention::new | heads={} kv_heads={} head_dim={} max_seq={} rope_dim={} rope_base={}",
                i_heads, i_kv_heads, i_head_dim, i_max_seq, i_rope_dim, i_rope_base
            );
        }*/

        Self {
            n_heads: i_heads,
            n_kv_heads: i_kv_heads,
            head_dim: i_head_dim,
            max_seq_len: i_max_seq,
            rope_dim: i_rope_dim,
            rope_base: i_rope_base,
            w_q: Linear::new(i_hidden, i_hidden),
            w_k: Linear::new(i_hidden, i_kv_out),
            w_v: Linear::new(i_hidden, i_kv_out),
            w_o: Linear::new(i_hidden, i_hidden),
            k_cache: vec![0.0; i_kv_heads * i_max_seq * i_head_dim],
            v_cache: vec![0.0; i_kv_heads * i_max_seq * i_head_dim],
        }
    }

    pub fn reset_cache(&mut self) {
        for v in self.k_cache.iter_mut() {
            *v = 0.0;
        }
        for v in self.v_cache.iter_mut() {
            *v = 0.0;
        }
    }

    #[inline]
    fn kv_index(&self, i_kvh: usize, i_pos: usize) -> usize {
        (i_kvh * self.max_seq_len + i_pos) * self.head_dim
    }

    pub fn forward(&mut self, x: &[f32], i_pos: usize) -> Vec<f32> {
        // positions größer als Kontext -> ring-buffer (schützt gegen oob)
        let i_pos_eff = if self.max_seq_len == 0 {
            0
        } else {
            i_pos % self.max_seq_len
        };
        /*if dbg_on() && i_pos != i_pos_eff {
            println!(
                "Attention::forward | warn: pos={} wrapped-> {} (max_seq_len={})",
                i_pos, i_pos_eff, self.max_seq_len
            );
        }*/

        // Projektionen
        let v_q_all = self.w_q.forward(x); // [n_heads * head_dim]
        let v_k_all = self.w_k.forward(x); // [n_kv_heads * head_dim]
        let v_v_all = self.w_v.forward(x); // [n_kv_heads * head_dim]

        /*if dbg_on() && i_pos_eff < 2 {
            println!(
                "dbg attn proj: pos={} mean|q|={:.6} mean|k|={:.6} mean|v|={:.6}",
                i_pos_eff,
                mean_abs(&v_q_all),
                mean_abs(&v_k_all),
                mean_abs(&v_v_all)
            );
        }*/

        // RoPE nur auf K (aktueller Schritt), dann in Cache
        for i_kvh in 0..self.n_kv_heads {
            let i_start = i_kvh * self.head_dim;
            let i_end = i_start + self.head_dim;
            let mut v_kh = v_k_all[i_start..i_end].to_vec();

            let mut v_dummy_q = vec![0f32; self.head_dim];
            apply_rope_partial_base(
                &mut v_dummy_q,
                &mut v_kh,
                self.head_dim,
                self.rope_dim,
                i_pos_eff,
                self.rope_base,
            );

            // schreibe K/V in Cache für pos_eff
            let i_k_base = self.kv_index(i_kvh, i_pos_eff);
            let i_v_base = i_k_base;

            for j in 0..self.head_dim {
                self.k_cache[i_k_base + j] = v_kh[j];
                self.v_cache[i_v_base + j] = v_v_all[i_start + j];
            }
        }

        // Gruppengröße für GQA
        let i_group = if self.n_kv_heads > 0 {
            // exakt, da oben geprüft
            self.n_heads / self.n_kv_heads
        } else {
            1
        };

        // Output Heads
        let mut v_out_heads = vec![0f32; self.n_heads * self.head_dim];

        // Für jeden Q-Head: zugehörigen KV-Head wählen, RoPE anwenden, Scores berechnen
        for i_h in 0..self.n_heads {
            let i_q_start = i_h * self.head_dim;
            let i_q_end = i_q_start + self.head_dim;

            let mut v_qh = v_q_all[i_q_start..i_q_end].to_vec();

            // RoPE auf Q
            let mut v_dummy_k = vec![0f32; self.head_dim];
            apply_rope_partial_base(
                &mut v_qh,
                &mut v_dummy_k,
                self.head_dim,
                self.rope_dim,
                i_pos_eff,
                self.rope_base,
            );

            // zugeordneter KV-Head (geclamped)
            let mut i_kvh = i_h / i_group;
            if self.n_kv_heads > 0 {
                if i_kvh >= self.n_kv_heads {
                    i_kvh = self.n_kv_heads - 1;
                }
            } else {
                i_kvh = 0;
            }

            let i_steps = if i_pos_eff + 1 > self.max_seq_len {
                self.max_seq_len
            } else {
                i_pos_eff + 1
            };
            let mut v_scores = vec![0f32; i_steps];

            for t in 0..i_steps {
                let i_kbase = self.kv_index(i_kvh, t);
                let mut d_s = 0f32;
                for j in 0..self.head_dim {
                    d_s += v_qh[j] * self.k_cache[i_kbase + j];
                }
                v_scores[t] = d_s / (self.head_dim as f32).sqrt();
            }

            softmax_inplace(&mut v_scores);

            // Weighted sum über V
            let mut v_ctx = vec![0f32; self.head_dim];
            for t in 0..i_steps {
                let wt = v_scores[t];
                let i_vbase = self.kv_index(i_kvh, t);
                for j in 0..self.head_dim {
                    v_ctx[j] += wt * self.v_cache[i_vbase + j];
                }
            }

            // Sammle Head-Output
            for j in 0..self.head_dim {
                v_out_heads[i_q_start + j] = v_ctx[j];
            }
        }

        // Output-Projektion
        let y = self.w_o.forward(&v_out_heads);

        /*if dbg_on() && i_pos_eff < 2 {
            println!(
                "dbg attn out: pos={} mean|head_out|={:.6} mean|y|={:.6}",
                i_pos_eff,
                mean_abs(&v_out_heads),
                mean_abs(&y)
            );
        }*/

        y
    }
}

// FeedForward mit SwiGLU
#[derive(Clone)]
pub struct FeedForward {
    pub w1: Linear, // up
    pub w2: Linear, // down
    pub w3: Linear, // gate
}

impl FeedForward {
    pub fn new(i_hidden: usize, i_inter: usize) -> Self {
        Self {
            w1: Linear::new(i_hidden, i_inter),
            w2: Linear::new(i_inter, i_hidden),
            w3: Linear::new(i_hidden, i_inter),
        }
    }

    // SwiGLU korrekt: y = W2( SiLU(W_gate x) * (W_up x) )
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut v_gate = self.w3.forward(x); // gate
        silu_inplace(&mut v_gate);

        let v_up = self.w1.forward(x); // up

        let mut v_act = v_gate;
        hadamard_inplace(&mut v_act, &v_up);

        let y = self.w2.forward(&v_act); // down

        /*if dbg_on() {
            println!(
                "dbg ffn: mean|up|={:.6} mean|gate|={:.6} mean|act|={:.6} mean|y|={:.6}",
                mean_abs(&v_up),
                mean_abs(&v_act), // schon gate*silu(up), näherungsweise
                mean_abs(&v_act),
                mean_abs(&y)
            );
        }*/

        y
    }
}

// Ein Transformer-Block (Pre-LN)
#[derive(Clone)]
pub struct TransformerBlock {
    pub ln1: RMSNorm,
    pub attn: Attention,
    pub ln2: RMSNorm,
    pub ffn: FeedForward,
}

impl TransformerBlock {
    pub fn new(cfg: &ModelConfig) -> Self {
        /*if dbg_on() {
            println!(
                "TransformerBlock::new | hidden={} heads={} kv_heads={} inter={} max_seq={} rope_dim={} rope_base={} rms_eps={} rope_scaling={:?} factor={:?}",
                cfg.hidden_size,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.intermediate_size,
                cfg.max_seq_len,
                cfg.rope_dim,
                cfg.rope_base,
                cfg.rms_eps,
                cfg.rope_scaling_type,
                cfg.rope_scaling_factor
            );
        }*/

        // hier ggf. rope_base mit Scaling anwenden
        let rope_base_eff = apply_rope_scaling_if_enabled(
            cfg.rope_base,
            cfg.rope_scaling_type.as_deref(),
            cfg.rope_scaling_factor,
        );

        Self {
            // RMSNorm jetzt mit cfg.rms_eps
            ln1: RMSNorm::new(cfg.hidden_size, cfg.rms_eps),
            attn: Attention::new(
                cfg.hidden_size,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.max_seq_len,
                cfg.rope_dim,
                rope_base_eff, // der (ggf.) skalierte Wert
            ),
            ln2: RMSNorm::new(cfg.hidden_size, cfg.rms_eps),
            ffn: FeedForward::new(cfg.hidden_size, cfg.intermediate_size),
        }
    }

    pub fn reset_cache(&mut self) {
        self.attn.reset_cache();
    }

    pub fn forward(&mut self, x: &[f32], i_pos: usize) -> Vec<f32> {
        let a = self.ln1.forward(x);
        /*if dbg_on() && i_pos < 2 {
            println!("dbg ln1: pos={} mean|a|={:.6}", i_pos, mean_abs(&a));
        }*/

        let a2 = self.attn.forward(&a, i_pos);
        /*if dbg_on() && i_pos < 2 {
            println!("dbg attn: pos={} mean|a2|={:.6}", i_pos, mean_abs(&a2));
        }*/

        let mut h = vec![0f32; x.len()];
        for i in 0..x.len() {
            h[i] = x[i] + a2[i];
        }

        let m = self.ln2.forward(&h);
        let m2 = self.ffn.forward(&m);
        /*if dbg_on() && i_pos < 2 {
            println!("dbg ffn: pos={} mean|m2|={:.6}", i_pos, mean_abs(&m2));
        }*/

        let mut y = vec![0f32; h.len()];
        for i in 0..h.len() {
            y[i] = h[i] + m2[i];
        }

        y
    }
}

// Vollständiges Modell (nur Inferenz-Pfad next-token)
#[derive(Clone)]
pub struct TransformerModel {
    pub cfg: ModelConfig,
    pub tok_emb: Vec<f32>, // [vocab, hidden]
    pub lm_head: Vec<f32>, // [hidden, vocab]
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
}

impl TransformerModel {
    pub fn new_empty(cfg: ModelConfig) -> Self {
        /*if dbg_on() {
            println!(
                "TransformerModel::new_empty | vocab={} hidden={} layers={} heads={} kv_heads={} inter={} ctx={} rope_dim={} rope_base={} rms_eps={} rope_scaling={:?} factor={:?}",
                cfg.vocab_size,
                cfg.hidden_size,
                cfg.n_layers,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.intermediate_size,
                cfg.max_seq_len,
                cfg.rope_dim,
                cfg.rope_base,
                cfg.rms_eps,
                cfg.rope_scaling_type,
                cfg.rope_scaling_factor
            );
        }*/

        Self {
            tok_emb: vec![0.0; cfg.vocab_size * cfg.hidden_size],
            lm_head: vec![0.0; cfg.hidden_size * cfg.vocab_size],
            blocks: (0..cfg.n_layers)
                .map(|_| TransformerBlock::new(&cfg))
                .collect(),
            // finale Norm auch mit cfg.rms_eps
            final_norm: RMSNorm::new(cfg.hidden_size, cfg.rms_eps),
            cfg,
        }
    }

    pub fn reset_kv_cache(&mut self) {
        for blk in self.blocks.iter_mut() {
            blk.reset_cache();
        }
    }

    // Ein Schritt: ein Token, Position i_pos, KV-Cache wird aktualisiert
    pub fn forward_next(&mut self, i_token_id: usize, i_pos: usize) -> Result<Vec<f32>, String> {
        if i_token_id >= self.cfg.vocab_size {
            return Err("token id out of range".to_string());
        }

        // Embedding-Zeile holen
        let mut x = vec![0f32; self.cfg.hidden_size];
        let i_row = i_token_id * self.cfg.hidden_size;
        let row = &self.tok_emb[i_row..i_row + self.cfg.hidden_size];
        for i in 0..self.cfg.hidden_size {
            x[i] = row[i];
        }

        /*if dbg_on() && i_pos < 2 {
            println!(
                "dbg emb: pos={} tok_id={} mean|emb|={:.6}",
                i_pos,
                i_token_id,
                mean_abs(&x)
            );
        }*/

        // durch die Blöcke
        for (i_b, blk) in self.blocks.iter_mut().enumerate() {
            x = blk.forward(&x, i_pos);


            /*if dbg_on() && i_pos == 0 && i_b == 0 {
                println!(
                    "dbg block0 out: mean|x|={:.6} (nach attn+ffn)",
                    mean_abs(&x)
                );
            }*/
        }

        // finale Norm
        x = self.final_norm.forward(&x);

        /*if dbg_on() && i_pos < 2 {
            println!("dbg final norm: pos={} mean|x|={:.6}", i_pos, mean_abs(&x));
        }*/

        // LM-Head
        let mut logits = vec![0f32; self.cfg.vocab_size];
        matmul_blocked(
            &x,
            1,
            self.cfg.hidden_size,
            &self.lm_head,
            self.cfg.vocab_size,
            &mut logits,
        );

        /*if dbg_on() && i_pos < 2 {
            // Mini-Statistik statt Top-N hier (Top-N gern im Aufrufer)
            let mut max_abs = 0.0f32;
            for &v in &logits {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            println!(
                "dbg logits: pos={} mean|logits|={:.6} max|logits|={:.6}",
                i_pos,
                mean_abs(&logits),
                max_abs
            );
        }*/

        Ok(logits)
    }
}
