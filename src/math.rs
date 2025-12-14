// math.rs
// Einfache Mathe-Routinen: matmul, matmul_blocked, softmax, silu, rmsnorm, sampling
// CPU only, keine externen Abhaengigkeiten

pub fn matmul(a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize, out: &mut [f32]) {
    for r in 0..a_rows {
        for c in 0..b_cols {
            let mut d_acc = 0f32;
            let ar = &a[r * a_cols..(r + 1) * a_cols];
            for k in 0..a_cols {
                d_acc += ar[k] * b[k * b_cols + c];
            }
            out[r * b_cols + c] = d_acc;
        }
    }
}

// Blocked Matmul: besser fuer CPU Cache.
// Funktioniert fuer beliebige Groessen. Fallback auf klein bei winzigen Matrizen.
pub fn matmul_blocked(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_cols: usize,
    out: &mut [f32],
) {
    // einfache Schwelle: kleine Matrizen direkt
    let i_work = a_rows.saturating_mul(a_cols).saturating_mul(b_cols);
    if i_work <= 4096 {
        matmul(a, a_rows, a_cols, b, b_cols, out);
        return;
    }

    // Kachelgroessen (heuristik)
    let i_bm = 64usize;
    let i_bn = 64usize;
    let i_bk = 64usize;

    // out nullen
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let i_m = a_rows;
    let i_n = b_cols;
    let i_k = a_cols;

    let mut i_r0 = 0usize;
    while i_r0 < i_m {
        let i_r1 = (i_r0 + i_bm).min(i_m);
        let mut i_k0 = 0usize;
        while i_k0 < i_k {
            let i_k1 = (i_k0 + i_bk).min(i_k);
            let mut i_c0 = 0usize;
            while i_c0 < i_n {
                let i_c1 = (i_c0 + i_bn).min(i_n);

                for i_r in i_r0..i_r1 {
                    for i_kk in i_k0..i_k1 {
                        let d_a = a[i_r * i_k + i_kk];
                        let i_b_row = i_kk * i_n;
                        let i_out_row = i_r * i_n;
                        let b_slice = &b[i_b_row + i_c0..i_b_row + i_c1];
                        let out_slice = &mut out[i_out_row + i_c0..i_out_row + i_c1];
                        for i_c in 0..(i_c1 - i_c0) {
                            out_slice[i_c] += d_a * b_slice[i_c];
                        }
                    }
                }

                i_c0 += i_bn;
            }
            i_k0 += i_bk;
        }
        i_r0 += i_bm;
    }
}

pub fn add_bias(vec: &mut [f32], bias: &[f32]) {
    for (v, b) in vec.iter_mut().zip(bias.iter()) {
        *v += *b;
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let mut d_maxv = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > d_maxv {
            d_maxv = v;
        }
    }
    let mut d_sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - d_maxv).exp();
        d_sum += *v;
    }
    if d_sum > 0.0 {
        for v in x.iter_mut() {
            *v /= d_sum;
        }
    }
}

pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + (0.79788456 * (*v + 0.044715 * *v * *v * *v)).tanh());
    }
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let d = *v;
        *v = d / (1.0 + (-d).exp());
    }
}

pub fn hadamard_inplace(x: &mut [f32], y: &[f32]) {
    for i in 0..x.len() {
        x[i] *= y[i];
    }
}

pub fn rms_norm(x: &mut [f32], d_eps: f32) -> f32 {
    let mut d_ss = 0f32;
    for &v in x.iter() {
        d_ss += v * v;
    }
    let d_mean = d_ss / (x.len() as f32);
    let d_scale = 1.0 / (d_mean + d_eps).sqrt();
    for v in x.iter_mut() {
        *v *= d_scale;
    }
    d_scale
}

// Rotary embeddings (einfach)
pub fn apply_rope(q: &mut [f32], k: &mut [f32], i_head_dim: usize, i_pos: usize) {
    let d_theta_base: f32 = 10000.0;
    let mut v_inv_freq: Vec<f32> = Vec::with_capacity(i_head_dim / 2);
    for i_i in 0..(i_head_dim / 2) {
        let d = 1.0 / d_theta_base.powf((2.0 * i_i as f32) / i_head_dim as f32);
        v_inv_freq.push(d);
    }
    let mut v_angle: Vec<f32> = Vec::with_capacity(v_inv_freq.len());
    let d_pos = i_pos as f32;
    for &w in v_inv_freq.iter() {
        v_angle.push(w * d_pos);
    }
    for i in 0..(i_head_dim / 2) {
        let d_cos = v_angle[i].cos();
        let d_sin = v_angle[i].sin();
        let d_q0 = q[i];
        let d_q1 = q[i + i_head_dim / 2];
        q[i] = d_q0 * d_cos - d_q1 * d_sin;
        q[i + i_head_dim / 2] = d_q0 * d_sin + d_q1 * d_cos;

        let d_k0 = k[i];
        let d_k1 = k[i + i_head_dim / 2];
        k[i] = d_k0 * d_cos - d_k1 * d_sin;
        k[i + i_head_dim / 2] = d_k0 * d_sin + d_k1 * d_cos;
    }
}

// Simple RNG und Sampler bleiben unveraendert aus deiner letzten Version
pub struct SimpleRng {
    pub i_state: u64,
}
impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 {
            0x9E3779B97F4A7C15u64
        } else {
            seed
        };
        SimpleRng { i_state: s }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.i_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.i_state = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        let v = (u >> 40) as u32;
        (v as f32) / ((1u32 << 24) as f32)
    }
}

pub fn sample_top_k_top_p_temperature(
    v_logits: &[f32],
    d_temperature: f32,
    i_top_k: usize,
    d_top_p: f32,
    rng: &mut SimpleRng,
) -> usize {
    let d_temp = if d_temperature <= 0.0 {
        1.0
    } else {
        d_temperature
    };
    let mut v_scaled = vec![0f32; v_logits.len()];
    for i in 0..v_logits.len() {
        v_scaled[i] = v_logits[i] / d_temp;
    }
    let mut v_probs = v_scaled.clone();
    softmax_inplace(&mut v_probs);

    let mut v_idx: Vec<usize> = (0..v_probs.len()).collect();
    v_idx.sort_by(|&a, &b| {
        v_probs[b]
            .partial_cmp(&v_probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut v_keep: Vec<usize> = if i_top_k > 0 && i_top_k < v_idx.len() {
        v_idx[..i_top_k].to_vec()
    } else {
        v_idx
    };

    if d_top_p > 0.0 && d_top_p < 1.0 {
        let mut d_cum = 0f32;
        let mut v_cut: Vec<usize> = Vec::new();
        for &ix in v_keep.iter() {
            v_cut.push(ix);
            d_cum += v_probs[ix];
            if d_cum >= d_top_p {
                break;
            }
        }
        if !v_cut.is_empty() {
            v_keep = v_cut;
        }
    }

    let mut d_sum = 0f32;
    for &ix in v_keep.iter() {
        d_sum += v_probs[ix];
    }
    if d_sum <= 0.0 {
        let mut i_best = 0usize;
        let mut d_best = f32::NEG_INFINITY;
        for (i, &x) in v_logits.iter().enumerate() {
            if x > d_best {
                d_best = x;
                i_best = i;
            }
        }
        return i_best;
    }

    let mut v_sel: Vec<(usize, f32)> = Vec::with_capacity(v_keep.len());
    for &ix in v_keep.iter() {
        v_sel.push((ix, v_probs[ix] / d_sum));
    }
    let d_r = rng.next_f32();
    let mut d_acc = 0f32;
    for (ix, p) in v_sel.iter() {
        d_acc += *p;
        if d_r <= d_acc {
            return *ix;
        }
    }
    v_sel.last().map(|t| t.0).unwrap_or(0)
}

pub fn apply_rope_partial_base(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    base: f32,
) {
    let half = head_dim / 2;
    let rdim = rope_dim.min(head_dim);
    let pairs = rdim / 2;
    if pairs == 0 { return; }

    // WICHTIG: Typen angeben (Vec<f32>)
    let inv: Vec<f32> = (0..pairs)
        .map(|i| 1.0_f32 / base.powf((2.0_f32 * i as f32) / rdim as f32))
        .collect();
    let ang: Vec<f32> = inv.iter().map(|w| *w * pos as f32).collect();

    for i in 0..pairs {
        // Paarung über die halbe Breite
        let i0 = i;
        let i1 = i + half;

        let cos = ang[i].cos();
        let sin = ang[i].sin();

        let q0 = q[i0]; let q1 = q[i1];
        q[i0] = q0 * cos - q1 * sin;
        q[i1] = q0 * sin + q1 * cos;

        let k0 = k[i0]; let k1 = k[i1];
        k[i0] = k0 * cos - k1 * sin;
        k[i1] = k0 * sin + k1 * cos;
    }
}
