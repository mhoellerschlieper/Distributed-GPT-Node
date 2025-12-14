// train.rs
// Feintuning-Skelett auf CPU. Keine externen ML-Module.
// Idee: LoRA-Adapter fuer Linears. Hier: Minimal-SGD Platzhalter.

use crate::layer::{Linear, TransformerModel};

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn step_vec(params: &mut [f32], grads: &[f32], lr: f32) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p -= lr * *g;
        }
    }
}

// LoRA-Adapter fuer Linear: W + A * B, rank r
pub struct LoraAdapter {
    pub rank: usize,
    pub a: Vec<f32>, // [in_dim, rank]
    pub b: Vec<f32>, // [rank, out_dim]
}

impl LoraAdapter {
    pub fn new(in_dim: usize, out_dim: usize, rank: usize) -> Self {
        Self {
            rank,
            a: vec![0.0; in_dim * rank],
            b: vec![0.0; rank * out_dim],
        }
    }

    pub fn apply(&self, lin: &Linear, x: &[f32]) -> Vec<f32> {
        // y = x W + x A B + b
        let mut base = lin.forward(x);

        // x A
        let mut xa = vec![0f32; self.rank];
        super::math::matmul(x, 1, lin.in_dim, &self.a, self.rank, &mut xa);
        // (x A) B
        let mut xab = vec![0f32; lin.out_dim];
        super::math::matmul(&xa, 1, self.rank, &self.b, lin.out_dim, &mut xab);
        for i in 0..lin.out_dim {
            base[i] += xab[i];
        }
        base
    }
}

// Ein sehr einfacher Trainings-Placeholder (Pseudo-Backprop)
pub fn finetune_step_placeholder(_model: &mut TransformerModel, _tokens: &[usize], _targets: &[usize], _opt: &mut SGD) {
    // TODO:
    // 1) Forward ueber Tokens
    // 2) Cross-Entropy Loss
    // 3) Rueckwaertsableitung fuer LM-Head und ggf. LoRA-Adapter
    // 4) SGD-Update
    // Hier nur Skelett, damit Code strukturiert ist.
}
