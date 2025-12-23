// layer.rs
// Ein Layer-Baukasten, der Candle-Tensors nutzt.
// Das echte TransformerModel kommt später separat.
// Autor: Marcus Schlieper | ExpChat.ai

use candle::{Device, IndexOp, Tensor};
use candle::quantized::{QMatMul, QTensor};
use candle_nn::{ops, LayerNorm};
use crate::rope::precompute_rope; // s. u.

/// Ein einfaches RMSNorm-Wrapper (Candle hat nur LayerNorm).
pub struct RmsNorm {
    inner: LayerNorm,
}

impl RmsNorm {
    pub fn new(weight: &QTensor, eps: f32) -> candle::Result<Self> {
        let w = weight.dequantize(&weight.device())?;
        Ok(Self { inner: LayerNorm::rms_norm(w, eps as f64) })
    }
    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        self.inner.forward(x)
    }
}

/// Feed-Forward (SwiGLU) mit quantisierten Gewichten.
pub struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl Mlp {
    pub fn new(w1: QTensor, w2: QTensor, w3: QTensor) -> candle::Result<Self> {
        Ok(Self {
            w1: QMatMul::from_qtensor(w1)?,
            w2: QMatMul::from_qtensor(w2)?,
            w3: QMatMul::from_qtensor(w3)?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let up = self.w1.forward(x)?;
        let gate = ops::silu(&self.w3.forward(x)?)?;
        let act = gate * up?;
        self.w2.forward(&act)
    }
}

/// Eine Attention-Projektion (Q, K, V, O) + RoPE.
/// KV-Cache wird außen verwaltet (z. B. im späteren TransformerModel).
pub struct Attention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: QTensor,
        wk: QTensor,
        wv: QTensor,
        wo: QTensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        rope_base: f32,
        ctx: usize,
        dev: &Device,
    ) -> candle::Result<Self> {
        let (cos, sin) = precompute_rope(head_dim, rope_base, ctx, dev)?;
        Ok(Self {
            wq: QMatMul::from_qtensor(wq)?,
            wk: QMatMul::from_qtensor(wk)?,
            wv: QMatMul::from_qtensor(wv)?,
            wo: QMatMul::from_qtensor(wo)?,
            n_head,
            n_kv_head,
            head_dim,
            cos,
            sin,
        })
    }

    /// Führt nur einen Schritt aus (Seq-Len = 1) – ideal für Auto-Reg.
    pub fn forward(
        &self,
        x: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        pos: usize,
    ) -> candle::Result<Tensor> {
        let (bs, _seq, _h) = x.dims3()?;
        // Projektionen
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Formate anpassen: (bs, head, 1, dim)
        let q = q.reshape((bs, 1, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((bs, 1, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((bs, 1, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        // RoPE
        let cos = self.cos.i(pos)?;
        let sin = self.sin.i(pos)?;
        let rope_q = apply_rope(&q, &cos, &sin)?;
        let rope_k = apply_rope(&k, &cos, &sin)?;

        // KV-Cache aktualisieren
        k_cache.copy_()
            .slice_assign(&[pos], &rope_k)?;
        v_cache.copy_()
            .slice_assign(&[pos], &v)?;

        // Scaled-Dot-Product
        let att = (rope_q.matmul(&k_cache.t()?)?
            / (self.head_dim as f64).sqrt())?;
        let att = ops::softmax_last_dim(&att)?;
        let ctx = att.matmul(&v_cache)?;

        // Output proj.
        let ctx = ctx.transpose(1, 2)?
            .reshape((bs, 1, self.n_head * self.head_dim))?;
        self.wo.forward(&ctx)
    }
}
