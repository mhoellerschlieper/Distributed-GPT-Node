// rope.rs
// ------------------------------------------------------------
// Rotary Embedding Hilfsfunktionen
// ------------------------------------------------------------
use candle::{Device, Result, Tensor, DType};
use candle::IndexOp;

pub fn precompute_rope(
    i_head_dim: usize,
    d_base: f32,
    i_max_seq: usize,
    o_dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut v_theta: Vec<f32> = Vec::with_capacity((i_head_dim + 1) / 2);
    for i_idx in (0..i_head_dim).step_by(2) {
        let d_val = 1.0f32 / d_base.powf(i_idx as f32 / i_head_dim as f32);
        v_theta.push(d_val);
    }
    let o_theta = Tensor::new(v_theta.as_slice(), o_dev)?;
    let o_idx = Tensor::arange(0u32, i_max_seq as u32, o_dev)?
        .to_dtype(DType::F32)?
        .reshape((i_max_seq, 1))?;
    let o_idx_theta = o_idx.matmul(&o_theta.reshape((1, o_theta.elem_count()))?)?;
    Ok((o_idx_theta.cos()?, o_idx_theta.sin()?))
}

// Erwartet x: (bs, head, seq, dim)
// o_cos/o_sin: Form (head_dim/2) oder broadcast-faehig
pub fn apply_rope(o_x: &Tensor, o_cos: &Tensor, o_sin: &Tensor) -> Result<Tensor> {
    let (i_bs, i_head, i_seq, i_dim) = o_x.dims4()?;
    let i_half = i_dim / 2;
    let o_x_rs = o_x.reshape((i_bs, i_head, i_seq, i_half, 2))?;
    let o_x0 = o_x_rs.i((.., .., .., .., 0))?;
    let o_x1 = o_x_rs.i((.., .., .., .., 1))?;

    let a0 = (&o_x0 * o_cos)?;
    let b0 = (&o_x1 * o_sin)?;
    let o_y0 = (a0 - b0)?;

    let a1 = (&o_x0 * o_sin)?;
    let b1 = (&o_x1 * o_cos)?;
    let o_y1 = (a1 + b1)?;

    Tensor::cat(&[o_y0, o_y1], 4)?.flatten_from(3)
}
