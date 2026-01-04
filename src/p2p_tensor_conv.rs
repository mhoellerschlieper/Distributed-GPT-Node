// src/p2p_tensor_conv.rs
// ------------------------------------------------------------
// Konvertierung Candle Tensor <-> WireTensor
//
// Hinweis
// - wir senden als bytes immer f32 little endian
// - dtype im wire ist nur info fuer rueckgabe
//
// Autor: Marcus Schlieper, ExpChat.ai
// Historie:
// - 2025-12-26 Marcus Schlieper: initiale version
// ------------------------------------------------------------

use crate::p2p_wire::{WireDType, WireTensor};
use crate::device_select::get_default_device;
use candle::{DType, Device, Tensor};

pub fn tensor_to_wire(o_x: &Tensor) -> Result<WireTensor, String> {
    let v_shape = o_x
        .dims()
        .iter()
        .map(|&d| d as usize)
        .collect::<Vec<usize>>();

    let e_dtype = match o_x.dtype() {
        DType::F32 => WireDType::F32,
        DType::F16 => WireDType::F16,
        DType::BF16 => WireDType::BF16,
        _ => return Err("tensor_to_wire: dtype nicht unterstuetzt".to_string()),
    };

    let o_cpu = o_x
        .to_device(&get_default_device())
        .map_err(|e| format!("tensor_to_wire: to_cpu: {}", e))?;

    let o_f32 = o_cpu
        .to_dtype(DType::F32)
        .map_err(|e| format!("tensor_to_wire: to_dtype f32: {}", e))?;

    let v_f32: Vec<f32> = o_f32
        .flatten_all()
        .map_err(|e| format!("tensor_to_wire: flatten: {}", e))?
        .to_vec1()
        .map_err(|e| format!("tensor_to_wire: to_vec1: {}", e))?;

    Ok(WireTensor {
        v_shape,
        e_dtype,
        v_data: cast_f32_to_bytes(&v_f32),
    })
}

pub fn wire_to_tensor(o_w: &WireTensor) -> Result<Tensor, String> {
    if o_w.v_shape.is_empty() {
        return Err("wire_to_tensor: shape ist leer".to_string());
    }
    if o_w.v_data.is_empty() {
        return Err("wire_to_tensor: data ist leer".to_string());
    }

    let v_f32 = cast_bytes_to_f32(&o_w.v_data)?;
    let o_t = Tensor::new(v_f32, &get_default_device()).map_err(|e| format!("Tensor::new: {}", e))?;
    let o_t = o_t
        .reshape(o_w.v_shape.as_slice())
        .map_err(|e| format!("reshape: {}", e))?;

    match o_w.e_dtype {
        WireDType::F32 => o_t.to_dtype(DType::F32).map_err(|e| e.to_string()),
        WireDType::F16 => o_t.to_dtype(DType::F16).map_err(|e| e.to_string()),
        WireDType::BF16 => o_t.to_dtype(DType::BF16).map_err(|e| e.to_string()),
        WireDType::I64 => o_t.to_dtype(DType::I64).map_err(|e| e.to_string()),
        WireDType::U32 => o_t.to_dtype(DType::U32).map_err(|e| e.to_string()),
    }
}

fn cast_f32_to_bytes(v_val: &[f32]) -> Vec<u8> {
    let mut v_out: Vec<u8> = Vec::with_capacity(v_val.len() * 4);
    for &f in v_val {
        v_out.extend_from_slice(&f.to_le_bytes());
    }
    v_out
}

fn cast_bytes_to_f32(v_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if v_bytes.len() % 4 != 0 {
        return Err("cast_bytes_to_f32: bytes laenge ist nicht durch 4 teilbar".to_string());
    }

    let i_n = v_bytes.len() / 4;
    let mut v_out: Vec<f32> = Vec::with_capacity(i_n);

    let mut i_off = 0usize;
    while i_off < v_bytes.len() {
        let f = f32::from_le_bytes([
            v_bytes[i_off],
            v_bytes[i_off + 1],
            v_bytes[i_off + 2],
            v_bytes[i_off + 3],
        ]);
        v_out.push(f);
        i_off += 4;
    }
    Ok(v_out)
}
