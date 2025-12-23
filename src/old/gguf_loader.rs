// gguf_loader.rs
// Projekt: CPU-only GGUF v3 Loader (Rust, ohne unsafe)
// Autor: Marcus Schlieper, ExpChat.ai
// Kontakt: mschlieper@ylook.de
// Firma: ExpChat.ai - Der KI Chat Client fuer den Mittelstand aus Breckerfeld
// Zweck:
//  - GGUF v3 Header + Key-Value lesen (inkl. Arrays)
//  - Tensor-Infos lesen, Datenbereich mit Alignment bestimmen
//  - Packed Daten laden (F32, F16, Q4_0, Q8_0, sowie Typcodes 12, 13, 14)
//  - Konvertierung zu f32 on-demand pro Tensor
//  - Tokenizer: Byte-Level und SPM-Unigram (mit Byte-Fallback), Decoder
//  - Architektur-Hinweise fuer: general.architecture = mistral, qwen, phi, llama, vibethinker
//
// Sicherheit:
//  - Kein unsafe
//  - Strikte Pruefungen, Rueckgabe Result<T, String>
//
// Hinweise:
//  - Q8_0: strikt 36 Bytes je Block (QK=32)
//  - Q4_0: 20 (f32 scale) oder 18 (f16 scale) Bytes je Block (QK=32)
//  - Typcodes 12, 13, 14: Daten werden geladen; to_f32_vec liefert aktuell Err (TODO)
//  - Alignment: KV "general.alignment" (Default 32)
//
// Historie:
//  - 2025-12-07: Version mit Typcodes 12, 13, 14 (Laden), Arch-Hilfen fuer mehrere Modelle.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

// GGUF v3 KV-Typcodes
const GGUF_T_U8: u32 = 0;
const GGUF_T_I8: u32 = 1;
const GGUF_T_U16: u32 = 2;
const GGUF_T_I16: u32 = 3;
const GGUF_T_U32: u32 = 4;
const GGUF_T_I32: u32 = 5;
const GGUF_T_F32: u32 = 6;
const GGUF_T_BOOL: u32 = 7;
const GGUF_T_STR: u32 = 8;
const GGUF_T_ARR: u32 = 9;
const GGUF_T_U64: u32 = 10;
const GGUF_T_I64: u32 = 11;
const GGUF_T_F64: u32 = 12;

// ggml Tensor-Datentypcodes (wichtig fuer Packed-Layout)
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
// (3 existiert, hier nicht implementiert)
const GGML_TYPE_Q8_0: u32 = 8;
// Neue angeforderte Typcodes (werden geladen; Dequant TODO):
const GGML_TYPE_Q4_K: u32 = 12; // K-Quant Familie (verbreitet)
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;

const QK_K: usize = 256;

#[derive(Clone, Debug)]
pub enum GgufValue {
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    Str(String),
    ArrBool(Vec<bool>),
    ArrI8(Vec<i8>),
    ArrU8(Vec<u8>),
    ArrI16(Vec<i16>),
    ArrU16(Vec<u16>),
    ArrI32(Vec<i32>),
    ArrU32(Vec<u32>),
    ArrI64(Vec<i64>),
    ArrU64(Vec<u64>),
    ArrF32(Vec<f32>),
    ArrF64(Vec<f64>),
    ArrStr(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<usize>,    // ggml order: ne0, ne1, ...
    pub type_code: u32,       // ggml type code
    pub offset: u64,          // relativ zur Datenbasis
    pub n_elems: usize,       // Produkt der shape
    pub nbytes_packed: usize, // tatsaechlich gelesene Bytes
    pub data: Vec<u8>,        // packed Bytes (Dequant on-demand)
}

#[derive(Clone, Debug)]
pub struct GgufModel {
    pub kv: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensor>,
}

// KV Helpers
impl GgufModel {
    pub fn get_kv_u32(&self, s_key: &str) -> Option<u32> {
        match self.kv.get(s_key) {
            Some(GgufValue::U32(v)) => Some(*v),
            Some(GgufValue::U64(v)) => Some(*v as u32),
            Some(GgufValue::I32(v)) => Some((*v).max(0) as u32),
            Some(GgufValue::I64(v)) => Some((*v).max(0) as u32),
            _ => None,
        }
    }
    pub fn get_kv_str(&self, s_key: &str) -> Option<String> {
        match self.kv.get(s_key) {
            Some(GgufValue::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }
}

fn k6_block_bytes_from_env_or_default() -> Result<usize, String> {
    if let Ok(s) = std::env::var("K6_BLOCK_BYTES") {
        if let Ok(v) = s.trim().parse::<usize>() {
            if v > 0 {
                return Ok(v);
            }
        }
        return Err("K6_BLOCK_BYTES ist gesetzt, aber ungültig".to_string());
    }
    // Default-Layout: d(f16)=2 + scales(16) + qh(64) + ql(128) = 210
    Ok(210usize)
}

// Laden einer GGUF-Datei (v3)
pub fn load_gguf(s_path: &str) -> Result<GgufModel, String> {
    let mut o_f = File::open(s_path).map_err(|e| format!("Datei oeffnen fehlgeschlagen: {}", e))?;

    // Header
    let mut a_magic = [0u8; 4];
    o_f.read_exact(&mut a_magic).map_err(|e| e.to_string())?;
    if &a_magic != b"GGUF" {
        return Err("Kein GGUF File (Magic)".to_string());
    }
    let i_version = read_u32(&mut o_f)?;
    if i_version != 3 {
        return Err(format!("Nur GGUF v3 unterstuetzt, gefunden v{}", i_version));
    }
    let i_n_tensors = read_u64(&mut o_f)? as usize;
    let i_n_kv = read_u64(&mut o_f)? as usize;

    // KV lesen
    let mut m_kv: HashMap<String, GgufValue> = HashMap::new();
    for _ in 0..i_n_kv {
        let s_key = read_string(&mut o_f)?;
        let i_ty = read_u32(&mut o_f)?;
        let o_val = read_kv_value(&mut o_f, i_ty)?;
        m_kv.insert(s_key, o_val);
    }

    // Tensor-Infos lesen
    let mut v_infos: Vec<(String, Vec<usize>, u32, u64)> = Vec::with_capacity(i_n_tensors);
    for _ in 0..i_n_tensors {
        let s_name = read_string(&mut o_f)?;
        let i_nd = read_u32(&mut o_f)? as usize;
        let mut v_shape: Vec<usize> = Vec::with_capacity(i_nd);
        for _ in 0..i_nd {
            v_shape.push(read_u64(&mut o_f)? as usize);
        }
        let i_type_code = read_u32(&mut o_f)?;
        let i_rel_off = read_u64(&mut o_f)?;
        v_infos.push((s_name, v_shape, i_type_code, i_rel_off));
    }

    // Datenbasis (Alignment)
    let d_meta_end = o_f.seek(SeekFrom::Current(0)).map_err(|e| e.to_string())?;
    let i_alignment = match m_kv.get("general.alignment") {
        Some(GgufValue::U32(v)) => *v as usize,
        Some(GgufValue::U64(v)) => *v as usize,
        _ => 32usize,
    };
    let i_data_base = align_to(d_meta_end as usize, i_alignment) as u64;
    if i_data_base > d_meta_end {
        o_f.seek(SeekFrom::Start(i_data_base))
            .map_err(|e| e.to_string())?;
    }

    // Dateiende
    let d_file_len = o_f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    o_f.seek(SeekFrom::Start(i_data_base))
        .map_err(|e| e.to_string())?;

    // Nach Offset sortieren
    let mut v_sorted = v_infos.clone();
    v_sorted.sort_by_key(|(_, _, _, i_off)| *i_off);

    // Hilfsfunktionen fuer Groessen
    let row_size_bytes = |i_type: u32, i_ne0: usize| -> Option<usize> {
        match i_type {
            GGML_TYPE_F32 => i_ne0.checked_mul(4),
            GGML_TYPE_F16 => i_ne0.checked_mul(2),
            GGML_TYPE_Q4_0 => {
                let i_qk = 32usize;
                let i_blocks = (i_ne0 + i_qk - 1) / i_qk;
                i_blocks.checked_mul(20) // 20 B (f32 scale) – genaue Pruefung tiefer
            }
            GGML_TYPE_Q8_0 => {
                let i_qk = 32usize;
                let i_blocks = (i_ne0 + i_qk - 1) / i_qk;
                i_blocks.checked_mul(36) // strikt 36 B
            }
            // K-Quant Typen: Blockgroessen variieren je Variante.
            // Wir berechnen hier KEINE feste Blockgroesse, da Formate je nach Upstream variieren.
            // Die genaue Pruefung erfolgt spaeter dynamisch (anhand allowed).
            GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => None,
            _ => None,
        }
    };

    let packed_nbytes_by_shape_fixed = |i_type: u32, v_shape: &[usize]| -> Option<usize> {
        if v_shape.is_empty() {
            return Some(0);
        }
        let i_ne0 = v_shape[0];
        let i_rows = v_shape[1..]
            .iter()
            .try_fold(1usize, |i_acc, &d| i_acc.checked_mul(d))?;
        let i_row = row_size_bytes(i_type, i_ne0)?;
        i_row.checked_mul(i_rows)
    };

    // Tensor-Daten lesen
    let mut m_tensors: HashMap<String, GgufTensor> = HashMap::new();
    for i_idx in 0..v_sorted.len() {
        let (ref s_name, ref v_shape, i_type_code, i_rel_off) = v_sorted[i_idx];

        // Anzahl Elemente pruefen
        let i_n_elems =
            checked_product(v_shape).ok_or_else(|| format!("Overflow bei shape von {}", s_name))?;

        // erlaubte Spannweite
        let i_next_off: u64 = if i_idx + 1 < v_sorted.len() {
            v_sorted[i_idx + 1].3
        } else {
            (d_file_len - i_data_base) as u64
        };
        if i_rel_off > i_next_off {
            return Err(format!(
                "{}: rel_off > next_off ({} > {})",
                s_name, i_rel_off, i_next_off
            ));
        }
        let u_allowed_by_next = (i_next_off as u128).saturating_sub(i_rel_off as u128);
        let i_allowed_by_next =
            usize::try_from(u_allowed_by_next).map_err(|_| "span overflow".to_string())?;
        let d_abs_off = i_data_base
            .checked_add(i_rel_off)
            .ok_or_else(|| "offset overflow".to_string())?;
        let u_file_rem = (d_file_len as u128).saturating_sub(d_abs_off as u128);
        let i_file_rem =
            usize::try_from(u_file_rem).map_err(|_| "file_rem overflow".to_string())?;
        let i_allowed = i_allowed_by_next.min(i_file_rem);

        // erwartete Groesse
        let i_ne0 = v_shape.get(0).copied().unwrap_or(0);
        let i_rows = v_shape
            .get(1..)
            .unwrap_or(&[])
            .iter()
            .copied()
            .fold(1usize, |a, b| a.saturating_mul(b));

        let i_nbytes_packed = match i_type_code {
            GGML_TYPE_F32 | GGML_TYPE_F16 => {
                let i_need = packed_nbytes_by_shape_fixed(i_type_code, v_shape)
                    .ok_or_else(|| format!("{}: row size calc failed", s_name))?;
                if i_need > i_allowed {
                    return Err(format!(
                        "Tensor {}: expected={} > allowed={} (type={}, shape={:?}, ne0={}, rows={}, rel_off={}, data_base={}, file_len={})",
                        s_name,
                        i_need,
                        i_allowed,
                        i_type_code,
                        v_shape,
                        i_ne0,
                        i_rows,
                        i_rel_off,
                        i_data_base,
                        d_file_len
                    ));
                }
                i_need
            }
            // GGML_TYPE_Q8_0: dynamisch 36 oder 34 Bytes pro Block
            GGML_TYPE_Q8_0 => {
                let qk = 32usize;
                let blocks_per_row = if i_ne0 == 0 { 0 } else { (i_ne0 + qk - 1) / qk };

                let row36 = blocks_per_row.checked_mul(36);
                let row34 = blocks_per_row.checked_mul(34);
                let need36 = row36.and_then(|r| r.checked_mul(i_rows));
                let need34 = row34.and_then(|r| r.checked_mul(i_rows));

                // Bevorzuge 36B, falle auf 34B zurück, wenn nötig
                let pick = if let Some(n36) = need36 {
                    if n36 <= i_allowed {
                        Some(n36)
                    } else if let Some(n34) = need34 {
                        if n34 <= i_allowed { Some(n34) } else { None }
                    } else {
                        None
                    }
                } else if let Some(n34) = need34 {
                    if n34 <= i_allowed { Some(n34) } else { None }
                } else {
                    None
                };

                match pick {
                    Some(n) => n,
                    None => {
                        return Err(format!(
                            "Tensor {}: Q8_0: neither 36B nor 34B fit allowed={} (36B={:?}, 34B={:?}, shape={:?}, ne0={}, rows={}, rel_off={}, data_base={}, file_len={})",
                            s_name,
                            i_allowed,
                            need36,
                            need34,
                            v_shape,
                            i_ne0,
                            i_rows,
                            i_rel_off,
                            i_data_base,
                            d_file_len
                        ));
                    }
                }
            }

            GGML_TYPE_Q4_0 => {
                // Versuche 20, sonst 18 Bytes je Block
                let i_qk = 32usize;
                let i_blocks = if i_ne0 == 0 {
                    0
                } else {
                    (i_ne0 + i_qk - 1) / i_qk
                };
                let i_need20 = i_blocks.checked_mul(20).and_then(|r| r.checked_mul(i_rows));
                let i_need18 = i_blocks.checked_mul(18).and_then(|r| r.checked_mul(i_rows));
                if let Some(n) = i_need20 {
                    if n <= i_allowed {
                        n
                    } else if let Some(n18) = i_need18 {
                        if n18 <= i_allowed {
                            n18
                        } else {
                            return Err(format!(
                                "Tensor {}: Q4_0: neither 20B nor 18B fit allowed={} (20B={:?}, 18B={:?})",
                                s_name, i_allowed, i_need20, i_need18
                            ));
                        }
                    } else {
                        return Err(format!(
                            "Tensor {}: Q4_0: 20B overflow and 18B not computable",
                            s_name
                        ));
                    }
                } else if let Some(n18) = i_need18 {
                    if n18 <= i_allowed {
                        n18
                    } else {
                        return Err(format!(
                            "Tensor {}: Q4_0: 18B too large for allowed={}",
                            s_name, i_allowed
                        ));
                    }
                } else {
                    return Err(format!(
                        "Tensor {}: Q4_0: cannot compute required bytes",
                        s_name
                    ));
                }
            }
            GGML_TYPE_Q6_K => {
                let blocks_per_row = if i_ne0 == 0 {
                    0
                } else {
                    (i_ne0 + QK_K - 1) / QK_K
                };
                let blk = k6_block_bytes_from_env_or_default()?;
                let row_bytes = blocks_per_row
                    .checked_mul(blk)
                    .ok_or_else(|| format!("Q6_K row_bytes overflow for {}", s_name))?;
                let need = row_bytes
                    .checked_mul(i_rows)
                    .ok_or_else(|| format!("Q6_K total bytes overflow for {}", s_name))?;
                if need > i_allowed {
                    return Err(format!(
                        "Tensor {}: Q6_K expects {} bytes but only {} allowed (type={}, shape={:?}, ne0={}, rows={}, rel_off={}, data_base={}, file_len={})",
                        s_name,
                        need,
                        i_allowed,
                        i_type_code,
                        v_shape,
                        i_ne0,
                        i_rows,
                        i_rel_off,
                        i_data_base,
                        d_file_len
                    ));
                }
                // Wir bestehen auf exakter Übereinstimmung, um stille Fehler zu vermeiden
                if need != i_allowed {
                    return Err(format!(
                        "Tensor {}: Q6_K size mismatch: expected={}, allowed={}",
                        s_name, need, i_allowed
                    ));
                }
                need
            }

            // K-Quant: wir lesen bis zum naechsten Offset (sicheres Laden).
            // Exakte Block-Pruefung ist modellabhaengig; to_f32_vec gibt TODO-Fehler zurueck.
            GGML_TYPE_Q4_K | GGML_TYPE_Q5_K => i_allowed,
            _ => {
                // Unbekannter Typ: lese was moeglich ist
                i_allowed
            }
        };

        // Daten lesen
        o_f.seek(SeekFrom::Start(d_abs_off))
            .map_err(|e| e.to_string())?;
        let mut v_data = vec![0u8; i_nbytes_packed];
        if i_nbytes_packed > 0 {
            o_f.read_exact(&mut v_data).map_err(|e| {
                format!(
                    "read_exact fehlgeschlagen fuer {} an abs_off={} need={} err={}",
                    s_name, d_abs_off, i_nbytes_packed, e
                )
            })?;
        }

        // Eintragen
        m_tensors.insert(
            s_name.clone(),
            GgufTensor {
                name: s_name.clone(),
                shape: v_shape.clone(),
                type_code: i_type_code,
                offset: i_rel_off,
                n_elems: i_n_elems,
                nbytes_packed: i_nbytes_packed,
                data: v_data,
            },
        );
    }

    Ok(GgufModel {
        kv: m_kv,
        tensors: m_tensors,
    })
}

// Konvertierung zu f32
impl GgufTensor {
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        match self.type_code {
            GGML_TYPE_F32 => {
                let i_need = self.n_elems.checked_mul(4).ok_or("len overflow")?;
                if self.data.len() != i_need {
                    return Err("F32 bytes len mismatch".to_string());
                }
                let mut v_out = vec![0f32; self.n_elems];
                for i_i in 0..self.n_elems {
                    let a_b = [
                        self.data[i_i * 4],
                        self.data[i_i * 4 + 1],
                        self.data[i_i * 4 + 2],
                        self.data[i_i * 4 + 3],
                    ];
                    v_out[i_i] = f32::from_le_bytes(a_b);
                }
                Ok(v_out)
            }
            GGML_TYPE_F16 => {
                let i_need = self.n_elems.checked_mul(2).ok_or("len overflow")?;
                if self.data.len() != i_need {
                    return Err("F16 bytes len mismatch".to_string());
                }
                let mut v_out = vec![0f32; self.n_elems];
                for i_i in 0..self.n_elems {
                    let a_b = [self.data[i_i * 2], self.data[i_i * 2 + 1]];
                    v_out[i_i] = f16_to_f32_bits(u16::from_le_bytes(a_b));
                }
                Ok(v_out)
            }
            GGML_TYPE_Q4_0 => {
                if self.shape.is_empty() {
                    return Ok(Vec::new());
                }
                let i_ne0 = self.shape[0];
                let i_rows = self.shape[1..]
                    .iter()
                    .fold(1usize, |a, &d| a.saturating_mul(d));
                if i_rows == 0 {
                    return Ok(Vec::new());
                }
                let i_row_bytes = self.nbytes_packed / i_rows;
                let i_blocks_per_row = (i_ne0 + 31) / 32;
                if i_blocks_per_row == 0 {
                    return Err("Q4_0 blocks_per_row = 0".to_string());
                }
                let i_block_bytes = i_row_bytes / i_blocks_per_row;
                if i_block_bytes != 20 && i_block_bytes != 18 {
                    return Err(format!("Q4_0 unerwartete block_bytes={}", i_block_bytes));
                }
                let mut v_out = vec![0f32; self.n_elems];
                let mut i_src = 0usize;
                for i_r in 0..i_rows {
                    let i_base_out = i_r * i_ne0;
                    for i_b in 0..i_blocks_per_row {
                        let d_scale: f32 = if i_block_bytes == 20 {
                            let a_b = [
                                self.data[i_src],
                                self.data[i_src + 1],
                                self.data[i_src + 2],
                                self.data[i_src + 3],
                            ];
                            i_src += 4;
                            f32::from_le_bytes(a_b)
                        } else {
                            let h = u16::from_le_bytes([self.data[i_src], self.data[i_src + 1]]);
                            i_src += 2;
                            f16_to_f32_bits(h)
                        };
                        for i_byte in 0..16 {
                            let i_bb = self.data[i_src];
                            i_src += 1;
                            let i_q0 = (i_bb & 0x0F) as i32 - 8;
                            let i_col0 = i_b * 32 + (i_byte * 2) as usize;
                            if i_col0 < i_ne0 {
                                v_out[i_base_out + i_col0] = d_scale * i_q0 as f32;
                            }
                            let i_q1 = (i_bb >> 4) as i32 - 8;
                            let i_col1 = i_col0 + 1;
                            if i_col1 < i_ne0 {
                                v_out[i_base_out + i_col1] = d_scale * i_q1 as f32;
                            }
                        }
                    }
                }
                Ok(v_out)
            }
            GGML_TYPE_Q8_0 => {
                if self.shape.is_empty() {
                    return Ok(Vec::new());
                }
                let ne0 = self.shape[0];
                let rows = self.shape[1..]
                    .iter()
                    .fold(1usize, |a, &d| a.saturating_mul(d));
                if rows == 0 {
                    return Ok(Vec::new());
                }
                let row_bytes = self.nbytes_packed / rows;
                let blocks_per_row = (ne0 + 31) / 32;
                if blocks_per_row == 0 {
                    return Err("Q8_0 blocks_per_row = 0".to_string());
                }
                let block_bytes = row_bytes / blocks_per_row;
                if block_bytes != 36 && block_bytes != 34 {
                    return Err(format!("Q8_0 unerwartete block_bytes={}", block_bytes));
                }

                let mut out = vec![0f32; self.n_elems];
                let mut src = 0usize;
                for r in 0..rows {
                    let base_out = r * ne0;
                    for b in 0..blocks_per_row {
                        // Skala lesen: 36B => f32, 34B => f16
                        let scale: f32 = if block_bytes == 36 {
                            let sbytes = [
                                self.data[src],
                                self.data[src + 1],
                                self.data[src + 2],
                                self.data[src + 3],
                            ];
                            src += 4;
                            f32::from_le_bytes(sbytes)
                        } else {
                            let h = u16::from_le_bytes([self.data[src], self.data[src + 1]]);
                            src += 2;
                            crate::gguf_loader::f16_to_f32_bits(h)
                        };

                        // 32 Werte int8
                        for j in 0..32 {
                            let col = b * 32 + j;
                            let q = self.data[src] as i8 as f32;
                            src += 1;
                            if col < ne0 {
                                out[base_out + col] = scale * q;
                            }
                        }
                    }
                }
                Ok(out)
            }
            GGML_TYPE_Q6_K => {
                if self.shape.is_empty() {
                    return Ok(Vec::new());
                }
                let ne0 = self.shape[0];
                let rows = self.shape[1..]
                    .iter()
                    .fold(1usize, |a, &d| a.saturating_mul(d));
                if rows == 0 {
                    return Ok(Vec::new());
                }

                // Blöcke/Zeile
                let qk = 256usize;
                let blocks_per_row = if ne0 == 0 { 0 } else { (ne0 + qk - 1) / qk };
                if blocks_per_row == 0 {
                    return Err("Q6_K blocks_per_row = 0".to_string());
                }

                // Bytegrößen ableiten
                let row_bytes = self.nbytes_packed / rows;
                if row_bytes == 0 {
                    return Err("Q6_K row_bytes = 0".to_string());
                }
                let block_bytes = row_bytes / blocks_per_row;

                // Optional per ENV überschreibbar
                let block_bytes_env = std::env::var("K6_BLOCK_BYTES")
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(block_bytes);

                // Prüfung erwarteter Größen
                let supported = [210usize, 208usize];
                if !supported.contains(&block_bytes_env) {
                    return Err(format!(
                        "Q6_K unerwartete block_bytes={}, unterstützt: {:?}",
                        block_bytes_env, supported
                    ));
                }

                // Layout-Option (ENV): "d-s-qh-ql" (Default) oder "d-s-ql-qh"
                let order = std::env::var("K6_ORDER")
                    .unwrap_or_else(|_| "d-s-qh-ql".to_string())
                    .to_lowercase();
                let use_d_s_qh_ql = match order.as_str() {
                    "d-s-qh-ql" => true,
                    "d-s-ql-qh" => false,
                    _ => true, // Default
                };

                // Ausgabepuffer
                let mut out = vec![0f32; self.n_elems];

                // Hilfsfunktionen
                #[inline]
                fn f16_to_f32_bits(h: u16) -> f32 {
                    crate::gguf_loader::f16_to_f32_bits(h)
                }

                // Rekonstruiere eine 6-bit Zahl aus ql (low 4 bit) und qh (2 bit)
                // ql: 128 Bytes => 2 Werte je Byte
                // qh: 64 Bytes  => 2 Bits je Wert, 16 Werte => 32 Bits = 4 Bytes pro Gruppe
                #[inline]
                fn get_low4(ql: &[u8], idx_in_block: usize) -> u8 {
                    let byte = ql[idx_in_block / 2];
                    if (idx_in_block & 1) == 0 {
                        byte & 0x0F
                    } else {
                        (byte >> 4) & 0x0F
                    }
                }

                // qh ist gruppiert: 16er-Gruppen; pro Gruppe 4 Bytes (2 Bits je Wert)
                // Rückgabe: 0..3 (die zwei High-Bits)
                #[inline]
                fn get_high2(qh: &[u8], idx_in_block: usize) -> u8 {
                    let g = (idx_in_block >> 4) & 0x0F; // Gruppe 0..15 in diesem 256er-Block
                    let j = idx_in_block & 0x0F; // Index 0..15 innerhalb der Gruppe
                    let base = g * 4; // 4 Bytes je Gruppe
                    let byte = qh[base + (j >> 2)]; // 4 Werte je Byte
                    let shift = (j & 3) * 2; // 2 Bits je Wert
                    (byte >> shift) & 0x03
                }

                // Indizes für die Blockfelder nach Reihenfolge bestimmen
                let bytes_d = 2usize;
                let bytes_scales = 16usize;
                let bytes_qh = 64usize;
                let bytes_ql = 128usize;

                // Prüfe, ob block_bytes_env groß genug ist
                let need = bytes_d + bytes_scales + bytes_qh + bytes_ql;
                if block_bytes_env < need {
                    return Err(format!(
                        "Q6_K block_bytes={} kleiner als benötigte {}",
                        block_bytes_env, need
                    ));
                }

                let mut src = 0usize; // globaler Reader
                for r in 0..rows {
                    let base_out_row = r * ne0;
                    let row_base = r * row_bytes;

                    for b in 0..blocks_per_row {
                        let blk_base = row_base + b * block_bytes_env;

                        // d (f16) lesen
                        let d_bytes = [self.data[blk_base], self.data[blk_base + 1]];
                        let d = f16_to_f32_bits(u16::from_le_bytes(d_bytes));

                        // Offsets nach ORDER
                        let (off_scales, off_qh, off_ql) = if use_d_s_qh_ql {
                            (
                                bytes_d,
                                bytes_d + bytes_scales,
                                bytes_d + bytes_scales + bytes_qh,
                            )
                        } else {
                            (
                                bytes_d,
                                bytes_d + bytes_scales + bytes_ql, // qh hinter ql
                                bytes_d + bytes_scales,
                            ) // ql direkt nach scales
                        };

                        let scales_slice =
                            &self.data[blk_base + off_scales..blk_base + off_scales + bytes_scales];
                        let qh_slice = &self.data[blk_base + off_qh..blk_base + off_qh + bytes_qh];
                        let ql_slice = &self.data[blk_base + off_ql..blk_base + off_ql + bytes_ql];

                        // 16 Skalen (s8) für 16 Gruppen à 16 Werte
                        let mut scales: [i8; 16] = [0; 16];
                        for i in 0..16 {
                            scales[i] = scales_slice[i] as i8;
                        }

                        // 256 Werte im Block
                        for i in 0..qk {
                            let col = b * qk + i;
                            if col >= ne0 {
                                break; // letzte Blockkante (Padding)
                            }

                            let lo = get_low4(ql_slice, i) as i32; // 0..15
                            let hi = get_high2(qh_slice, i) as i32; // 0..3
                            let q = ((hi << 4) | lo) - 32; // -32..31 (symmetrisch)

                            let g = (i >> 4) & 0x0F; // Gruppe 0..15
                            let s = scales[g] as f32;

                            // Dequant: y = d * s * q
                            let y = d * s * (q as f32);
                            out[base_out_row + col] = y;
                        }

                        // Fortschritt für Debug (optional)
                        src = blk_base + block_bytes_env;
                    }
                }

                Ok(out)
            }
            // K-Quant: hier bewusst noch nicht nach f32 dequantisieren.
            // Damit gibt es keinen stillen Fehler. Bitte Dequant-Layout gemaess Upstream nachruesten.
            GGML_TYPE_Q4_K | GGML_TYPE_Q5_K => Err(format!(
                "to_f32_vec: ggml type {} (K-Quant) derzeit nicht implementiert",
                self.type_code
            )),
            other => Err(format!("ggml type {} nicht unterstuetzt", other)),
        }
    }
}

// Einfache Tokenizer

#[derive(Clone)]
pub struct SimpleTokenizer;
impl SimpleTokenizer {
    pub fn new_byte_level() -> Self {
        SimpleTokenizer
    }
    pub fn encode(&self, s_text: &str) -> Vec<usize> {
        s_text.as_bytes().iter().map(|&b| b as usize).collect()
    }
    pub fn decode(&self, v_ids: &[usize]) -> String {
        let v_bytes: Vec<u8> = v_ids.iter().map(|&i| i as u8).collect();
        String::from_utf8_lossy(&v_bytes).to_string()
    }
}

// Decoder fuer GGUF-Tokenlisten
#[derive(Clone)]
pub struct GgufTokenDecoder {
    pub tokens: Vec<String>,
}
impl GgufTokenDecoder {
    pub fn from_kv(m_kv: &std::collections::HashMap<String, GgufValue>) -> Option<Self> {
        if let Some(GgufValue::ArrStr(v)) = m_kv.get("tokenizer.ggml.tokens") {
            let mut v_tokens = Vec::with_capacity(v.len());
            for s in v {
                v_tokens.push(s.clone());
            }
            Some(GgufTokenDecoder { tokens: v_tokens })
        } else {
            None
        }
    }
    pub fn decode_one(&self, i_id: usize) -> String {
        if i_id < self.tokens.len() {
            self.tokens[i_id].clone()
        } else {
            "<unk>".to_string()
        }
    }
}

// SPM Unigram Tokenizer (mit Byte-Fallback)
#[derive(Clone)]
pub struct GgufSpmUnigram {
    pub tokens: Vec<String>,
    pub id_by_token: std::collections::HashMap<String, usize>,
    pub scores: Vec<f32>,
    pub bos_id: Option<usize>,
    pub eos_id: Option<usize>,
    pub unk_id: Option<usize>,
    pub add_bos: bool,
    pub add_eos: bool,
}
impl GgufSpmUnigram {
    pub fn from_kv(m_kv: &std::collections::HashMap<String, GgufValue>) -> Option<Self> {
        let v_tokens = match m_kv.get("tokenizer.ggml.tokens") {
            Some(GgufValue::ArrStr(v)) => v.clone(),
            _ => return None,
        };
        let v_scores = match m_kv.get("tokenizer.ggml.scores") {
            Some(GgufValue::ArrF32(v)) => v.clone(),
            _ => vec![0.0; v_tokens.len()],
        };
        let mut m_id_by_token = std::collections::HashMap::new();
        for (i, t) in v_tokens.iter().enumerate() {
            m_id_by_token.insert(t.clone(), i);
        }
        let get_usize = |v_keys: &[&str]| -> Option<usize> {
            for k in v_keys {
                match m_kv.get(*k) {
                    Some(GgufValue::U32(v)) => return Some(*v as usize),
                    Some(GgufValue::U64(v)) => return Some(*v as usize),
                    Some(GgufValue::I32(v)) => return Some((*v).max(0) as usize),
                    Some(GgufValue::I64(v)) => return Some((*v).max(0) as usize),
                    _ => {}
                }
            }
            None
        };
        let o_bos_id = get_usize(&["tokenizer.ggml.bos_token_id", "tokenizer.ggml.bos_id"]);
        let o_eos_id = get_usize(&["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eos_id"]);
        let o_unk_id = get_usize(&["tokenizer.ggml.unknown_token_id", "tokenizer.ggml.unk_id"]);
        let b_add_bos = matches!(
            m_kv.get("tokenizer.ggml.add_bos_token"),
            Some(GgufValue::Bool(true))
        );
        let b_add_eos = matches!(
            m_kv.get("tokenizer.ggml.add_eos_token"),
            Some(GgufValue::Bool(true))
        );
        Some(Self {
            tokens: v_tokens,
            id_by_token: m_id_by_token,
            scores: v_scores,
            bos_id: o_bos_id,
            eos_id: o_eos_id,
            unk_id: o_unk_id,
            add_bos: b_add_bos,
            add_eos: b_add_eos,
        })
    }

    fn byte_tok_id(&self, b: u8) -> Option<usize> {
        let s = format!("<0x{:02X}>", b);
        self.id_by_token.get(&s).copied()
    }

    // Einfache Viterbi ueber Zeichen; Byte-Fallback statt UNK
    pub fn encode(&self, s_text: &str) -> Vec<usize> {
        let mut s = s_text.replace(' ', "\u{2581}");
        if !s.starts_with('\u{2581}') {
            s = format!("\u{2581}{}", s);
        }
        let v_chars: Vec<char> = s.chars().collect();
        let i_n = v_chars.len();
        let mut v_best_score = vec![f32::NEG_INFINITY; i_n + 1];
        let mut v_best_len = vec![0usize; i_n + 1];
        v_best_score[0] = 0.0;
        let i_max_try = 32usize;

        for i in 0..i_n {
            if v_best_score[i].is_infinite() {
                continue;
            }
            let i_limit = (i_n - i).min(i_max_try);
            for l in 1..=i_limit {
                let s_sub: String = v_chars[i..i + l].iter().collect();
                if let Some(&i_tid) = self.id_by_token.get(&s_sub) {
                    let d_sc = self.scores.get(i_tid).cloned().unwrap_or(0.0);
                    let d_cand = v_best_score[i] + d_sc;
                    if d_cand > v_best_score[i + l] {
                        v_best_score[i + l] = d_cand;
                        v_best_len[i + l] = l;
                    }
                }
            }
            // kein Match: markiere 1 Zeichen; Bytes spaeter
            if v_best_len[i + 1] == 0 {
                v_best_score[i + 1] = v_best_score[i];
                v_best_len[i + 1] = 1;
            }
        }

        let mut v_pieces: Vec<String> = Vec::new();
        let mut i_pos = i_n;
        while i_pos > 0 {
            let i_l = v_best_len[i_pos].max(1);
            let s_sub: String = v_chars[i_pos - i_l..i_pos].iter().collect();
            v_pieces.push(s_sub);
            i_pos -= i_l;
        }
        v_pieces.reverse();

        let mut v_ids: Vec<usize> = Vec::new();
        if self.add_bos {
            if let Some(i_bos) = self.bos_id {
                v_ids.push(i_bos);
            }
        }
        for p in v_pieces {
            if let Some(&i_tid) = self.id_by_token.get(&p) {
                v_ids.push(i_tid);
            } else {
                for b in p.as_bytes() {
                    if let Some(i_tid) = self.byte_tok_id(*b) {
                        v_ids.push(i_tid);
                    } else if let Some(i_unk) = self.unk_id {
                        v_ids.push(i_unk);
                    }
                }
            }
        }
        if self.add_eos {
            if let Some(i_eos) = self.eos_id {
                v_ids.push(i_eos);
            }
        }
        v_ids
    }

    pub fn decode(&self, v_ids: &[usize]) -> String {
        let mut v_out_bytes: Vec<u8> = Vec::new();
        for &i_id in v_ids {
            if i_id >= self.tokens.len() {
                continue;
            }
            let s_tok = &self.tokens[i_id];
            // BOS/EOS/UNK unterdruecken
            if s_tok == "<s>" || s_tok == "</s>" || s_tok == "<unk>" {
                continue;
            }
            // Byte-Token <0xAB>
            if s_tok.len() == 6 && s_tok.starts_with("<0x") && s_tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&s_tok[3..5], 16) {
                    v_out_bytes.push(b);
                    continue;
                }
            }
            v_out_bytes.extend_from_slice(s_tok.as_bytes());
        }
        let mut s = String::from_utf8_lossy(&v_out_bytes).to_string();
        s = s.replace('\u{2581}', " ");
        s
    }
}

// Architektur-Hinweise (generisch fuer mehrere Modelle)
//
// Liefert robuste Groessen aus KV oder Tensorformen.
// Diese Struktur ist nur ein Hint-Container. Die eigentliche ModelConfig
// baust du in deinem Model-Modul.
#[derive(Clone, Debug)]
pub struct ArchHints {
    pub s_arch: String,
    pub i_vocab_size: usize,
    pub i_hidden_size: usize,
    pub i_n_layers: usize,
    pub i_n_heads: usize,
    pub i_n_kv_heads: usize,
    pub i_intermediate_size: usize,
    pub i_max_seq_len: usize,
    pub i_rope_dim: usize,
    pub d_rope_base: f32,
}

pub fn build_arch_hints(o_gguf: &GgufModel) -> ArchHints {
    let s_arch = o_gguf
        .get_kv_str("general.architecture")
        .unwrap_or_else(|| "llama".to_string());

    // Vokab
    let i_vocab_size = o_gguf
        .get_kv_u32("tokenizer.vocab_size")
        .map(|v| v as usize)
        .or_else(|| {
            if let Some(GgufValue::ArrStr(v)) = o_gguf.kv.get("tokenizer.ggml.tokens") {
                Some(v.len())
            } else {
                None
            }
        })
        .unwrap_or(32000);

    // Embedding-Tensor fuer Fallback finden
    let f_find_tensor = |names: &[&str]| -> Option<&GgufTensor> {
        for n in names {
            if let Some(t) = o_gguf.tensors.get(*n) {
                return Some(t);
            }
        }
        None
    };
    let o_emb_t = f_find_tensor(&[
        "tok_embeddings.weight",
        "token_embd.weight",
        "token_embeddings.weight",
        "embed_tokens.weight",
    ]);
    let (i_hidden_guess, i_vocab_guess) = if let Some(t) = o_emb_t {
        if t.shape.len() == 2 {
            // ggml: ne0=cols(embedding_dim), ne1=rows(vocab)
            (t.shape[0], t.shape[1])
        } else {
            (128usize, i_vocab_size)
        }
    } else {
        (128usize, i_vocab_size)
    };

    let f_kv_u32_any = |vv: &[String]| -> Option<u32> {
        for k in vv {
            if let Some(v) = o_gguf.get_kv_u32(k) {
                return Some(v);
            }
        }
        None
    };

    // Keys pro arch
    let v_heads_keys = vec![
        format!("{}.attention.head_count", s_arch),
        "attention.head_count".to_string(),
        "llama.attention.head_count".to_string(),
    ];
    let v_kv_heads_keys = vec![
        format!("{}.attention.head_count_kv", s_arch),
        "attention.head_count_kv".to_string(),
        "llama.attention.head_count_kv".to_string(),
    ];
    let v_layers_keys = vec![
        format!("{}.block_count", s_arch),
        "block_count".to_string(),
        "llama.block_count".to_string(),
    ];
    let v_hidden_keys = vec![
        format!("{}.embedding_length", s_arch),
        "embedding_length".to_string(),
        "llama.embedding_length".to_string(),
    ];
    let v_ffn_keys = vec![
        format!("{}.feed_forward_length", s_arch),
        "feed_forward_length".to_string(),
        "llama.feed_forward_length".to_string(),
    ];
    let v_ctx_keys = vec![
        format!("{}.context_length", s_arch),
        "context_length".to_string(),
        "llama.context_length".to_string(),
    ];
    let v_rope_dim_keys = vec![
        format!("{}.rope.dimension_count", s_arch),
        "rope.dimension_count".to_string(),
        "llama.rope.dimension_count".to_string(),
    ];
    let v_rope_base_keys = vec![
        format!("{}.rope.freq_base", s_arch),
        "rope.freq_base".to_string(),
        "llama.rope.freq_base".to_string(),
        "rope.freq_base".to_string(),
    ];

    let i_hidden_size = f_kv_u32_any(
        &v_hidden_keys
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .map(|v| v as usize)
    .unwrap_or(i_hidden_guess);

    let i_n_layers = f_kv_u32_any(
        &v_layers_keys
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .map(|v| v as usize)
    .unwrap_or_else(|| {
        // Fallback: zaehle blk.i.* Praefixe
        let mut i = 0usize;
        loop {
            let s_p = format!("blk.{}", i);
            let b_has = o_gguf.tensors.keys().any(|k| k.starts_with(&s_p));
            if !b_has {
                break;
            }
            i += 1;
        }
        i.max(1)
    });

    let i_n_heads = f_kv_u32_any(
        &v_heads_keys
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .map(|v| v as usize)
    .unwrap_or((i_hidden_size / 64).max(1));

    let i_n_kv_heads = f_kv_u32_any(
        &v_kv_heads_keys
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .map(|v| v as usize)
    .unwrap_or(i_n_heads);

    let i_intermediate_size =
        f_kv_u32_any(&v_ffn_keys.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .map(|v| v as usize)
            .or_else(|| {
                // Fallback: aus blk.0.ffn_up.weight [hidden, intermediate]
                if let Some(t) = o_gguf.tensors.get("blk.0.ffn_up.weight") {
                    if t.shape.len() == 2 {
                        return Some(t.shape[1]);
                    }
                }
                None
            })
            .unwrap_or(i_hidden_size * 4);

    let i_max_seq_len = f_kv_u32_any(&v_ctx_keys.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .map(|v| v as usize)
        .unwrap_or(2048);

    let i_rope_dim = f_kv_u32_any(
        &v_rope_dim_keys
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .map(|v| v as usize)
    .unwrap_or(i_hidden_size / i_n_heads);

    let d_rope_base = f_kv_u32_any(
        &v_rope_base_keys
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    )
    .map(|v| v as f32)
    .unwrap_or(10000.0);

    ArchHints {
        s_arch,
        i_vocab_size: i_vocab_size.max(i_vocab_guess),
        i_hidden_size,
        i_n_layers,
        i_n_heads,
        i_n_kv_heads,
        i_intermediate_size,
        i_max_seq_len,
        i_rope_dim,
        d_rope_base,
    }
}

// Lese-Utilities

fn read_u32(o_f: &mut File) -> Result<u32, String> {
    let mut a_b = [0u8; 4];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(a_b))
}

fn read_u64(o_f: &mut File) -> Result<u64, String> {
    let mut a_b = [0u8; 8];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(a_b))
}

fn read_i32(o_f: &mut File) -> Result<i32, String> {
    let mut a_b = [0u8; 4];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(i32::from_le_bytes(a_b))
}

fn read_i64(o_f: &mut File) -> Result<i64, String> {
    let mut a_b = [0u8; 8];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(i64::from_le_bytes(a_b))
}

fn read_f32(o_f: &mut File) -> Result<f32, String> {
    let mut a_b = [0u8; 4];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(f32::from_le_bytes(a_b))
}

fn read_f64(o_f: &mut File) -> Result<f64, String> {
    let mut a_b = [0u8; 8];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(f64::from_le_bytes(a_b))
}

fn read_u16(o_f: &mut File) -> Result<u16, String> {
    let mut a_b = [0u8; 2];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(u16::from_le_bytes(a_b))
}

fn read_i16(o_f: &mut File) -> Result<i16, String> {
    let mut a_b = [0u8; 2];
    o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
    Ok(i16::from_le_bytes(a_b))
}

fn read_string(o_f: &mut File) -> Result<String, String> {
    let i_n = read_u64(o_f)? as usize;
    let mut v_buf = vec![0u8; i_n];
    o_f.read_exact(&mut v_buf).map_err(|e| e.to_string())?;
    let s = String::from_utf8(v_buf).map_err(|e| e.to_string())?;
    Ok(s)
}

fn read_array(o_f: &mut File) -> Result<GgufValue, String> {
    // Array: [elem_type:u32][count:u64][data...]
    let i_elem_ty = read_u32(o_f)?;
    let i_count = read_u64(o_f)? as usize;
    match i_elem_ty {
        GGUF_T_BOOL => {
            let mut v = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                let mut a_b = [0u8; 1];
                o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
                v.push(a_b[0] != 0);
            }
            Ok(GgufValue::ArrBool(v))
        }
        GGUF_T_U8 => {
            let mut v = vec![0u8; i_count];
            o_f.read_exact(&mut v).map_err(|e| e.to_string())?;
            Ok(GgufValue::ArrU8(v))
        }
        GGUF_T_I8 => {
            let mut v = vec![0u8; i_count];
            o_f.read_exact(&mut v).map_err(|e| e.to_string())?;
            let v_out: Vec<i8> = v.into_iter().map(|x| x as i8).collect();
            Ok(GgufValue::ArrI8(v_out))
        }
        GGUF_T_U16 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_u16(o_f)?);
            }
            Ok(GgufValue::ArrU16(v_out))
        }
        GGUF_T_I16 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_i16(o_f)?);
            }
            Ok(GgufValue::ArrI16(v_out))
        }
        GGUF_T_U32 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_u32(o_f)?);
            }
            Ok(GgufValue::ArrU32(v_out))
        }
        GGUF_T_I32 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_i32(o_f)?);
            }
            Ok(GgufValue::ArrI32(v_out))
        }
        GGUF_T_U64 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_u64(o_f)?);
            }
            Ok(GgufValue::ArrU64(v_out))
        }
        GGUF_T_I64 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_i64(o_f)?);
            }
            Ok(GgufValue::ArrI64(v_out))
        }
        GGUF_T_F32 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_f32(o_f)?);
            }
            Ok(GgufValue::ArrF32(v_out))
        }
        GGUF_T_F64 => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_f64(o_f)?);
            }
            Ok(GgufValue::ArrF64(v_out))
        }
        GGUF_T_STR => {
            let mut v_out = Vec::with_capacity(i_count);
            for _ in 0..i_count {
                v_out.push(read_string(&mut *o_f)?);
            }
            Ok(GgufValue::ArrStr(v_out))
        }
        _ => Err(format!("Array: unbekannter Elementtyp {}", i_elem_ty)),
    }
}

fn read_kv_value(o_f: &mut File, i_ty: u32) -> Result<GgufValue, String> {
    match i_ty {
        GGUF_T_BOOL => {
            let mut a_b = [0u8; 1];
            o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
            Ok(GgufValue::Bool(a_b[0] != 0))
        }
        GGUF_T_U8 => {
            let mut a_b = [0u8; 1];
            o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
            Ok(GgufValue::U8(a_b[0]))
        }
        GGUF_T_I8 => {
            let mut a_b = [0u8; 1];
            o_f.read_exact(&mut a_b).map_err(|e| e.to_string())?;
            Ok(GgufValue::I8(a_b[0] as i8))
        }
        GGUF_T_U16 => Ok(GgufValue::U16(read_u16(o_f)?)),
        GGUF_T_I16 => Ok(GgufValue::I16(read_i16(o_f)?)),
        GGUF_T_U32 => Ok(GgufValue::U32(read_u32(o_f)?)),
        GGUF_T_I32 => Ok(GgufValue::I32(read_i32(o_f)?)),
        GGUF_T_U64 => Ok(GgufValue::U64(read_u64(o_f)?)),
        GGUF_T_I64 => Ok(GgufValue::I64(read_i64(o_f)?)),
        GGUF_T_F32 => Ok(GgufValue::F32(read_f32(o_f)?)),
        GGUF_T_F64 => Ok(GgufValue::F64(read_f64(o_f)?)),
        GGUF_T_STR => Ok(GgufValue::Str(read_string(o_f)?)),
        GGUF_T_ARR => read_array(o_f),
        i_other => Err(format!("Unbekannter KV-Typ: {}", i_other)),
    }
}

fn align_to(i_x: usize, i_a: usize) -> usize {
    if i_a == 0 {
        i_x
    } else {
        ((i_x + i_a - 1) / i_a) * i_a
    }
}

fn checked_product(v_shape: &[usize]) -> Option<usize> {
    let mut i_acc: usize = 1;
    for d in v_shape {
        i_acc = i_acc.checked_mul(*d)?;
    }
    Some(i_acc)
}

// F16 -> F32 (ohne unsafe)
pub fn f16_to_f32_bits(h: u16) -> f32 {
    let i_s = ((h >> 15) & 0x0001) as u32;
    let i_e = ((h >> 10) & 0x001f) as u32;
    let i_f = (h & 0x03ff) as u32;

    let u_bits: u32 = if i_e == 0 {
        if i_f == 0 {
            i_s << 31
        } else {
            // subnormal
            let mut i_e32: i32 = 127 - 15 + 1;
            let mut i_f32m = i_f;
            while (i_f32m & 0x0400) == 0 {
                i_f32m <<= 1;
                i_e32 -= 1;
            }
            let i_f32m = i_f32m & 0x03ff;
            (i_s << 31) | ((i_e32 as u32) << 23) | (i_f32m << 13)
        }
    } else if i_e == 31 {
        if i_f == 0 {
            (i_s << 31) | 0x7f800000
        } else {
            (i_s << 31) | 0x7f800000 | (i_f << 13)
        }
    } else {
        let i_e32 = i_e + (127 - 15);
        (i_s << 31) | (i_e32 << 23) | (i_f << 13)
    };

    f32::from_le_bytes(u_bits.to_le_bytes())
}
