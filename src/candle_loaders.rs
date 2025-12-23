// candle_loaders.rs
// ------------------------------------------------------------
// GGUF Loader: Tensoren und Metadaten einlesen
// Autor: Marcus Schlieper, ExpChat.ai
// Stand: 22-12-2025
// ------------------------------------------------------------

use candle::Device;
use candle::quantized::{gguf_file, QTensor};
use candle::quantized::gguf_file::Value;
use std::collections::HashMap;

/// Typ-Alias: Map von Tensor-Namen auf QTensor
pub type TensorMap = HashMap<String, QTensor>;
/// Typ-Alias: GGUF Metadaten
pub type MetaMap = HashMap<String, Value>;

/// Lädt alle Tensoren und Metadaten aus einer GGUF-Datei.
/// Sichere Fehlerbehandlung mit Result.
pub fn load_gguf_with_meta(
    s_path: &str,
    o_dev: &Device,
) -> candle::Result<(TensorMap, MetaMap)> {
    // Datei öffnen
    let mut o_reader = std::fs::File::open(s_path)?;

    // Content lesen (Signatur: nur &mut reader)
    let o_content = gguf_file::Content::read(&mut o_reader)?;

    // Tensoren sammeln
    let mut m_tensors: TensorMap = HashMap::new();

    // Achtung: tensor_infos ist eine Liste von (String, TensorInfo)
    for (s_name, _ti) in o_content.tensor_infos.iter() {
        // Tensor mit Name laden (benötigt reader, name, device)
        let o_qt = o_content.tensor(&mut o_reader, s_name, o_dev)?;
        // In Map eintragen
        m_tensors.insert(s_name.clone(), o_qt);
    }

    // Metadaten kopieren
    let m_meta: MetaMap = o_content.metadata.clone();

    Ok((m_tensors, m_meta))
}
