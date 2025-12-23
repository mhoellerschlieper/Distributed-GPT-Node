// gguf_loader_hint.rs
// ------------------------------------------------------------
// Zusatz: Freundliche Meldung, wenn ein nicht unterstützter
//         K-Quant Typ gefunden wird.
// Autor: Marcus Schlieper, ExpChat.ai
// Stand: 2025-12-23
// ------------------------------------------------------------
use crate::gguf_loader::{GgufModel, GgufTensor};

pub fn warn_unsupported_k_quant(o_gguf: &GgufModel) {
    let mut b_has_q4k = false;
    let mut b_has_q5k = false;
    for (_name, t) in &o_gguf.tensors {
        match t.type_code {
            12 => b_has_q4k = true, // GGML_TYPE_Q4_K
            13 => b_has_q5k = true, // GGML_TYPE_Q5_K
            _ => {}
        }
    }
    if b_has_q4k || b_has_q5k {
        eprintln!();
        eprintln!("Hinweis: Dieses GGUF enthält K-Quant Typen, die aktuell nicht dequantisiert werden:");
        if b_has_q4k { eprintln!("- GGML_TYPE_Q4_K"); }
        if b_has_q5k { eprintln!("- GGML_TYPE_Q5_K"); }
        eprintln!("Bitte quantisiere das Modell vorübergehend in Q8_0 oder Q6_K um.");
        eprintln!("Beispiel mit llama.cpp (Windows):");
        eprintln!("  - quantize.exe input.gguf output.Q8_0.gguf Q8_0");
        eprintln!("Oder nutze ein bereits Q8_0 / Q6_K Modell.");
        eprintln!();
    }
}
