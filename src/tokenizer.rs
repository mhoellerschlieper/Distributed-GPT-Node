// src/tokenizer.rs
use std::fs::File;
use std::io::Read;
use std::collections::HashSet;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct GgufTokenizer {
    tok: Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    special_token_ids: HashSet<u32>,
}

impl GgufTokenizer {
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        let enc = self.tok.encode(text, false).map_err(|e| format!("encode failed: {}", e))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if add_special {
            if let Some(bos) = self.bos_id { if ids.first().copied() != Some(bos) { ids.insert(0, bos); } }
            if let Some(eos) = self.eos_id { if ids.last().copied() != Some(eos) { ids.push(eos); } }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        let out = if skip_special {
            let filtered: Vec<u32> = ids.iter().copied().filter(|id| !self.special_token_ids.contains(id)).collect();
            self.tok.decode(&filtered, true).map_err(|e| format!("decode failed: {}", e))?
        } else {
            self.tok.decode(ids, false).map_err(|e| format!("decode failed: {}", e))?
        };
        Ok(out.replace('\u{2581}', " "))
    }

    pub fn bos_id(&self) -> Option<u32> { self.bos_id }
    pub fn eos_id(&self) -> Option<u32> { self.eos_id }
}

// Nur JSON-Loader
pub fn load_tokenizer_from_json_force<P: AsRef<std::path::Path>>(json_path: P) -> Result<GgufTokenizer, String> {
    let mut f = File::open(json_path.as_ref()).map_err(|e| format!("tokenizer.json open Fehler: {}", e))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("tokenizer.json read Fehler: {}", e))?;
    let tok = tokenizers::Tokenizer::from_bytes(&buf).map_err(|e| format!("Tokenizer JSON Fehler: {}", e))?;

    // BOS/EOS heuristisch
    let bos_candidates = ["<s>", "<|bos|>", "<|begin_of_text|>", "bos", "[BOS]"];
    let eos_candidates = ["</s>", "<|eos|>", "<|end_of_text|>", "eos", "[EOS]"];
    let bos_id = bos_candidates.iter().filter_map(|t| tok.token_to_id(t)).next();
    let eos_id = eos_candidates.iter().filter_map(|t| tok.token_to_id(t)).next();

    Ok(GgufTokenizer {
        tok,
        bos_id,
        eos_id,
        special_token_ids: HashSet::new(),
    })
}
