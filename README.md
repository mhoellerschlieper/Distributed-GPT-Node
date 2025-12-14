# GGUF CPU Inferenz in Rust (Marcus Schlieper)

Willkommen! Du willst ein LLM im GGUF-Format lokal auf der CPU laufen lassen. Dieses Projekt lädt GGUF v3 Dateien, baut ein kleines Transformer‑Modell in Rust und führt Inferenz durch. Alles ohne unsafe. Viele Tests sind dabei. Du kannst es leicht anpassen.

Hinweis: Ich schreibe in einfachem Deutsch. Kurze Sätze. So lernst du es schnell.

---

## Features

- CPU‑only, kein unsafe
- GGUF v3 Loader: F32, F16, Q4_0 (18/20 B), Q8_0 (34/36 B), Q6_K (dequant vorhanden)
  - Q4_K und Q5_K werden geladen, aber to_f32_vec gibt noch Err (geplant)
- Tokenizer:
  - Llama (SentencePiece Unigram)
  - GPT‑2 (Byte‑Level BPE)
- Transformer:
  - KV‑Cache, GQA (Grouped‑Query Attention)
  - RoPE (mit Basis/Scaling per ENV)
  - SwiGLU‑MLP
- Sampling: Temperatur, Top‑K, Top‑P
- Viele Unit‑Tests und Golden‑Step‑Vergleich

---

## Schnellstart

Voraussetzungen:
- Rust stable (empfohlen: aktuelle stable)
- Ein GGUF‑Modell (z. B. TinyLlama)

Installieren und bauen:
```bash
git clone <dein-repo-url>
cd <repo>
cargo build --release
```

Starten (Linux/macOS):
```bash
export MODEL_PATH="/pfad/zum/model.gguf"
export PROMPT_TPL="simple"        # optional: simple | inst | chatml | im
cargo run --release
```

Starten (Windows PowerShell):
```powershell
$env:MODEL_PATH="C:\Pfad\zum\model.gguf"
$env:PROMPT_TPL="simple"
cargo run --release
```

Konsole zeigt:
- Model Infos
- Eingabezeile
- Tippe deine Frage
- Tippe "exit" für Ende

Tipp: Starte mit kleiner Temperatur (z. B. 0.2) für klare Antworten.

---

## Kurzes Beispiel

Linux/macOS:
```bash
export MODEL_PATH="/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
export TEMP=0.2
export TOP_K=20
export TOP_P=0.90
export MAX_NEW=200
cargo run --release
```

Windows PowerShell:
```powershell
$env:MODEL_PATH="C:\models\tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
$env:TEMP="0.2"
$env:TOP_K="20"
$env:TOP_P="0.90"
$env:MAX_NEW="200"
cargo run --release
```

Dann eingeben:
```
> Was ist 2 + 2?
```

---

## Projektstruktur

- src/main.rs: CLI, Prompt‑Loop, Checks
- src/gguf_loader.rs: GGUF v3 Laden, Dequant, Tokenizer‑Rohdaten
- src/model.rs: Config, Mapping der Gewichte, Checks
- src/layer.rs: Transformer, Attention (GQA), RoPE, SwiGLU, KV‑Cache
- src/math.rs: Matmul, Softmax, SiLU, RMSNorm, Sampler
- src/tokenizer.rs: Llama Unigram, GPT‑2 BPE (über tokenizers‑crate)
- src/tests.rs: Viele Unit‑ und Integrations‑Tests, Golden‑Step

---

## Wichtige Umgebungsvariablen

Allgemein:
- MODEL_PATH: Pfad zur .gguf Datei
- PROMPT_TPL: simple | inst | chatml | im (Template)
- TEMP, TOP_K, TOP_P, MAX_NEW, SEED: Sampling‑Regeln
- MODEL_DEBUG: 1/true → viele Debug‑Logs
- RUST_DECODE_TEST: 1 → kleiner Tokenizer Test in main, dann Ende
- RUST_LLAMA_CHECK: 1 → kompakter Modell‑Check, dann Ende

RoPE / Heads:
- ROPE_THETA: Basis überschreiben (z. B. 10000)
- ROPE_SCALE_APPLY: 1 → einfache Skalierung aktiv
- FORCE_KV_HEADS: 1 → KV‑Heads aus Gewichten schätzen und erzwingen

Tokenizer (BPE):
- BPE_ADD_PREFIX_SPACE: 0/1
- BPE_TRIM_OFFSETS: 0/1

K‑Quant Layout:
- K6_BLOCK_BYTES: 210 oder 208 (Standard 210)
- K6_ORDER: d-s-qh-ql (Standard) oder d-s-ql-qh

Golden‑Step‑Test:
- GOLDEN_JSON: Pfad zu golden.json
- MODEL_PATH: Pfad zum Modell (gleich zur Golden‑Datei)
- GOLDEN_TOLERANT: 1/0
- GOLDEN_TOP5_OVERLAP_MIN: z. B. 4
- GOLDEN_LOGITS_EPS: z. B. 0.02 bis 0.05
- GOLDEN_ALLOW_MISMATCH_STEPS: z. B. 0…N
- GOLDEN_TOP1_IN_TOP5_BIDIR: 1/0

---

## Tests

Alle Tests:
```bash
cargo test
```

Golden‑Step (Vergleich mit Referenz):
```bash
export GOLDEN_JSON="/pfad/golden.json"
export MODEL_PATH="/pfad/model.gguf"
export GOLDEN_TOLERANT=1
export GOLDEN_TOP5_OVERLAP_MIN=4
export GOLDEN_LOGITS_EPS=0.02
cargo test -- tests::golden_step_tests::test_golden_step_top1_top5_parity --nocapture
```

Hinweise:
- Top‑1 muss gleich sein. Mit Toleranz kannst du “ähnlich” erlauben.
- Top‑5 kann als Menge geprüft werden (z. B. 4/5 Überschneidung).
- Logits werden zentriert (minus max) verglichen.

---

## Grenzen und Status

- Nur CPU. Kein CUDA.
- Kein Batch. Ein Token pro Schritt.
- Kein fused QKV Laden (wirft Err, wenn so im Modell).
- Q4_K und Q5_K: Laden ok, Dequant noch TODO.
- Tokenizer nutzt huggingface tokenizers‑crate.

---

## Tipps bei Fehlern

- “Kein GGUF File (Magic)”
  - Datei ist nicht GGUF. Nutze eine .gguf Datei.

- “n_heads must be divisible by n_kv_heads”
  - KV‑Heads passen nicht. Prüfe GGUF‑Meta oder FORCE_KV_HEADS.

- “Layer X hat nur Null‑Gewichte”
  - Mapping schief gelaufen. Prüfe Tensors‑Namen oder Typen.

- “to_f32_vec: … K‑Quant nicht implementiert”
  - Modell nutzt Q4_K oder Q5_K. Wähle anderes Modell oder baue Dequant.

- Tokenizer‑Fehler
  - Prüfe tokenizer.ggml.model und tokens/merges/scores im GGUF.

---

## Beispiele: kleine Tools in main

Tokenizer Test:
```bash
export RUST_DECODE_TEST=1
export MODEL_PATH="/pfad/model.gguf"
cargo run --release
```

Llama Check (Infos + 0..29 Tokens + kleiner Encode/Decode):
```bash
export RUST_LLAMA_CHECK=1
cargo run --release
```

---

## Roadmap (Ideen)

- Dequant für Q4_K / Q5_K
- Fused‑QKV Pfad
- Batch‑Inferenz
- AVX/Parallelisierung für Matmul
- Mehr RoPE‑Scaling‑Modi (NTK, YaRN präziser)
- Mehr Prompt‑Vorlagen

---

## Beiträge

- PRs sind willkommen.
- Schreibe kurz, was du änderst.
- Bitte mit Tests.
- Stil: sicherer Code, klare Fehlertexte.

---

## Lizenz

- Siehe LICENSE (bitte ergänzen, z. B. MIT)

---

## Kontakt

Marcus Schlieper — ExpChat.ai  
Der KI Chat Client für den Mittelstand aus Breckerfeld im Sauerland.  
RPA, KI Agents, KI Internet Research, KI Wissensmanagement.  
Wir bringen KI in den Mittelstand.  
Adresse: Epscheider Str21, 58339 Breckerfeld

- E‑Mail: mschlieper@ylook.de
- Telefon: +49 2338 8748862
- Mobil: +49 151 15751864

Viel Erfolg! Du schaffst das. Wenn etwas klemmt, melde dich gern.
