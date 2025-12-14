# Large Language Models with GGUF CPU with Inference in Rust

Welcome! You want to run an LLM in the GGUF format locally on the CPU. This project loads GGUF v3 files, builds a small Transformer model in Rust, and performs inference. All without unsafe. Many tests are included. You can adapt it easily.

Note: I write in simple German. Short sentences. This way you learn it quickly.

---

## Features

- CPU-only, no unsafe
- GGUF v3 loader: F32, F16, Q4_0 (18/20 B), Q8_0 (34/36 B), Q6_K (dequant available)
  - Q4_K and Q5_K are loaded, but to_f32_vec still returns Err (planned)
- Tokenizer:
  - Llama (SentencePiece Unigram)
  - GPT-2 (Byte-Level BPE)
- Transformer:
  - KV cache, GQA (Grouped-Query Attention)
  - RoPE (with base/scaling via ENV)
  - SwiGLU MLP
- Sampling: Temperature, Top-K, Top-P
- Many unit tests and golden-step comparison

---

## Quick start

Requirements:
- Rust stable (recommended: latest stable)
- A GGUF model (e.g., TinyLlama)

Install and build:
bash
git clone 
cd 
cargo build --release


Start (Linux/macOS):
bash
export MODEL_PATH=&quot;/path/to/model.gguf&quot;
export PROMPT_TPL=&quot;simple&quot;        # optional: simple | inst | chatml | im
cargo run --release


Start (Windows PowerShell):
powershell
$env:MODEL_PATH=&quot;C:\path\to\model.gguf&quot;
$env:PROMPT_TPL=&quot;simple&quot;
cargo run --release


Console shows:
- Model info
- Input line
- Type your question
- Type &quot;exit&quot; to quit

Tip: Start with a small temperature (e.g., 0.2) for clear answers.

---

## Short example

Linux/macOS:
bash
export MODEL_PATH=&quot;/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf&quot;
export TEMP=0.2
export TOP_K=20
export TOP_P=0.90
export MAX_NEW=200
cargo run --release


Windows PowerShell:
powershell
$env:MODEL_PATH=&quot;C:\models\tinyllama-1.1b-chat-v1.0.Q8_0.gguf&quot;
$env:TEMP=&quot;0.2&quot;
$env:TOP_K=&quot;20&quot;
$env:TOP_P=&quot;0.90&quot;
$env:MAX_NEW=&quot;200&quot;
cargo run --release


Then type:

&gt; What is 2 + 2?


---

## Project structure

- src/main.rs: CLI, prompt loop, checks
- src/gguf_loader.rs: GGUF v3 loading, dequant, tokenizer raw data
- src/model.rs: Config, weight mapping, checks
- src/layer.rs: Transformer, attention (GQA), RoPE, SwiGLU, KV cache
- src/math.rs: Matmul, softmax, SiLU, RMSNorm, sampler
- src/tokenizer.rs: Llama Unigram, GPT-2 BPE (via tokenizers crate)
- src/tests.rs: Many unit and integration tests, golden-step

---

## Important environment variables

General:
- MODEL_PATH: Path to the .gguf file
- PROMPT_TPL: simple | inst | chatml | im (template)
- TEMP, TOP_K, TOP_P, MAX_NEW, SEED: sampling parameters
- MODEL_DEBUG: 1/true &rarr; verbose debug logs
- RUST_DECODE_TEST: 1 &rarr; small tokenizer test in main, then exit
- RUST_LLAMA_CHECK: 1 &rarr; compact model check, then exit

RoPE / heads:
- ROPE_THETA: override base (e.g., 10000)
- ROPE_SCALE_APPLY: 1 &rarr; enable simple scaling
- FORCE_KV_HEADS: 1 &rarr; estimate KV heads from weights and enforce

Tokenizer (BPE):
- BPE_ADD_PREFIX_SPACE: 0/1
- BPE_TRIM_OFFSETS: 0/1

K-quant layout:
- K6_BLOCK_BYTES: 210 or 208 (default 210)
- K6_ORDER: d-s-qh-ql (default) or d-s-ql-qh

Golden-step test:
- GOLDEN_JSON: Path to golden.json
- MODEL_PATH: Path to the model (same as for the golden file)
- GOLDEN_TOLERANT: 1/0
- GOLDEN_TOP5_OVERLAP_MIN: e.g., 4
- GOLDEN_LOGITS_EPS: e.g., 0.02 to 0.05
- GOLDEN_ALLOW_MISMATCH_STEPS: e.g., 0&hellip;N
- GOLDEN_TOP1_IN_TOP5_BIDIR: 1/0

---

## Tests

All tests:
bash
cargo test


Golden-step (comparison with reference):
bash
export GOLDEN_JSON=&quot;/path/golden.json&quot;
export MODEL_PATH=&quot;/path/model.gguf&quot;
export GOLDEN_TOLERANT=1
export GOLDEN_TOP5_OVERLAP_MIN=4
export GOLDEN_LOGITS_EPS=0.02
cargo test -- tests::golden_step_tests::test_golden_step_top1_top5_parity --nocapture


Notes:
- Top-1 must match. With tolerance you can allow &ldquo;similar&rdquo;.
- Top-5 can be checked as a set (e.g., 4/5 overlap).
- Logits are compared after centering (minus max).

---

## Limitations and status

- CPU only. No CUDA.
- No batching. One token per step.
- No fused QKV loading (returns Err if the model uses it).
- Q4_K and Q5_K: loading works, dequant is still TODO.
- Tokenizer uses the Hugging Face tokenizers crate.

---

## Troubleshooting tips

- &ldquo;No GGUF file (magic)&rdquo;
  - File is not GGUF. Use a .gguf file.

- &ldquo;n_heads must be divisible by n_kv_heads&rdquo;
  - KV heads do not match. Check GGUF metadata or FORCE_KV_HEADS.

- &ldquo;Layer X has only zero weights&rdquo;
  - Mapping went wrong. Check tensor names or types.

- &ldquo;to_f32_vec: &hellip; K-quant not implemented&rdquo;
  - Model uses Q4_K or Q5_K. Choose another model or implement dequant.

- Tokenizer errors
  - Check tokenizer.ggml.model and tokens/merges/scores in the GGUF.

---

## Examples: small tools in main

Tokenizer test:
bash
export RUST_DECODE_TEST=1
export MODEL_PATH=&quot;/path/model.gguf&quot;
cargo run --release


Llama check (info + 0..29 tokens + small encode/decode):
bash
export RUST_LLAMA_CHECK=1
cargo run --release


---

## Roadmap (ideas)

- Dequant for Q4_K / Q5_K
- Fused-QKV path
- Batch inference
- AVX/parallelization for matmul
- More RoPE scaling modes (NTK, more precise YaRN)
- More prompt templates

---

## Contributions

- PRs are welcome.
- Briefly describe what you change.
- Please include tests.
- Style: safe code, clear error messages.

---

## License

- See LICENSE (please add, e.g., MIT)

---

## Contact

Marcus Schlieper &mdash; ExpChat.ai  
The AI chat client for SMEs from Breckerfeld in the Sauerland.  
RPA, AI agents, AI internet research, AI knowledge management.  
We bring AI to SMEs.  
Address: Epscheider Str21, 58339 Breckerfeld

- Email: mschlieper@expchat.ai
- Phone: +49 2338 8748862
- Mobile: +49 151 15751864

Good luck! You can do it. If something gets stuck, feel free to reach out.
