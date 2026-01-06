# LLM_Node

LLM_Node is a Rust project for **local and distributed Llama inference** built on **Candle**. It is designed as an **LLM node**: one node can run a model fully locally, or—via routing with a `BlocksMap`—load only selected Transformer layers and execute the remaining layers through a **P2P network (libp2p)**.

The focus is on practical inference, an interactive **CLI chat** with **token streaming**, **KV-cache optimizations**, and **distributed block forwarding** (layer sharding).

---

## What this repository does

- provides an interactive **CLI chat app** (token streaming, stop sequences, prompt templates)
- loads Llama weights (mmap safetensors) and runs inference with Candle
- supports **distributed execution**: individual Transformer layers can be loaded per peer and computed remotely
- manages server-side **per-peer KV caches**, so different peers keep separate contexts

---

## Key features

### Inference & performance (local)
- **Paged KV cache** (chunked growth) instead of `Tensor::cat` per token
- **Mixed precision path** (attention scores in F32, remaining tensors in model dtype)
- **Cached additive -inf masks** instead of expensive broadcast masked-fill patterns
- **GQA/MQA handling** with reduced duplication (fallback to `repeat_kv`)

### Distributed execution (P2P)
- **Selective layer loading** per block based on `BlocksMap`
- **Block forwarding**: a peer can execute `forward_one_block` server-side
- **Auto-connect** to all required peers (derived from routing)
- **Server handler** with **cache per peer hash**, plus clear/reset logic

### CLI / UX
- Token streaming (delta decoding)
- Stop sequences as strings or token-id sequences
- Prompt-template auto detection:
  - ChatML (two variants), Llama2, Llama3, Mistral, Gemma, Alpaca, fallback tags
- Commands: `help`, `exit`, `peers`, `connect <peer_id> <addr>`, `clear`
- Save/load token context to a local file (CSV)
- Space key aborts ongoing streaming output

---

## Federated model concept (and benefits)

The repository is designed for **distributed LLM models**. Instead of a single node holding all model weights and executing the full forward pass, routing (via `BlocksMap`) distributes **Transformer layers** across multiple peers. In that sense, the project can act as a technical foundation for a **federated AI model**: multiple independent nodes cooperate, and no single node must own or compute everything.

Benefits of a federated/distributed approach:
- **Scale across multiple machines**: large models become usable even if a single node lacks enough RAM/VRAM.
- **Better resource utilization**: heterogeneous hardware (different GPUs/CPUs) can be used more efficiently by assigning layers accordingly.
- **More control over model access**: nodes do not necessarily need all weights (depending on routing and deployment strategy).
- **More robust operating models**: routing can be adjusted (by orchestration outside this code) when peers fail.
- **Practical deployments**: updates can roll out step-by-step because not everything must be replaced at once.

---

## Repository layout (key modules)

- `main.rs`  
  CLI chat client, prompt template detection, streaming output, save/load context, startup of P2P runtime and server handlers, backend selection.

- `local_llama.rs`  
  Candle-based model implementation plus optimizations:
  `LocalLlama`, `Llama`, `Block`, `CausalSelfAttention`, `Cache`, `PagedKv`.

- `p2p_*` modules  
  libp2p runtime, wire format, codec, block forwarding, BlocksMap routing, identity/keys.

---

## Requirements

- Rust (stable)
- Candle working (CPU or GPU backend depending on the setup)
- Model assets:
  - safetensors weights (possibly sharded files)
  - JSON config (`LlamaConfig`)
  - tokenizer JSON

---

## Build

```bash
cargo build --release
