markdown
# Distributed-GPT-Node

Distributed-GPT-Node is a Rust project for **local and distributed Llama inference** based on **Candle**. It is intended as an **LLM node**: a node can run a model entirely locally or&mdash;via routing with a `BlocksMap`&mdash;load only selected Transformer layers and execute the remaining layers over a **P2P network (libp2p)**.

The focus is on practical inference: an interactive **CLI chat application** with **token streaming**, **KV-cache optimizations**, and **distributed block forwarding** (layer sharding).

&gt; Note: The text cites no external sources. Therefore, no APA citations are included here.

---

## Why Distributed-GPT-Node (advantages)

Distributed-GPT-Node enables distributed use of LLMs without requiring a single machine to bear the entire load. The project offers several practical advantages:

- **Distributing AI models across many different nodes**  
  Layers can be split among peers instead of keeping everything on one system.

- **Using large models despite limited capacity**  
  Even if individual computers have little RAM/VRAM, the overall model can run across multiple machines.

- **No expensive hardware required**  
  Instead of a single large GPU machine, existing hardware is combined.

- **Node PCs are often idle**  
  Free resources on existing computers can be used for inference.

- **Improved fault tolerance**  
  If a node fails, routing can be adjusted so the system continues running.

- **Flexible scaling**  
  More nodes mean more resources and potentially higher throughput.

- **Incremental expansion instead of big-bang upgrades**  
  New machines can be added gradually without a complete migration.

- **Heterogeneous hardware is usable**  
  Different CPUs/GPUs and systems can work together.

- **Better resource utilization**  
  Load can be distributed instead of overloading individual machines.

- **More control over model and data distribution**  
  Depending on routing, not all weights need to reside everywhere.

- **Practical for local networks and edge setups**  
  Inference can run closer to usage, without forcing a cloud dependency.

- **Fewer bottlenecks through layer sharding**  
  Layers can be distributed in ways that reduce bottlenecks.

- **Faster experimentation with distributed deployments**  
  Routing via `BlocksMap` makes new distributions easier to test.

- **Better cost transparency**  
  Capacity grows through many small contributions rather than one large investment.

---

## What this repository does

- provides an interactive **CLI chat application** (token streaming, stop sequences, prompt templates)
- loads Llama weights (mmap safetensors) and performs inference with Candle
- supports **distributed execution**: individual Transformer layers can be loaded per peer and computed remotely
- manages server-side **KV caches per peer**, so different peers maintain separate contexts

---

## Key features

### Inference &amp; performance (local)

- **Paged KV cache** (chunked growth) instead of `Tensor::cat` per token
- **Mixed precision path** (attention scores in F32, the rest in the model dtype)
- **Cached additive -inf masks** instead of expensive broadcast masked-fill patterns
- **GQA/MQA handling** with reduced duplication (fallback to `repeat_kv`)

### Distributed execution (P2P)

- **Selective layer loading** per block via `BlocksMap`
- **Block forwarding**: a peer can execute `forward_one_block` server-side
- **Auto-connect** to all required peers (derived from routing)
- **Server handler** with **cache per peer hash** plus clear/reset logic

### CLI / UX

- Token streaming (delta decoding)
- Stop sequences as strings or token-ID sequences
- Prompt-template auto-detection:
  - ChatML (two variants), Llama2, Llama3, Mistral, Gemma, Alpaca, fallback tags
- Commands: `help`, `exit`, `peers`, `connect  `, `clear`
- Save/load token context to a local file (CSV)
- Space key aborts ongoing streaming output

---

## Federated / distributed model concept

Distributed-GPT-Node is designed for **distributed LLM models**. Instead of a single node holding all weights and computing the full forward pass, routing (via `BlocksMap`) distributes the **Transformer layers** across multiple peers.

This makes the project a technical basis for a form of **federated inference**: multiple independent nodes cooperate, and no single node must own everything or compute everything.

---

## Repository layout (high level)

- `main.rs`  
  CLI chat, prompt-template detection, streaming output, save/load context, starting the P2P runtime and server handler, backend selection.
- `local_llama.rs`  
  Candle-based model implementation plus optimizations: `LocalLlama`, `Llama`, `Block`, `CausalSelfAttention`, `Cache`, `PagedKv`.
- `p2p_*` modules  
  libp2p runtime, wire format, codec, block forwarding, BlocksMap routing, identity/keys.

---

## Requirements

- Rust (stable)
- Candle (CPU or GPU backend depending on setup)
- Model assets:
  - safetensors weights (optionally sharded files)
  - JSON config (`LlamaConfig`)
  - tokenizer JSON

---

## Build

bash
cargo build --release


---

## Status &amp; contribution

The project is designed for practical inference and distributed experimentation. Issues and PRs are welcome, especially regarding:

- routing/orchestration around `BlocksMap`
- performance (KV cache, transfer, quantization)
- stability in the libp2p layer
- better observability (logs/metrics)

---
