# vibe-rust

Rust port of [vibe-demo](../vibe-demo) -- hands-on exploration of
[VibeVoice](https://github.com/microsoft/VibeVoice), Microsoft's open-source
family of frontier Voice AI models (ASR, TTS, Streaming TTS).

## What This Is

A from-scratch Rust rewrite of the Python `vibe-demo` project. All model
inference runs through ONNX Runtime (`ort` crate) with fp16 weights, so no
Python, PyTorch, or transformers are needed at runtime.

**Current status:**

| Module | Status | Notes |
|--------|--------|-------|
| `utils` | Complete | Audio I/O (hound/symphonia), device detection, timing, test-tone generation |
| `tts` | Complete (stub) | Placeholder -- upstream inference code removed due to misuse concerns |
| `asr` | Scaffold | API surface defined; inference backend not yet integrated |
| `realtime` | Complete | Full ONNX inference pipeline, DPM-Solver++ diffusion, voice presets |
| `onnx_backend` | Complete | 5-model ONNX pipeline: text LM, TTS LM, diffusion head, vocoder, acoustic connector |

**Lines of code:** ~2,400 Rust (15 files), plus Taskfile and shell scripts.

## VibeVoice Models

| Model | Size | Capability | HuggingFace |
|-------|------|------------|-------------|
| VibeVoice-ASR | 7B | 60-min long-form speech-to-text, 50+ languages, hotwords | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) |
| VibeVoice-TTS | 1.5B | 90-min multi-speaker (up to 4) text-to-speech, EN/ZH | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-Realtime | 0.5B | Real-time streaming TTS, ~200ms first-chunk latency | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |

## Quick Start

### Prerequisites

- Rust (latest stable, edition 2024)
- [Taskfile](https://taskfile.dev) (optional, for convenience tasks)
- ~2 GB disk for ONNX model weights

### Download Model Weights

```sh
# ONNX export (required for Realtime TTS)
hf download nenad1002/microsoft-vibevoice-0.5B-onnx-fp16

# Or download to a local directory
hf download nenad1002/microsoft-vibevoice-0.5B-onnx-fp16 --local-dir models/vibevoice-onnx
```

### Download Voice Presets

```sh
# All voices (default + experimental, 61 total)
task voices

# Or just defaults (25 voices)
task voices:default
```

### Build and Run

```sh
# Build
cargo build

# Run Realtime TTS with default text
cargo run --bin vibe-realtime -- --text "Hello from VibeVoice!"

# Run with a specific voice and text file
cargo run --bin vibe-realtime -- --file data/text/vibevoice_intro.txt --speaker emma
```

## Examples

### Realtime TTS

```sh
# Basic single-voice synthesis
task run:realtime:basic

# Multi-voice demo (EN/SP/KR, 9 utterances)
task run:realtime:multi-voice

# Bilingual EN/KR long-form demo (22 segments)
task run:realtime:bilingual

# Korean folktale narration (20 segments, parallel synthesis)
task run:realtime:folktale
```

### ASR (scaffold -- not yet functional)

```sh
cargo run --example basic_transcribe
cargo run --example basic_transcribe -- --audio data/sample.wav
```

### TTS (status check only)

```sh
cargo run --example status
```

## Project Layout

```
src/
  lib.rs              Library root -- re-exports all modules
  utils.rs            Audio I/O, device detection, timing, test-tone generation
  asr.rs              ASR model wrapper (scaffold)
  tts.rs              TTS stub (upstream code removed)
  realtime.rs         Streaming TTS wrapper, model loading, synthesis API
  onnx_backend.rs     ONNX Runtime inference: 5-model pipeline, DPM-Solver++, voice presets
  bin/
    asr.rs            CLI binary for ASR
    tts.rs            CLI binary for TTS status
    realtime.rs       CLI binary for Realtime TTS
examples/
  asr/
    basic_transcribe.rs
  tts/
    status.rs
  realtime/
    basic_synthesis.rs      Single-voice synthesis
    multi_voice_demo.rs     Multi-language, multi-voice (EN/SP/KR)
    bilingual_long_demo.rs  EN/KR bilingual long-form narration
    korean_folktale.rs      Full Korean folktale with parallel synthesis
data/
  text/                     Sample text inputs
demo/
  download_voices.sh        Voice preset downloader
  voices/                   Downloaded .pt voice prompts
output/                     Generated WAV files (git-ignored except showcase results)
docs/
  rationale/                Architecture decision records
```

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `ort` | ONNX Runtime bindings (model inference) |
| `half` | f16 support for ONNX tensor I/O |
| `ndarray` | N-dimensional array operations |
| `tokenizers` | HuggingFace tokenizer for text encoding |
| `hound` | WAV file read/write |
| `symphonia` | Audio decoding (WAV, MP3) |
| `clap` | CLI argument parsing (derive API) |
| `anyhow` / `thiserror` | Error handling |
| `rayon` | Parallel synthesis (worker pool) |
| `hf-hub` | HuggingFace model cache resolution |
| `serde` / `serde_json` | Config/schedule deserialization |
| `zip` | NPZ voice preset extraction |
| `rand` / `rand_distr` | DPM-Solver++ noise sampling |

## Development

```sh
# Quality checks (format, lint, build)
task check

# Auto-fix lint + format
task fix

# Run tests
task test

# Lines-of-code summary
task loc
```

### Pre-commit

All three must pass before committing:

```sh
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

## Hardware

- **Realtime TTS (0.5B):** Runs on Mac M4 Pro (~14s wall-clock for 20-segment folktale). NVIDIA T4+ also works.
- **ASR (7B):** Requires ~16 GB VRAM (CUDA) or ~28 GB RAM (CPU/MPS). Not yet functional in Rust.
- All models use MIT license.

## Upstream References

- [VibeVoice repo](https://github.com/microsoft/VibeVoice)
- [Project page](https://microsoft.github.io/VibeVoice)
- [HuggingFace collection](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
- [ASR docs](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md)
- [Realtime docs](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
- [TTS docs](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-tts.md)

## License

MIT
