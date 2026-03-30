---
description: "vibe-rust: Rust port of vibe-demo -- VibeVoice exploration demos for Microsoft's open-source frontier Voice AI models (ASR, TTS, Streaming TTS)."
globs: "*.rs, *.toml, *.yaml, *.sh, *.wav, *.mp3, *.txt"
alwaysApply: true
---

## Project Overview

A **Rust port** of [`../vibe-demo`](../vibe-demo) (Python) -- hands-on exploration of **[VibeVoice](https://github.com/microsoft/VibeVoice)**, Microsoft's open-source family of frontier Voice AI models. The goal is to rewrite all Python modules and examples from `vibe-demo` into idiomatic Rust, preserving equivalent functionality and CLI interfaces.

### Port Scope

The Python source lives in `../vibe-demo/src/vibe_demo/` with these modules:

| Python module | Purpose | Rust port feasibility |
|---|---|---|
| `utils.py` | Device detection, audio I/O (load/save WAV), timing, test-tone generation, timestamp formatting | ✅ Straightforward — use `hound`/`symphonia` for audio, `std::time` for timing |
| `asr.py` | ASR model wrapper (VibeVoice-ASR 7B) — load model, transcribe audio, structured output, CLI | ⚠️ Depends on `transformers`/`torch` — use `candle` or `ort` (ONNX Runtime) for inference |
| `realtime.py` | Streaming TTS wrapper (VibeVoice-Realtime-0.5B) — voice presets, synthesis, CLI | ⚠️ Same inference dependency — `candle` or `ort` |
| `tts.py` | Placeholder/stub — upstream code removed, just status check | ✅ Trivial stub port |

### Key Python → Rust Mapping

| Python | Rust equivalent |
|---|---|
| `torch` / `transformers` | `candle` (native Rust ML) or `ort` (ONNX Runtime bindings) |
| `librosa` / `soundfile` | `symphonia` (decode) + `hound` (WAV read/write) |
| `numpy` | `ndarray` |
| `argparse` | `clap` |
| `pathlib.Path` | `std::path::PathBuf` |
| `huggingface_hub` | `hf-hub` crate |

### Port Priority

1. **`utils`** — foundational, no ML deps, port first
2. **`tts`** — trivial stub, port second
3. **`asr`** — core model inference, needs ML runtime research
4. **`realtime`** — streaming synthesis, most complex

## VibeVoice Models

| Model | Size | Capability | HuggingFace |
|-------|------|------------|-------------|
| **VibeVoice-ASR** | 7B | 60-min long-form speech-to-text (who/when/what), 50+ languages, hotwords | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) |
| **VibeVoice-TTS** | 1.5B | 90-min multi-speaker (up to 4) text-to-speech, EN/ZH | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| **VibeVoice-Realtime** | 0.5B | Real-time streaming TTS, ~200ms first-chunk latency, ~10 min | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |

> **Note:** VibeVoice-TTS code was removed upstream due to misuse concerns. TTS demos may be limited to inference via the HuggingFace model weights only.

## Toolchain

- **Language:** Rust (latest stable)
- **Build:** `cargo`
- **Task runner:** [Taskfile](https://taskfile.dev) (`Taskfile.yml`)
- **Linter:** `clippy`
- **Formatter:** `rustfmt`

### Pre-commit Requirement

**IMPORTANT:** Always run these checks before committing:

```sh
cargo fmt --check           # format check
cargo clippy -- -D warnings # lint
cargo test                  # tests
```

All three must pass.

## Code Style

- Use `anyhow` for application-level error handling, `thiserror` for library errors
- Use `clap` (derive API) for CLI argument parsing
- Prefer strong typing — define domain structs, avoid stringly-typed data
- Use type aliases and newtypes where they improve clarity
- Keep `unsafe` to an absolute minimum; document any `unsafe` blocks
- Prefer clarity over cleverness; add comments for non-obvious logic

## Project Layout

```
src/
  lib.rs                     # Library root — re-export modules
  utils.rs                   # Audio I/O, device detection, timing (port of utils.py)
  asr.rs                     # ASR model wrapper (port of asr.py)
  tts.rs                     # TTS stub (port of tts.py)
  realtime.rs                # Streaming TTS wrapper (port of realtime.py)
  bin/
    asr.rs                   # CLI binary for ASR
    tts.rs                   # CLI binary for TTS
    realtime.rs              # CLI binary for Realtime TTS
examples/                    # Standalone example scripts
  asr/
  tts/
  realtime/
data/                        # Sample audio files, text inputs, transcripts
Cargo.toml
Taskfile.yml
```

## VibeVoice Conventions

### ASR (Speech-to-Text)
- Model: `microsoft/VibeVoice-ASR` (7B params, requires GPU with ~16 GB VRAM)
- Accepts up to 60 minutes of audio in a single pass
- Outputs structured transcription: speaker labels, timestamps, and content
- Supports customized hotwords for domain-specific accuracy
- Audio format: WAV preferred

### TTS (Text-to-Speech)
- Model: `microsoft/VibeVoice-1.5B` (code removed upstream; weights-only usage)
- Generates up to 90 minutes of conversational speech
- Supports up to 4 distinct speakers with natural turn-taking

### Streaming TTS (Real-time)
- Model: `microsoft/VibeVoice-Realtime-0.5B` (0.5B params, deployment-friendly)
- ~200ms first-chunk latency, real-time streaming text input
- Primarily English; experimental multilingual voices available

### General
- Recommend NVIDIA GPU (T4 minimum for Realtime, A100/H100 for ASR-7B)
- Mac M4 Pro verified for Realtime model
- All models use MIT license

## Upstream References

- Repo: https://github.com/microsoft/VibeVoice
- Project page: https://microsoft.github.io/VibeVoice
- ASR docs: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md
- Realtime docs: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md
- TTS docs: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-tts.md
- Python reference: [`../vibe-demo`](../vibe-demo)
- HuggingFace collection: https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f
