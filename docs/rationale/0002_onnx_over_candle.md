# 0002 -- ONNX Runtime Over Candle

**Status:** Accepted
**Date:** 2026-03-30

## Context

Porting the VibeVoice-Realtime inference pipeline from Python/PyTorch to Rust
requires an ML runtime. Two serious candidates exist:

1. **candle** -- Hugging Face's native Rust ML framework
2. **ort** -- Rust bindings for Microsoft's ONNX Runtime

## Decision

Use ONNX Runtime via the `ort` crate.

## Rationale

### Model availability

The VibeVoice-Realtime pipeline consists of 5 neural network components:

1. `text_lm_kv.onnx` -- Text language model with KV-cache
2. `tts_lm_kv.onnx` -- TTS language model with KV-cache
3. `diffusion_head.onnx` -- Diffusion denoiser (per-step)
4. `vocoder.onnx` -- Latent-to-waveform decoder
5. `acoustic_connector.onnx` -- Feedback loop connector

A community ONNX export already existed (`nenad1002/microsoft-vibevoice-0.5B-onnx-fp16`)
with all 5 models, tokenizer, config, schedule, and `.npz` voice presets.
This meant we could start building immediately without implementing model
architecture code.

With candle, we would have needed to reimplement each model architecture
(transformer blocks, diffusion UNet, vocoder) in Rust, matching the exact
layer configurations. This is months of work for a 5-model pipeline.

### KV-cache support

Both language models use KV-cache for autoregressive generation. The ONNX
export bakes this into the model interface -- past key/value tensors are
explicit inputs and outputs. ONNX Runtime handles them as regular tensors
with no special framework support needed.

In candle, KV-cache management would require manual implementation inside
each transformer layer, matching the original PyTorch model's cache layout
exactly.

### fp16 inference

The ONNX models are exported in fp16. ONNX Runtime handles fp16 natively
on all backends (CPU, CoreML, CUDA). The `ort` crate exposes this through
the `half::f16` type.

candle supports f16 but requires more manual dtype management, especially
for mixed-precision operations (f16 tensors with f64 DPM-Solver++ math).

### Hardware portability

ONNX Runtime has mature execution providers:

- CPU (default, works everywhere)
- CoreML (macOS/Apple Silicon -- automatic acceleration)
- CUDA (NVIDIA GPUs)
- DirectML (Windows)

One binary, multiple backends. The `ort` crate selects the provider at
runtime based on availability.

### What we had to build ourselves

ONNX Runtime handles the neural network forward passes, but several pieces
still needed manual Rust implementation:

- **DPM-Solver++ scheduler** -- The 5-step diffusion sampling loop with
  classifier-free guidance. Implemented in pure Rust arithmetic (`f64`),
  matching the Python reference step-for-step.
- **NPZ parser** -- Voice presets are numpy `.npz` files containing f16
  arrays. The `ndarray-npy` crate doesn't support f16, so we wrote a custom
  parser that reads the `.npy` headers and reinterprets raw bytes as `f16`.
- **KV-cache management** -- Extracting, reshaping (3D to 4D), and feeding
  back KV tensors between autoregressive steps.
- **Text sanitization** -- Unicode smart quote replacement before tokenization.

## Tradeoffs

- **Binary size:** ONNX Runtime adds ~30 MB to the binary (shared library).
  candle would produce a smaller, fully static binary.
- **Debugging:** ONNX models are opaque. When something goes wrong inside a
  session, you get a cryptic ONNX error rather than a Rust stack trace. With
  candle, every layer would be debuggable Rust code.
- **Version coupling:** Tied to the specific ONNX export. If the upstream
  model architecture changes, we need a new ONNX export rather than just
  updating Rust model code.

## Consequences

- The `nenad1002/microsoft-vibevoice-0.5B-onnx-fp16` model is a hard
  dependency. If it disappears from HuggingFace, we need a replacement export.
- We carry a manual NPZ/f16 parser (~90 lines) that could be replaced if
  `ndarray-npy` adds f16 support.
- The DPM-Solver++ implementation must be validated against the Python
  reference whenever the schedule parameters change.
