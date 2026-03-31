# 0001 -- Why Rust

**Status:** Accepted
**Date:** 2026-03-30

## Context

The original `vibe-demo` project is written in Python, using PyTorch and
HuggingFace Transformers for model inference. We needed to decide whether to
continue iterating in Python or port to a different language.

## Decision

Port the entire project to Rust.

## Rationale

### Performance without a GIL

Python's Global Interpreter Lock makes true CPU parallelism difficult.
The Korean folktale demo synthesizes 20 segments -- in Python, parallelizing
this requires multiprocessing (with process-spawn overhead and memory
duplication for each model copy). In Rust, `rayon` gives us work-stealing
thread parallelism with zero-copy sharing of immutable data. Each worker
holds its own `Mutex<RealtimeTts>`, and scheduling is automatic.

### Single binary deployment

`cargo build --release` produces a self-contained binary with no runtime
dependencies beyond the ONNX Runtime shared library. No virtualenv, no pip,
no conda, no Python version conflicts. The binary can be shipped alongside
the ONNX model files and voice presets -- that's the entire deployment.

### Type safety for tensor shapes

The ONNX inference pipeline juggles 5 neural network sessions with KV caches
in f16, mixed with f32/f64 intermediate computations. Rust's type system
catches shape mismatches and precision bugs at compile time that would be
silent runtime errors in Python/numpy.

### Memory control

Voice presets are large (~200 MB of f16 KV cache per voice). Rust gives
explicit control over allocation and lifetime. There's no garbage collector
pause during synthesis, which matters for the real-time streaming use case
where first-chunk latency is a key metric.

### Ecosystem maturity

The Rust ML ecosystem has reached a point where this port is practical:

- `ort` (ONNX Runtime bindings) supports KV-cache models with f16 tensors
- `ndarray` handles the array operations that would use numpy in Python
- `tokenizers` is the same HuggingFace library, just called from Rust instead
- `hound` / `symphonia` cover audio I/O that librosa/soundfile handle in Python

## Tradeoffs

- **Iteration speed:** Rust compile times are slower than Python's edit-run
  cycle. Mitigated by incremental compilation and keeping the project small.
- **Ecosystem gaps:** No equivalent of `transformers.AutoModelForCausalLM`.
  We had to implement the ONNX pipeline manually (see 0002).
- **Community:** Fewer Rust developers in the voice/ML space, so less
  copy-paste-able reference code.

## Consequences

- All Python code in `vibe-demo` needs a Rust equivalent
- The ONNX model export (`nenad1002/microsoft-vibevoice-0.5B-onnx-fp16`) is
  required -- we cannot load PyTorch `.pt` model weights directly
- Voice presets must be `.npz` (numpy-compatible) format, parsed manually
  since `ndarray-npy` doesn't support f16
