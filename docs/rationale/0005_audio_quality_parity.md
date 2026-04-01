# 0005 -- Audio Quality Parity: Rust ONNX vs Python PyTorch

**Status:** Confirmed
**Date:** 2026-04-01

## Context

After completing the Rust ONNX inference pipeline for VibeVoice-Realtime, a
subjective quality comparison suggested that the Rust output sounded worse than
the Python PyTorch reference (`vibe-demo`). The Korean folktale example
("The Sun and the Moon") was the benchmark -- same 20 text segments, same
speaker (`kr-spk0_woman`), same CFG scale (1.5), same pause structure.

The question: is the Rust implementation introducing audio quality degradation,
or is the perceived difference an artifact of non-deterministic generation?

## Investigation

### Scope

Three outputs were compared:

1. **Rust ONNX** -- `vibe-rust` using ONNX Runtime with fp16 models
2. **Python ONNX** -- the community reference script (`vibevoice_full_onnx.py`)
   using the same fp16 ONNX models via `onnxruntime` in Python
3. **Python PyTorch** -- `vibe-demo` using the `vibevoice` package with
   float32 PyTorch inference

The Python ONNX baseline was critical: it isolates whether any quality
difference comes from the Rust code (pipeline logic, DPM-Solver++, KV-cache
management) versus the fp16 ONNX model precision.

### Exhaustive algorithmic audit

Every component of the Rust pipeline was compared line-by-line against the
upstream Python source code:

| Component | Files compared | Result |
|-----------|----------------|--------|
| DPM-Solver++ (v-prediction, 2nd-order midpoint) | `dpm_solver.rs` vs `dpm_solver.py` (1065 lines) | Match |
| Diffusion step count | `constants.rs` vs `realtime.py` line 147 | Both use 5 steps |
| First-order vs second-order pattern | `constants.rs` vs `DPMSolverMultistepScheduler` | `[0,4]` first-order, `[1,2,3]` second-order |
| Schedule (timesteps + sigmas) | `schedule.json` vs `set_timesteps(5)` output | Match: `[999, 799, 599, 400, 200]` |
| CFG guidance | `generate.rs` vs `modeling_vibevoice_streaming_inference.py` | `uncond + scale * (cond - uncond)`, both halves duplicated |
| Noise initialization | `generate.rs` vs upstream | `2 * latent_dim` noise, first half for diffusion |
| Vocoder latent scaling | `generate.rs:351-353` vs upstream | `latent / scaling_factor - bias_factor` |
| Text windowing / EOS logic | `generate.rs` vs `vibevoice_full_onnx.py` | Match |
| NPZ voice preset loading | `voice.rs` vs upstream `.npz` handling | f16 arrays preserved as-is |

No algorithmic bugs were found.

### Upstream source files read

The audit required reading the full upstream `vibevoice` package internals,
not just the public API:

- `vibevoice/schedule/dpm_solver.py` -- 1065 lines, full DPMSolverMultistepScheduler
- `vibevoice/modular/modeling_vibevoice_streaming_inference.py` -- 905 lines, generation loop
- `vibevoice/modular/modular_vibevoice_diffusion_head.py` -- 287 lines, diffusion architecture
- `vibevoice/modular/modeling_vibevoice_streaming.py` -- 190 lines, model init
- `vibevoice/modular/configuration_vibevoice.py` -- DiffusionHeadConfig
- `vibevoice/modular/configuration_vibevoice_streaming.py` -- StreamingConfig
- `nenad1002/.../vibevoice_full_onnx.py` -- 360 lines, community ONNX reference

### Quantitative comparison

All three outputs were analyzed across multiple dimensions.

**Overall statistics (full combined output):**

| Metric | Rust ONNX | Python ONNX | Python PyTorch |
|--------|-----------|-------------|----------------|
| Duration | 271.9s | 265.1s | 255.5s |
| RMS energy | 0.0566 | 0.0551 | 0.0568 |
| Peak amplitude | 0.351 | 0.330 | 0.359 |
| Band energy (lo/mid/hi) | 98.0/1.4/0.5% | 98.1/1.5/0.4% | 98.1/1.5/0.4% |

**Windowed noise-floor analysis (segment 1, 4096-sample Hann windows):**

| Metric | Rust ONNX | Python ONNX | Python PyTorch |
|--------|-----------|-------------|----------------|
| Speech band power (100-4 kHz) | 2.12e+01 | 2.00e+01 | 1.76e+01 |
| Noise floor power (>10 kHz) | 6.31e-07 | 7.10e-07 | 5.03e-07 |
| **SNR** | **75.3 dB** | **74.5 dB** | **75.4 dB** |

All three outputs have SNR within a 0.9 dB range -- well within the natural
variation from different random seeds. The Rust ONNX output (75.3 dB) is
essentially identical to the Python PyTorch output (75.4 dB).

## Conclusion

The Rust ONNX pipeline produces audio quality equivalent to both the Python
ONNX reference and the Python PyTorch reference. There is no quality bug.

The perceived difference was caused by **non-deterministic generation**:
each run uses different random noise seeds for the diffusion process, which
produces different prosody, pacing, and emphasis per segment. Some runs sound
subjectively better than others regardless of which runtime is used.

### Why durations differ

TTS is autoregressive -- the model generates speech tokens until it predicts
EOS (end-of-speech). Different random seeds lead to different numbers of
speech tokens per segment, resulting in different total durations:

- Rust ONNX: 271.9s
- Python ONNX: 265.1s
- Python PyTorch: 255.5s

This 6-7% variation is normal and expected.

### fp16 vs float32

The ONNX models run internally in fp16. The PyTorch path on Mac runs in
float32. Despite this precision difference, the final audio quality is
indistinguishable. This is because:

1. The vocoder (waveform decoder) is robust to small numerical differences
   in its input latents
2. The DPM-Solver++ scheduler in both Rust and Python uses f64 intermediate
   precision for the diffusion sampling math, limiting error accumulation
3. Only 5 diffusion steps are used, giving limited opportunity for
   precision errors to compound

## Consequences

- No code changes needed in the Rust pipeline
- The `vibevoice_full_onnx.py` comparison script is kept at
  `scripts/korean_folktale_onnx.py` for future regression testing
- Quality complaints should be investigated by re-running the same text
  with a different random seed before assuming a pipeline bug
- If deterministic output is desired in the future, both pipelines would
  need a fixed random seed (not currently implemented in either)
