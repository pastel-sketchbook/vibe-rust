# 0004 -- Audio I/O: hound + symphonia

**Status:** Accepted
**Date:** 2026-03-30

## Context

The project needs to read and write audio files. The Python version uses
`soundfile` (libsndfile wrapper) for WAV I/O and `librosa` for loading with
resampling. We needed Rust equivalents.

## Decision

Use `hound` for WAV read/write and `symphonia` for multi-format audio
decoding.

## Rationale

### hound for WAV write

WAV is the primary output format (model produces 24 kHz mono f32 PCM).
hound is the standard Rust WAV library:

- Pure Rust, no C dependencies
- Supports f32 and i16/i24/i32 sample formats
- Simple streaming writer API -- write samples one at a time, finalize
- Handles spec (channels, sample rate, bit depth) correctly

The `save_audio` function is 20 lines: create spec, open writer, write
samples, finalize. It also creates parent directories automatically.

### hound for WAV read

`load_audio` uses hound's reader to load WAV files for ASR input and
testing. It handles both float and integer sample formats, normalizes
integer samples to `[-1.0, 1.0]`, and mixes multi-channel to mono.

### symphonia for format flexibility

symphonia provides codec-agnostic audio decoding. We enable the `wav`,
`pcm`, and `mp3` features. This allows the project to accept MP3 inputs
for ASR without requiring a separate MP3 library. symphonia is pure Rust
with no system dependencies.

Currently, the actual decoding path uses hound directly (since all our
test data is WAV), but symphonia is available for future format support.

### Why not rodio?

rodio is an audio playback library, not a file I/O library. It wraps
symphonia for decoding but adds playback infrastructure we don't need.
The project generates WAV files -- it doesn't play audio through speakers.

### Why not cpal?

cpal is a cross-platform audio I/O library for recording and playback
from audio devices (microphones, speakers). We're doing file-based
batch processing, not real-time audio device I/O.

## Sample format decisions

- **Output:** 32-bit float WAV. This preserves full precision from the
  vocoder (which outputs f32) without quantization. Files are larger than
  16-bit PCM but avoid any clipping or rounding artifacts.
- **Input:** Accept both float and integer WAV. Integer samples are
  normalized by dividing by `2^(bits-1)`.
- **Channel mixing:** Multi-channel inputs are averaged to mono. All
  models expect single-channel audio.

## Consequences

- All output WAV files are 32-bit float, mono, 24 kHz. This is ~192 KB/s
  (about 11.5 MB per minute). The Korean folktale combined output is ~25 MB
  for ~2.2 minutes of audio.
- symphonia is a dependency but currently underused. It will become more
  relevant when ASR supports non-WAV input formats.
- No resampling is implemented yet. If ASR input arrives at a different
  sample rate than the model expects, we'll need to add a resampler
  (rubato crate or symphonia's built-in resampling).
