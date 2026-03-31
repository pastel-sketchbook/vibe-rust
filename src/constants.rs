//! Shared constants used across the crate.

// ---------------------------------------------------------------------------
// Audio constants
// ---------------------------------------------------------------------------

/// Default output sample rate for `VibeVoice` models.
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// WAV output channels (mono).
pub const WAV_CHANNELS: u16 = 1;

/// WAV output bits per sample (32-bit float).
pub const WAV_BITS_PER_SAMPLE: u16 = 32;

// ---------------------------------------------------------------------------
// Test tone defaults
// ---------------------------------------------------------------------------

/// Default test tone duration in seconds.
pub const TEST_TONE_DURATION_SECS: f32 = 3.0;

/// Default test tone frequency in Hz (A4).
pub const TEST_TONE_FREQ_HZ: f32 = 440.0;

/// Default test tone amplitude.
pub const TEST_TONE_AMPLITUDE: f32 = 0.5;

/// Default path for generated test tone WAV.
pub const TEST_TONE_PATH: &str = "data/test_tone.wav";

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Minimum buffer size for resampling.
pub const RESAMPLE_MIN_BUFFER_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// TTS synthesis defaults
// ---------------------------------------------------------------------------

/// Default CFG scale for TTS synthesis.
pub const DEFAULT_CFG_SCALE: f32 = 1.5;

/// Default speaker name for TTS synthesis.
pub const DEFAULT_SPEAKER: &str = "carter";

/// Default text for TTS synthesis when no input is provided.
pub const DEFAULT_TEXT: &str = "VibeVoice is an open-source family of frontier Voice AI models \
     from Microsoft, including speech-to-text and text-to-speech capabilities.";

/// RTF threshold: below this value means generation is faster than real-time.
pub const RTF_REALTIME_THRESHOLD: f64 = 1.0;

// ---------------------------------------------------------------------------
// ASR defaults
// ---------------------------------------------------------------------------

/// Default maximum new tokens for ASR transcription.
pub const ASR_MAX_TOKENS: u32 = 32_768;

/// Default sampling temperature for ASR (0 = greedy).
pub const ASR_TEMPERATURE: f32 = 0.0;

/// Default nucleus sampling threshold for ASR (1.0 = disabled).
pub const ASR_TOP_P: f32 = 1.0;

// ---------------------------------------------------------------------------
// Display formatting
// ---------------------------------------------------------------------------

/// Maximum characters to display for raw transcription text.
pub const MAX_DISPLAY_CHARS: usize = 2000;

/// Maximum number of structured segments to display.
pub const MAX_DISPLAY_SEGMENTS: usize = 60;

/// Default width for banner separators.
pub const BANNER_SEPARATOR_WIDTH: usize = 72;

/// Default truncation length for text display.
pub const DEFAULT_TRUNCATE_LENGTH: usize = 120;

// ---------------------------------------------------------------------------
// Generation loop thresholds
// ---------------------------------------------------------------------------

/// EOS probability threshold for stopping generation.
pub const EOS_PROB_THRESHOLD: f64 = 0.5;

/// Minimum number of speech tokens before EOS can trigger.
pub const MIN_SPEECH_TOKENS: usize = 6;

/// Safety stop multiplier: generation stops when speech tokens exceed
/// `total_tokens * SAFETY_STOP_MULTIPLIER`.
pub const SAFETY_STOP_MULTIPLIER: usize = 3;

// ---------------------------------------------------------------------------
// DPM-Solver++ constants
// ---------------------------------------------------------------------------

/// Number of diffusion steps in DPM-Solver++.
pub const DPM_SOLVER_STEPS: usize = 5;

/// Minimum `sigma_t` value before lambda is clamped.
pub const SIGMA_T_MIN: f64 = 1e-10;

/// Maximum lambda value (clamped when `sigma_t` is near zero).
pub const LAMBDA_MAX: f64 = 20.0;

/// Step indices that use first-order DPM-Solver++ steps.
pub const DPM_FIRST_ORDER_STEPS: [usize; 2] = [0, 4];

/// Noise dimension multiplier (positive + negative trajectories).
pub const NOISE_DIM_MULTIPLIER: usize = 2;

// ---------------------------------------------------------------------------
// ONNX backend
// ---------------------------------------------------------------------------

/// Default number of intra-op threads for ONNX Runtime.
pub const DEFAULT_ONNX_INTRA_THREADS: usize = 4;

/// Text suffix: newline appended before tokenization.
pub const TEXT_SUFFIX_NEWLINE: &str = "\n";

// ---------------------------------------------------------------------------
// Model IDs
// ---------------------------------------------------------------------------

/// Default `HuggingFace` model id for `VibeVoice`-ASR.
pub const ASR_MODEL_ID: &str = "microsoft/VibeVoice-ASR";

/// Default `HuggingFace` model id for `VibeVoice`-Realtime.
pub const REALTIME_MODEL_ID: &str = "microsoft/VibeVoice-Realtime-0.5B";

/// Default ONNX model id (fp16 export with voice presets).
pub const REALTIME_ONNX_MODEL_ID: &str = "nenad1002/microsoft-vibevoice-0.5B-onnx-fp16";

/// Default `HuggingFace` model id for `VibeVoice`-TTS.
pub const TTS_MODEL_ID: &str = "microsoft/VibeVoice-1.5B";

// ---------------------------------------------------------------------------
// File paths
// ---------------------------------------------------------------------------

/// Default local ONNX model directory.
pub const DEFAULT_LOCAL_MODEL_DIR: &str = "models/vibevoice-onnx";

/// Legacy voice preset directory.
pub const LEGACY_VOICE_DIR: &str = "demo/voices/streaming_model";

/// Default output directory for generated audio.
pub const DEFAULT_OUTPUT_DIR: &str = "output";

// ---------------------------------------------------------------------------
// Download hints
// ---------------------------------------------------------------------------

/// Hint message for downloading the ONNX model.
pub const HF_DOWNLOAD_ONNX_HINT: &str = "hf download nenad1002/microsoft-vibevoice-0.5B-onnx-fp16";

/// Hint message for downloading the ASR model.
pub const HF_DOWNLOAD_ASR_HINT: &str = "hf download microsoft/VibeVoice-ASR";
