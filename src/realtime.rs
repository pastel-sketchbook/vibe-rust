//! VibeVoice-Realtime: streaming text-to-speech with ~200ms first-chunk latency.
//!
//! Rust port of `vibe_demo/realtime.py`.
//!
//! **Status:** Scaffold only. The actual model inference requires a Rust ML
//! runtime (candle or ort/ONNX) and VibeVoice model support. This module
//! defines the public API surface and data types.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::utils::{self, Device};

/// Default HuggingFace model id.
pub const DEFAULT_MODEL: &str = "microsoft/VibeVoice-Realtime-0.5B";

/// Output sample rate produced by the model.
pub const OUTPUT_SR: u32 = 24_000;

// ---------------------------------------------------------------------------
// Voice-preset discovery
// ---------------------------------------------------------------------------

/// Find voice preset `.pt` files in the project or upstream package.
///
/// Searches `demo/voices/streaming_model/` relative to the project root.
pub fn list_voices(project_root: &Path) -> HashMap<String, PathBuf> {
    let mut voices = HashMap::new();
    let voice_dir = project_root
        .join("demo")
        .join("voices")
        .join("streaming_model");
    if voice_dir.is_dir()
        && let Ok(entries) = std::fs::read_dir(&voice_dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "pt")
                && let Some(stem) = path.file_stem()
            {
                voices.insert(stem.to_string_lossy().to_lowercase(), path);
            }
        }
    }
    voices
}

// ---------------------------------------------------------------------------
// Synthesis result
// ---------------------------------------------------------------------------

/// Result of a synthesis operation.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Generated audio samples (mono, `OUTPUT_SR` Hz).
    pub audio: Vec<f32>,
    /// Duration of generated audio in seconds.
    pub duration_secs: f64,
    /// Wall-clock time for generation in seconds.
    pub generation_time_secs: f64,
    /// Real-time factor (generation_time / duration).
    pub rtf: f64,
    /// Path where audio was saved, if any.
    pub output_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Model wrapper
// ---------------------------------------------------------------------------

/// Configuration for the Realtime TTS model.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    pub model_path: String,
    pub device: Device,
    pub attn_impl: String,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        let device = utils::detect_device();
        Self {
            model_path: DEFAULT_MODEL.to_string(),
            device,
            attn_impl: utils::detect_attn_impl(device).to_string(),
        }
    }
}

/// Realtime TTS model wrapper.
///
/// Currently a scaffold — `load()` and `synthesize()` will return errors
/// until a Rust ML inference backend is integrated.
pub struct RealtimeTts {
    #[allow(dead_code)]
    config: RealtimeConfig,
    #[allow(dead_code)]
    voices: HashMap<String, PathBuf>,
}

impl RealtimeTts {
    /// Load the Realtime TTS model.
    ///
    /// **Not yet implemented** — returns an error describing the missing backend.
    pub fn load(config: RealtimeConfig, project_root: &Path) -> Result<Self> {
        println!("Loading VibeVoice-Realtime from {}", config.model_path);
        println!("  device={}  attn={}", config.device, config.attn_impl);

        let voices = list_voices(project_root);
        if voices.is_empty() {
            println!("  no voice presets found (run download_experimental_voices.sh upstream)");
        } else {
            let mut names: Vec<_> = voices.keys().map(|s| s.as_str()).collect();
            names.sort();
            println!("  voices: {}", names.join(", "));
        }

        // TODO: integrate candle or ort for actual model loading
        anyhow::bail!(
            "Realtime TTS model inference is not yet implemented in Rust. \
             Requires candle or ONNX Runtime integration for VibeVoice-Realtime."
        );
    }

    /// Synthesize speech from text.
    ///
    /// **Not yet implemented.**
    pub fn synthesize(
        &self,
        _text: &str,
        _speaker: &str,
        _cfg_scale: f32,
        _output_path: Option<&Path>,
    ) -> Result<SynthesisResult> {
        anyhow::bail!("Realtime TTS synthesis not yet implemented")
    }
}
