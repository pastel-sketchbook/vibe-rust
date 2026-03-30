//! VibeVoice-ASR: speech-to-text model wrapper.
//!
//! Rust port of `vibe_demo/asr.py`.
//!
//! **Status:** Scaffold only. The actual model inference requires a Rust ML
//! runtime (candle or ort/ONNX) and VibeVoice model support. This module
//! defines the public API surface and data types so downstream code can be
//! written against it.

use std::path::Path;

use anyhow::Result;

use crate::utils::{self, Device};

/// Default HuggingFace model id.
pub const DEFAULT_MODEL: &str = "microsoft/VibeVoice-ASR";

/// A single transcription segment with speaker/timing info.
#[derive(Debug, Clone)]
pub struct Segment {
    pub start_time: f64,
    pub end_time: f64,
    pub speaker_id: String,
    pub text: String,
}

/// Result of a transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub raw_text: String,
    pub segments: Vec<Segment>,
}

/// Configuration for the ASR model.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    pub model_path: String,
    pub device: Device,
    pub attn_impl: String,
}

impl Default for AsrConfig {
    fn default() -> Self {
        let device = utils::detect_device();
        Self {
            model_path: DEFAULT_MODEL.to_string(),
            device,
            attn_impl: utils::detect_attn_impl(device).to_string(),
        }
    }
}

/// ASR model wrapper.
///
/// Currently a scaffold — `load()` and `transcribe()` will return errors
/// until a Rust ML inference backend is integrated.
pub struct AsrModel {
    #[allow(dead_code)]
    config: AsrConfig,
}

impl AsrModel {
    /// Load the ASR model.
    ///
    /// **Not yet implemented** — returns an error describing the missing backend.
    pub fn load(config: AsrConfig) -> Result<Self> {
        println!("Loading VibeVoice-ASR from {}", config.model_path);
        println!("  device={}  attn={}", config.device, config.attn_impl);

        // TODO: integrate candle or ort for actual model loading
        anyhow::bail!(
            "ASR model inference is not yet implemented in Rust. \
             Requires candle or ONNX Runtime integration for VibeVoice-ASR."
        );
    }

    /// Transcribe a single audio file.
    ///
    /// **Not yet implemented.**
    pub fn transcribe(
        &self,
        _audio: &Path,
        _max_new_tokens: u32,
        _temperature: f32,
    ) -> Result<TranscriptionResult> {
        anyhow::bail!("ASR transcription not yet implemented")
    }
}

/// Pretty-print a transcription result to stdout.
pub fn print_transcription(result: &TranscriptionResult) {
    println!("\n--- Raw transcription ---");
    let display_len = result.raw_text.len().min(2000);
    print!("{}", &result.raw_text[..display_len]);
    if result.raw_text.len() > 2000 {
        println!("  ... ({} chars total)", result.raw_text.len());
    }
    println!();

    if !result.segments.is_empty() {
        println!(
            "\n--- Structured output ({} segments) ---",
            result.segments.len()
        );
        for seg in result.segments.iter().take(60) {
            let start = utils::format_timestamp(seg.start_time);
            let end = utils::format_timestamp(seg.end_time);
            println!(
                "  [{start} -> {end}] Speaker {}: {}",
                seg.speaker_id, seg.text
            );
        }
        if result.segments.len() > 60 {
            println!("  ... and {} more", result.segments.len() - 60);
        }
    }
}
