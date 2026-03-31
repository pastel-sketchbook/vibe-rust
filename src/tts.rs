//! `VibeVoice`-TTS placeholder — upstream inference code was removed.
//!
//! Rust port of `vibe_demo/tts.py`.
//!
//! The `VibeVoice`-TTS (1.5B) model weights are available on `HuggingFace` at
//! `microsoft/VibeVoice-1.5B`, but the inference code was removed from the
//! upstream `VibeVoice` repository due to misuse concerns.
//!
//! This module provides:
//! - A status-check utility that reports whether the weights are cached locally.
//! - A `main()` so the `vibe-tts` binary exits cleanly with a helpful message.

use std::path::{Path, PathBuf};

use crate::constants;

/// Default `HuggingFace` model ID.
pub const MODEL_ID: &str = constants::TTS_MODEL_ID;

const NOTE: &str = "\
VibeVoice-TTS inference code was removed upstream due to misuse concerns.
Model weights are still hosted at: https://huggingface.co/microsoft/VibeVoice-1.5B
This module is a placeholder -- real synthesis will be added when inference
code is re-published.";

/// Check whether the TTS model weights are cached locally via `hf-hub`.
#[must_use]
pub fn hf_cache_dir(model_id: &str) -> Option<PathBuf> {
    let api = hf_hub::api::sync::Api::new().ok()?;
    let repo = api.model(model_id.to_string());
    let config_path = repo.get("config.json").ok()?;
    config_path.parent().map(Path::to_path_buf)
}

/// Print whether the TTS model weights are cached locally.
pub fn check_tts_status() {
    if let Some(cached) = hf_cache_dir(MODEL_ID) {
        println!("Weights cached at: {}", cached.display());
    } else {
        println!("Weights not found in local HuggingFace cache.");
        println!("  Download with:  hf download {MODEL_ID}");
    }
    println!();
    println!("{NOTE}");
}
