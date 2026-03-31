//! VibeVoice-Realtime: streaming text-to-speech with ~200ms first-chunk latency.
//!
//! Rust port of `vibe_demo/realtime.py`.
//!
//! Uses ONNX Runtime via the `ort` crate for model inference, with the
//! `nenad1002/microsoft-vibevoice-0.5B-onnx-fp16` model export.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::constants;
use crate::onnx_backend::{self, OnnxConfig, OnnxSessions};
use crate::utils::{self, Device};

/// Default HuggingFace model id.
pub const DEFAULT_MODEL: &str = constants::REALTIME_MODEL_ID;

/// Default ONNX model id (fp16 export with voice presets).
pub const DEFAULT_ONNX_MODEL: &str = constants::REALTIME_ONNX_MODEL_ID;

/// Output sample rate produced by the model.
pub const OUTPUT_SR: u32 = constants::DEFAULT_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Voice-preset discovery (legacy .pt files)
// ---------------------------------------------------------------------------

/// Find voice preset `.pt` files in the project or upstream package.
///
/// Searches `demo/voices/streaming_model/` relative to the project root.
pub fn list_voices(project_root: &Path) -> HashMap<String, PathBuf> {
    let mut voices = HashMap::new();
    let voice_dir = project_root.join(constants::LEGACY_VOICE_DIR);
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

/// Realtime TTS model wrapper backed by ONNX Runtime.
pub struct RealtimeTts {
    #[allow(dead_code)]
    config: RealtimeConfig,
    sessions: OnnxSessions,
    tokenizer: tokenizers::Tokenizer,
    onnx_config: OnnxConfig,
    schedule: onnx_backend::DpmSchedule,
    /// Path to the ONNX model directory (where .npz voice presets live).
    model_dir: PathBuf,
    /// Legacy .pt voice list for display purposes.
    #[allow(dead_code)]
    voices: HashMap<String, PathBuf>,
}

impl RealtimeTts {
    /// Load the Realtime TTS model from an ONNX model directory.
    ///
    /// The `onnx_model_dir` should contain the ONNX files, tokenizer.json,
    /// config.json, schedule.json, and .npz voice presets.
    pub fn load(config: RealtimeConfig, project_root: &Path) -> Result<Self> {
        println!("Loading VibeVoice-Realtime from {}", config.model_path);
        println!("  device={}  attn={}", config.device, config.attn_impl);

        let voices = list_voices(project_root);
        if voices.is_empty() {
            println!("  no legacy .pt voice presets found");
        } else {
            let mut names: Vec<_> = voices.keys().map(|s| s.as_str()).collect();
            names.sort();
            println!("  voices: {}", names.join(", "));
        }

        // Resolve the ONNX model directory
        let model_dir = resolve_onnx_model_dir(&config.model_path)?;
        println!("  ONNX model dir: {}", model_dir.display());

        // Load config + schedule
        let onnx_config: OnnxConfig = {
            let cfg_path = model_dir.join("config.json");
            let data = std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("Missing config.json in {}", model_dir.display()))?;
            serde_json::from_str(&data)?
        };

        let schedule: onnx_backend::DpmSchedule = {
            let sched_path = model_dir.join("schedule.json");
            let data = std::fs::read_to_string(&sched_path)
                .with_context(|| format!("Missing schedule.json in {}", model_dir.display()))?;
            serde_json::from_str(&data)?
        };

        // Load tokenizer
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Load ONNX sessions
        println!("  Loading ONNX sessions...");
        let sessions = OnnxSessions::load(&model_dir)?;

        // List available .npz voice presets
        let npz_presets = onnx_backend::list_voice_presets(&model_dir);
        if npz_presets.is_empty() {
            println!("  WARNING: no .npz voice presets found in ONNX model dir");
        } else {
            let names: Vec<String> = npz_presets
                .iter()
                .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                .collect();
            println!("  ONNX voices: {}", names.join(", "));
        }

        println!("  Model loaded successfully.");

        Ok(Self {
            config,
            sessions,
            tokenizer,
            onnx_config,
            schedule,
            model_dir,
            voices,
        })
    }

    /// Synthesize speech from text.
    ///
    /// The `speaker` name is matched against `.npz` voice presets in the model
    /// directory using case-insensitive substring matching.
    pub fn synthesize(
        &mut self,
        text: &str,
        speaker: &str,
        cfg_scale: f32,
        output_path: Option<&Path>,
    ) -> Result<SynthesisResult> {
        // Load voice preset
        let voice_path = onnx_backend::resolve_voice(&self.model_dir, speaker)?;
        let voice = onnx_backend::load_voice_preset(&voice_path)?;

        let start = Instant::now();

        let audio = onnx_backend::generate(
            &mut self.sessions,
            &self.tokenizer,
            &self.onnx_config,
            &self.schedule,
            &voice,
            text,
            cfg_scale as f64,
        )?;

        let gen_time = start.elapsed().as_secs_f64();

        let audio = audio.unwrap_or_default();
        let duration = audio.len() as f64 / OUTPUT_SR as f64;
        let rtf = if duration > 0.0 {
            gen_time / duration
        } else {
            0.0
        };

        // Save if requested
        let saved_path = if let Some(path) = output_path {
            utils::save_audio(&audio, path, OUTPUT_SR)?;
            Some(path.to_path_buf())
        } else {
            None
        };

        Ok(SynthesisResult {
            audio,
            duration_secs: duration,
            generation_time_secs: gen_time,
            rtf,
            output_path: saved_path,
        })
    }
}

// ---------------------------------------------------------------------------
// Model directory resolution
// ---------------------------------------------------------------------------

/// Check whether a directory looks like a valid ONNX model directory.
fn is_valid_onnx_dir(dir: &Path) -> bool {
    OnnxSessions::is_valid_onnx_dir(dir)
}

/// Resolve the ONNX model directory.
///
/// Checks in order:
/// 1. If `model_path` is an existing directory with ONNX files, use it
/// 2. Look for the **ONNX export** repo (`DEFAULT_ONNX_MODEL`) in HF cache
/// 3. Look for `model_path` (e.g. original Microsoft repo) in HF cache
///    — but only accept it if it contains ONNX files
/// 4. Check a well-known local path `models/vibevoice-onnx`
fn resolve_onnx_model_dir(model_path: &str) -> Result<PathBuf> {
    // 1. Direct path
    let p = PathBuf::from(model_path);
    if p.is_dir() && is_valid_onnx_dir(&p) {
        return Ok(p);
    }

    let hf_cache = dirs_hf_cache();

    // Helper: look for a repo in HF cache and validate it has ONNX files.
    let find_in_cache = |repo_id: &str| -> Option<PathBuf> {
        let repo_dir_name = format!("models--{}", repo_id.replace('/', "--"));
        for cache_root in &hf_cache {
            let snapshots_dir = cache_root.join(&repo_dir_name).join("snapshots");
            if snapshots_dir.is_dir()
                && let Ok(entries) = std::fs::read_dir(&snapshots_dir)
            {
                let mut snapshots: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_dir() && is_valid_onnx_dir(p))
                    .collect();
                snapshots.sort();
                if let Some(snap) = snapshots.last() {
                    return Some(snap.clone());
                }
            }
        }
        None
    };

    // 2. Always check the ONNX export repo first
    if let Some(dir) = find_in_cache(DEFAULT_ONNX_MODEL) {
        return Ok(dir);
    }

    // 3. Check the user-specified model path in HF cache (only if it has ONNX files)
    if model_path != DEFAULT_ONNX_MODEL
        && let Some(dir) = find_in_cache(model_path)
    {
        return Ok(dir);
    }

    // 4. Check well-known local path
    let local = PathBuf::from(constants::DEFAULT_LOCAL_MODEL_DIR);
    if local.is_dir() && is_valid_onnx_dir(&local) {
        return Ok(local);
    }

    anyhow::bail!(
        "Could not find ONNX model directory.\n\n\
         The ONNX export (not the original Microsoft model) is required.\n\
         Download it with:\n\n  \
         hf download {DEFAULT_ONNX_MODEL}\n\n\
         Or download to a local directory:\n\n  \
         hf download {DEFAULT_ONNX_MODEL} --local-dir {}",
        constants::DEFAULT_LOCAL_MODEL_DIR,
    )
}

/// Return HuggingFace cache directories to search.
fn dirs_hf_cache() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // $HF_HOME/hub or $HF_HUB_CACHE
    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        dirs.push(PathBuf::from(hub_cache));
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        dirs.push(PathBuf::from(hf_home).join("hub"));
    }

    // Default: ~/.cache/huggingface/hub
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(
            PathBuf::from(home)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }

    dirs
}
