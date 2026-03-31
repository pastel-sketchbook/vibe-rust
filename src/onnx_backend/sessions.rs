//! ONNX session management.
//!
//! Manages the 5 neural network components:
//! - text_lm: Text language model with KV-cache
//! - tts_lm: TTS language model with KV-cache
//! - diffusion_head: Diffusion model for speech synthesis
//! - vocoder: Neural vocoder for waveform generation
//! - acoustic_connector: Connects acoustic features to TTS LM

use std::path::Path;

use anyhow::Result;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::constants;

/// Required ONNX model files.
pub const REQUIRED_ONNX_FILES: &[&str] = &[
    "text_lm_kv.onnx",
    "tts_lm_kv.onnx",
    "diffusion_head.onnx",
    "vocoder.onnx",
    "acoustic_connector.onnx",
];

/// Collection of ONNX sessions for the VibeVoice-Realtime model.
pub struct OnnxSessions {
    pub text_lm: Session,
    pub tts_lm: Session,
    pub diffusion_head: Session,
    pub vocoder: Session,
    pub acoustic_connector: Session,
}

/// Builder for creating ONNX sessions.
pub struct OnnxSessionsBuilder {
    optimization_level: GraphOptimizationLevel,
    intra_threads: usize,
}

impl Default for OnnxSessionsBuilder {
    fn default() -> Self {
        Self {
            optimization_level: GraphOptimizationLevel::Level3,
            intra_threads: constants::DEFAULT_ONNX_INTRA_THREADS,
        }
    }
}

impl OnnxSessionsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn build(self, model_dir: &Path) -> Result<OnnxSessions> {
        let load = |name: &str| -> Result<Session> {
            let path = model_dir.join(format!("{name}.onnx"));
            anyhow::ensure!(path.exists(), "Missing ONNX model: {}", path.display());
            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_optimization_level(self.optimization_level)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_threads(self.intra_threads)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .commit_from_file(&path)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(session)
        };

        Ok(OnnxSessions {
            text_lm: load("text_lm_kv")?,
            tts_lm: load("tts_lm_kv")?,
            diffusion_head: load("diffusion_head")?,
            vocoder: load("vocoder")?,
            acoustic_connector: load("acoustic_connector")?,
        })
    }
}

impl OnnxSessions {
    /// Load all ONNX sessions from a model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        OnnxSessionsBuilder::default().build(model_dir)
    }

    /// Check whether a directory looks like a valid ONNX model directory.
    pub fn is_valid_onnx_dir(dir: &Path) -> bool {
        let mut files = vec!["config.json"];
        files.extend(REQUIRED_ONNX_FILES.iter());
        files.iter().all(|f| dir.join(f).exists())
    }
}
