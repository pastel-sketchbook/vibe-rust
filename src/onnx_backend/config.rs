//! Configuration types for the ONNX backend.

use serde::Deserialize;

/// Model configuration loaded from `config.json`.
#[derive(Debug, Deserialize)]
pub struct OnnxConfig {
    pub hidden_size: usize,
    pub latent_dim: usize,
    pub sample_rate: u32,
    pub speech_scaling_factor: f64,
    pub speech_bias_factor: f64,
    pub text_window_size: usize,
    pub speech_window_size: usize,
    pub num_diffusion_steps: usize,
    pub cfg_scale: f64,
    pub text_lm_layers: usize,
    pub tts_lm_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// DPM-Solver++ schedule loaded from `schedule.json`.
#[derive(Debug, Deserialize)]
pub struct DpmSchedule {
    pub sigmas: Vec<f64>,
    pub timesteps: Vec<i64>,
}
