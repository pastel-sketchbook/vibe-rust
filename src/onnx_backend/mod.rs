//! ONNX Runtime inference backend for VibeVoice-Realtime.
//!
//! Rust port of `vibevoice_full_onnx.py` (nenad1002/microsoft-vibevoice-0.5B-onnx-fp16).
//!
//! All 5 neural network components run via ONNX Runtime with KV-cache.
//! DPM-Solver++ diffusion scheduler implemented in pure Rust.
//! Speaker voice presets loaded from `.npz` files.

mod config;
mod dpm_solver;
mod generate;
mod sessions;
mod voice;

pub use config::{DpmSchedule, OnnxConfig};
pub use dpm_solver::{sigma_to_alpha_sigma, sigma_to_lambda};
pub use generate::generate;
pub use sessions::{OnnxSessions, OnnxSessionsBuilder};
pub use voice::{KvCache, extract_kv, list_voice_presets, load_voice_preset, resolve_voice};
