//! Shared utilities for audio I/O, device detection, and timing.
//!
//! Rust port of `vibe_demo/utils.py`.

use std::f32::consts::PI;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

// ---------------------------------------------------------------------------
// Device / dtype helpers
// ---------------------------------------------------------------------------

/// Compute device for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Mps,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda => write!(f, "cuda"),
            Device::Mps => write!(f, "mps"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cuda" => Ok(Device::Cuda),
            "mps" => Ok(Device::Mps),
            "cpu" => Ok(Device::Cpu),
            other => anyhow::bail!("unknown device: {other}"),
        }
    }
}

/// Auto-detect the best available compute device.
///
/// Currently returns `Mps` on Apple Silicon macOS, `Cpu` otherwise.
/// CUDA detection will require runtime checks against a GPU backend.
pub fn detect_device() -> Device {
    #[cfg(target_os = "macos")]
    {
        // Apple Silicon (aarch64) → MPS
        if cfg!(target_arch = "aarch64") {
            return Device::Mps;
        }
    }
    Device::Cpu
}

/// Floating-point precision for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    BFloat16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float32 => write!(f, "float32"),
            DType::BFloat16 => write!(f, "bfloat16"),
        }
    }
}

/// Return the preferred dtype for `device` (`BFloat16` on CUDA, `Float32` elsewhere).
pub fn detect_dtype(device: Device) -> DType {
    match device {
        Device::Cuda => DType::BFloat16,
        _ => DType::Float32,
    }
}

/// Attention implementation name suitable for the given device.
pub fn detect_attn_impl(device: Device) -> &'static str {
    match device {
        Device::Cuda => "sdpa", // flash_attention_2 if available at runtime
        _ => "sdpa",
    }
}

// ---------------------------------------------------------------------------
// Audio I/O
// ---------------------------------------------------------------------------

/// Audio sample data with its sample rate.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Mono audio samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioData {
    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Load a WAV file and return audio samples + sample rate.
pub fn load_audio(path: &Path) -> Result<AudioData> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read float samples")?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read int samples")?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Mix to mono if multi-channel
    let mono = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples
            .chunks(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };

    Ok(AudioData {
        samples: mono,
        sample_rate,
    })
}

/// Write mono audio samples to a WAV file, creating parent directories as needed.
pub fn save_audio(samples: &[f32], path: &Path, sample_rate: u32) -> Result<PathBuf> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;

    for &s in samples {
        writer.write_sample(s).context("failed to write sample")?;
    }

    writer.finalize().context("failed to finalize WAV")?;
    Ok(path.to_path_buf())
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// A simple scoped timer that prints elapsed time on drop.
pub struct Timer {
    label: String,
    start: Instant,
}

impl Timer {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let tag = if self.label.is_empty() {
            String::new()
        } else {
            format!(" [{}]", self.label)
        };
        println!("  elapsed{tag}: {:.2}s", elapsed.as_secs_f64());
    }
}

// ---------------------------------------------------------------------------
// Test tone generation
// ---------------------------------------------------------------------------

/// Generate a short sine-wave WAV file for smoke-testing pipelines.
pub fn generate_test_tone(
    path: &Path,
    sample_rate: u32,
    duration_secs: f32,
    freq_hz: f32,
) -> Result<PathBuf> {
    let n_samples = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * freq_hz * t).sin()
        })
        .collect();

    save_audio(&samples, path, sample_rate)
}

/// Generate a test tone with default parameters (3s, 440 Hz, 24 kHz).
pub fn generate_test_tone_default() -> Result<PathBuf> {
    generate_test_tone(Path::new("data/test_tone.wav"), 24_000, 3.0, 440.0)
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format seconds as `HH:MM:SS.mmm`.
pub fn format_timestamp(seconds: f64) -> String {
    let h = (seconds / 3600.0) as u32;
    let m = ((seconds % 3600.0) / 60.0) as u32;
    let s = seconds % 60.0;
    format!("{h:02}:{m:02}:{s:06.3}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp() {
        assert_eq!(format_timestamp(0.0), "00:00:00.000");
        assert_eq!(format_timestamp(61.5), "00:01:01.500");
        assert_eq!(format_timestamp(3661.123), "01:01:01.123");
    }

    #[test]
    fn test_detect_device() {
        let dev = detect_device();
        // On macOS ARM we expect MPS, otherwise CPU
        assert!(dev == Device::Cpu || dev == Device::Mps);
    }

    #[test]
    fn test_generate_and_load_tone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone.wav");

        let saved = generate_test_tone(&path, 24_000, 1.0, 440.0).unwrap();
        assert!(saved.exists());

        let audio = load_audio(&saved).unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples.len(), 24_000);
        assert!((audio.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_save_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("deep").join("test.wav");
        let samples = vec![0.0f32; 100];
        save_audio(&samples, &path, 16_000).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_device_from_str() {
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("MPS".parse::<Device>().unwrap(), Device::Mps);
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert!("gpu".parse::<Device>().is_err());
    }

    #[test]
    fn test_detect_dtype() {
        assert_eq!(detect_dtype(Device::Cuda), DType::BFloat16);
        assert_eq!(detect_dtype(Device::Mps), DType::Float32);
        assert_eq!(detect_dtype(Device::Cpu), DType::Float32);
    }
}
