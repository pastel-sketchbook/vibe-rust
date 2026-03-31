//! Shared utilities for audio I/O, device detection, and timing.
//!
//! Rust port of `vibe_demo/utils.py`.

use std::f32::consts::PI;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::constants;

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
#[must_use]
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
#[must_use]
pub fn detect_dtype(device: Device) -> DType {
    match device {
        Device::Cuda => DType::BFloat16,
        _ => DType::Float32,
    }
}

/// Attention implementation name suitable for the given device.
#[must_use]
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
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Load an audio file (WAV, MP3, or PCM) and return audio samples + sample rate.
///
/// WAV files are decoded via `hound` for speed. Other formats use `symphonia`.
pub fn load_audio(path: &Path) -> Result<AudioData> {
    load_audio_with_sr(path, None)
}

/// Load an audio file and optionally resample to `target_sr`.
///
/// If `target_sr` is `None`, the audio is returned at its native sample rate.
/// WAV files are decoded via `hound`; MP3 and other formats use `symphonia`.
pub fn load_audio_with_sr(path: &Path, target_sr: Option<u32>) -> Result<AudioData> {
    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let audio = if extension == "wav" {
        load_wav_hound(path)?
    } else {
        load_audio_symphonia(path)?
    };

    // Resample if a target rate is specified and differs from native rate
    let (samples, sample_rate) = if let Some(target) = target_sr {
        if target != audio.sample_rate && !audio.samples.is_empty() {
            let resampled =
                resample(&audio.samples, audio.sample_rate, target).with_context(|| {
                    format!(
                        "failed to resample from {} Hz to {} Hz",
                        audio.sample_rate, target
                    )
                })?;
            (resampled, target)
        } else {
            (audio.samples, audio.sample_rate)
        }
    } else {
        (audio.samples, audio.sample_rate)
    };

    Ok(AudioData {
        samples,
        sample_rate,
    })
}

/// Load a WAV file using hound (fast path).
fn load_wav_hound(path: &Path) -> Result<AudioData> {
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

    let mono = mix_to_mono(&samples, spec.channels as usize);

    Ok(AudioData {
        samples: mono,
        sample_rate,
    })
}

/// Load audio using symphonia (supports MP3, PCM, and other formats).
fn load_audio_symphonia(path: &Path) -> Result<AudioData> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::conv::FromSample;

    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let source = symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = symphonia::core::probe::Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(&ext.to_string_lossy());
    }

    let mut probed = symphonia::default::get_probe()
        .format(&hint, source, &Default::default(), &Default::default())
        .context("failed to probe audio format")?;

    let track = probed
        .format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("no supported audio tracks found")?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("failed to create decoder")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .context("unknown sample rate")?;

    // Note: mono mixing is done per-frame inside the decode loop below,
    // so `samples` is already mono regardless of the channel count.
    let mut samples = Vec::new();

    while let Ok(packet) = probed.format.next_packet() {
        while !probed.format.metadata().is_latest() {
            probed.format.metadata().pop();
        }

        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F32(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += ab.chan(ch)[frame_idx];
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::F64(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f64;
                    for ch in 0..n_ch {
                        sum += ab.chan(ch)[frame_idx];
                    }
                    samples.push((sum / n_ch as f64) as f32);
                }
            }
            AudioBufferRef::S16(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::S32(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::S8(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::U8(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::U16(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::U32(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::U24(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
            AudioBufferRef::S24(ref ab) => {
                let n_ch = ab.spec().channels.count();
                for frame_idx in 0..ab.frames() {
                    let mut sum = 0.0f32;
                    for ch in 0..n_ch {
                        sum += f32::from_sample(ab.chan(ch)[frame_idx]);
                    }
                    samples.push(sum / n_ch as f32);
                }
            }
        }
    }

    Ok(AudioData {
        samples,
        sample_rate,
    })
}

/// Mix multi-channel interleaved samples to mono by averaging channels.
fn mix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample audio from `from_sr` to `to_sr` using FFT-based resampling.
///
/// Uses `rubato` with a high-quality FFT resampler for audio-grade resampling.
pub fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    use audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{Fft, FixedSync, Resampler};

    let mut resampler = Fft::<f32>::new(
        from_sr as usize,
        to_sr as usize,
        samples.len().max(constants::RESAMPLE_MIN_BUFFER_SIZE),
        1,
        1,
        FixedSync::Input,
    )
    .context("failed to create resampler")?;

    let input_frames = samples.len();
    let output_len = resampler.process_all_needed_output_len(input_frames);

    let input_data = vec![samples.to_vec()];
    let mut output_data = vec![vec![0.0f32; output_len]];

    let input =
        SequentialSliceOfVecs::new(&input_data, 1, input_frames).context("invalid input buffer")?;
    let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .context("invalid output buffer")?;

    let (_nbr_in, nbr_out) = resampler
        .process_all_into_buffer(&input, &mut output, input_frames, None)
        .context("resampling failed")?;

    Ok(output_data[0][..nbr_out].to_vec())
}

/// Write mono audio samples to a WAV file, creating parent directories as needed.
pub fn save_audio(samples: &[f32], path: &Path, sample_rate: u32) -> Result<PathBuf> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    let spec = hound::WavSpec {
        channels: constants::WAV_CHANNELS,
        sample_rate,
        bits_per_sample: constants::WAV_BITS_PER_SAMPLE,
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
#[must_use = "consider saving the generated test tone"]
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
#[must_use = "consider saving the generated test tone"]
pub fn generate_test_tone_default() -> Result<PathBuf> {
    generate_test_tone(
        Path::new(constants::TEST_TONE_PATH),
        constants::DEFAULT_SAMPLE_RATE,
        constants::TEST_TONE_DURATION_SECS,
        constants::TEST_TONE_FREQ_HZ,
    )
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

/// Truncate a string to at most `max_chars` characters, appending `...` if truncated.
///
/// Unlike byte-based slicing this is safe for multi-byte UTF-8 text (e.g. CJK, emoji).
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => format!("{}...", &s[..byte_idx]),
        None => s.to_string(),
    }
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

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 5), "hello...");
        // Korean: each char is 3 bytes — must not panic
        assert_eq!(truncate_str("가나다라마바사", 3), "가나다...");
        assert_eq!(truncate_str("가나다", 3), "가나다");
        assert_eq!(truncate_str("가나다", 10), "가나다");
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn test_resample_upsample() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone_16k.wav");
        let saved = generate_test_tone(&path, 16_000, 1.0, 440.0).unwrap();
        let audio = load_audio_with_sr(&saved, Some(24_000)).unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples.len(), 24_000);
    }

    #[test]
    fn test_resample_downsample() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone_48k.wav");
        let saved = generate_test_tone(&path, 48_000, 1.0, 440.0).unwrap();
        let audio = load_audio_with_sr(&saved, Some(24_000)).unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples.len(), 24_000);
    }

    #[test]
    fn test_resample_noop_when_same_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone_24k.wav");
        let saved = generate_test_tone(&path, 24_000, 1.0, 440.0).unwrap();
        let audio = load_audio_with_sr(&saved, Some(24_000)).unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples.len(), 24_000);
    }
}
