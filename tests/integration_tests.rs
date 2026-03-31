//! Integration tests for vibe-rust.

use vibe_rust::utils::{self, Device};

// ---------------------------------------------------------------------------
// Audio I/O integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_wav_roundtrip_float() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip_float.wav");
    let original: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 2.0 - 1.0).collect();
    let saved = utils::save_audio(&original, &path, 24_000).unwrap();
    assert!(saved.exists());

    let audio = utils::load_audio(&saved).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    assert_eq!(audio.samples.len(), original.len());
    for (a, b) in audio.samples.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-6, "sample mismatch: {a} vs {b}");
    }
}

#[test]
fn test_wav_roundtrip_int16() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip_int16.wav");

    // Write int16 WAV manually using hound
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    let original: Vec<f32> = (0..500).map(|i| (i as f32 / 500.0) * 2.0 - 1.0).collect();
    let max_val = i16::MAX as f32;
    for &s in &original {
        writer.write_sample((s * max_val) as i16).unwrap();
    }
    writer.finalize().unwrap();

    let audio = utils::load_audio(&path).unwrap();
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.samples.len(), original.len());
    // Allow larger tolerance for int16 roundtrip
    for (a, b) in audio.samples.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-3, "sample mismatch: {a} vs {b}");
    }
}

#[test]
fn test_load_audio_with_resampling() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resample_test.wav");
    let samples: Vec<f32> = (0..48_000)
        .map(|i| (i as f32 / 48_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();
    utils::save_audio(&samples, &path, 48_000).unwrap();

    // Load at native rate
    let native = utils::load_audio(&path).unwrap();
    assert_eq!(native.sample_rate, 48_000);
    assert_eq!(native.samples.len(), 48_000);

    // Load with target rate
    let resampled = utils::load_audio_with_sr(&path, Some(24_000)).unwrap();
    assert_eq!(resampled.sample_rate, 24_000);
    // Should be approximately half the samples
    assert!(
        (resampled.samples.len() as f64 - 24_000.0).abs() < 100.0,
        "expected ~24000 samples, got {}",
        resampled.samples.len()
    );
}

#[test]
fn test_load_audio_no_resampling_when_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_resample.wav");
    let samples: Vec<f32> = vec![0.0; 16_000];
    utils::save_audio(&samples, &path, 16_000).unwrap();

    let audio = utils::load_audio_with_sr(&path, None).unwrap();
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.samples.len(), 16_000);
}

// ---------------------------------------------------------------------------
// Resampling tests
// ---------------------------------------------------------------------------

#[test]
fn test_resample_preserves_duration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("duration_test.wav");

    // Generate 1 second at 44.1 kHz
    let sr_in = 44_100;
    let samples: Vec<f32> = (0..sr_in)
        .map(|i| (i as f32 / sr_in as f32 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();
    utils::save_audio(&samples, &path, sr_in).unwrap();

    let audio = utils::load_audio_with_sr(&path, Some(24_000)).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    // Duration should be preserved: ~1 second
    assert!(
        (audio.duration_secs() - 1.0).abs() < 0.01,
        "duration mismatch: expected ~1.0s, got {}s",
        audio.duration_secs()
    );
}

#[test]
fn test_resample_various_ratios() {
    use vibe_rust::utils::resample;

    let samples: Vec<f32> = (0..10000).map(|i| i as f32 / 10000.0).collect();

    // Upsample 2x
    let up = resample(&samples, 16_000, 32_000).unwrap();
    assert!(
        (up.len() as f64 - 20_000.0).abs() < 100.0,
        "expected ~20000 samples, got {}",
        up.len()
    );

    // Downsample 2x
    let down = resample(&samples, 32_000, 16_000).unwrap();
    assert!(
        (down.len() as f64 - 5_000.0).abs() < 100.0,
        "expected ~5000 samples, got {}",
        down.len()
    );

    // Upsample 3x
    let up3 = resample(&samples, 16_000, 48_000).unwrap();
    assert!(
        (up3.len() as f64 - 30_000.0).abs() < 100.0,
        "expected ~30000 samples, got {}",
        up3.len()
    );
}

// ---------------------------------------------------------------------------
// Device detection tests
// ---------------------------------------------------------------------------

#[test]
fn test_device_display() {
    assert_eq!(format!("{}", Device::Cpu), "cpu");
    assert_eq!(format!("{}", Device::Cuda), "cuda");
    assert_eq!(format!("{}", Device::Mps), "mps");
}

#[test]
fn test_device_parse_case_insensitive() {
    assert_eq!("CUDA".parse::<Device>().unwrap(), Device::Cuda);
    assert_eq!("Mps".parse::<Device>().unwrap(), Device::Mps);
    assert_eq!("CPU".parse::<Device>().unwrap(), Device::Cpu);
}

// ---------------------------------------------------------------------------
// Timestamp formatting tests
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_edge_cases() {
    assert_eq!(utils::format_timestamp(0.0), "00:00:00.000");
    assert_eq!(utils::format_timestamp(3600.0), "01:00:00.000");
    assert_eq!(utils::format_timestamp(86400.0), "24:00:00.000");
    assert_eq!(utils::format_timestamp(0.001), "00:00:00.001");
}

// ---------------------------------------------------------------------------
// String truncation tests
// ---------------------------------------------------------------------------

#[test]
fn test_truncate_boundary_conditions() {
    assert_eq!(utils::truncate_str("", 0), "");
    assert_eq!(utils::truncate_str("a", 0), "...");
    assert_eq!(utils::truncate_str("abc", 3), "abc");
    assert_eq!(utils::truncate_str("abcd", 3), "abc...");
}
