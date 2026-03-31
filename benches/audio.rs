//! Benchmarks for audio I/O and resampling.

use criterion::{Criterion, criterion_group, criterion_main};
use vibe_rust::utils;

fn bench_load_wav(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.wav");

    // 1 second of audio at 24 kHz
    let samples: Vec<f32> = (0..24_000)
        .map(|i| (i as f32 / 24_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();
    utils::save_audio(&samples, &path, 24_000).unwrap();

    c.bench_function("load_wav_1s", |b| {
        b.iter(|| utils::load_audio(&path).unwrap())
    });
}

fn bench_save_wav(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench_save.wav");

    let samples: Vec<f32> = (0..24_000)
        .map(|i| (i as f32 / 24_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();

    c.bench_function("save_wav_1s", |b| {
        b.iter(|| utils::save_audio(&samples, &path, 24_000).unwrap())
    });
}

fn bench_resample_up(c: &mut Criterion) {
    let samples: Vec<f32> = (0..24_000)
        .map(|i| (i as f32 / 24_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();

    c.bench_function("resample_24k_to_48k", |b| {
        b.iter(|| utils::resample(&samples, 24_000, 48_000).unwrap())
    });
}

fn bench_resample_down(c: &mut Criterion) {
    let samples: Vec<f32> = (0..48_000)
        .map(|i| (i as f32 / 48_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();

    c.bench_function("resample_48k_to_24k", |b| {
        b.iter(|| utils::resample(&samples, 48_000, 24_000).unwrap())
    });
}

fn bench_load_and_resample(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench_resample.wav");

    let samples: Vec<f32> = (0..48_000)
        .map(|i| (i as f32 / 48_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();
    utils::save_audio(&samples, &path, 48_000).unwrap();

    c.bench_function("load_and_resample_48k_to_24k", |b| {
        b.iter(|| utils::load_audio_with_sr(&path, Some(24_000)).unwrap())
    });
}

fn bench_load_wav_5s(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench_5s.wav");

    let samples: Vec<f32> = (0..120_000)
        .map(|i| (i as f32 / 24_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5)
        .collect();
    utils::save_audio(&samples, &path, 24_000).unwrap();

    c.bench_function("load_wav_5s", |b| {
        b.iter(|| utils::load_audio(&path).unwrap())
    });
}

criterion_group!(
    benches,
    bench_load_wav,
    bench_save_wav,
    bench_resample_up,
    bench_resample_down,
    bench_load_and_resample,
    bench_load_wav_5s,
);
criterion_main!(benches);
