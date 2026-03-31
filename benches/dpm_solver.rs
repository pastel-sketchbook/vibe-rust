//! Benchmarks for the DPM-Solver++ diffusion scheduler.

use std::collections::HashMap;

use criterion::{Criterion, criterion_group, criterion_main};
use half::f16;
use ndarray::{ArrayD, IxDyn};

use vibe_rust::onnx_backend::{DpmSchedule, extract_kv, sigma_to_alpha_sigma, sigma_to_lambda};

fn bench_sigma_conversions(c: &mut Criterion) {
    let sigmas: Vec<f64> = vec![0.999, 0.8, 0.5, 0.2, 0.05, 0.0];

    c.bench_function("sigma_to_alpha_sigma", |b| {
        b.iter(|| {
            for &s in &sigmas {
                let _ = sigma_to_alpha_sigma(s);
            }
        })
    });

    c.bench_function("sigma_to_lambda", |b| {
        b.iter(|| {
            for &s in &sigmas {
                let _ = sigma_to_lambda(s);
            }
        })
    });
}

fn bench_extract_kv_cache(c: &mut Criterion) {
    // Simulate a voice preset with 24 layers of KV cache
    let mut voice: HashMap<String, ArrayD<f16>> = HashMap::new();
    let n_layers = 24;
    for i in 0..n_layers {
        let key_data: Vec<f16> = (0..108 * 896)
            .map(|j| f16::from_f64(j as f64 / 1000.0))
            .collect();
        let val_data: Vec<f16> = (0..108 * 896)
            .map(|j| f16::from_f64(j as f64 / 1000.0))
            .collect();
        voice.insert(
            format!("lm_key_{i}"),
            ArrayD::from_shape_vec(IxDyn(&[1, 108, 896]), key_data.clone()).unwrap(),
        );
        voice.insert(
            format!("lm_value_{i}"),
            ArrayD::from_shape_vec(IxDyn(&[1, 108, 896]), val_data.clone()).unwrap(),
        );
        voice.insert(
            format!("tts_lm_key_{i}"),
            ArrayD::from_shape_vec(IxDyn(&[1, 108, 896]), key_data).unwrap(),
        );
        voice.insert(
            format!("tts_lm_value_{i}"),
            ArrayD::from_shape_vec(IxDyn(&[1, 108, 896]), val_data).unwrap(),
        );
    }

    c.bench_function("extract_kv_cache_24_layers", |b| {
        b.iter(|| {
            let _ = extract_kv(&voice, "lm", n_layers);
            let _ = extract_kv(&voice, "tts_lm", n_layers);
        })
    });
}

fn bench_dpm_schedule_creation(c: &mut Criterion) {
    c.bench_function("create_dpm_schedule_5_steps", |b| {
        b.iter(|| DpmSchedule {
            sigmas: vec![0.999, 0.8, 0.5, 0.2, 0.05, 0.0],
            timesteps: vec![999, 800, 500, 200, 50, 0],
        })
    });
}

criterion_group!(
    benches,
    bench_sigma_conversions,
    bench_extract_kv_cache,
    bench_dpm_schedule_creation,
);
criterion_main!(benches);
