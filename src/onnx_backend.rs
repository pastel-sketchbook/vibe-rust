//! ONNX Runtime inference backend for VibeVoice-Realtime.
//!
//! Rust port of `vibevoice_full_onnx.py` (nenad1002/microsoft-vibevoice-0.5B-onnx-fp16).
//!
//! All 5 neural network components run via ONNX Runtime with KV-cache.
//! DPM-Solver++ diffusion scheduler implemented in pure Rust.
//! Speaker voice presets loaded from `.npz` files.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use half::f16;
use ndarray::{Array2, Array3, Array4, ArrayD, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

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

#[derive(Debug, Deserialize)]
pub struct DpmSchedule {
    pub sigmas: Vec<f64>,
    pub timesteps: Vec<i64>,
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// KV cache as flat HashMap of named f16 arrays.
pub type KvCache = HashMap<String, ArrayD<f16>>;

/// Load a voice preset from an `.npz` file.
///
/// The voice preset `.npz` files from the ONNX export store all arrays as
/// numpy float16 (`<f2`).  The `ndarray-npy` crate doesn't support float16,
/// so we manually parse the `.npy` entries inside the zip.
pub fn load_voice_preset(path: &Path) -> Result<KvCache> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open voice preset: {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)?;
    let mut cache = KvCache::new();

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry
            .name()
            .strip_suffix(".npy")
            .unwrap_or(entry.name())
            .to_string();

        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;

        let arr = parse_npy_f16(&buf)
            .with_context(|| format!("Failed to parse array '{name}' from {}", path.display()))?;
        cache.insert(name, arr);
    }

    Ok(cache)
}

/// Parse a `.npy` byte buffer containing float16 data into an `ArrayD<f16>`.
///
/// Numpy `.npy` v1 format:
///   - 6 bytes magic: `\x93NUMPY`
///   - 1 byte major version
///   - 1 byte minor version
///   - 2 bytes (v1) or 4 bytes (v2+) header length, little-endian
///   - Header: ASCII Python dict, e.g. `{'descr': '<f2', 'fortran_order': False, 'shape': (1, 108, 896), }`
///   - Raw data bytes
fn parse_npy_f16(buf: &[u8]) -> Result<ArrayD<f16>> {
    anyhow::ensure!(buf.len() >= 10, "NPY buffer too short");
    anyhow::ensure!(&buf[..6] == b"\x93NUMPY", "Invalid NPY magic");

    let major = buf[6];
    let header_len_offset: usize;
    let header_len: usize;

    if major == 1 {
        header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        header_len_offset = 10;
    } else {
        // v2/v3: 4-byte header length
        anyhow::ensure!(buf.len() >= 12, "NPY v2+ buffer too short");
        header_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        header_len_offset = 12;
    }

    let header_end = header_len_offset + header_len;
    anyhow::ensure!(buf.len() >= header_end, "NPY header extends past buffer");

    let header = std::str::from_utf8(&buf[header_len_offset..header_end])
        .context("NPY header is not valid UTF-8")?;

    // Parse descr — we only support float16 (`<f2` or `=f2`)
    let descr = extract_npy_field(header, "descr").context("Missing 'descr' in NPY header")?;
    anyhow::ensure!(
        descr == "<f2" || descr == "=f2" || descr == "|f2",
        "Unsupported NPY dtype '{descr}' — only float16 (<f2) is supported"
    );

    // Parse shape
    let shape = parse_npy_shape(header).context("Failed to parse 'shape' from NPY header")?;

    // Compute expected element count
    let n_elements: usize = shape.iter().product();
    let data_bytes = &buf[header_end..];
    anyhow::ensure!(
        data_bytes.len() >= n_elements * 2,
        "NPY data too short: expected {} bytes for {} f16 elements, got {}",
        n_elements * 2,
        n_elements,
        data_bytes.len()
    );

    // Reinterpret raw bytes as f16 (little-endian, same as in-memory on LE platforms)
    let f16_data: Vec<f16> = data_bytes[..n_elements * 2]
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Ok(ArrayD::from_shape_vec(IxDyn(&shape), f16_data)?)
}

/// Extract a string field from a numpy header dict.
///
/// Example header: `{'descr': '<f2', 'fortran_order': False, 'shape': (1, 108, 896), }`
fn extract_npy_field(header: &str, field: &str) -> Option<String> {
    let pattern = format!("'{field}': '");
    let start = header.find(&pattern)? + pattern.len();
    let end = header[start..].find('\'')? + start;
    Some(header[start..end].to_string())
}

/// Parse the shape tuple from a numpy header dict.
fn parse_npy_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape': (")? + "'shape': (".len();
    let end = header[start..].find(')')? + start;
    let shape_str = &header[start..end];

    if shape_str.trim().is_empty() {
        return Some(vec![]); // scalar
    }

    Some(
        shape_str
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse::<usize>().ok()
                }
            })
            .collect(),
    )
}

/// Extract LM-style KV cache from voice preset with a given prefix.
fn extract_kv(voice: &KvCache, prefix: &str, n_layers: usize) -> KvCache {
    let mut kv = KvCache::new();
    for i in 0..n_layers {
        let key_name = format!("{prefix}_key_{i}");
        let val_name = format!("{prefix}_value_{i}");
        if let Some(k) = voice.get(&key_name) {
            kv.insert(format!("key_{i}"), k.clone());
        }
        if let Some(v) = voice.get(&val_name) {
            kv.insert(format!("value_{i}"), v.clone());
        }
    }
    kv
}

// ---------------------------------------------------------------------------
// ONNX sessions
// ---------------------------------------------------------------------------

pub struct OnnxSessions {
    pub text_lm: Session,
    pub tts_lm: Session,
    pub diffusion_head: Session,
    pub vocoder: Session,
    pub acoustic_connector: Session,
}

impl OnnxSessions {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let load = |name: &str| -> Result<Session> {
            let path = model_dir.join(format!("{name}.onnx"));
            anyhow::ensure!(path.exists(), "Missing ONNX model: {}", path.display());
            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_threads(4)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .commit_from_file(&path)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(session)
        };

        Ok(Self {
            text_lm: load("text_lm_kv")?,
            tts_lm: load("tts_lm_kv")?,
            diffusion_head: load("diffusion_head")?,
            vocoder: load("vocoder")?,
            acoustic_connector: load("acoustic_connector")?,
        })
    }
}

// ---------------------------------------------------------------------------
// ONNX model calls
// ---------------------------------------------------------------------------

/// Helper: extract a tensor output as an ndarray, given the tuple from `try_extract_tensor`.
///
/// `ort::value::Shape` derefs to `&[i64]`, so we index it directly.
fn extract_array_3d(shape: &ort::value::Shape, data: &[f16]) -> Result<Array3<f16>> {
    anyhow::ensure!(shape.len() == 3, "Expected 3D tensor, got {}D", shape.len());
    let (d0, d1, d2) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    Ok(Array3::from_shape_vec((d0, d1, d2), data.to_vec())?)
}

fn extract_array_dyn(shape: &ort::value::Shape, data: &[f16]) -> Result<ArrayD<f16>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())?)
}

/// Run text LM with KV-cache.
///
/// Returns (hidden_states [1, seq, 896], updated_kv).
fn run_text_lm(
    sess: &mut Session,
    input_ids: &Array2<i64>,
    past_kvs: &KvCache,
    n_layers: usize,
) -> Result<(Array3<f16>, KvCache)> {
    let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    inputs.push((
        "input_ids".to_string(),
        Tensor::from_array(input_ids.clone())?.into_dyn(),
    ));

    for i in 0..n_layers {
        let key = past_kvs
            .get(&format!("key_{i}"))
            .context("Missing LM key cache")?;
        let val = past_kvs
            .get(&format!("value_{i}"))
            .context("Missing LM value cache")?;

        // Reshape to 4D if needed: [batch, heads, seq, dim]
        let k4 = reshape_kv_4d(key)?;
        let v4 = reshape_kv_4d(val)?;

        inputs.push((format!("past_key_{i}"), Tensor::from_array(k4)?.into_dyn()));
        inputs.push((
            format!("past_value_{i}"),
            Tensor::from_array(v4)?.into_dyn(),
        ));
    }

    let outputs = sess.run(inputs)?;

    // Output 0: hidden_states
    let (hidden_shape, hidden_data) = outputs[0].try_extract_tensor::<f16>()?;
    let hidden_arr = extract_array_3d(hidden_shape, hidden_data)?;

    // Outputs 1..2*n_layers: present_key_0, present_value_0, ...
    let mut new_kvs = KvCache::new();
    for i in 0..n_layers {
        let (k_shape, k_data) = outputs[1 + 2 * i].try_extract_tensor::<f16>()?;
        let (v_shape, v_data) = outputs[2 + 2 * i].try_extract_tensor::<f16>()?;
        new_kvs.insert(format!("key_{i}"), extract_array_dyn(k_shape, k_data)?);
        new_kvs.insert(format!("value_{i}"), extract_array_dyn(v_shape, v_data)?);
    }

    Ok((hidden_arr, new_kvs))
}

/// Run TTS LM with KV-cache.
///
/// Returns (hidden_states, eos_logits, updated_kv).
fn run_tts_lm(
    sess: &mut Session,
    inputs_embeds: &Array3<f16>,
    tts_text_mask: &Array2<i64>,
    past_kvs: &KvCache,
    n_layers: usize,
) -> Result<(Array3<f16>, f32, KvCache)> {
    let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    inputs.push((
        "inputs_embeds".to_string(),
        Tensor::from_array(inputs_embeds.clone())?.into_dyn(),
    ));
    inputs.push((
        "tts_text_mask".to_string(),
        Tensor::from_array(tts_text_mask.clone())?.into_dyn(),
    ));

    for i in 0..n_layers {
        let key = past_kvs
            .get(&format!("key_{i}"))
            .context("Missing TTS LM key cache")?;
        let val = past_kvs
            .get(&format!("value_{i}"))
            .context("Missing TTS LM value cache")?;

        let k4 = reshape_kv_4d(key)?;
        let v4 = reshape_kv_4d(val)?;

        inputs.push((format!("past_key_{i}"), Tensor::from_array(k4)?.into_dyn()));
        inputs.push((
            format!("past_value_{i}"),
            Tensor::from_array(v4)?.into_dyn(),
        ));
    }

    let outputs = sess.run(inputs)?;

    // Output 0: hidden_states
    let (hs_shape, hs_data) = outputs[0].try_extract_tensor::<f16>()?;
    let hidden_arr = extract_array_3d(hs_shape, hs_data)?;

    // Output 1: eos_logits
    let (_eos_shape, eos_data) = outputs[1].try_extract_tensor::<f16>()?;
    let eos_val = eos_data[0].to_f32();

    // Outputs 2..2+2*n_layers: present_key/value
    let mut new_kvs = KvCache::new();
    for i in 0..n_layers {
        let (k_shape, k_data) = outputs[2 + 2 * i].try_extract_tensor::<f16>()?;
        let (v_shape, v_data) = outputs[3 + 2 * i].try_extract_tensor::<f16>()?;
        new_kvs.insert(format!("key_{i}"), extract_array_dyn(k_shape, k_data)?);
        new_kvs.insert(format!("value_{i}"), extract_array_dyn(v_shape, v_data)?);
    }

    Ok((hidden_arr, eos_val, new_kvs))
}

type Array1F16 = ndarray::Array1<f16>;

/// Run diffusion head (single step): noisy_latent + timestep + condition → v_prediction.
fn run_diffusion(
    sess: &mut Session,
    noisy: &Array2<f16>,
    timestep: &Array1F16,
    condition: &Array2<f16>,
) -> Result<Array2<f16>> {
    let outputs = sess.run(ort::inputs![
        "noisy_latent" => Tensor::from_array(noisy.clone())?,
        "timestep" => Tensor::from_array(timestep.clone())?,
        "condition" => Tensor::from_array(condition.clone())?
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    Ok(Array2::from_shape_vec(
        (shape[0] as usize, shape[1] as usize),
        data.to_vec(),
    )?)
}

/// Run vocoder: latents → audio waveform.
fn run_vocoder(sess: &mut Session, latents: &Array3<f16>) -> Result<Vec<f32>> {
    let outputs = sess.run(ort::inputs![
        "latents" => Tensor::from_array(latents.clone())?
    ])?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    Ok(data.iter().map(|v| v.to_f32()).collect())
}

/// Run acoustic connector: latent → embedding for feedback loop.
fn run_connector(sess: &mut Session, latent: &Array3<f16>) -> Result<Array3<f16>> {
    let outputs = sess.run(ort::inputs![
        "latent" => Tensor::from_array(latent.clone())?
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    extract_array_3d(shape, data)
}

// ---------------------------------------------------------------------------
// DPM-Solver++ (pure Rust, matches Python reference)
// ---------------------------------------------------------------------------

fn sigma_to_alpha_sigma(sigma: f64) -> (f64, f64) {
    let alpha = 1.0 / (1.0 + sigma * sigma).sqrt();
    (alpha, sigma * alpha)
}

fn sigma_to_lambda(sigma: f64) -> f64 {
    let (alpha, sigma_t) = sigma_to_alpha_sigma(sigma);
    if sigma_t < 1e-10 {
        return 20.0;
    }
    (alpha / sigma_t).ln()
}

/// Sample one speech latent via DPM-Solver++ with classifier-free guidance.
///
/// Maintains 2 parallel denoising trajectories (positive + negative) matching
/// the PyTorch reference behavior.
fn sample_speech_token(
    diff_sess: &mut Session,
    sigmas: &[f64],
    timesteps: &[i64],
    pos_cond: &Array2<f16>, // [1, 896]
    neg_cond: &Array2<f16>, // [1, 896]
    cfg_scale: f64,
    latent_dim: usize,
) -> Result<Vec<f32>> {
    use rand::RngExt;
    use rand_distr::StandardNormal;
    let mut rng = rand::rng();

    // Start with random noise [2, latent_dim]
    let noise: Vec<f64> = (0..2 * latent_dim)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();

    let mut sample = noise.clone(); // [2, latent_dim] as flat vec, f64
    let mut m_list: Vec<Vec<f64>> = Vec::new();

    for step_idx in 0..5 {
        // half = sample[:1] (positive trajectory only)
        let half: Vec<f16> = sample[..latent_dim]
            .iter()
            .map(|&v| f16::from_f64(v))
            .collect();
        let half_arr = Array2::from_shape_vec((1, latent_dim), half)?;
        let t_val = f16::from_f64(timesteps[step_idx] as f64);
        let t_arr = ndarray::Array1::from_vec(vec![t_val]);

        // Run conditional diffusion
        let cond_eps = run_diffusion(diff_sess, &half_arr, &t_arr, pos_cond)?;
        // Run unconditional diffusion
        let uncond_eps = run_diffusion(diff_sess, &half_arr, &t_arr, neg_cond)?;

        // CFG: guided = uncond + cfg_scale * (cond - uncond)
        let mut half_eps = Vec::with_capacity(latent_dim);
        for j in 0..latent_dim {
            let c = cond_eps[[0, j]].to_f64();
            let u = uncond_eps[[0, j]].to_f64();
            half_eps.push(u + cfg_scale * (c - u));
        }

        // v_pred for both halves = [half_eps, half_eps]
        let mut v_pred = Vec::with_capacity(2 * latent_dim);
        v_pred.extend_from_slice(&half_eps);
        v_pred.extend_from_slice(&half_eps);

        let sig = sigmas[step_idx];
        let (alpha_s0, sigma_s0) = sigma_to_alpha_sigma(sig);

        // x0 = alpha_s0 * sample - sigma_s0 * v_pred
        let x0: Vec<f64> = sample
            .iter()
            .zip(v_pred.iter())
            .map(|(&s, &vp)| alpha_s0 * s - sigma_s0 * vp)
            .collect();
        m_list.push(x0.clone());

        let sig_next = sigmas[step_idx + 1];
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma(sig_next);
        let lam_t = sigma_to_lambda(sig_next);
        let lam_s0 = sigma_to_lambda(sig);

        if step_idx == 0 || step_idx == 4 {
            // First-order step
            let h = lam_t - lam_s0;
            let expm1_neg_h = (-h).exp_m1(); // expm1(-h)
            for i in 0..sample.len() {
                sample[i] = (sigma_t / sigma_s0) * sample[i] - alpha_t * expm1_neg_h * x0[i];
            }
        } else {
            // Second-order (midpoint) step
            let sig_prev = sigmas[step_idx - 1];
            let lam_s1 = sigma_to_lambda(sig_prev);
            let h = lam_t - lam_s0;
            let h_0 = lam_s0 - lam_s1;
            let r0 = h_0 / h;
            let expm1_neg_h = (-h).exp_m1();
            let prev_x0 = &m_list[m_list.len() - 2];
            for i in 0..sample.len() {
                let d0 = x0[i];
                let d1 = (1.0 / r0) * (x0[i] - prev_x0[i]);
                sample[i] = (sigma_t / sigma_s0) * sample[i]
                    - alpha_t * expm1_neg_h * d0
                    - 0.5 * alpha_t * expm1_neg_h * d1;
            }
        }
    }

    // Return positive half only [0..latent_dim] as f32
    Ok(sample[..latent_dim].iter().map(|&v| v as f32).collect())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reshape a potentially flat/dynamic KV array to 4D [batch, heads, seq, dim].
fn reshape_kv_4d(arr: &ArrayD<f16>) -> Result<Array4<f16>> {
    match arr.ndim() {
        4 => {
            let s = arr.shape();
            Ok(Array4::from_shape_vec(
                (s[0], s[1], s[2], s[3]),
                arr.iter().copied().collect(),
            )?)
        }
        _ => {
            anyhow::bail!(
                "Expected 4D KV cache array, got {}D shape {:?}",
                arr.ndim(),
                arr.shape()
            );
        }
    }
}

/// Sanitize text: replace Unicode smart quotes with ASCII equivalents.
fn sanitize_text(text: &str) -> String {
    text.replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{201c}', '\u{201d}'], "\"")
}

// ---------------------------------------------------------------------------
// Voice discovery for .npz presets
// ---------------------------------------------------------------------------

/// Find `.npz` voice preset files in the model directory.
pub fn list_voice_presets(model_dir: &Path) -> Vec<PathBuf> {
    let mut presets: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "npz") {
                presets.push(path);
            }
        }
    }
    presets.sort();
    presets
}

/// Resolve a speaker name to an `.npz` voice preset path.
///
/// Uses case-insensitive substring matching, falling back to the first preset.
pub fn resolve_voice(model_dir: &Path, speaker_name: &str) -> Result<PathBuf> {
    let presets = list_voice_presets(model_dir);
    if presets.is_empty() {
        anyhow::bail!("No .npz voice presets found in {}", model_dir.display());
    }

    let speaker_lower = speaker_name.to_lowercase();
    for p in &presets {
        let stem = p
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        if stem.contains(&speaker_lower) {
            return Ok(p.clone());
        }
    }

    // Fallback to first preset
    let chosen = presets[0].clone();
    println!(
        "  No match for '{}', using {}",
        speaker_name,
        chosen.file_name().unwrap_or_default().to_string_lossy()
    );
    Ok(chosen)
}

// ---------------------------------------------------------------------------
// Main generation loop
// ---------------------------------------------------------------------------

/// Generate speech audio from text using the ONNX pipeline.
///
/// Returns raw f32 PCM samples at 24 kHz.
pub fn generate(
    sessions: &mut OnnxSessions,
    tokenizer: &tokenizers::Tokenizer,
    config: &OnnxConfig,
    schedule: &DpmSchedule,
    voice: &KvCache,
    text: &str,
    cfg_scale: f64,
) -> Result<Option<Vec<f32>>> {
    let text = sanitize_text(text);
    let text_with_nl = text.trim().to_string() + "\n";

    let encoding = tokenizer
        .encode(text_with_nl.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;
    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let total_tokens = token_ids.len();

    if total_tokens == 0 {
        return Ok(None);
    }

    // Initialize KV caches from voice preset
    let mut lm_kvs = extract_kv(voice, "lm", config.text_lm_layers);
    let mut tts_kvs = extract_kv(voice, "tts_lm", config.tts_lm_layers);
    let mut neg_tts_kvs = extract_kv(voice, "neg_tts_lm", config.tts_lm_layers);

    // Negative TTS hidden state for CFG
    let neg_tts_hidden_raw = voice
        .get("neg_tts_lm_hidden")
        .context("Missing neg_tts_lm_hidden in voice preset")?;
    let neg_hidden_shape = neg_tts_hidden_raw.shape();
    let mut neg_tts_hidden = Array3::<f16>::from_shape_vec(
        (
            neg_hidden_shape[0],
            neg_hidden_shape[1],
            neg_hidden_shape[2],
        ),
        neg_tts_hidden_raw.iter().copied().collect(),
    )?;

    let mut tts_hidden: Option<Array3<f16>> = None;
    let mut all_latents: Vec<Vec<f32>> = Vec::new();
    let mut win_idx: usize = 0;
    let mut finished = false;

    let h = config.hidden_size;

    println!("Generating speech for {total_tokens} tokens...");

    while !finished {
        let start = win_idx * config.text_window_size;
        let end = ((win_idx + 1) * config.text_window_size).min(total_tokens);
        win_idx += 1;

        if start < total_tokens {
            let cur_ids: Vec<i64> = token_ids[start..end].to_vec();
            let seq_len = cur_ids.len();
            let ids_arr = Array2::from_shape_vec((1, seq_len), cur_ids)?;

            let (lm_hidden, new_lm_kvs) = run_text_lm(
                &mut sessions.text_lm,
                &ids_arr,
                &lm_kvs,
                config.text_lm_layers,
            )?;
            lm_kvs = new_lm_kvs;

            let tts_mask = Array2::from_elem((1, seq_len), 1_i64);
            let (tts_h, _eos, new_tts_kvs) = run_tts_lm(
                &mut sessions.tts_lm,
                &lm_hidden,
                &tts_mask,
                &tts_kvs,
                config.tts_lm_layers,
            )?;
            tts_kvs = new_tts_kvs;
            tts_hidden = Some(tts_h);
        }

        if tts_hidden.is_none() {
            continue;
        }

        for _si in 0..config.speech_window_size {
            // Extract last hidden state as condition [1, H]
            // Must read from tts_hidden each iteration (it's updated in the feedback loop below).
            let tts_h = tts_hidden.as_ref().unwrap();
            let last_idx = tts_h.shape()[1] - 1;
            let pos_cond_vec: Vec<f16> = (0..h).map(|j| tts_h[[0, last_idx, j]]).collect();
            let pos_cond = Array2::from_shape_vec((1, h), pos_cond_vec)?;

            let neg_last_idx = neg_tts_hidden.shape()[1] - 1;
            let neg_cond_vec: Vec<f16> = (0..h)
                .map(|j| neg_tts_hidden[[0, neg_last_idx, j]])
                .collect();
            let neg_cond = Array2::from_shape_vec((1, h), neg_cond_vec)?;

            // Sample speech latent via DPM-Solver++ with CFG
            let latent = sample_speech_token(
                &mut sessions.diffusion_head,
                &schedule.sigmas,
                &schedule.timesteps,
                &pos_cond,
                &neg_cond,
                cfg_scale,
                config.latent_dim,
            )?;
            all_latents.push(latent.clone());

            // Feedback: connector → TTS LM
            let latent_f16: Vec<f16> = latent.iter().map(|&v| f16::from_f32(v)).collect();
            let latent_3d = Array3::from_shape_vec((1, 1, config.latent_dim), latent_f16)?;
            let embed = run_connector(&mut sessions.acoustic_connector, &latent_3d)?;

            let speech_mask = Array2::from_elem((1, 1), 0_i64);

            let (new_tts_h, eos_logits, new_tts_kvs) = run_tts_lm(
                &mut sessions.tts_lm,
                &embed,
                &speech_mask,
                &tts_kvs,
                config.tts_lm_layers,
            )?;
            tts_kvs = new_tts_kvs;
            tts_hidden = Some(new_tts_h);

            // Also run negative TTS LM for CFG
            let (new_neg_h, _neg_eos, new_neg_kvs) = run_tts_lm(
                &mut sessions.tts_lm,
                &embed,
                &speech_mask,
                &neg_tts_kvs,
                config.tts_lm_layers,
            )?;
            neg_tts_kvs = new_neg_kvs;
            neg_tts_hidden = new_neg_h;

            // Check EOS
            let eos_prob = 1.0 / (1.0 + (-eos_logits as f64).exp());
            let all_text_done = win_idx * config.text_window_size >= total_tokens;
            let min_speech = total_tokens.max(6);
            if eos_prob > 0.5 && all_latents.len() >= min_speech && all_text_done {
                println!(
                    "  EOS at token {} (prob={:.3})",
                    all_latents.len(),
                    eos_prob
                );
                finished = true;
                break;
            }

            // Safety stop: too many speech tokens (runaway generation)
            if all_latents.len() > total_tokens * 3 && all_text_done {
                println!("  Safety stop at {} speech tokens", all_latents.len());
                finished = true;
                break;
            }
        }

        println!("  Window {}: {} speech tokens", win_idx, all_latents.len());
    }

    if all_latents.is_empty() {
        return Ok(None);
    }

    // Stack all latents → [1, N, latent_dim] and apply scaling
    let n_frames = all_latents.len();
    let mut latent_seq: Vec<f16> = Vec::with_capacity(n_frames * config.latent_dim);
    for lat in &all_latents {
        for &v in lat {
            let scaled =
                (v as f64 / config.speech_scaling_factor - config.speech_bias_factor) as f32;
            latent_seq.push(f16::from_f32(scaled));
        }
    }
    let latent_arr = Array3::from_shape_vec((1, n_frames, config.latent_dim), latent_seq)?;

    println!("Decoding {n_frames} frames...");
    let audio = run_vocoder(&mut sessions.vocoder, &latent_arr)?;

    Ok(Some(audio))
}
