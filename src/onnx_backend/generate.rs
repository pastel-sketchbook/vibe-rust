//! Main generation loop for speech synthesis.
//!
//! Orchestrates text windowing, speech generation, EOS detection,
//! and vocoder decoding.

use anyhow::{Context, Result};
use half::f16;
use ndarray::{Array2, Array3};
use ort::session::Session;

use crate::constants;
use crate::onnx_backend::config::{DpmSchedule, OnnxConfig};
use crate::onnx_backend::dpm_solver::sample_speech_token;
use crate::onnx_backend::sessions::OnnxSessions;
use crate::onnx_backend::voice::{KvCache, extract_kv};

/// Text LM output: hidden states + updated KV cache.
type TextLmOutput = (Array3<f16>, KvCache);

/// TTS LM output: hidden states + EOS logit + updated KV cache.
type TtsLmOutput = (Array3<f16>, f32, KvCache);

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
    let text_with_nl = text.trim().to_string() + constants::TEXT_SUFFIX_NEWLINE;

    let encoding = tokenizer
        .encode(text_with_nl.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;
    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let total_tokens = token_ids.len();

    if total_tokens == 0 {
        return Ok(None);
    }

    let mut lm_kvs = extract_kv(voice, "lm", config.text_lm_layers);
    let mut tts_kvs = extract_kv(voice, "tts_lm", config.tts_lm_layers);
    let mut neg_tts_kvs = extract_kv(voice, "neg_tts_lm", config.tts_lm_layers);

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
            let tts_h = tts_hidden.as_ref().unwrap();
            let last_idx = tts_h.shape()[1] - 1;
            let pos_cond_vec: Vec<f16> = (0..h).map(|j| tts_h[[0, last_idx, j]]).collect();
            let pos_cond = Array2::from_shape_vec((1, h), pos_cond_vec)?;

            let neg_last_idx = neg_tts_hidden.shape()[1] - 1;
            let neg_cond_vec: Vec<f16> = (0..h)
                .map(|j| neg_tts_hidden[[0, neg_last_idx, j]])
                .collect();
            let neg_cond = Array2::from_shape_vec((1, h), neg_cond_vec)?;

            let latent = sample_speech_token(
                &mut sessions.diffusion_head,
                schedule,
                &pos_cond,
                &neg_cond,
                cfg_scale,
                config.latent_dim,
            )?;
            all_latents.push(latent.clone());

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

            let (new_neg_h, _neg_eos, new_neg_kvs) = run_tts_lm(
                &mut sessions.tts_lm,
                &embed,
                &speech_mask,
                &neg_tts_kvs,
                config.tts_lm_layers,
            )?;
            neg_tts_kvs = new_neg_kvs;
            neg_tts_hidden = new_neg_h;

            let eos_prob = 1.0 / (1.0 + (-eos_logits as f64).exp());
            let all_text_done = win_idx * config.text_window_size >= total_tokens;
            let min_speech = total_tokens.max(constants::MIN_SPEECH_TOKENS);
            if eos_prob > constants::EOS_PROB_THRESHOLD
                && all_latents.len() >= min_speech
                && all_text_done
            {
                println!(
                    "  EOS at token {} (prob={:.3})",
                    all_latents.len(),
                    eos_prob
                );
                finished = true;
                break;
            }

            if all_latents.len() > total_tokens * constants::SAFETY_STOP_MULTIPLIER && all_text_done
            {
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

/// Run text LM with KV-cache.
fn run_text_lm(
    sess: &mut Session,
    input_ids: &Array2<i64>,
    past_kvs: &KvCache,
    n_layers: usize,
) -> Result<TextLmOutput> {
    let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    inputs.push((
        "input_ids".to_string(),
        ort::value::Tensor::from_array(input_ids.clone())?.into_dyn(),
    ));

    for i in 0..n_layers {
        let key = past_kvs
            .get(&format!("key_{i}"))
            .context("Missing LM key cache")?;
        let val = past_kvs
            .get(&format!("value_{i}"))
            .context("Missing LM value cache")?;

        let k4 = reshape_kv_4d(key)?;
        let v4 = reshape_kv_4d(val)?;

        inputs.push((
            format!("past_key_{i}"),
            ort::value::Tensor::from_array(k4)?.into_dyn(),
        ));
        inputs.push((
            format!("past_value_{i}"),
            ort::value::Tensor::from_array(v4)?.into_dyn(),
        ));
    }

    let outputs = sess.run(inputs)?;

    let (hidden_shape, hidden_data) = outputs[0].try_extract_tensor::<f16>()?;
    let hidden_arr = extract_array_3d(hidden_shape, hidden_data)?;

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
fn run_tts_lm(
    sess: &mut Session,
    inputs_embeds: &Array3<f16>,
    tts_text_mask: &Array2<i64>,
    past_kvs: &KvCache,
    n_layers: usize,
) -> Result<TtsLmOutput> {
    let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

    inputs.push((
        "inputs_embeds".to_string(),
        ort::value::Tensor::from_array(inputs_embeds.clone())?.into_dyn(),
    ));
    inputs.push((
        "tts_text_mask".to_string(),
        ort::value::Tensor::from_array(tts_text_mask.clone())?.into_dyn(),
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

        inputs.push((
            format!("past_key_{i}"),
            ort::value::Tensor::from_array(k4)?.into_dyn(),
        ));
        inputs.push((
            format!("past_value_{i}"),
            ort::value::Tensor::from_array(v4)?.into_dyn(),
        ));
    }

    let outputs = sess.run(inputs)?;

    let (hs_shape, hs_data) = outputs[0].try_extract_tensor::<f16>()?;
    let hidden_arr = extract_array_3d(hs_shape, hs_data)?;

    let (_eos_shape, eos_data) = outputs[1].try_extract_tensor::<f16>()?;
    let eos_val = eos_data[0].to_f32();

    let mut new_kvs = KvCache::new();
    for i in 0..n_layers {
        let (k_shape, k_data) = outputs[2 + 2 * i].try_extract_tensor::<f16>()?;
        let (v_shape, v_data) = outputs[3 + 2 * i].try_extract_tensor::<f16>()?;
        new_kvs.insert(format!("key_{i}"), extract_array_dyn(k_shape, k_data)?);
        new_kvs.insert(format!("value_{i}"), extract_array_dyn(v_shape, v_data)?);
    }

    Ok((hidden_arr, eos_val, new_kvs))
}

/// Run acoustic connector: latent → embedding for feedback loop.
fn run_connector(sess: &mut Session, latent: &Array3<f16>) -> Result<Array3<f16>> {
    let outputs = sess.run(ort::inputs![
        "latent" => ort::value::Tensor::from_array(latent.clone())?
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    extract_array_3d(shape, data)
}

/// Run vocoder: latents → audio waveform.
fn run_vocoder(sess: &mut Session, latents: &Array3<f16>) -> Result<Vec<f32>> {
    let outputs = sess.run(ort::inputs![
        "latents" => ort::value::Tensor::from_array(latents.clone())?
    ])?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f16>()?;
    Ok(data.iter().map(|v| v.to_f32()).collect())
}

/// Helper: extract a tensor output as an ndarray.
fn extract_array_3d(shape: &ort::value::Shape, data: &[f16]) -> Result<Array3<f16>> {
    anyhow::ensure!(shape.len() == 3, "Expected 3D tensor, got {}D", shape.len());
    let (d0, d1, d2) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    Ok(Array3::from_shape_vec((d0, d1, d2), data.to_vec())?)
}

fn extract_array_dyn(shape: &ort::value::Shape, data: &[f16]) -> Result<ndarray::ArrayD<f16>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&dims),
        data.to_vec(),
    )?)
}

/// Reshape a potentially flat/dynamic KV array to 4D [batch, heads, seq, dim].
fn reshape_kv_4d(arr: &ndarray::ArrayD<f16>) -> Result<ndarray::Array4<f16>> {
    match arr.ndim() {
        4 => {
            let s = arr.shape();
            Ok(ndarray::Array4::from_shape_vec(
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
